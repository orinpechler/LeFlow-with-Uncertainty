from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from latent_planner import (
    InverseDynamics,
    UncertaintyLatentPathFlow,
    checkpoint_payload,
    inverse_dynamics_loss,
    lewm_consistency_loss,
    load_lewm,
    smoothness_loss,
    stablewm_cache_dir,
    uncertainty_flow_matching_loss,
)
from train_latent_planner import (
    encode_latents,
    freeze,
    init_wandb,
    make_loaders,
    metric_float,
)


def step_batch(
    *,
    batch: dict,
    lewm: torch.nn.Module,
    flow: UncertaintyLatentPathFlow,
    inverse_dynamics: InverseDynamics,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch["action"] = torch.nan_to_num(batch["action"].to(device), 0.0)
    z_path = encode_latents(lewm, batch, device)
    actions = batch["action"][:, : cfg.planner.horizon]

    loss_flow, uncertainty_metrics = uncertainty_flow_matching_loss(
        flow,
        z_path,
        beta=cfg.uncertainty.beta,
        correction_weight=cfg.uncertainty.correction_weight,
        density_eps=cfg.uncertainty.density_eps,
    )
    loss_inv, pred_actions = inverse_dynamics_loss(
        inverse_dynamics, z_path, actions
    )

    if cfg.loss.consistency.weight:
        if cfg.loss.consistency.detach_inverse:
            with torch.no_grad():
                loss_dyn = lewm_consistency_loss(
                    lewm,
                    z_path,
                    pred_actions.detach(),
                    history_size=cfg.lewm_history_size,
                )
        else:
            loss_dyn = lewm_consistency_loss(
                lewm,
                z_path,
                pred_actions,
                history_size=cfg.lewm_history_size,
            )
    else:
        loss_dyn = z_path.new_tensor(0.0)
    loss_smooth = smoothness_loss(z_path)
    total = (
        cfg.loss.flow.weight * loss_flow
        + cfg.loss.inverse.weight * loss_inv
        + cfg.loss.consistency.weight * loss_dyn
        + cfg.loss.smoothness.weight * loss_smooth
    )
    return {
        "loss": total,
        "flow_loss": loss_flow.detach(),
        "inverse_loss": loss_inv.detach(),
        "consistency_loss": loss_dyn.detach(),
        "smoothness_loss": loss_smooth.detach(),
        **uncertainty_metrics,
    }


@torch.no_grad()
def validate(
    *,
    loader,
    lewm: torch.nn.Module,
    flow: UncertaintyLatentPathFlow,
    inverse_dynamics: InverseDynamics,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    flow.eval()
    inverse_dynamics.eval()
    sums: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        out = step_batch(
            batch=batch,
            lewm=lewm,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            cfg=cfg,
            device=device,
        )
        batch_size = batch["pixels"].size(0)
        count += batch_size
        for key, value in out.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
        if cfg.val_batches is not None and i + 1 >= cfg.val_batches:
            break
    flow.train()
    inverse_dynamics.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="latent_uncert_planner",
)
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(cfg)
    lewm = freeze(load_lewm(cfg.lewm_checkpoint).to(device))

    first_batch = next(iter(train_loader))
    z = encode_latents(lewm, first_batch, device)
    latent_dim = z.size(-1)
    action_dim = first_batch["action"].size(-1)

    flow = UncertaintyLatentPathFlow(
        latent_dim=latent_dim,
        max_horizon=cfg.planner.max_horizon,
        min_log_variance=cfg.uncertainty.min_log_variance,
        max_log_variance=cfg.uncertainty.max_log_variance,
        variance_init=cfg.uncertainty.variance_init,
        **cfg.flow,
    ).to(device)
    inverse_dynamics = InverseDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim,
        **cfg.inverse_dynamics,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(flow.parameters()) + list(inverse_dynamics.parameters()),
        **cfg.optimizer,
    )

    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "latent_uncert_planner_config.yaml")
    wandb_run = init_wandb(cfg, run_dir)

    global_step = 0
    try:
        for epoch in range(cfg.epochs):
            print(f"epoch={epoch + 1} train_start", flush=True)
            flow.train()
            inverse_dynamics.train()
            for batch_idx, batch in enumerate(train_loader):
                out = step_batch(
                    batch=batch,
                    lewm=lewm,
                    flow=flow,
                    inverse_dynamics=inverse_dynamics,
                    cfg=cfg,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                grad_norm = None
                if cfg.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(flow.parameters())
                        + list(inverse_dynamics.parameters()),
                        cfg.grad_clip_norm,
                    )
                optimizer.step()
                global_step += 1

                if global_step % cfg.log_interval == 0:
                    metrics = " ".join(
                        f"{key}={metric_float(value):.4f}"
                        for key, value in out.items()
                    )
                    print(
                        f"epoch={epoch + 1} step={global_step} {metrics}",
                        flush=True,
                    )

                if wandb_run is not None:
                    log_data = {
                        f"train/{key}": metric_float(value)
                        for key, value in out.items()
                    }
                    log_data["train/epoch"] = epoch + 1
                    log_data["train/lr"] = optimizer.param_groups[0]["lr"]
                    if grad_norm is not None:
                        log_data["train/grad_norm"] = float(grad_norm)
                    wandb_run.log(log_data, step=global_step)

                if (
                    cfg.max_train_batches is not None
                    and batch_idx + 1 >= cfg.max_train_batches
                ):
                    break

            print(f"epoch={epoch + 1} validation_start", flush=True)
            val_metrics = validate(
                loader=val_loader,
                lewm=lewm,
                flow=flow,
                inverse_dynamics=inverse_dynamics,
                cfg=cfg,
                device=device,
            )
            print(
                f"epoch={epoch + 1} validation "
                + " ".join(
                    f"{key}={value:.4f}"
                    for key, value in val_metrics.items()
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"val/{key}": value
                        for key, value in val_metrics.items()
                    }
                    | {"val/epoch": epoch + 1},
                    step=global_step,
                )

            payload = checkpoint_payload(
                lewm_checkpoint=str(cfg.lewm_checkpoint),
                action_block=cfg.planner.action_block,
                flow=flow,
                inverse_dynamics=inverse_dynamics,
                cfg=OmegaConf.to_container(cfg, resolve=True),
            )
            epoch_path = (
                run_dir / f"{cfg.output_model_name}_epoch_{epoch + 1}.pt"
            )
            latest_path = run_dir / f"{cfg.output_model_name}.pt"
            print(
                f"epoch={epoch + 1} checkpoint_start path={epoch_path}",
                flush=True,
            )
            torch.save(payload, epoch_path)
            torch.save(payload, latest_path)
            print(
                f"epoch={epoch + 1} checkpoint_done path={latest_path}",
                flush=True,
            )
            if wandb_run is not None and cfg.wandb.log_model:
                wandb_run.save(str(latest_path), base_path=str(run_dir))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    run()
