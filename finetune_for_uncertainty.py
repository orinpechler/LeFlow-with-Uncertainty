from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from latent_planner import (
    InverseDynamics,
    UncertaintyLatentPathFlow,
    checkpoint_payload,
    load_lewm,
    stablewm_cache_dir,
    trajectory_uncertainty_loss,
    uncertainty_flow_matching_loss,
)
from train_latent_planner import (
    encode_latents,
    freeze,
    init_wandb,
    make_loaders,
    metric_float,
)


class FlowEMA:
    """EMA used by the official UA-Flow fine-tuning setup."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        # UA-Flow corrects the decay during the first updates instead of using
        # the full 0.9999 value immediately.
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - decay)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        parameter.copy_(backup[name])


def load_pretrained_planner(
    checkpoint_path: str | Path,
    *,
    uncertainty_cfg: DictConfig,
    device: torch.device,
) -> tuple[Path, dict[str, Any], UncertaintyLatentPathFlow, InverseDynamics]:
    """Load a vanilla LeFlow checkpoint and add a zero-initialized sigma head."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained LeFlow checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict checkpoint at {path}, got {type(payload)}")
    required = {
        "lewm_checkpoint",
        "action_block",
        "arch",
        "flow_state_dict",
        "inverse_dynamics_state_dict",
    }
    missing_payload = required.difference(payload)
    if missing_payload:
        raise KeyError(
            f"Checkpoint {path} is missing required keys: {sorted(missing_payload)}"
        )

    arch = payload["arch"]
    if arch.get("flow_type", "standard") != "standard":
        raise ValueError(
            f"Expected a vanilla LeFlow checkpoint, found flow_type={arch.get('flow_type')!r}"
        )

    flow = UncertaintyLatentPathFlow(
        **arch["flow"],
        min_log_sigma=float(uncertainty_cfg.min_log_sigma),
        max_log_sigma=float(uncertainty_cfg.max_log_sigma),
        log_sigma_init=float(uncertainty_cfg.log_sigma_init),
        uncertainty_hidden_dim=uncertainty_cfg.get("hidden_dim"),
        uncertainty_depth=int(uncertainty_cfg.get("depth", 1)),
        uncertainty_dropout=float(uncertainty_cfg.get("dropout", 0.0)),
    )
    incompatible = flow.load_state_dict(payload["flow_state_dict"], strict=False)
    expected_missing = {
        f"uncertainty_out.{name}" for name in flow.uncertainty_out.state_dict()
    }
    actual_missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if actual_missing != expected_missing or unexpected:
        raise RuntimeError(
            "The checkpoint architecture is incompatible with the uncertainty flow. "
            f"Expected only {sorted(expected_missing)} to be new; "
            f"missing={sorted(actual_missing)}, unexpected={sorted(unexpected)}"
        )

    inverse_dynamics = InverseDynamics(**arch["inverse_dynamics"])
    inverse_dynamics.load_state_dict(
        payload["inverse_dynamics_state_dict"], strict=True
    )
    flow = flow.to(device)
    inverse_dynamics = freeze(inverse_dynamics.to(device))
    return path, payload, flow, inverse_dynamics


def validate_source_configuration(payload: dict[str, Any], cfg: DictConfig) -> None:
    source_cfg = payload.get("config", {})
    source_planner = source_cfg.get("planner", {}) if isinstance(source_cfg, dict) else {}
    checks = {
        "horizon": int(cfg.planner.horizon),
        "action_block": int(cfg.planner.action_block),
        "max_horizon": int(cfg.planner.max_horizon),
    }
    for name, configured in checks.items():
        if name in source_planner and int(source_planner[name]) != configured:
            raise ValueError(
                f"planner.{name}={configured} does not match the pretrained "
                f"checkpoint value {source_planner[name]}"
            )
    if int(payload["action_block"]) != int(cfg.planner.action_block):
        raise ValueError(
            f"planner.action_block={cfg.planner.action_block} does not match "
            f"checkpoint action_block={payload['action_block']}"
        )


def step_batch(
    *,
    batch: dict,
    lewm: torch.nn.Module,
    flow: UncertaintyLatentPathFlow,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    z_path = encode_latents(lewm, batch, device)
    target = str(cfg.uncertainty.get("target", "velocity"))
    if target == "trajectory":
        loss, uncertainty_metrics = trajectory_uncertainty_loss(
            flow,
            z_path,
            beta=float(cfg.uncertainty.beta),
            flow_steps=int(cfg.uncertainty.flow_steps),
        )
    elif target == "velocity":
        loss, uncertainty_metrics = uncertainty_flow_matching_loss(
            flow,
            z_path,
            beta=float(cfg.uncertainty.beta),
            correction_weight=(
                float(cfg.uncertainty.correction_weight)
                if cfg.uncertainty.include_hat_term
                else 0.0
            ),
            density_eps=float(cfg.uncertainty.density_eps),
        )
    else:
        raise ValueError(
            f"Unknown uncertainty.target={target!r}; expected trajectory or velocity"
        )
    return {"loss": loss, **uncertainty_metrics}


@torch.no_grad()
def validate(
    *,
    loader,
    lewm: torch.nn.Module,
    flow: UncertaintyLatentPathFlow,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    flow.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        out = step_batch(
            batch=batch,
            lewm=lewm,
            flow=flow,
            cfg=cfg,
            device=device,
        )
        batch_size = batch["pixels"].size(0)
        count += batch_size
        for key, value in out.items():
            sums[key] = sums.get(key, 0.0) + metric_float(value) * batch_size
        if cfg.val_batches is not None and batch_idx + 1 >= cfg.val_batches:
            break
    flow.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def make_optimizer(
    flow: UncertaintyLatentPathFlow,
    cfg: DictConfig,
    *,
    head_only: bool = False,
) -> torch.optim.Optimizer:
    sigma_parameters = list(flow.uncertainty_out.parameters())
    if head_only:
        return torch.optim.AdamW(
            [
                {
                    "params": sigma_parameters,
                    "lr": float(cfg.optimizer.lr),
                    "name": "log_sigma_head",
                }
            ],
            betas=tuple(float(x) for x in cfg.optimizer.betas),
            weight_decay=float(cfg.optimizer.weight_decay),
        )

    sigma_ids = {id(parameter) for parameter in sigma_parameters}
    backbone_parameters = [
        parameter
        for parameter in flow.parameters()
        if parameter.requires_grad and id(parameter) not in sigma_ids
    ]
    head_lr = float(cfg.optimizer.lr)
    backbone_lr = head_lr * float(cfg.optimizer.backbone_lr_scale)
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "name": "mean_backbone",
            },
            {
                "params": sigma_parameters,
                "lr": head_lr,
                "name": "log_sigma_head",
            },
        ],
        betas=tuple(float(x) for x in cfg.optimizer.betas),
        weight_decay=float(cfg.optimizer.weight_decay),
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: DictConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(cfg.optimizer.warmup_steps)
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    start_factor = min(
        1.0,
        max(float(cfg.optimizer.warmup_start_lr) / float(cfg.optimizer.lr), 1e-8),
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    constant = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=1
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, constant],
        milestones=[warmup_steps],
    )


def train(cfg: DictConfig, *, head_only: bool) -> None:
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    source_path, source_payload, flow, inverse_dynamics = load_pretrained_planner(
        cfg.pretrained_checkpoint,
        uncertainty_cfg=cfg.uncertainty,
        device=device,
    )
    validate_source_configuration(source_payload, cfg)
    if head_only:
        freeze(flow)
        flow.uncertainty_out.requires_grad_(True)
    with open_dict(cfg):
        cfg.lewm_checkpoint = str(source_payload["lewm_checkpoint"])

    print(f"pretrained_checkpoint={source_path}", flush=True)
    print(
        "checkpoint_load=ok "
        f"flow={'frozen' if head_only else 'trainable'} "
        "inverse_dynamics=frozen log_sigma_head=zero_initialized",
        flush=True,
    )

    train_loader, val_loader = make_loaders(cfg)
    lewm = freeze(load_lewm(cfg.lewm_checkpoint).to(device))

    optimizer = make_optimizer(flow, cfg, head_only=head_only)
    scheduler = make_scheduler(optimizer, cfg)
    ema = FlowEMA(flow, decay=float(cfg.ema.decay)) if cfg.ema.enabled else None

    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_filename = (
        "train_uncertainty_head_config.yaml"
        if head_only
        else "finetune_for_uncertainty_config.yaml"
    )
    OmegaConf.save(cfg, run_dir / config_filename)
    wandb_run = init_wandb(cfg, run_dir)

    trainable_parameters = sum(
        parameter.numel() for parameter in flow.parameters() if parameter.requires_grad
    )
    all_parameters = sum(parameter.numel() for parameter in flow.parameters())
    group_lrs = " ".join(
        f"{group['name']}_initial_lr={group['lr']:.3e}"
        for group in optimizer.param_groups
    )
    print(
        f"optimizer {group_lrs} warmup_steps={cfg.optimizer.warmup_steps} "
        f"ema={bool(ema)} trainable_parameters={trainable_parameters} "
        f"total_parameters={all_parameters}",
        flush=True,
    )

    global_step = 0
    try:
        for epoch in range(int(cfg.epochs)):
            print(f"epoch={epoch + 1} train_start", flush=True)
            if head_only:
                flow.eval()
                flow.uncertainty_out.train()
            else:
                flow.train()
            for batch_idx, batch in enumerate(train_loader):
                out = step_batch(
                    batch=batch,
                    lewm=lewm,
                    flow=flow,
                    cfg=cfg,
                    device=device,
                )
                if not torch.isfinite(out["loss"]):
                    raise ValueError(
                        f"Non-finite uncertainty loss at step {global_step + 1}: "
                        f"{metric_float(out['loss'])}"
                    )

                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    flow.parameters(), float(cfg.grad_clip_norm)
                )
                optimizer.step()
                if ema is not None:
                    ema.update(flow)
                scheduler.step()
                global_step += 1

                if global_step % int(cfg.log_interval) == 0:
                    metrics = " ".join(
                        f"{key}={metric_float(value):.4f}"
                        for key, value in out.items()
                    )
                    print(
                        f"epoch={epoch + 1} step={global_step} {metrics} "
                        + " ".join(
                            f"{group['name']}_lr={group['lr']:.3e}"
                            for group in optimizer.param_groups
                        ),
                        flush=True,
                    )

                if wandb_run is not None:
                    log_data = {
                        f"train/{key}": metric_float(value)
                        for key, value in out.items()
                    }
                    log_data.update(
                        {
                            "train/epoch": epoch + 1,
                            "train/grad_norm": metric_float(grad_norm),
                        }
                    )
                    log_data.update(
                        {
                            f"train/{group['name']}_lr": group["lr"]
                            for group in optimizer.param_groups
                        }
                    )
                    wandb_run.log(log_data, step=global_step)

                if (
                    cfg.max_train_batches is not None
                    and batch_idx + 1 >= int(cfg.max_train_batches)
                ):
                    break

            print(f"epoch={epoch + 1} validation_start", flush=True)
            if ema is None:
                val_metrics = validate(
                    loader=val_loader,
                    lewm=lewm,
                    flow=flow,
                    cfg=cfg,
                    device=device,
                )
            else:
                with ema.average_parameters(flow):
                    val_metrics = validate(
                        loader=val_loader,
                        lewm=lewm,
                        flow=flow,
                        cfg=cfg,
                        device=device,
                    )
            print(
                f"epoch={epoch + 1} validation "
                + " ".join(
                    f"{key}={value:.4f}" for key, value in val_metrics.items()
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"val/{key}": value for key, value in val_metrics.items()
                    }
                    | {"val/epoch": epoch + 1},
                    step=global_step,
                )

            ema_context = (
                ema.average_parameters(flow) if ema is not None else nullcontext()
            )
            with ema_context:
                payload = checkpoint_payload(
                    lewm_checkpoint=str(cfg.lewm_checkpoint),
                    action_block=int(source_payload["action_block"]),
                    flow=flow,
                    inverse_dynamics=inverse_dynamics,
                    cfg=OmegaConf.to_container(cfg, resolve=True),
                )
                # state_dict() tensors share storage with the module. Clone
                # them before restoring online weights when leaving the EMA
                # context so the serialized planner really contains EMA.
                payload["flow_state_dict"] = {
                    key: value.detach().clone()
                    for key, value in payload["flow_state_dict"].items()
                }
            payload.update(
                {
                    "source_checkpoint": str(source_path),
                    "source_flow_type": source_payload["arch"].get(
                        "flow_type", "standard"
                    ),
                    "training_mode": (
                        "trajectory_uncertainty_head"
                        if head_only
                        else "full_flow_finetune"
                    ),
                    "uncertainty_target": str(
                        cfg.uncertainty.get("target", "velocity")
                    ),
                    "finetune_epoch": epoch + 1,
                    "global_step": global_step,
                    "ema_decay": float(cfg.ema.decay) if ema is not None else None,
                    "ema_num_updates": ema.num_updates if ema is not None else 0,
                    "flow_weights": "ema" if ema is not None else "online",
                }
            )
            epoch_path = run_dir / f"{cfg.output_model_name}_epoch_{epoch + 1}.pt"
            latest_path = run_dir / f"{cfg.output_model_name}.pt"
            print(
                f"epoch={epoch + 1} checkpoint_start path={epoch_path}", flush=True
            )
            torch.save(payload, epoch_path)
            torch.save(payload, latest_path)
            print(
                f"epoch={epoch + 1} checkpoint_done path={latest_path}", flush=True
            )
            if wandb_run is not None and cfg.wandb.log_model:
                wandb_run.save(str(latest_path), base_path=str(run_dir))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="finetune_for_uncertainty",
)
def run(cfg: DictConfig) -> None:
    train(cfg, head_only=False)


if __name__ == "__main__":
    run()
