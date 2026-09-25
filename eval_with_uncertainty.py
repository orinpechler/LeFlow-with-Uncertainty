import os

os.environ["MUJOCO_GL"] = "egl"

import json
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def load_heldout_episodes(cfg: DictConfig) -> set[int] | None:
    split_cfg = cfg.get("episode_split")
    if not split_cfg or not split_cfg.get("enabled", False):
        return None
    if not split_cfg.get("split_file"):
        raise ValueError("episode_split.enabled=true requires episode_split.split_file for eval.")

    split_file = Path(split_cfg.split_file)
    with split_file.open("r") as f:
        payload = json.load(f)

    mode = split_cfg.get("mode", "eval")
    key = "eval_episodes" if mode == "eval" else "train_episodes"
    episodes = {int(x) for x in payload[key]}
    if not episodes:
        raise ValueError(f"No {key} found in episode split file {split_file}.")
    print(
        f"episode_split enabled mode={mode} split_file={split_file} "
        f"episodes={len(episodes)}",
        flush=True,
    )
    return episodes


def jsonable(value: Any) -> Any:
    """Convert evaluation outputs to standard JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


@hydra.main(version_base=None, config_path="./config/eval", config_name="reacher")
def run(cfg: DictConfig):
    """Evaluate after replacing high-uncertainty candidates before normal ranking."""
    if cfg.solver.get("_target_") != "uncertainty_planner.UncertaintyLatentPathSolver":
        raise ValueError("Use solver=latent_flow_uncertainty with eval_with_uncertainty.py.")
    if cfg.get("policy", "random") == "random":
        raise ValueError("Set policy to the trained uncertainty checkpoint path.")
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    heldout_episodes = load_heldout_episodes(cfg)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        config = swm.PlanConfig(**cfg.plan_config)
        solver_target = cfg.solver.get("_target_", "")
        if solver_target in {
            "latent_planner.LearnedLatentPathSolver",
            "uncertainty_planner.UncertaintyLatentPathSolver",
            "action_flow_planner.LearnedActionFlowSolver",
        }:
            solver = hydra.utils.instantiate(cfg.solver)
        else:
            model = swm.policy.AutoCostModel(cfg.policy)
            model = model.to("cuda")
            model = model.eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    default_results_dir = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )
    video_dir = cfg.output.get("video_dir", "checkpoint")
    if video_dir is None:
        video_path = None
    elif video_dir == "checkpoint":
        video_path = default_results_dir
    else:
        video_path = Path(video_dir)

    # sample the episodes and the starting indices
    episode_positions = (
        sorted(heldout_episodes)
        if heldout_episodes is not None
        else range(len(dataset.lengths))
    )
    invalid_positions = [
        ep for ep in episode_positions if ep < 0 or ep >= len(dataset.lengths)
    ]
    if invalid_positions:
        raise ValueError(
            f"Episode split contains IDs outside the dataset: {invalid_positions[:10]}"
        )
    valid_ranges = [
        np.arange(
            int(dataset.offsets[ep]),
            int(dataset.offsets[ep])
            + max(int(dataset.lengths[ep]) - cfg.eval.goal_offset_steps, 0),
        )
        for ep in episode_positions
    ]
    valid_indices = np.concatenate(valid_ranges)
    print(len(valid_indices), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=video_path,
    )
    end_time = time.time()

    print(metrics)
    uncertainty_report = solver.uncertainty_report()
    uncertainty_summary = {
        key: value for key, value in uncertainty_report.items() if key != "planning_batches"
    }
    print("Uncertainty sampling:", uncertainty_summary)

    results_path = default_results_dir / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        f.write(f"uncertainty_sampling: {json.dumps(uncertainty_summary)}\n")

    json_filename = cfg.output.get("json_filename")
    if json_filename:
        json_path = Path(json_filename)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": int(cfg.seed),
            "checkpoint": str(cfg.policy),
            "num_episodes": int(cfg.eval.num_eval),
            "metrics": jsonable(metrics),
            "evaluation_time_seconds": end_time - start_time,
            "uncertainty_sampling": uncertainty_report,
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    run()

