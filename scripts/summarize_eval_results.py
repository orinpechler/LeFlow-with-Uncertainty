#!/usr/bin/env python3
"""Aggregate per-seed JSON files produced by the LeFlow evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-epoch", type=int, choices=(1, 2))
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    return parser.parse_args()


def load_seed_result(results_dir: Path, seed: int) -> dict[str, Any]:
    path = results_dir / f"seed_{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing result for seed {seed}: {path}")
    with path.open() as f:
        result = json.load(f)
    if int(result["seed"]) != seed:
        raise ValueError(f"{path} contains seed {result['seed']}, expected {seed}")
    return result


def main() -> None:
    args = parse_args()
    results = [load_seed_result(args.results_dir, seed) for seed in args.seeds]

    checkpoints = {result["checkpoint"] for result in results}
    episode_counts = {int(result["num_episodes"]) for result in results}
    if len(checkpoints) != 1:
        raise ValueError(f"Per-seed results use different checkpoints: {checkpoints}")
    if len(episode_counts) != 1:
        raise ValueError(f"Per-seed results use different episode counts: {episode_counts}")

    success_rates = [float(result["metrics"]["success_rate"]) for result in results]
    evaluation_times = [float(result["evaluation_time_seconds"]) for result in results]
    episodes_per_seed = episode_counts.pop()

    per_seed = {
        str(seed): {
            "success_rate_percent": success_rate,
            "evaluation_time_seconds": evaluation_time,
            "episode_successes": result["metrics"].get("episode_successes"),
        }
        for seed, success_rate, evaluation_time, result in zip(
            args.seeds, success_rates, evaluation_times, results
        )
    }
    summary = {
        "checkpoint": checkpoints.pop(),
        "checkpoint_epoch": args.checkpoint_epoch,
        "seeds": args.seeds,
        "num_seeds": len(args.seeds),
        "episodes_per_seed": episodes_per_seed,
        "total_episodes": episodes_per_seed * len(args.seeds),
        "per_seed": per_seed,
        "success_rate_percent": {
            "mean": mean(success_rates),
            "std": population_std(success_rates),
            "std_ddof": 0,
        },
        "evaluation_time_seconds": {
            "mean": mean(evaluation_times),
            "std": population_std(evaluation_times),
            "total": sum(evaluation_times),
            "std_ddof": 0,
        },
    }

    sampling = [result.get("uncertainty_sampling") for result in results]
    if any(row is not None for row in sampling):
        if not all(row is not None for row in sampling):
            raise ValueError("Cannot mix ordinary and uncertainty-filtered evaluation results.")
        setting_keys = ("source", "score_definition", "threshold", "max_resample_rounds", "num_samples")
        settings = {key: sampling[0][key] for key in setting_keys}
        if any(any(row[key] != value for key, value in settings.items()) for row in sampling):
            raise ValueError("Per-seed results use different uncertainty sampling settings.")
        totals = {key: sum(row[key] for row in sampling) for key in ("proposed", "accepted", "rejected")}
        summary["uncertainty_sampling"] = {
            **settings, **totals,
            "acceptance_rate": totals["accepted"] / totals["proposed"] if totals["proposed"] else None,
        }

    json_path = args.results_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    success_summary = summary["success_rate_percent"]
    timing_summary = summary["evaluation_time_seconds"]
    lines = [
        "LeFlow Reacher evaluation summary",
        f"checkpoint: {summary['checkpoint']}",
        f"checkpoint epoch: {args.checkpoint_epoch}",
        f"seeds: {', '.join(map(str, args.seeds))}",
        f"episodes per seed: {episodes_per_seed}",
        f"total episodes: {summary['total_episodes']}",
        "",
        "seed  success_rate_percent  evaluation_time_seconds",
    ]
    lines.extend(
        f"{seed:<5} {success_rate:>20.2f}  {evaluation_time:>23.2f}"
        for seed, success_rate, evaluation_time in zip(
            args.seeds, success_rates, evaluation_times
        )
    )
    lines.extend(
        [
            "",
            f"success rate: {success_summary['mean']:.2f} +/- "
            f"{success_summary['std']:.2f}% (population std across seeds)",
            f"evaluation time: {timing_summary['mean']:.2f} +/- "
            f"{timing_summary['std']:.2f} seconds per seed",
            f"total evaluation time: {timing_summary['total']:.2f} seconds",
        ]
    )
    if "uncertainty_sampling" in summary:
        lines.extend(["", "uncertainty sampling: " + json.dumps(summary["uncertainty_sampling"])])
    (args.results_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
