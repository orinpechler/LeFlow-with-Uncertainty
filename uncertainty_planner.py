"""Thresholded rejection sampling for both latent-planner uncertainty targets."""

from __future__ import annotations

import math
from typing import Any

import torch

from latent_planner import (
    LatentPlannerRuntime,
    LearnedLatentPathSolver,
    UncertaintyLatentPathFlow,
)


class UncertaintyLatentPlannerRuntime(LatentPlannerRuntime):
    def configure_uncertainty(
        self, threshold: float, source: str = "auto", max_resample_rounds: int = 100
    ) -> None:
        if not isinstance(self.flow, UncertaintyLatentPathFlow):
            raise ValueError("Uncertainty evaluation requires an uncertainty checkpoint.")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("uncertainty_threshold must be finite and nonnegative.")
        if max_resample_rounds < 0:
            raise ValueError("max_resample_rounds must be nonnegative.")
        if source not in {"auto", "velocity", "trajectory"}:
            raise ValueError("uncertainty_source must be auto, velocity, or trajectory.")

        config = getattr(self, "checkpoint_config", {}) or {}
        uncertainty_config = config.get("uncertainty", {}) or {}
        target = getattr(self, "uncertainty_target", None) or uncertainty_config.get("target")
        if target is None:
            mode = getattr(self, "training_mode", None)
            if mode == "trajectory_uncertainty_head":
                target = "trajectory"
            elif mode == "full_flow_finetune" or "include_hat_term" in uncertainty_config:
                # Earlier velocity-finetuning checkpoints predate the target field.
                target = "velocity"
        if target is not None and target not in {"velocity", "trajectory"}:
            raise ValueError(f"Unsupported checkpoint uncertainty target: {target!r}")
        if source == "auto":
            if target is None:
                raise ValueError(
                    "Checkpoint has no uncertainty target metadata; set "
                    "uncertainty_source=velocity or trajectory to match its training."
                )
            source = target
        elif target is not None and source != target:
            raise ValueError(f"uncertainty_source={source} conflicts with checkpoint target={target}.")

        self.uncertainty_threshold = float(threshold)
        self.uncertainty_source = source
        self.max_resample_rounds = max_resample_rounds
        self.uncertainty_trace: list[dict] = []

    @torch.no_grad()
    def sample_paths_with_uncertainty(
        self, z_start, z_goal, *, horizon, num_samples, flow_steps, generator=None
    ):
        if horizon < 2 or num_samples < 1 or flow_steps < 1:
            raise ValueError("Require horizon >= 2, num_samples >= 1, and flow_steps >= 1.")
        if self.uncertainty_source == "trajectory":
            paths = self.sample_paths(
                z_start, z_goal, horizon=horizon, num_samples=num_samples,
                flow_steps=flow_steps, generator=generator,
            )
            variance = self.trajectory_variance(paths)
            return paths, variance[:, :, 1:-1].mean(dim=(-1, -2))

        batch, dim = z_start.shape
        count = batch * num_samples
        z0 = z_start[:, None].expand(batch, num_samples, dim).reshape(count, dim)
        zg = z_goal[:, None].expand(batch, num_samples, dim).reshape(count, dim)
        x = torch.randn(
            count, horizon - 1, dim, device=z_start.device,
            dtype=z_start.dtype, generator=generator,
        )
        variance_sum = torch.zeros(count, device=x.device, dtype=x.dtype)
        dt = 1.0 / flow_steps
        for step in range(flow_steps):
            t = torch.full((count,), step * dt, device=x.device, dtype=x.dtype)
            velocity, log_sigma = self.flow.forward_with_uncertainty(x, t, z0, zg)
            variance_sum += torch.exp(2.0 * log_sigma).mean(dim=(-1, -2))
            x = x + velocity * dt
        paths = torch.cat([z0[:, None], x, zg[:, None]], dim=1)
        # Time-averaged velocity variance, not propagated final-path variance.
        return (
            paths.reshape(batch, num_samples, horizon + 1, dim),
            (variance_sum / flow_steps).reshape(batch, num_samples),
        )

    @torch.no_grad()
    def plan(
        self, info_dict, *, horizon, num_samples, flow_steps,
        score_mode="rollout_goal", goal_weight=1.0, consistency_weight=0.0,
        smoothness_weight=0.0, history_size=3, generator=None,
    ):
        z_start, z_goal = self.encode_current_and_goal(info_dict)
        initial_paths, initial_scores = self.sample_paths_with_uncertainty(
            z_start, z_goal, horizon=horizon, num_samples=num_samples,
            flow_steps=flow_steps, generator=generator,
        )
        accepted_paths, accepted_scores = [], []
        proposal_counts, resample_rounds = [], []
        for env in range(z_start.size(0)):
            paths, scores = initial_paths[env], initial_scores[env]
            kept_paths, kept_scores = [], []
            accepted, proposed, rounds = 0, 0, 0
            while True:
                approved = (
                    torch.isfinite(scores) & (scores <= self.uncertainty_threshold)
                    & torch.isfinite(paths).all(dim=(-1, -2))
                )
                kept_paths.append(paths[approved])
                kept_scores.append(scores[approved])
                accepted += int(approved.sum().item())
                proposed += scores.numel()
                missing = num_samples - accepted
                if missing == 0:
                    break
                if rounds >= self.max_resample_rounds:
                    finite = scores[torch.isfinite(scores)]
                    score_range = (
                        f"{finite.min().item():.6g}..{finite.max().item():.6g}"
                        if finite.numel() else "no finite scores"
                    )
                    raise RuntimeError(
                        f"Uncertainty rejection sampling exhausted for batch environment {env}: "
                        f"accepted {accepted}/{num_samples} from {proposed} proposals after "
                        f"{rounds} replacement rounds; source={self.uncertainty_source}, "
                        f"threshold={self.uncertainty_threshold}, last score range={score_range}. "
                        "Choose a higher calibrated threshold or increase max_resample_rounds."
                    )
                rounds += 1
                replacement_paths, replacement_scores = self.sample_paths_with_uncertainty(
                    z_start[env:env + 1], z_goal[env:env + 1], horizon=horizon,
                    num_samples=missing, flow_steps=flow_steps, generator=generator,
                )
                paths, scores = replacement_paths[0], replacement_scores[0]
            accepted_paths.append(torch.cat(kept_paths))
            accepted_scores.append(torch.cat(kept_scores))
            proposal_counts.append(proposed)
            resample_rounds.append(rounds)

        paths = torch.stack(accepted_paths)
        scores = torch.stack(accepted_scores)
        result = self.rank_paths(
            z_start, z_goal, paths, score_mode=score_mode, goal_weight=goal_weight,
            consistency_weight=consistency_weight, smoothness_weight=smoothness_weight,
            history_size=history_size, uncertainty_scores=scores,
        )
        result["proposal_counts"] = torch.tensor(proposal_counts)
        result["resample_rounds"] = torch.tensor(resample_rounds)
        self.uncertainty_trace.append({
            "candidate_uncertainties": result["all_uncertainties"].tolist(),
            "selected_index": result["selected_index"].tolist(),
            "selected_uncertainty": result["uncertainty"].tolist(),
            "proposal_counts": proposal_counts,
            "resample_rounds": resample_rounds,
        })
        return result

    def uncertainty_report(self) -> dict:
        proposed = sum(sum(row["proposal_counts"]) for row in self.uncertainty_trace)
        accepted = sum(
            len(scores) for row in self.uncertainty_trace
            for scores in row["candidate_uncertainties"]
        )
        return {
            "source": self.uncertainty_source,
            "score_definition": (
                "mean_velocity_variance_over_flow_steps_and_interior_elements"
                if self.uncertainty_source == "velocity"
                else "mean_completed_path_variance_over_interior_elements"
            ),
            "threshold": self.uncertainty_threshold,
            "max_resample_rounds": self.max_resample_rounds,
            "proposed": proposed, "accepted": accepted, "rejected": proposed - accepted,
            "acceptance_rate": accepted / proposed if proposed else None,
            "planning_batches": self.uncertainty_trace,
        }


class UncertaintyLatentPathSolver(LearnedLatentPathSolver):
    runtime_class = UncertaintyLatentPlannerRuntime

    def __init__(
        self, checkpoint, uncertainty_threshold: float, uncertainty_source: str = "auto",
        max_resample_rounds: int = 100, **kwargs: Any,
    ):
        super().__init__(checkpoint=checkpoint, **kwargs)
        if self.batch_size < 1 or self.num_samples < 1 or self.flow_steps < 1:
            raise ValueError("batch_size, num_samples, and flow_steps must be positive.")
        self.model.configure_uncertainty(
            uncertainty_threshold, uncertainty_source, max_resample_rounds
        )

    def configure(self, **kwargs):
        super().configure(**kwargs)
        if self.horizon < 2:
            raise ValueError("Uncertainty filtering requires horizon >= 2 (an interior token).")

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action
        total_envs = len(next(iter(info_dict.values())))
        outputs = []
        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            out = self.model.plan(
                {key: value[start:end] for key, value in info_dict.items()},
                horizon=self.horizon, num_samples=self.num_samples, flow_steps=self.flow_steps,
                score_mode=self.score_mode, goal_weight=self.goal_weight,
                consistency_weight=self.consistency_weight,
                smoothness_weight=self.smoothness_weight, history_size=self.history_size,
                generator=self.torch_gen,
            )
            self.model.uncertainty_trace[-1].update({
                "environment_start": start, "environment_end": end,
            })
            outputs.append(out)
        keys = (
            "actions", "costs", "goal_costs", "uncertainty", "all_uncertainties",
            "selected_index", "proposal_counts", "resample_rounds",
        )
        result = {key: torch.cat([out[key] for out in outputs]) for key in keys}
        result["costs"] = result["costs"].tolist()
        result["rollout_count"] = self.model.rollout_count
        return result

    def uncertainty_report(self) -> dict:
        return {"num_samples": self.num_samples, **self.model.uncertainty_report()}
