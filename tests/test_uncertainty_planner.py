import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from latent_planner import InverseDynamics, LatentPlannerRuntime, UncertaintyLatentPathFlow, checkpoint_payload
from uncertainty_planner import UncertaintyLatentPathSolver, UncertaintyLatentPlannerRuntime


class TinyWorldModel(torch.nn.Module):
    def encode(self, info):
        return {"emb": info["pixels"]}

    def action_encoder(self, actions):
        return torch.cat([actions, actions], dim=-1)

    def predict(self, embeddings, actions):
        return embeddings + actions


class UncertaintyPlannerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def runtime(self, target="trajectory", threshold=1.0, rounds=3):
        runtime = UncertaintyLatentPlannerRuntime(
            lewm=TinyWorldModel(),
            flow=UncertaintyLatentPathFlow(
                latent_dim=4, hidden_dim=16, depth=1, max_horizon=5, time_dim=8,
            ),
            inverse_dynamics=InverseDynamics(latent_dim=4, action_dim=2, hidden_dim=8, depth=1),
            action_block=1,
        ).eval()
        runtime.uncertainty_target = target
        runtime.configure_uncertainty(threshold, max_resample_rounds=rounds)
        return runtime

    def info(self, batch=2):
        return {"pixels": torch.randn(batch, 1, 4), "goal": torch.randn(batch, 1, 4)}

    def test_velocity_scores_average_variance_at_integration_times(self):
        runtime = self.runtime("velocity")
        times = []

        def forward(x, t, z0, zg):
            times.append(t[0].item())
            log_sigma = (1 + 2 * t[:, None, None]).log().expand_as(x) / 2
            return torch.zeros_like(x), log_sigma

        with patch.object(runtime.flow, "forward_with_uncertainty", side_effect=forward):
            with patch.object(runtime.flow, "trajectory_log_sigma", side_effect=AssertionError):
                paths, scores = runtime.sample_paths_with_uncertainty(
                    torch.zeros(2, 4), torch.ones(2, 4), horizon=3, num_samples=3, flow_steps=4,
                )
        self.assertEqual(times, [0, 0.25, 0.5, 0.75])
        torch.testing.assert_close(scores, torch.full((2, 3), 1.75))
        torch.testing.assert_close(paths[:, :, 0], torch.zeros(2, 3, 4))
        torch.testing.assert_close(paths[:, :, -1], torch.ones(2, 3, 4))

    def test_head_scores_completed_interior_variance(self):
        runtime = self.runtime()
        with torch.no_grad():
            runtime.flow.uncertainty_out[-1].bias.fill_(torch.tensor(2.0).log())
        with patch.object(runtime.flow, "forward_with_uncertainty", side_effect=AssertionError):
            paths, scores = runtime.sample_paths_with_uncertainty(
                torch.zeros(2, 4), torch.ones(2, 4), horizon=3, num_samples=3, flow_steps=2,
            )
        torch.testing.assert_close(scores, torch.full((2, 3), 4.0))
        self.assertEqual(paths.shape, (2, 3, 4, 4))

    def test_only_missing_candidates_are_replaced_per_environment(self):
        runtime = self.runtime(threshold=0.5)
        initial = torch.arange(6.0).reshape(2, 3, 1, 1).expand(2, 3, 4, 4).clone()
        proposals = [
            (initial, torch.tensor([[0.1, 0.9, 0.2], [0.9, 0.9, 0.9]])),
            (torch.full((1, 1, 4, 4), 6.0), torch.tensor([[0.3]])),
            (torch.full((1, 3, 4, 4), 7.0), torch.tensor([[float("nan"), 0.5, 0.9]])),
            (torch.full((1, 2, 4, 4), 8.0), torch.tensor([[0.1, 0.2]])),
        ]
        info = self.info()
        with patch.object(runtime, "sample_paths_with_uncertainty", side_effect=proposals) as sample:
            with patch.object(runtime, "rank_paths", wraps=runtime.rank_paths) as rank:
                out = runtime.plan(info, horizon=3, num_samples=3, flow_steps=2, score_mode="first")
        self.assertEqual([call.kwargs["num_samples"] for call in sample.call_args_list], [3, 1, 3, 2])
        torch.testing.assert_close(sample.call_args_list[1].args[0], info["pixels"][0:1, 0])
        torch.testing.assert_close(sample.call_args_list[2].args[0], info["pixels"][1:2, 0])
        self.assertEqual(rank.call_count, 1)
        ranked_paths = rank.call_args.args[2]
        torch.testing.assert_close(ranked_paths[:, :, 1, 0], torch.tensor([[0., 2., 6.], [7., 8., 8.]]))
        torch.testing.assert_close(out["all_uncertainties"], torch.tensor([[.1, .2, .3], [.5, .1, .2]]))
        self.assertEqual(out["proposal_counts"].tolist(), [4, 8])
        self.assertEqual(out["resample_rounds"].tolist(), [1, 2])
        report = runtime.uncertainty_report()
        self.assertEqual((report["accepted"], report["rejected"], report["acceptance_rate"]), (6, 6, 0.5))
        json.dumps(report, allow_nan=False)

    def test_nonfinite_scores_and_paths_never_reach_ranking(self):
        runtime = self.runtime(threshold=1.0, rounds=0)
        paths = torch.zeros(1, 3, 4, 4)
        paths[0, 2, 1, 0] = float("nan")
        with patch.object(runtime, "sample_paths_with_uncertainty", return_value=(
            paths, torch.tensor([[float("inf"), float("nan"), 0.1]])
        )):
            with patch.object(runtime, "rank_paths") as rank:
                with self.assertRaisesRegex(RuntimeError, "accepted 0/3 from 3 proposals"):
                    runtime.plan(self.info(1), horizon=3, num_samples=3, flow_steps=2)
        rank.assert_not_called()

    def test_exhaustion_stops_without_fallback(self):
        runtime = self.runtime(threshold=0.5, rounds=2)
        with patch.object(runtime, "sample_paths_with_uncertainty", return_value=(
            torch.zeros(1, 3, 4, 4), torch.ones(1, 3)
        )) as sample:
            with self.assertRaisesRegex(RuntimeError, "after 2 replacement rounds"):
                runtime.plan(self.info(1), horizon=3, num_samples=3, flow_steps=2)
        self.assertEqual(sample.call_count, 3)

    def test_all_accepted_matches_normal_sampling_and_rollout_ranking(self):
        for target in ("velocity", "trajectory"):
            with self.subTest(target=target):
                runtime = self.runtime(target)
                info = self.info()
                baseline = LatentPlannerRuntime.plan(
                    runtime, info, horizon=3, num_samples=4, flow_steps=2,
                    generator=torch.Generator().manual_seed(23),
                )
                out = runtime.plan(
                    info, horizon=3, num_samples=4, flow_steps=2,
                    generator=torch.Generator().manual_seed(23),
                )
                for key in ("actions", "paths", "costs", "all_costs", "selected_index"):
                    torch.testing.assert_close(out[key], baseline[key])
                self.assertEqual(out["proposal_counts"].tolist(), [4, 4])
                self.assertEqual(out["resample_rounds"].tolist(), [0, 0])

    def test_metadata_detection_and_invalid_options(self):
        runtime = self.runtime()
        runtime.uncertainty_target = None
        runtime.checkpoint_config = {"uncertainty": {"include_hat_term": True}}
        runtime.configure_uncertainty(1.0)
        self.assertEqual(runtime.uncertainty_source, "velocity")
        runtime.checkpoint_config = {"uncertainty": {"target": "trajectory"}}
        runtime.configure_uncertainty(1.0)
        self.assertEqual(runtime.uncertainty_source, "trajectory")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            runtime.configure_uncertainty(1.0, source="velocity")
        runtime.checkpoint_config = {}
        with self.assertRaisesRegex(ValueError, "no uncertainty target"):
            runtime.configure_uncertainty(1.0)
        runtime.configure_uncertainty(1.0, source="velocity")
        for threshold in (-1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                runtime.configure_uncertainty(threshold)
        with self.assertRaisesRegex(ValueError, "max_resample_rounds"):
            runtime.configure_uncertainty(1., max_resample_rounds=-1)

    def test_checkpoint_to_solver_batches_and_report(self):
        for target in ("velocity", "trajectory"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                runtime = self.runtime(target)
                payload = checkpoint_payload(
                    lewm_checkpoint="unused.pt", action_block=1, flow=runtime.flow,
                    inverse_dynamics=runtime.inverse_dynamics, cfg={"uncertainty": {"target": target}},
                )
                payload["uncertainty_target"] = target
                path = Path(directory) / "model.pt"
                torch.save(payload, path)
                with patch("latent_planner.load_lewm", return_value=TinyWorldModel()):
                    solver = UncertaintyLatentPathSolver(
                        checkpoint=path, uncertainty_threshold=1.0, batch_size=2,
                        num_samples=3, flow_steps=2,
                    )
                solver.configure(
                    action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32), n_envs=3,
                    config=SimpleNamespace(horizon=3, action_block=1),
                )
                result = solver(self.info(3))
                self.assertEqual(result["actions"].shape, (3, 3, 2))
                self.assertEqual(result["all_uncertainties"].shape, (3, 3))
                self.assertEqual(result["rollout_count"], 9)
                report = solver.uncertainty_report()
                self.assertEqual(report["accepted"], 9)
                self.assertEqual(report["source"], target)
                self.assertEqual([r["environment_start"] for r in report["planning_batches"]], [0, 2])
                json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
