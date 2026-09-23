import copy
import unittest

import torch

from latent_planner import (
    InverseDynamics,
    LatentPlannerRuntime,
    LatentPathFlow,
    UncertaintyLatentPathFlow,
    checkpoint_payload,
    trajectory_uncertainty_loss,
    uncertainty_flow_matching_loss,
)


class UncertaintyHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.flow_kwargs = {
            "latent_dim": 6,
            "hidden_dim": 16,
            "depth": 1,
            "max_horizon": 5,
            "time_dim": 8,
            "dropout": 0.0,
        }

    def make_frozen_flow(self):
        pretrained = LatentPathFlow(**self.flow_kwargs)
        flow = UncertaintyLatentPathFlow(
            **self.flow_kwargs,
            uncertainty_hidden_dim=12,
            uncertainty_depth=2,
        )
        incompatible = flow.load_state_dict(pretrained.state_dict(), strict=False)
        expected_missing = {
            f"uncertainty_out.{key}" for key in flow.uncertainty_out.state_dict()
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])
        flow.eval().requires_grad_(False)
        flow.uncertainty_out.train().requires_grad_(True)
        return flow

    def test_head_step_preserves_pretrained_flow(self):
        flow = self.make_frozen_flow()
        frozen_before = {
            name: parameter.detach().clone()
            for name, parameter in flow.named_parameters()
            if not name.startswith("uncertainty_out.")
        }
        optimizer = torch.optim.AdamW(flow.uncertainty_out.parameters(), lr=1e-3)

        z_path = torch.randn(4, 5, self.flow_kwargs["latent_dim"])
        loss, metrics = trajectory_uncertainty_loss(
            flow,
            z_path,
            flow_steps=2,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(parameter.grad is not None for parameter in flow.uncertainty_out.parameters())
        )
        for name, parameter in flow.named_parameters():
            if name in frozen_before:
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(parameter, frozen_before[name]))

    def test_completed_path_variance_covers_interior_tokens(self):
        flow = self.make_frozen_flow()
        inverse_dynamics = InverseDynamics(
            latent_dim=self.flow_kwargs["latent_dim"],
            action_dim=2,
            hidden_dim=8,
            depth=1,
        )
        runtime = LatentPlannerRuntime(
            lewm=torch.nn.Identity(),
            flow=flow,
            inverse_dynamics=inverse_dynamics,
        )
        paths = torch.randn(2, 3, 5, self.flow_kwargs["latent_dim"])

        variance = runtime.trajectory_variance(paths)

        self.assertEqual(variance.shape, paths.shape)
        self.assertTrue(torch.equal(variance[:, :, 0], torch.zeros_like(variance[:, :, 0])))
        self.assertTrue(torch.equal(variance[:, :, -1], torch.zeros_like(variance[:, :, -1])))
        self.assertTrue(torch.all(variance[:, :, 1:-1] > 0))

    def test_uncertainty_checkpoint_round_trip(self):
        flow = self.make_frozen_flow()
        inverse_dynamics = InverseDynamics(
            latent_dim=self.flow_kwargs["latent_dim"],
            action_dim=2,
            hidden_dim=8,
            depth=1,
        ).eval().requires_grad_(False)
        inverse_before = copy.deepcopy(inverse_dynamics.state_dict())

        payload = checkpoint_payload(
            lewm_checkpoint="lewm.pt",
            action_block=1,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            cfg={},
        )
        restored = UncertaintyLatentPathFlow(**payload["arch"]["flow"])
        restored.load_state_dict(payload["flow_state_dict"], strict=True)

        self.assertEqual(payload["arch"]["flow_type"], "uncertainty")
        self.assertEqual(restored.uncertainty_depth, 2)
        for key, value in inverse_dynamics.state_dict().items():
            self.assertTrue(torch.equal(value, inverse_before[key]))

    def test_rejects_invalid_uncertainty_settings(self):
        with self.assertRaisesRegex(ValueError, "min_log_sigma"):
            UncertaintyLatentPathFlow(
                **self.flow_kwargs,
                min_log_sigma=1.0,
                max_log_sigma=1.0,
            )
        flow = self.make_frozen_flow()
        with self.assertRaisesRegex(ValueError, "density_eps"):
            uncertainty_flow_matching_loss(
                flow,
                torch.randn(2, 5, self.flow_kwargs["latent_dim"]),
                density_eps=0.0,
            )


if __name__ == "__main__":
    unittest.main()
