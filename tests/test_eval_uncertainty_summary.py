import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import summarize_eval_results


class EvaluationSummaryTest(unittest.TestCase):
    def write_results(self, directory, with_uncertainty):
        for seed in (42, 43):
            result = {
                "seed": seed, "checkpoint": "/tmp/model.pt", "num_episodes": 50,
                "metrics": {"success_rate": 60 + 2 * (seed - 42)},
                "evaluation_time_seconds": 100,
            }
            if with_uncertainty:
                result["uncertainty_sampling"] = {
                    "source": "velocity", "score_definition": "mean_velocity_variance",
                    "threshold": 0.5, "max_resample_rounds": 100, "num_samples": 64,
                    "proposed": 128, "accepted": 64, "rejected": 64,
                    "planning_batches": [],
                }
            (directory / f"seed_{seed}.json").write_text(json.dumps(result))

    def summarize(self, directory, epoch=None):
        args = SimpleNamespace(results_dir=directory, checkpoint_epoch=epoch, seeds=[42, 43])
        with patch.object(summarize_eval_results, "parse_args", return_value=args):
            summarize_eval_results.main()
        return json.loads((directory / "summary.json").read_text())

    def test_uncertainty_counts_and_settings_are_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_results(directory, True)
            summary = self.summarize(directory)
            self.assertEqual(summary["success_rate_percent"]["mean"], 61)
            sampling = summary["uncertainty_sampling"]
            self.assertEqual(sampling["proposed"], 256)
            self.assertEqual(sampling["accepted"], 128)
            self.assertEqual(sampling["rejected"], 128)
            self.assertEqual(sampling["acceptance_rate"], 0.5)
            self.assertIn('"threshold": 0.5', (directory / "summary.txt").read_text())

    def test_original_evaluation_results_still_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_results(directory, False)
            summary = self.summarize(directory, epoch=2)
            self.assertEqual(summary["checkpoint_epoch"], 2)
            self.assertNotIn("uncertainty_sampling", summary)

    def test_mismatched_thresholds_are_not_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_results(directory, True)
            path = directory / "seed_43.json"
            result = json.loads(path.read_text())
            result["uncertainty_sampling"]["threshold"] = 2.0
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "different uncertainty sampling"):
                self.summarize(directory)


if __name__ == "__main__":
    unittest.main()
