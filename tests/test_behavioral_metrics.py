import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "behavioral_metrics.py"
SPEC = importlib.util.spec_from_file_location("behavioral_metrics", MODULE_PATH)
behavioral_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = behavioral_metrics
SPEC.loader.exec_module(behavioral_metrics)


class BehavioralMetricsTests(unittest.TestCase):
    def test_detects_pose_and_frontier_revisits(self):
        rows = [
            {"agent_location": point, "selected_frontier_location": point}
            for point in ([0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0])
        ]
        result = behavioral_metrics.compute_episode_metrics(
            rows, cell_size_m=0.5, min_gap_steps=2
        )
        self.assertAlmostEqual(result.revisit_ratio, 0.4)
        self.assertEqual(result.loop_closure_count, 1)
        self.assertEqual(result.frontier_revisit_count, 1)
        self.assertAlmostEqual(result.redundant_path_ratio, 0.5)

    def test_stationary_episode_is_well_defined(self):
        rows = [{"agent_location": [0, 0, 0]}]
        result = behavioral_metrics.compute_episode_metrics(rows)
        self.assertEqual(result.path_length_m, 0.0)
        self.assertEqual(result.redundant_path_ratio, 0.0)
        self.assertIsNone(result.frontier_revisit_count)

    def test_paired_bootstrap_preserves_difference_direction(self):
        result = behavioral_metrics.paired_bootstrap_difference(
            [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], n_resamples=500, seed=7
        )
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean_difference"], -1.0)
        self.assertLess(result["ci95_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
