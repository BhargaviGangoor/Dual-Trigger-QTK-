import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_classification_metrics, calculate_qtk_system_metrics
from evaluation.confidence import compute_confidence_intervals, aggregate_run_metrics

class TestMetrics(unittest.TestCase):
    def test_classification_metrics(self):
        # 10 TP, 2 FP, 0 FN, 18 TN
        metrics = calculate_classification_metrics(tp=10, fp=2, fn=0, tn=18)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["detection_rate"], 1.0)
        self.assertEqual(metrics["precision"], round(10/12, 4))
        self.assertEqual(metrics["false_positive_rate"], 0.1)
        self.assertEqual(metrics["accuracy"], round(28/30, 4))
        self.assertAlmostEqual(metrics["f1_score"], round(2*(10/12)*1.0/((10/12)+1.0), 4), places=3)

    def test_qtk_system_metrics(self):
        # Injection at epoch 10, quarantined at epoch 12
        sys_m = calculate_qtk_system_metrics(
            quarantined_rogue_epochs=[12],
            injection_epoch=10,
            total_epochs=30,
            false_quarantined_legit_count=0,
            total_legitimate_devices=3
        )
        self.assertEqual(sys_m["avg_detection_latency"], 2.0)
        self.assertEqual(sys_m["avg_evasion_duration"], 2.0)
        self.assertEqual(sys_m["false_quarantine_rate"], 0.0)

    def test_confidence_intervals(self):
        vals = [10.0, 12.0, 11.0, 9.0, 13.0]
        ci = compute_confidence_intervals(vals)
        self.assertEqual(ci["mean"], 11.0)
        self.assertGreater(ci["std"], 0.0)
        self.assertGreater(ci["ci_margin"], 0.0)
        self.assertLess(ci["ci_lower"], ci["mean"])
        self.assertGreater(ci["ci_upper"], ci["mean"])

if __name__ == "__main__":
    unittest.main()
