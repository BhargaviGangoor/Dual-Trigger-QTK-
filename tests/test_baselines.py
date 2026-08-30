import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.qtk_baseline import QTKBaseline
from baselines.threshold_detector import ThresholdDetector
from baselines.hmm_baseline import HMMBaseline
from baselines.isolation_forest import IsolationForestBaseline
from baselines.lstm_baseline import LSTMBaseline
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice
from simulator.silent_device import SilentDevice

class TestBaselines(unittest.TestCase):
    def test_qtk_baseline(self):
        base = QTKBaseline(delta_inact=5)
        phone = LegitimateDevice(device_id="phone_01", owner_id="u0", initial_epoch=0)
        silent = SilentDevice(device_id="silent_01", owner_id="u0", initial_epoch=0)
        rogue = RogueDevice(device_id="rogue_01", owner_id="u0", initial_epoch=0)

        # At epoch 6:
        # Phone rotated at epoch 3 -> age 3 < 5
        phone.update_key(3)
        self.assertFalse(base.evaluate_device(phone, current_epoch=6))

        # Silent never rotated -> age 6 >= 5 -> quarantined
        self.assertTrue(base.evaluate_device(silent, current_epoch=6))

        # Rogue rotated at epoch 4 -> age 2 < 5 -> evades
        rogue.update_key(4)
        self.assertFalse(base.evaluate_device(rogue, current_epoch=6))

    def test_threshold_detector(self):
        detector = ThresholdDetector()
        normal_ctx = {"sync_frequency": 4.0, "session_duration_sec": 120.0, "is_vpn": 0.0}
        self.assertLess(detector.evaluate_telemetry(normal_ctx), 0.50)

        vpn_ctx = {"sync_frequency": 18.0, "session_duration_sec": 600.0, "is_vpn": 1.0}
        self.assertGreaterEqual(detector.evaluate_telemetry(vpn_ctx), 0.50)

    def test_isolation_forest_baseline(self):
        iforest = IsolationForestBaseline()
        iforest.fit_on_normal([])
        score_norm = iforest.evaluate_telemetry({"session_duration_sec": 120.0, "sync_frequency": 4.0, "message_count_sent": 5})
        self.assertGreaterEqual(score_norm, 0.0)
        self.assertLessEqual(score_norm, 1.0)

    def test_lstm_baseline(self):
        lstm_base = LSTMBaseline()
        records = [
            {"session_duration_sec": 120.0, "sync_frequency": 4.0, "message_count_sent": 5} for _ in range(10)
        ]
        score = lstm_base.evaluate_history(records)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
