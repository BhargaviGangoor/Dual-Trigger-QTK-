import unittest
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice
from simulator.rogue_device import RogueDevice
from simulator.mimicry_attacker import MimicryAttacker
from simulator.irregular_legitimate import IrregularLegitimateDevice
from simulator.telemetry_generator import TelemetryGenerator

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_legitimate_device_key_updates(self):
        phone = LegitimateDevice(device_id="phone_01", owner_id="user_0", name="Pixel Phone")
        self.assertEqual(phone.device_id, "phone_01")
        self.assertEqual(phone.epoch_last_key_update, 0)

        # Simulate 5 epochs; legitimate device should update key before delta_inact = 5
        for ep in range(1, 6):
            obs = phone.simulate_epoch(ep, rng=self.rng)
            self.assertEqual(obs["ground_truth_label"], 0)
            self.assertIn("context_telemetry", obs)
            self.assertIn("protocol_telemetry", obs)

        # Key age should stay <= 4
        self.assertLess(5 - phone.epoch_last_key_update, 5)

    def test_silent_device_inactivity(self):
        silent = SilentDevice(device_id="silent_01", owner_id="user_0", name="Dormant Tablet")
        self.assertEqual(silent.epoch_last_key_update, 0)

        for ep in range(1, 10):
            obs = silent.simulate_epoch(ep, rng=self.rng)
            self.assertEqual(obs["ground_truth_label"], 0)

        # Key update epoch should NOT have changed
        self.assertEqual(silent.epoch_last_key_update, 0)
        self.assertEqual(silent.get_key_age(9), 9)

    def test_rogue_device_key_update_evasion(self):
        rogue = RogueDevice(device_id="rogue_01", owner_id="user_0", name="Rogue Terminal")
        self.assertEqual(rogue.epoch_last_key_update, 0)

        for ep in range(1, 15):
            obs = rogue.simulate_epoch(ep, rng=self.rng)
            self.assertEqual(obs["ground_truth_label"], 1)
            # Rogue keeps key age strictly below 5
            self.assertLessEqual(rogue.get_key_age(ep), 3)

    def test_mimicry_attacker(self):
        mimic = MimicryAttacker(device_id="mimic_01", owner_id="user_0", mimicry_strength="strong_mimicry")
        obs = mimic.simulate_epoch(1, rng=self.rng)
        self.assertEqual(obs["ground_truth_label"], 1)
        self.assertIn("features", obs)
        self.assertEqual(len(obs["features"]), 5)

    def test_irregular_legitimate_device(self):
        traveler = IrregularLegitimateDevice(device_id="travel_01", owner_id="user_0")
        for ep in range(1, 10):
            obs = traveler.simulate_epoch(ep, rng=self.rng)
            self.assertEqual(obs["ground_truth_label"], 0)

if __name__ == "__main__":
    unittest.main()
