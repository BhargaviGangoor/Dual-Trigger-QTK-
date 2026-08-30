import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice
from simulator.rogue_device import RogueDevice
from qtk.inactivity_trigger import InactivityTrigger
from qtk.dual_trigger import DualTrigger, TriggerReason
from qtk.epoch_tracker import EpochTracker
from qtk.quarantine_state import ShamirSecretSharing, QuarantineManager

class TestQTK(unittest.TestCase):
    def setUp(self):
        self.inact = InactivityTrigger(delta_inact=5)
        self.dual = DualTrigger(delta_inact=5, theta_R=0.65)
        self.tracker = EpochTracker()

    def test_inactivity_trigger_silent_device(self):
        silent = SilentDevice(device_id="silent_01", owner_id="u0", initial_epoch=0)
        # Epoch 4: age = 4 < 5 -> not triggered
        self.assertFalse(self.inact.check(silent, current_epoch=4))
        # Epoch 5: age = 5 >= 5 -> triggered
        self.assertTrue(self.inact.check(silent, current_epoch=5))
        # Epoch 8: age = 8 >= 5 -> triggered
        self.assertTrue(self.inact.check(silent, current_epoch=8))

    def test_inactivity_trigger_active_rogue_evasion(self):
        rogue = RogueDevice(device_id="rogue_01", owner_id="u0", initial_epoch=0)
        # Rogue rotates keys regularly
        rogue.update_key(3)
        # Epoch 6: age = 3 < 5 -> evaded
        self.assertFalse(self.inact.check(rogue, current_epoch=6))
        rogue.update_key(6)
        # Epoch 8: age = 2 < 5 -> evaded
        self.assertFalse(self.inact.check(rogue, current_epoch=8))

    def test_dual_trigger_decision_categories(self):
        dev = LegitimateDevice(device_id="phone_01", owner_id="u0", name="Phone", initial_epoch=10)

        # 1. Compliant Case (neither trigger) -> NONE
        should_q, reason, _ = self.dual.dual_trigger_decision(dev, current_epoch=12, R_dt=0.15)
        self.assertFalse(should_q)
        self.assertEqual(reason, TriggerReason.NONE)

        # 2. Inactivity Trigger Only (age = 6 >= 5, risk = 0.2 < 0.65) -> INACTIVITY
        should_q, reason, _ = self.dual.dual_trigger_decision(dev, current_epoch=16, R_dt=0.20)
        self.assertTrue(should_q)
        self.assertEqual(reason, TriggerReason.INACTIVITY)

        # 3. Behavioral Trigger Only (age = 2 < 5, risk = 0.85 >= 0.65) -> BEHAVIORAL
        should_q, reason, _ = self.dual.dual_trigger_decision(dev, current_epoch=12, R_dt=0.85)
        self.assertTrue(should_q)
        self.assertEqual(reason, TriggerReason.BEHAVIORAL)

        # 4. Both Triggers (age = 7 >= 5, risk = 0.90 >= 0.65) -> BOTH
        should_q, reason, _ = self.dual.dual_trigger_decision(dev, current_epoch=17, R_dt=0.90)
        self.assertTrue(should_q)
        self.assertEqual(reason, TriggerReason.BOTH)

    def test_shamir_secret_sharing_math(self):
        secret = 987654321
        t = 3
        m = 5
        shares = ShamirSecretSharing.split_secret(secret, t, m)
        self.assertEqual(len(shares), 5)

        # Reconstruct with exactly t shares
        rec_exact = ShamirSecretSharing.reconstruct_secret(shares[:3])
        self.assertEqual(rec_exact, secret)

        # Reconstruct with all m shares
        rec_all = ShamirSecretSharing.reconstruct_secret(shares)
        self.assertEqual(rec_all, secret)

        # Reconstruct with different subset of t shares
        rec_subset = ShamirSecretSharing.reconstruct_secret([shares[0], shares[2], shares[4]])
        self.assertEqual(rec_subset, secret)

    def test_quarantine_manager_lifecycle(self):
        target = LegitimateDevice(device_id="target_dev", owner_id="u0")
        peer1 = LegitimateDevice(device_id="peer_1", owner_id="u0")
        peer2 = LegitimateDevice(device_id="peer_2", owner_id="u0")
        peer3 = LegitimateDevice(device_id="peer_3", owner_id="u0")

        peers = [peer1, peer2, peer3]
        q_info = QuarantineManager.quarantine_device(target, peers)
        self.assertIn("shares", q_info)
        self.assertEqual(q_info["total_shares"], 3)
        self.assertEqual(q_info["threshold"], 2)

        # Recovery with 2 peers (meets threshold t=2)
        success, secret, msg = QuarantineManager.recover_device(target, ["peer_1", "peer_3"])
        self.assertTrue(success)
        self.assertEqual(secret, q_info["secret"])
        self.assertFalse(target.is_quarantined)

if __name__ == "__main__":
    unittest.main()
