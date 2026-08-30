import os
import sys
import random
from typing import Dict, Any, Tuple

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice
from simulator.rogue_device import RogueDevice
from simulator.irregular_legitimate import IrregularLegitimateDevice
from qtk.inactivity_trigger import InactivityTrigger
from qtk.dual_trigger import DualTrigger, TriggerReason
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

def run_smoke_test(seed: int = 42) -> bool:
    """
    End-to-End Smoke Test verifying the 5 critical invariant conditions:
    1. Silent device -> INACTIVITY quarantine
    2. Active rogue below inactivity threshold -> Baseline QTK misses it
    3. Active rogue with high behavioral risk -> Dual-Trigger QTK detects it
    4. Legitimate device -> No quarantine
    5. Irregular legitimate device -> No automatic quarantine
    """
    print("==================================================")
    print("        DUAL-TRIGGER QTK END-TO-END SMOKE TEST    ")
    print("==================================================")

    rng = random.Random(seed)
    inact_trigger = InactivityTrigger(delta_inact=5)
    dual_trigger = DualTrigger(delta_inact=5, theta_R=0.65)

    hmm = HMMDetector()
    graph_lstm = GraphLSTM(beta=0.8, seed=seed)
    fusion = RiskFusion()

    # Pre-train HMM on training traces if available
    train_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "train.jsonl"
    )
    if os.path.exists(train_path):
        import json
        train_records = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_records.append(json.loads(line))
        hmm.fit_from_dataset(train_records)

    # -------------------------------------------------------------
    # Test Case 1: Silent Device -> INACTIVITY Quarantine
    # -------------------------------------------------------------
    silent_dev = SilentDevice(device_id="silent_01", owner_id="user_0", initial_epoch=0)
    for ep in range(1, 7):
        silent_dev.simulate_epoch(ep, rng=rng)

    t1_inact = inact_trigger.check(silent_dev, current_epoch=6)
    t1_dual, t1_reason, _ = dual_trigger.dual_trigger_decision(silent_dev, current_epoch=6)
    pass_1 = (t1_inact is True) and (t1_dual is True) and (t1_reason == TriggerReason.INACTIVITY)
    print(f"Condition 1 [Silent Device -> INACTIVITY]:                 {'[PASS]' if pass_1 else '[FAIL]'} (Reason: {t1_reason.value})")

    # -------------------------------------------------------------
    # Test Case 2: Active Rogue Evasion -> Baseline QTK Misses
    # -------------------------------------------------------------
    rogue_dev = RogueDevice(device_id="rogue_01", owner_id="user_0", initial_epoch=0, attack_mode="stealth_burst")
    for ep in range(1, 15):
        rogue_dev.simulate_epoch(ep, rng=rng)

    t2_inact = inact_trigger.check(rogue_dev, current_epoch=14)
    pass_2 = (t2_inact is False)
    print(f"Condition 2 [Active Rogue -> Baseline QTK Misses]:         {'[PASS]' if pass_2 else '[FAIL]'} (Inact Fired: {t2_inact}, Key Age: {14 - rogue_dev.epoch_last_key_update})")

    # -------------------------------------------------------------
    # Test Case 3: Active Rogue -> Dual-Trigger QTK Detects
    # -------------------------------------------------------------
    phone = LegitimateDevice(device_id="phone_01", owner_id="user_0", initial_epoch=0)
    for ep in range(1, 15):
        p_meta = phone.get_latest_telemetry()
        phone.simulate_epoch(ep, rng=rng)

    # Evaluate ML models on rogue
    hmm.predict(rogue_dev)
    TrustScore.update(rogue_dev, rogue_dev.behavioral_risk, alpha=0.8)
    _, scores = graph_lstm.evaluate_devices([phone.telemetry_history, rogue_dev.telemetry_history])
    rogue_dev.graph_risk = scores[1]
    fusion.predict(rogue_dev)

    t3_dual, t3_reason, detail = dual_trigger.dual_trigger_decision(rogue_dev, current_epoch=14)
    pass_3 = (t3_dual is True) and (t3_reason in [TriggerReason.BEHAVIORAL, TriggerReason.BOTH])
    print(f"Condition 3 [Active Rogue -> Dual-Trigger QTK Detects]:   {'[PASS]' if pass_3 else '[FAIL]'} (Reason: {t3_reason.value}, Risk: {rogue_dev.final_risk:.3f})")

    # -------------------------------------------------------------
    # Test Case 4: Legitimate Device -> No Quarantine
    # -------------------------------------------------------------
    legit_phone = LegitimateDevice(device_id="phone_clean", owner_id="user_0", initial_epoch=0)
    for ep in range(1, 10):
        legit_phone.simulate_epoch(ep, rng=rng)
    hmm.predict(legit_phone)
    TrustScore.update(legit_phone, legit_phone.behavioral_risk, alpha=0.8)
    legit_phone.graph_risk = 0.05
    fusion.predict(legit_phone)

    t4_dual, t4_reason, _ = dual_trigger.dual_trigger_decision(legit_phone, current_epoch=9)
    pass_4 = (t4_dual is False) and (t4_reason == TriggerReason.NONE)
    print(f"Condition 4 [Legitimate Device -> No Quarantine]:          {'[PASS]' if pass_4 else '[FAIL]'} (Reason: {t4_reason.value}, Risk: {legit_phone.final_risk:.3f})")

    # -------------------------------------------------------------
    # Test Case 5: Irregular Legitimate (Traveler) -> No Auto Quarantine
    # -------------------------------------------------------------
    travel_dev = IrregularLegitimateDevice(device_id="travel_01", owner_id="user_0", initial_epoch=0)
    for ep in range(1, 6):
        travel_dev.simulate_epoch(ep, rng=rng)
    hmm.predict(travel_dev)
    TrustScore.update(travel_dev, travel_dev.behavioral_risk, alpha=0.8)
    travel_dev.graph_risk = 0.15
    fusion.predict(travel_dev)

    t5_dual, t5_reason, _ = dual_trigger.dual_trigger_decision(travel_dev, current_epoch=5)
    pass_5 = (t5_dual is False) and (t5_reason == TriggerReason.NONE)
    print(f"Condition 5 [Irregular Traveler -> No Auto Quarantine]:    {'[PASS]' if pass_5 else '[FAIL]'} (Reason: {t5_reason.value}, Risk: {travel_dev.final_risk:.3f})")

    print("==================================================")
    all_passed = pass_1 and pass_2 and pass_3 and pass_4 and pass_5
    if all_passed:
        print("ALL 5 SMOKE TEST INVARIANTS PASSED! READY FOR EXPERIMENTS.")
    else:
        print("SMOKE TEST INVARIANT FAILURE DETECTED!")
    print("==================================================")
    return all_passed

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
