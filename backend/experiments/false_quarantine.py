import numpy as np
from simulator.legitimate_device import LegitimateDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

def run_scenario(profile_name: str, has_travel: bool = False, idle_duration_epochs: int = 0):
    """
    Simulates legitimate user devices under different behavioral stress conditions:
    - Normal (Student profile, regular activity)
    - Travel (Traveler profile, shifting timezone and country)
    - Idle (long periods of silence on a secondary device)
    """
    delta_inact = 5
    theta_R = 0.65
    alpha = 0.8
    epochs = 40
    
    tracker = EpochTracker()
    dual_trigger = DualTrigger(delta_inact, theta_R)
    hmm_detector = HMMDetector()
    graph_lstm = GraphLSTM(beta=0.8)
    fusion = RiskFusion()
    
    hmm_detector.train_on_profile(profile_name)
    
    phone = LegitimateDevice("phone", "u1", "Phone", "primary", "Android 14", profile_name)
    laptop = LegitimateDevice("laptop", "u1", "Laptop", "linked", "Windows 11", profile_name)
    
    devices = [phone, laptop]
    prev_adj = None
    
    quarantined_count = 0
    
    for epoch in range(epochs):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch
        
        # Simulate actions
        for dev in devices:
            is_active = (current_epoch % 24) >= 9
            
            # Idle logic: laptop goes idle for the specified period
            if dev.device_id == "laptop" and idle_duration_epochs > 0:
                if 10 <= current_epoch < (10 + idle_duration_epochs):
                    # Inactive epoch (no action, no key update)
                    continue
                    
            dev.simulate_epoch_action(current_epoch, is_active)
            
            # Inject traveling IP/Country shifts manually if travel is enabled
            if has_travel and dev.telemetry_history:
                latest = dev.telemetry_history[-1]
                if current_epoch % 10 == 0:
                    latest["network_ip"] = f"82.102.15.{current_epoch}"
                    latest["location_country"] = "France" if current_epoch % 20 == 0 else "Germany"
                    latest["active_timezone"] = "Europe/Paris" if current_epoch % 20 == 0 else "Europe/Berlin"
                    
        # Model updates
        if all(len(d.telemetry_history) >= 2 for d in devices):
            histories = [d.telemetry_history for d in devices]
            
            # 1. HMM
            hmm_states = []
            hmm_confidences = []
            for dev in devices:
                state, conf = hmm_detector.evaluate_device(dev.telemetry_history)
                hmm_states.append(state)
                hmm_confidences.append(conf)
                
            # 2. Graph-LSTM
            adj, rel_scores = graph_lstm.evaluate_devices(histories, prev_adj)
            prev_adj = adj
            
            # 3. Trust decay
            for i, dev in enumerate(devices):
                evidence = 1.0
                if hmm_states[i] in [1, 2]:
                    evidence -= 0.3 * hmm_confidences[i]
                if rel_scores[i] > 0.4:
                    evidence -= 0.5 * rel_scores[i]
                    
                evidence = max(0.0, min(1.0, evidence))
                dev_alpha = TrustScore.get_dynamic_alpha(dev.device_type, dev.network_type, alpha)
                dev.trust_score = TrustScore.calculate_decay(dev.trust_score, evidence, dev_alpha)
                
                # 4. Risk Fusion
                R_dt = fusion.fuse(hmm_states[i], hmm_confidences[i], rel_scores[i], dev.trust_score)
                
                # Evaluate triggers
                triggered, reason = dual_trigger.evaluate(current_epoch, dev, R_dt)
                if triggered:
                    quarantined_count += 1
                    
    return quarantined_count

def run_experiment():
    print("Running Experiment: False Quarantine / Availability Analysis (RQ4)...")
    
    # Scenario A: Normal Student
    normal_q = run_scenario("Student")
    # Scenario B: Traveler with regular IP changes
    traveler_q = run_scenario("Traveler", has_travel=True)
    # Scenario C: Student with linked device going idle for 4 epochs (below delta_inact=5)
    idle_q = run_scenario("Student", idle_duration_epochs=4)
    # Scenario D: Student with linked device going idle for 8 epochs (above delta_inact=5, so standard QTK will quarantine it)
    long_idle_q = run_scenario("Student", idle_duration_epochs=8)
    
    print(f"  Scenario A (Normal Legitimate): Quarantined Devices={normal_q}")
    print(f"  Scenario B (Legitimate Travel): Quarantined Devices={traveler_q}")
    print(f"  Scenario C (Short Idle < delta): Quarantined Devices={idle_q}")
    print(f"  Scenario D (Long Idle >= delta): Quarantined Devices={long_idle_q} (Expected via inactivity check)")
    
    return {
        "normal_false_quarantines": normal_q,
        "travel_false_quarantines": traveler_q,
        "short_idle_false_quarantines": idle_q,
        "long_idle_false_quarantines": long_idle_q
    }

if __name__ == "__main__":
    run_experiment()
