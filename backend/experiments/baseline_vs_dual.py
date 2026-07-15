import numpy as np
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

def run_experiment():
    print("Running Experiment: QTK Baseline vs. Dual-Trigger QTK (RQ1)...")
    
    # Configuration
    delta_inact = 5
    theta_R = 0.65
    alpha = 0.8
    seq_len = 12
    epochs = 40
    
    # Initialize trackers, triggers, and detectors
    tracker = EpochTracker()
    inact_trigger = InactivityTrigger(delta_inact)
    dual_trigger = DualTrigger(delta_inact, theta_R)
    
    hmm_detector = HMMDetector()
    graph_lstm = GraphLSTM(beta=0.8)
    fusion = RiskFusion()
    
    # Train HMM on standard user profile
    hmm_detector.train_on_profile("Student")
    
    # Create devices
    # User has Phone (Primary) and Laptop (Linked)
    phone = LegitimateDevice("phone", "u1", "Android Phone", "primary", "Android 14", "Student")
    laptop = LegitimateDevice("laptop", "u1", "Chrome Browser", "linked", "Windows 11", "Student")
    
    # Rogue device is paired at epoch 10
    rogue = RogueDevice("rogue_laptop", "u1", "Chrome (Linux)", "linked", "Linux", "Student")
    
    devices = [phone, laptop]
    
    # Tracking results
    baseline_quarantined_epoch = None
    dual_quarantined_epoch = None
    
    # Store history for model evaluations
    # For GNN/LSTM, we need lists of historical telemetry per device
    # Format: [device1_history, device2_history, ...]
    
    prev_adj = None
    
    for epoch in range(epochs):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch
        
        # Inject rogue device at epoch 10
        if current_epoch == 10:
            devices.append(rogue)
            # Sync key update epoch initially
            rogue.update_key(current_epoch)
            
        # Simulate actions
        for dev in devices:
            # Active hour check (student active in afternoon/evening)
            is_active = (current_epoch % 24) >= 9
            dev.simulate_epoch_action(current_epoch, is_active)
            
        # Standard QTK check (Baseline)
        for dev in devices:
            if dev.device_id == "rogue_laptop" and not dev.is_quarantined:
                if inact_trigger.evaluate(current_epoch, dev):
                    baseline_quarantined_epoch = current_epoch
                    
        # Dual-Trigger QTK Evaluation
        # We need at least 2 telemetry records per device to run the HMM and Graph-LSTM
        if all(len(d.telemetry_history) >= 2 for d in devices):
            # Extract metadata histories
            histories = [d.telemetry_history for d in devices]
            
            # 1. Individual HMM score
            hmm_states = []
            hmm_confidences = []
            for dev in devices:
                state, conf = hmm_detector.evaluate_device(dev.telemetry_history)
                hmm_states.append(state)
                hmm_confidences.append(conf)
                
            # 2. Relational anomaly score (Graph-LSTM)
            adj, rel_scores = graph_lstm.evaluate_devices(histories, prev_adj)
            prev_adj = adj
            
            # 3. Trust Score Decay/Recovery
            # Evidence score is derived from anomaly scores: higher anomalies = lower evidence score
            for i, dev in enumerate(devices):
                # Penalty based on HMM state (Compromised/Ghost) and GNN anomalies
                evidence = 1.0
                if hmm_states[i] in [1, 2]:
                    evidence -= 0.3 * hmm_confidences[i]
                if rel_scores[i] > 0.4:
                    evidence -= 0.5 * rel_scores[i]
                    
                evidence = max(0.0, min(1.0, evidence))
                
                # Apply decay
                dev_alpha = TrustScore.get_dynamic_alpha(dev.device_type, dev.network_type, alpha)
                dev.trust_score = TrustScore.calculate_decay(dev.trust_score, evidence, dev_alpha)
                
                # Update FSM State
                if dev.trust_score < 0.2:
                    dev.current_trust_state = "Revoked"
                elif dev.trust_score < 0.5:
                    dev.current_trust_state = "Suspicious"
                
                # 4. Risk Fusion Layer
                R_dt = fusion.fuse(hmm_states[i], hmm_confidences[i], rel_scores[i], dev.trust_score)
                
                # Evaluate Dual-Trigger
                triggered, reason = dual_trigger.evaluate(current_epoch, dev, R_dt)
                if triggered and dev.device_id == "rogue_laptop" and dual_quarantined_epoch is None:
                    dual_quarantined_epoch = current_epoch
                    
    # Results trace
    print("\nSimulation Finished!")
    print(f"Rogue Device injection epoch: 10")
    print(f"Rogue Device quarantined under Plain QTK (Baseline): {baseline_quarantined_epoch if baseline_quarantined_epoch else 'Never quarantined (Evasion duration: 30+ epochs)'}")
    print(f"Rogue Device quarantined under Dual-Trigger QTK: {dual_quarantined_epoch if dual_quarantined_epoch else 'Never quarantined'}")
    
    evasion_baseline = epochs - 10 if baseline_quarantined_epoch is None else baseline_quarantined_epoch - 10
    evasion_dual = epochs - 10 if dual_quarantined_epoch is None else dual_quarantined_epoch - 10
    
    print(f"Evasion Duration (Baseline): {evasion_baseline} epochs")
    print(f"Evasion Duration (Dual-Trigger): {evasion_dual} epochs")
    
    return {
        "baseline_quarantined_epoch": baseline_quarantined_epoch,
        "dual_quarantined_epoch": dual_quarantined_epoch,
        "evasion_duration_baseline": evasion_baseline,
        "evasion_duration_dual": evasion_dual
    }

if __name__ == "__main__":
    run_experiment()
