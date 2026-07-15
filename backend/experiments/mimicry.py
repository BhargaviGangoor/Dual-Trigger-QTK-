import numpy as np
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice
from simulator.mimicry_attacker import MimicryAttacker
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

def run_simulation(attacker_type: str):
    """
    Simulates either 'naive' or 'mimicry' rogue attacker.
    Returns:
        caught (bool)
        caught_epoch (int)
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
    
    hmm_detector.train_on_profile("Student")
    
    phone = LegitimateDevice("phone", "u1", "Android Phone", "primary", "Android 14", "Student")
    laptop = LegitimateDevice("laptop", "u1", "Chrome Browser", "linked", "Windows 11", "Student")
    
    if attacker_type == 'naive':
        rogue = RogueDevice("rogue_laptop", "u1", "Chrome (Linux)", "linked", "Linux", "Student")
    else:
        rogue = MimicryAttacker("rogue_laptop", "u1", "Chrome (Linux)", "linked", "Linux", "Student")
        
    devices = [phone, laptop]
    prev_adj = None
    rogue_caught_epoch = None
    
    for epoch in range(epochs):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch
        
        if current_epoch == 10:
            devices.append(rogue)
            rogue.update_key(current_epoch)
            
        for dev in devices:
            is_active = (current_epoch % 24) >= 9
            if isinstance(dev, MimicryAttacker):
                # Pass phone's latest telemetry to mimic
                dev.simulate_epoch_action(current_epoch, is_active, phone.get_metadata())
            else:
                dev.simulate_epoch_action(current_epoch, is_active)
                
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
            
            # 3. Trust Decay
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
                triggered, _ = dual_trigger.evaluate(current_epoch, dev, R_dt)
                if triggered and dev.device_id == "rogue_laptop" and rogue_caught_epoch is None:
                    rogue_caught_epoch = current_epoch
                    
    return rogue_caught_epoch is not None, rogue_caught_epoch

def run_experiment():
    print("Running Experiment: Adversarial Mimicry Attack (RQ3)...")
    
    naive_caught, naive_epoch = run_simulation('naive')
    mimic_caught, mimic_epoch = run_simulation('mimicry')
    
    naive_latency = (naive_epoch - 10) if naive_caught else 30
    mimic_latency = (mimic_epoch - 10) if mimic_caught else 30
    
    print(f"  Naive Rogue: Caught={naive_caught}, Quarantine Epoch={naive_epoch}, Latency={naive_latency} epochs")
    print(f"  Mimicry Rogue: Caught={mimic_caught}, Quarantine Epoch={mimic_epoch}, Latency={mimic_latency} epochs")
    
    return {
        "naive_caught": naive_caught,
        "naive_latency": naive_latency,
        "mimic_caught": mimic_caught,
        "mimic_latency": mimic_latency
    }

if __name__ == "__main__":
    run_experiment()
