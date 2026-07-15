import numpy as np
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

def evaluate_thresholds(delta_inact: int, theta_R: float):
    """
    Evaluates a single threshold pair configuration.
    Returns:
        rogue_caught (bool)
        rogue_latency (int)
        false_positives (int)
    """
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
    rogue = RogueDevice("rogue_laptop", "u1", "Chrome (Linux)", "linked", "Linux", "Student")
    
    devices = [phone, laptop]
    prev_adj = None
    
    rogue_caught_epoch = None
    false_positives = 0
    
    for epoch in range(epochs):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch
        
        if current_epoch == 10:
            devices.append(rogue)
            rogue.update_key(current_epoch)
            
        for dev in devices:
            is_active = (current_epoch % 24) >= 9
            dev.simulate_epoch_action(current_epoch, is_active)
            
        if all(len(d.telemetry_history) >= 2 for d in devices):
            histories = [d.telemetry_history for d in devices]
            
            # HMM
            hmm_states = []
            hmm_confidences = []
            for dev in devices:
                state, conf = hmm_detector.evaluate_device(dev.telemetry_history)
                hmm_states.append(state)
                hmm_confidences.append(conf)
                
            # Graph-LSTM
            adj, rel_scores = graph_lstm.evaluate_devices(histories, prev_adj)
            prev_adj = adj
            
            # Trust Decay
            for i, dev in enumerate(devices):
                evidence = 1.0
                if hmm_states[i] in [1, 2]:
                    evidence -= 0.3 * hmm_confidences[i]
                if rel_scores[i] > 0.4:
                    evidence -= 0.5 * rel_scores[i]
                    
                evidence = max(0.0, min(1.0, evidence))
                dev_alpha = TrustScore.get_dynamic_alpha(dev.device_type, dev.network_type, alpha)
                dev.trust_score = TrustScore.calculate_decay(dev.trust_score, evidence, dev_alpha)
                
                # Risk Fusion
                R_dt = fusion.fuse(hmm_states[i], hmm_confidences[i], rel_scores[i], dev.trust_score)
                
                # Evaluate triggers
                triggered, _ = dual_trigger.evaluate(current_epoch, dev, R_dt)
                if triggered:
                    if dev.device_id == "rogue_laptop":
                        if rogue_caught_epoch is None:
                            rogue_caught_epoch = current_epoch
                    else:
                        false_positives += 1
                        
    latency = (rogue_caught_epoch - 10) if rogue_caught_epoch else 30
    caught = rogue_caught_epoch is not None
    
    return caught, latency, false_positives

def run_experiment():
    print("Running Experiment: Sensitivity Analysis Sweep (RQ5)...")
    
    delta_sweeps = [3, 5, 7]
    theta_sweeps = [0.5, 0.65, 0.8]
    
    results = []
    print(f"| delta_inact | theta_R | Rogue Caught | Latency (epochs) | False Positives |")
    print(f"|-------------|---------|--------------|------------------|-----------------|")
    for d in delta_sweeps:
        for t in theta_sweeps:
            caught, latency, fps = evaluate_thresholds(d, t)
            results.append({
                "delta_inact": d,
                "theta_R": t,
                "caught": caught,
                "latency": latency,
                "false_positives": fps
            })
            print(f"| {d:<11} | {t:<7.2f} | {str(caught):<12} | {latency:<16} | {fps:<15} |")
            
    return results

if __name__ == "__main__":
    run_experiment()
