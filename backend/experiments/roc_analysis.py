import os
import sys
import json
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.device import Device, DeviceType
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice, AttackStrategy
from qtk.epoch_tracker import EpochTracker
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

def run_experiment() -> Dict[str, Any]:
    print("==================================================")
    print("Running Experiment: ROC Analysis (Exp 6)")
    print("==================================================")
    
    epochs = 100
    num_users = 5
    devices_per_user = 2
    injection_epoch = 10
    
    tracker = EpochTracker()
    hmm_detector = HMMDetector()
    hmm_detector.train_on_profile("Student")
    graph_lstm = GraphLSTM(beta=0.8)
    fusion = RiskFusion.from_config()
    
    devices = []
    rogues = []
    
    for u in range(num_users):
        owner_id = f"user_{u}"
        phone = LegitimateDevice(
            device_id=f"{owner_id}_phone",
            owner_id=owner_id,
            name="Android Phone",
            device_type=DeviceType.PRIMARY,
            os_version="Android 14",
            profile_name="Student"
        )
        devices.append(phone)
        for d in range(1, devices_per_user):
            laptop = LegitimateDevice(
                device_id=f"{owner_id}_laptop_{d}",
                owner_id=owner_id,
                name=f"Chrome Browser {d}",
                device_type=DeviceType.LINKED,
                os_version="Windows 11",
                profile_name="Student"
            )
            devices.append(laptop)
            
    # Add a few rogues
    for r in range(3):
        rogue = RogueDevice(
            device_id=f"rogue_{r}",
            owner_id=f"user_{r}",
            name="Chrome (Linux)",
            device_type=DeviceType.LINKED,
            os_version="Linux",
            profile_name="Student",
            strategy=AttackStrategy.STEALTH
        )
        rogues.append(rogue)
        
    labels = []
    hmm_scores = []
    temporal_scores = []
    graph_scores = []
    fused_scores = []
    
    prev_adj = None
    
    for epoch in range(1, epochs + 1):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch
        
        if current_epoch == injection_epoch:
            for r in rogues:
                if r not in devices:
                    devices.append(r)
                    r.update_key(current_epoch)
                    
        for dev in devices:
            is_active_hour = (current_epoch % 24) >= 9
            dev.simulate_epoch_action(current_epoch, is_active_hour=is_active_hour)
            
        for dev in devices:
            if len(dev.telemetry_history) >= 2:
                hmm_detector.predict(dev)
                
        for dev in devices:
            if len(dev.telemetry_history) >= 2:
                p_c = getattr(dev, "behavioral_risk", 0.0)
                TrustScore.update(dev, 1.0 - p_c, 0.8)
                
        for u in range(num_users):
            u_devs = [d for d in devices if d.owner_id == f"user_{u}"]
            if all(len(d.telemetry_history) >= 2 for d in u_devs) and len(u_devs) >= 2:
                histories = [d.telemetry_history for d in u_devs]
                adj, rel_scores = graph_lstm.evaluate_devices(histories, prev_adj)
                prev_adj = adj
                for idx, dev in enumerate(u_devs):
                    dev.update_graph_risk(rel_scores[idx])
                    
        for dev in devices:
            if len(dev.telemetry_history) >= 2:
                fusion.predict(dev)
                
                # Only collect data after injection for fair ROC
                if current_epoch >= injection_epoch:
                    is_rogue = 1 if dev.device_id.startswith("rogue") else 0
                    labels.append(is_rogue)
                    
                    hmm_scores.append(getattr(dev, "behavioral_risk", 0.0))
                    
                    # Temporal score: raw inactivity gap mapped to a probability [0, 1]
                    # Centered around delta_inact = 5 (default QTK threshold)
                    epoch_gap = current_epoch - dev.epoch_last_key_update
                    temporal_risk = 1.0 / (1.0 + np.exp(-(epoch_gap - 5.0)))
                    temporal_scores.append(float(temporal_risk))
                    
                    graph_scores.append(dev.graph_risk)
                    fused_scores.append(dev.final_risk)
                    
    # Compute ROC and AUC
    results = {}
    
    models = {
        "HMM": hmm_scores,
        "Temporal": temporal_scores,
        "Graph-LSTM": graph_scores,
        "Fused": fused_scores
    }
    
    for name, scores in models.items():
        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            results[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc)
            }
        else:
            results[name] = {"fpr": [], "tpr": [], "auc": 0.0}
            
    print(f"ROC Analysis complete. Evaluated {len(labels)} data points.")
    
    # Save to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "roc_results.json")
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_experiment()
