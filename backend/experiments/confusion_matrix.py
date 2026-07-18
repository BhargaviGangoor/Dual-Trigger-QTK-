import os
import sys
import json
import yaml
from typing import List, Dict, Any, Tuple
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.device import Device, DeviceType
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice, AttackStrategy
from qtk.epoch_tracker import EpochTracker
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from qtk.dual_trigger import DualTrigger

def run_experiment() -> Dict[str, Any]:
    print("==================================================")
    print("Running Experiment: Confusion Matrix (Exp 7)")
    print("==================================================")
    
    epochs = 40
    num_users = 10
    devices_per_user = 3
    injection_epoch = 10
    
    tracker = EpochTracker()
    hmm_detector = HMMDetector()
    hmm_detector.train_on_profile("Student")
    graph_lstm = GraphLSTM(beta=0.8)
    fusion = RiskFusion.from_config()
    dual_trigger = DualTrigger()
    # Also need baseline trigger for comparison
    from qtk.inactivity_trigger import InactivityTrigger
    baseline_trigger = InactivityTrigger()
    
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
            
    # Add rogues to half the users
    for r in range(5):
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
    fused_preds = []
    baseline_preds = []
    
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
            # Simulate legitimate behavior
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
                
                # Check triggers
                fused_quarantine, reason = dual_trigger.evaluate(current_epoch, dev)
                baseline_quarantine = baseline_trigger.evaluate(current_epoch, dev)
                
                if current_epoch >= injection_epoch:
                    is_rogue = 1 if dev.device_id.startswith("rogue") else 0
                    labels.append(is_rogue)
                    fused_preds.append(1 if fused_quarantine else 0)
                    baseline_preds.append(1 if baseline_quarantine else 0)
                    
    # Generate confusion matrices
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(labels, fused_preds, labels=[0, 1]).ravel()
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(labels, baseline_preds, labels=[0, 1]).ravel()
    
    results = {
        "fused": {
            "tn": int(tn_f), "fp": int(fp_f), "fn": int(fn_f), "tp": int(tp_f)
        },
        "baseline": {
            "tn": int(tn_b), "fp": int(fp_b), "fn": int(fn_b), "tp": int(tp_b)
        }
    }
    
    # Save to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "confusion_matrix.json")
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("Confusion Matrix extraction complete.")
    return results

if __name__ == "__main__":
    run_experiment()
