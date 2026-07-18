import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.device import Device, DeviceType
from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice
from simulator.rogue_device import RogueDevice, AttackStrategy
from qtk.epoch_tracker import EpochTracker
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from qtk.dual_trigger import DualTrigger

def run_experiment() -> Dict[str, Any]:
    print("==================================================")
    print("Running Experiment: Case Study (Exp 9)")
    print("==================================================")
    
    epochs = 20
    tracker = EpochTracker()
    hmm_detector = HMMDetector()
    hmm_detector.train_on_profile("Student")
    graph_lstm = GraphLSTM(beta=0.8)
    fusion = RiskFusion.from_config()
    dual_trigger = DualTrigger()
    
    owner_id = "user_case"
    devices = [
        LegitimateDevice("legitimate", owner_id, "Smartphone", DeviceType.PRIMARY, "iOS 16", "Student"),
        SilentDevice("tablet_silent", owner_id, "Tablet", DeviceType.LINKED, "iPadOS 16", "Student"),
        RogueDevice("rogue", owner_id, "Attacker", DeviceType.LINKED, "Linux", "Student", strategy=AttackStrategy.STEALTH)
    ]
    
    # Rogue starts at epoch 5
    devices[-1].epoch_last_key_update = 5
    
    case_study_table = []
    prev_adj = None
    
    for epoch in range(1, epochs + 1):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch
        
        for dev in devices:
            if dev.device_id == "tablet_silent":
                # Only active until epoch 10
                dev.simulate_epoch_action(current_epoch, is_active_hour=(current_epoch <= 10))
            elif dev.device_id == "rogue":
                if current_epoch >= 5:
                    dev.simulate_epoch_action(current_epoch, is_active_hour=True)
            else:
                dev.simulate_epoch_action(current_epoch, is_active_hour=True)
                
        for dev in devices:
            if dev.device_id == "rogue" and current_epoch < 5:
                continue
            if len(dev.telemetry_history) >= 2:
                hmm_detector.predict(dev)
                
        for dev in devices:
            if dev.device_id == "rogue" and current_epoch < 5:
                continue
            if len(dev.telemetry_history) >= 2:
                p_c = getattr(dev, "behavioral_risk", 0.0)
                TrustScore.update(dev, 1.0 - p_c, 0.8)
                
        active_devices = [d for d in devices if not (d.device_id == "rogue" and current_epoch < 5)]
        
        if all(len(d.telemetry_history) >= 2 for d in active_devices) and len(active_devices) >= 2:
            histories = [d.telemetry_history for d in active_devices]
            adj, rel_scores = graph_lstm.evaluate_devices(histories, prev_adj)
            prev_adj = adj
            for idx, dev in enumerate(active_devices):
                dev.update_graph_risk(rel_scores[idx])
                
        for dev in active_devices:
            if len(dev.telemetry_history) >= 2:
                fusion.predict(dev)
                q_dual, reason = dual_trigger.evaluate(current_epoch, dev)
                
                if current_epoch == epochs:
                    case_study_table.append({
                        "Device": dev.name,
                        "HMM Anomaly": f"{getattr(dev, 'behavioral_risk', 0.0):.3f}",
                        "Behavioral Trust": f"{dev.trust_score:.3f}",
                        "Graph Risk": f"{dev.graph_risk:.3f}",
                        "Final Risk": f"{dev.final_risk:.3f}",
                        "Decision": "Quarantine" if q_dual else "Normal"
                    })
                    
    results = {"case_study": case_study_table}
    
    # Save to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "case_study.json")
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("Case Study extraction complete.")
    return results

if __name__ == "__main__":
    run_experiment()
