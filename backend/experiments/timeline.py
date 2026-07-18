import os
import sys
import json
from typing import List, Dict, Any

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
    print("Running Experiment: Detection Timeline (Exp 8)")
    print("==================================================")
    
    epochs = 30
    tracker = EpochTracker()
    hmm_detector = HMMDetector()
    hmm_detector.train_on_profile("Student")
    graph_lstm = GraphLSTM(beta=0.8)
    fusion = RiskFusion.from_config()
    dual_trigger = DualTrigger()
    
    # 1 legitimate, 1 silent, 1 rogue (added later)
    owner_id = "user_timeline"
    legit_device = LegitimateDevice(
        device_id="legitimate", owner_id=owner_id, name="Phone",
        device_type=DeviceType.PRIMARY, os_version="Android", profile_name="Student"
    )
    silent_device = LegitimateDevice(
        device_id="silent", owner_id=owner_id, name="Laptop",
        device_type=DeviceType.LINKED, os_version="Windows", profile_name="Student"
    )
    rogue_device = RogueDevice(
        device_id="rogue", owner_id=owner_id, name="Attacker",
        device_type=DeviceType.LINKED, os_version="Linux", profile_name="Student",
        strategy=AttackStrategy.BURST
    )
    
    devices = [legit_device, silent_device]
    
    timeline_data = {
        "legitimate": [],
        "silent": [],
        "rogue": []
    }
    
    prev_adj = None
    
    for epoch in range(1, epochs + 1):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch
        
        if current_epoch == 10:
            devices.append(rogue_device)
            rogue_device.update_key(current_epoch)
            
        for dev in devices:
            if dev.device_id == "legitimate":
                dev.simulate_epoch_action(current_epoch, is_active_hour=True)
            elif dev.device_id == "silent":
                # Only active for first 5 epochs
                dev.simulate_epoch_action(current_epoch, is_active_hour=(current_epoch <= 5))
            elif dev.device_id == "rogue":
                dev.simulate_epoch_action(current_epoch, is_active_hour=True)
                
        for dev in devices:
            if len(dev.telemetry_history) >= 2:
                hmm_detector.predict(dev)
                
        for dev in devices:
            if len(dev.telemetry_history) >= 2:
                p_c = getattr(dev, "behavioral_risk", 0.0)
                TrustScore.update(dev, 1.0 - p_c, 0.8)
                
        if len(devices) >= 2:
            if all(len(d.telemetry_history) >= 2 for d in devices):
                histories = [d.telemetry_history for d in devices]
                adj, rel_scores = graph_lstm.evaluate_devices(histories, prev_adj)
                prev_adj = adj
                for idx, dev in enumerate(devices):
                    dev.update_graph_risk(rel_scores[idx])
                    
        for dev in devices:
            if len(dev.telemetry_history) >= 2:
                fusion.predict(dev)
                q_dual, reason = dual_trigger.evaluate(current_epoch, dev)
                
                # Inactivity Trigger (Baseline)
                epoch_gap = current_epoch - dev.epoch_last_key_update
                q_baseline = (epoch_gap >= 5)
                
                record = {
                    "epoch": current_epoch,
                    "hmm_risk": getattr(dev, "behavioral_risk", 0.0),
                    "trust": dev.trust_score,
                    "graph_risk": dev.graph_risk,
                    "final_risk": dev.final_risk,
                    "dual_quarantine": q_dual,
                    "baseline_quarantine": q_baseline
                }
                timeline_data[dev.device_id].append(record)
                
    # Save to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "timeline.json")
    
    with open(out_path, "w") as f:
        json.dump(timeline_data, f, indent=4)
        
    print("Timeline data extraction complete.")
    return timeline_data

if __name__ == "__main__":
    run_experiment()
