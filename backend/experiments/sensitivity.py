import os
import sys
# Resolve python paths relative to backend root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import yaml
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Simulator imports
from simulator.device import Device, DeviceType, HMMState, TrustState
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice, AttackStrategy
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

class SensitivityTrialRunner:
    """Runs a single simulation trial for a specific (delta_inact, theta_R) threshold configuration."""
    def __init__(self, epochs: int, delta_inact: int, theta_R: float, alpha: float, 
                 profile_name: str, num_users: int, devices_per_user: int, injection_epoch: int):
        self.epochs = epochs
        self.delta_inact = delta_inact
        self.theta_R = theta_R
        self.alpha = alpha
        self.profile_name = profile_name
        self.num_users = num_users
        self.devices_per_user = devices_per_user
        self.injection_epoch = injection_epoch
        
        self.tracker = EpochTracker()
        self.inact_trigger = InactivityTrigger(delta_inact)
        self.dual_trigger = DualTrigger(delta_inact, theta_R)
        self.hmm_detector = HMMDetector()
        self.graph_lstm = GraphLSTM(beta=0.8)
        self.fusion = RiskFusion.from_config()
        
        self.hmm_detector.train_on_profile(self.profile_name)
        
        self.devices: List[Device] = []
        self.rogue: Optional[Device] = None

    def setup_simulation(self):
        self.devices = []
        
        # 1. Legitimate Devices
        for u in range(self.num_users):
            owner_id = f"user_{u}"
            phone = LegitimateDevice(
                device_id=f"{owner_id}_phone",
                owner_id=owner_id,
                name="Android Phone",
                device_type=DeviceType.PRIMARY,
                os_version="Android 14",
                profile_name=self.profile_name
            )
            self.devices.append(phone)
            
            for d in range(1, self.devices_per_user):
                laptop = LegitimateDevice(
                    device_id=f"{owner_id}_laptop_{d}",
                    owner_id=owner_id,
                    name=f"Chrome Browser {d}",
                    device_type=DeviceType.LINKED,
                    os_version="Windows 11",
                    profile_name=self.profile_name
                )
                self.devices.append(laptop)
                
        # 2. Rogue device (Stealthy attack strategy by default to benchmark)
        self.rogue = RogueDevice(
            device_id="rogue_laptop",
            owner_id="user_0",
            name="Chrome (Linux)",
            device_type=DeviceType.LINKED,
            os_version="Linux x86_64",
            profile_name=self.profile_name,
            strategy=AttackStrategy.STEALTH
        )

    def run(self) -> Dict[str, Any]:
        self.setup_simulation()
        prev_adj = None
        
        rogue_caught_epoch = None
        false_quarantines = 0
        legit_total = len(self.devices)
        quarantine_epochs = {}
        
        for epoch in range(1, self.epochs + 1):
            self.tracker.increment_epoch()
            current_epoch = self.tracker.current_epoch
            
            # Inject rogue device
            if current_epoch == self.injection_epoch:
                self.devices.append(self.rogue)
                self.rogue.update_key(current_epoch)
                
            # Simulate actions
            for dev in self.devices:
                if dev.is_quarantined:
                    continue
                is_active = (current_epoch % 24) >= 9
                dev.simulate_epoch_action(current_epoch, is_active)
                
            # 1. HMM Behavioral Prediction
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    self.hmm_detector.predict(dev)

            # 2. Trust updates (Equation 3 & 4)
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    p_c = getattr(dev, "behavioral_risk", 0.0)
                    evidence = 1.0 - p_c
                    TrustScore.update(dev, evidence, self.alpha)
                    
            # 3. Relational Graph Prediction
            u0_devs = [d for d in self.devices if d.owner_id == "user_0"]
            if all(len(d.telemetry_history) >= 2 for d in u0_devs) and len(u0_devs) >= 2:
                u0_histories = [d.telemetry_history for d in u0_devs]
                adj, rel_scores = self.graph_lstm.evaluate_devices(u0_histories, prev_adj)
                prev_adj = adj
                for idx, dev in enumerate(u0_devs):
                    dev.update_graph_risk(rel_scores[idx])

            # 4. Risk Fusion Layer
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    self.fusion.predict(dev)

            # 5. Dual Trigger QTK evaluate
            for dev in self.devices:
                if dev.is_quarantined or dev.current_trust_state == TrustState.REVOKED:
                    continue
                    
                triggered = False
                reason = "trust-compliant"
                if current_epoch <= 10:
                    triggered = self.inact_trigger.evaluate(current_epoch, dev)
                    reason = "Inactivity timer expired" if triggered else "trust-compliant"
                else:
                    triggered, reason = self.dual_trigger.evaluate(current_epoch, dev)
                    
                if triggered:
                    dev.quarantine(current_epoch)
                    quarantine_epochs[dev.device_id] = current_epoch
                    if dev.device_id == "rogue_laptop":
                        if rogue_caught_epoch is None:
                            rogue_caught_epoch = current_epoch
                    else:
                        false_quarantines += 1

        # 6. Compute metrics
        caught = rogue_caught_epoch is not None
        latency = (rogue_caught_epoch - self.injection_epoch) if caught else (self.epochs - self.injection_epoch)
        evasion_duration = latency
        
        # Binary stats
        tp = 1 if caught else 0
        fn = 0 if caught else 1
        fp = false_quarantines
        tn = legit_total - false_quarantines
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Availability loss: active quarantined epochs divided by total active epochs
        avail_epochs = legit_total * self.epochs
        loss_epochs = sum(self.epochs - q_ep + 1 for q_ep in quarantine_epochs.values() if q_ep is not None)
        availability_loss = loss_epochs / avail_epochs if avail_epochs > 0 else 0.0
        
        return {
            "detection_rate": float(tp),
            "latency": float(latency),
            "evasion_duration": float(evasion_duration),
            "false_positives": float(fp),
            "fpr": fpr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "availability_loss": availability_loss
        }

def run_experiment() -> List[Dict[str, Any]]:
    """Runs a parameters sweep for delta_inact and theta_R to evaluate tradeoffs."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "sensitivity.yaml"
    )
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    sim_cfg = config_data.get("simulation", {})
    sweep_cfg = config_data.get("sweep", {})
    
    epochs = sim_cfg.get("epochs", 40)
    num_trials = sim_cfg.get("num_trials", 5)
    alpha = sim_cfg.get("alpha", 0.8)
    profile_name = sim_cfg.get("profile_name", "Student")
    num_users = sim_cfg.get("num_users", 1)
    devices_per_user = sim_cfg.get("devices_per_user", 2)
    injection_epoch = sim_cfg.get("injection_epoch", 10)
    
    delta_sweeps = sweep_cfg.get("delta_inact", [3, 5, 7])
    theta_sweeps = sweep_cfg.get("theta_R", [0.5, 0.65, 0.8])
    
    print("==================================================")
    print("Running Experiment: Sensitivity Analysis Sweep (RQ5)")
    print("==================================================")
    print(f"Sweep ranges:")
    print(f"  delta_inact: {delta_sweeps}")
    print(f"  theta_R:     {theta_sweeps}")
    print("--------------------------------------------------")
    
    results = []
    
    print(f"| delta_inact | theta_R | Detection Rate | Latency (epochs) | False Positives | F1-Score | Avail Loss |")
    print(f"|-------------|---------|----------------|------------------|-----------------|----------|------------|")
    
    for delta in delta_sweeps:
        for theta in theta_sweeps:
            trial_metrics = []
            
            for trial in range(num_trials):
                runner = SensitivityTrialRunner(
                    epochs=epochs,
                    delta_inact=delta,
                    theta_R=theta,
                    alpha=alpha,
                    profile_name=profile_name,
                    num_users=num_users,
                    devices_per_user=devices_per_user,
                    injection_epoch=injection_epoch
                )
                metrics = runner.run()
                trial_metrics.append(metrics)
                
            # Aggregate trial outcomes
            agg = {}
            for k in trial_metrics[0].keys():
                vals = [m[k] for m in trial_metrics]
                agg[k] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
                
            results.append({
                "delta_inact": delta,
                "theta_R": theta,
                "detection_rate": agg["detection_rate"],
                "detection_rate_std": agg["detection_rate_std"],
                "latency": agg["latency"],
                "latency_std": agg["latency_std"],
                "false_positives": agg["false_positives"],
                "false_positives_std": agg["false_positives_std"],
                "fpr": agg["fpr"],
                "precision": agg["precision"],
                "recall": agg["recall"],
                "f1": agg["f1"],
                "availability_loss": agg["availability_loss"],
                "evasion_duration": agg["evasion_duration"]
            })
            
            print(f"| {delta:<11} | {theta:<7.2f} | {agg['detection_rate']:.2%}          | {agg['latency']:<16.2f} | {agg['false_positives']:<15.2f} | {agg['f1']:<8.4f} | {agg['availability_loss']:<10.2%} |")

    # Export results structured for heatmaps / surface plotting
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "sensitivity_results.csv")
    json_path = os.path.join(data_dir, "sensitivity_results.json")
    
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
            
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_experiment()
