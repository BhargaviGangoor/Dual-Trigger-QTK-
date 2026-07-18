import os
import sys
# Resolve python paths relative to backend root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import yaml
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Simulator imports
from simulator.device import Device, DeviceType, HMMState, TrustState
from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice, HeartbeatMode
from simulator.telemetry_generator import TelemetryGenerator
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

@dataclass
class ScenarioParams:
    name: str
    profile: str
    travel: bool
    idle_epochs: int
    battery_failure: bool
    network_switching: bool

class FalseQuarantineRunner:
    """
    Coordinates Legitimate Device Simulations to evaluate false quarantine rates (FQR)
    and availability metrics under multiple stress scenarios.
    """
    def __init__(self, epochs: int, delta_inact: int, theta_R: float, alpha: float):
        self.epochs = epochs
        self.delta_inact = delta_inact
        self.theta_R = theta_R
        self.alpha = alpha
        
        self.tracker = EpochTracker()
        self.inact_trigger = InactivityTrigger(delta_inact)
        self.dual_trigger = DualTrigger(delta_inact, theta_R)
        self.hmm_detector = HMMDetector()
        self.graph_lstm = GraphLSTM(beta=0.8)
        self.fusion = RiskFusion.from_config()
        
        self.devices: List[Device] = []
        self.epoch_logs: List[Dict[str, Any]] = []

    def setup_scenario(self, params: ScenarioParams):
        self.devices = []
        self.hmm_detector.train_on_profile(params.profile)
        
        # Standard user with Phone (primary) and Laptop (linked)
        phone = LegitimateDevice(
            device_id="phone",
            owner_id="user_0",
            name="Android Phone",
            device_type=DeviceType.PRIMARY,
            os_version="Android 14",
            profile_name=params.profile
        )
        laptop = LegitimateDevice(
            device_id="laptop",
            owner_id="user_0",
            name="Chrome Browser",
            device_type=DeviceType.LINKED,
            os_version="Windows 11",
            profile_name=params.profile
        )
        self.devices.extend([phone, laptop])

    def run_trial(self, params: ScenarioParams) -> Dict[str, Any]:
        self.setup_scenario(params)
        prev_adj = None
        
        correct_qtk_quarantines = 0
        incorrect_behavioral_quarantines = 0
        quarantine_epochs = {}
        
        for epoch in range(1, self.epochs + 1):
            self.tracker.increment_epoch()
            current_epoch = self.tracker.current_epoch
            
            # Simulate actions with specific stress conditions
            for dev in self.devices:
                if dev.is_quarantined:
                    continue
                    
                # Stress Condition 1: Short / Long Inactivity on Laptop
                if dev.device_id == "laptop" and params.idle_epochs > 0:
                    # Inactive between epochs 10 and 10 + idle_epochs
                    if 10 <= current_epoch < (10 + params.idle_epochs):
                        continue
                        
                # Stress Condition 2: Battery failure on primary device
                if dev.device_id == "phone" and params.battery_failure:
                    # Power off completely for epochs 15 to 18
                    if 15 <= current_epoch < 19:
                        continue
                        
                # Standard epoch actions
                is_active = (current_epoch % 24) >= 9
                dev.simulate_epoch_action(current_epoch, is_active)
                
                # Apply telemetry perturbations if stress conditions are enabled
                if dev.telemetry_history:
                    # Stress Condition 3: Legitimate Travel (VPN + Timezone shifts)
                    if params.travel and current_epoch % 8 == 0:
                        dev.telemetry_history[-1] = TelemetryGenerator.apply_vpn(dev.telemetry_history[-1])
                    # Stress Condition 4: Network switching / hopping
                    if params.network_switching:
                        dev.telemetry_history[-1] = TelemetryGenerator.apply_network_hopping(dev.telemetry_history[-1])
                        dev.telemetry_history[-1] = TelemetryGenerator.apply_ip_instability(dev.telemetry_history[-1])

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
            active_devs = [d for d in self.devices if len(d.telemetry_history) >= 2]
            if len(active_devs) >= 2:
                histories = [d.telemetry_history for d in active_devs]
                adj, rel_scores = self.graph_lstm.evaluate_devices(histories, prev_adj)
                prev_adj = adj
                for idx, dev in enumerate(active_devs):
                    dev.update_graph_risk(rel_scores[idx])

            # 4. Risk Fusion Layer
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    self.fusion.predict(dev)

            # 5. Dual Trigger QTK Decisions
            for dev in self.devices:
                if dev.is_quarantined:
                    continue
                    
                triggered = False
                reason = "trust-compliant"
                if current_epoch <= 10:
                    # Warm-up phase to build baseline history
                    triggered = self.inact_trigger.evaluate(current_epoch, dev)
                    reason = "Inactivity timer expired" if triggered else "trust-compliant"
                else:
                    triggered, reason = self.dual_trigger.evaluate(current_epoch, dev)
                if triggered:
                    dev.quarantine(current_epoch)
                    quarantine_epochs[dev.device_id] = current_epoch
                    
                    # Distinguish correct vs incorrect quarantines
                    # Inactivity on laptop >= delta_inact is expected (correct)
                    if dev.device_id == "laptop" and params.idle_epochs >= self.delta_inact and "Inactivity" in reason:
                        correct_qtk_quarantines += 1
                    else:
                        incorrect_behavioral_quarantines += 1
                        
                # Log stats
                self.epoch_logs.append({
                    "Epoch": current_epoch,
                    "Scenario": params.name,
                    "DeviceID": dev.device_id,
                    "HMM_State": dev.hmm_state.value if hasattr(dev.hmm_state, "value") else str(dev.hmm_state),
                    "AnomalyScore": round(getattr(dev, "behavioral_risk", 0.0), 4),
                    "BehavioralTrust": round(1.0 - getattr(dev, "behavioral_risk", 0.0), 4),
                    "TrustScore": round(dev.trust_score, 4),
                    "GraphRisk": round(dev.graph_risk, 4),
                    "FinalRisk": round(dev.final_risk, 4),
                    "Triggered": triggered,
                    "TriggerReason": reason,
                    "Quarantined": dev.is_quarantined,
                    "DetectionEpoch": dev.quarantined_epoch if dev.is_quarantined else None
                })
                
        # Compute standard evaluation metrics based on device states
        tp, fp, fn, tn = 0, 0, 0, 0
        loss_epochs = 0
        avail_epochs = 0
        
        for dev in self.devices:
            # laptop in idle scenario is essentially a legitimate device designed to trigger the inactivity baseline (True Positive for inactivity, but True Negative for behavioral). 
            # Wait, since there are NO ROGUES in this experiment, True Positives for behavioral attacks is inherently 0.
            # So ANY behavioral quarantine is a False Positive.
            # ANY non-quarantine is a True Negative.
            
            expected_inactivity_quarantine = (dev.device_id == "laptop" and params.idle_epochs >= self.delta_inact)
            
            if expected_inactivity_quarantine:
                continue # Skip for FQR calculation, as this is a correct baseline protocol quarantine
                
            avail_epochs += self.epochs
            if dev.is_quarantined:
                fp += 1
                loss_epochs += (self.epochs - dev.quarantined_epoch + 1)
            else:
                tn += 1
                
        false_quarantine_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        availability_loss = loss_epochs / avail_epochs if avail_epochs > 0 else 0.0
        
        # MTTFQ
        mttf = self.epochs
        if quarantine_epochs:
            incorrect_epochs = [epoch for dev_id, epoch in quarantine_epochs.items() 
                                if not (dev_id == "laptop" and params.idle_epochs >= self.delta_inact)]
            if incorrect_epochs:
                mttf = float(np.mean(incorrect_epochs))
                
        return {
            "false_quarantine_rate": false_quarantine_rate,
            "availability_loss": availability_loss,
            "mttf": mttf,
            "fp": fp,
            "tn": tn,
            "correct_qtk": correct_qtk_quarantines
        }

def run_experiment() -> Dict[str, Any]:
    """Runs RQ4 availability and false quarantine evaluation for all scenarios."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "availability.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    sim_cfg = config.get("simulation", {})
    scenarios_cfg = config.get("scenarios", {})
    
    epochs = sim_cfg.get("epochs", 40)
    delta_inact = sim_cfg.get("delta_inact", 5)
    theta_R = sim_cfg.get("theta_R", 0.65)
    alpha = sim_cfg.get("alpha", 0.8)
    num_trials = sim_cfg.get("num_trials", 10)
    
    print("==================================================")
    print("Running Experiment: False Quarantine / Availability Analysis (RQ4)")
    print("==================================================")
    
    all_results = {}
    logs_to_export = []
    
    for key, sc in scenarios_cfg.items():
        params = ScenarioParams(
            name=sc.get("name", key),
            profile=sc.get("profile", "Student"),
            travel=sc.get("travel", False),
            idle_epochs=sc.get("idle_epochs", 0),
            battery_failure=sc.get("battery_failure", False),
            network_switching=sc.get("network_switching", False)
        )
        
        trial_metrics = []
        for trial in range(num_trials):
            runner = FalseQuarantineRunner(epochs, delta_inact, theta_R, alpha)
            metrics = runner.run_trial(params)
            trial_metrics.append(metrics)
            logs_to_export.extend(runner.epoch_logs)
            
        # Statistical aggregations
        agg = {}
        for metric_key in trial_metrics[0].keys():
            vals = [m[metric_key] for m in trial_metrics]
            agg[f"{metric_key}_mean"] = float(np.mean(vals))
            agg[f"{metric_key}_std"] = float(np.std(vals))
            
        all_results[key] = agg
        
        print(f"\nScenario: {params.name} ({params.profile})")
        print(f"  Mean False Quarantine Rate:            {agg['false_quarantine_rate_mean']:.2%}")
        print(f"  Mean Availability Loss:                 {agg['availability_loss_mean']:.2%}")
        print(f"  Mean Time to False Quarantine (MTTFQ):  {agg['mttf_mean']:.2f} epochs")
        print(f"  Correct QTK Inactivity Quarantines:     {agg['correct_qtk_mean']:.2f}")
        print(f"  False Positives (Incorrect Quarantines):{agg['fp_mean']:.2f}")
        print("--------------------------------------------------")
        
    # Export logs
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "false_quarantine_results.csv")
    json_path = os.path.join(data_dir, "false_quarantine_results.json")
    
    if logs_to_export:
        keys = logs_to_export[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(logs_to_export)
            
    with open(json_path, "w") as f:
        json.dump(logs_to_export, f, indent=4)
        
    return all_results

if __name__ == "__main__":
    run_experiment()
