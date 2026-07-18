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
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

# Simulator imports
from simulator.device import Device, DeviceType, HMMState, TrustState
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice, AttackStrategy
from simulator.mimicry_attacker import MimicryAttacker
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

@dataclass
class MimicryConfig:
    epochs: int = 40
    num_trials: int = 10
    delta_inact: int = 5
    theta_R: float = 0.65
    alpha: float = 0.8
    profile_name: str = "Student"
    num_users: int = 1
    devices_per_user: int = 2
    injection_epoch: int = 10

    @classmethod
    def from_yaml(cls, path: str):
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        sim = data.get("simulation", {})
        return cls(
            epochs=sim.get("epochs", 40),
            num_trials=sim.get("num_trials", 10),
            delta_inact=sim.get("delta_inact", 5),
            theta_R=sim.get("theta_R", 0.65),
            alpha=sim.get("alpha", 0.8),
            profile_name=sim.get("profile_name", "Student"),
            num_users=sim.get("num_users", 1),
            devices_per_user=sim.get("devices_per_user", 2),
            injection_epoch=sim.get("injection_epoch", 10)
        )

class MimicryTrialRunner:
    """Runs a single simulation trial for a specific attacker strategy."""
    def __init__(self, config: MimicryConfig, strategy: str):
        self.cfg = config
        self.strategy = strategy
        
        self.tracker = EpochTracker()
        self.inact_trigger = InactivityTrigger(self.cfg.delta_inact)
        self.dual_trigger = DualTrigger(self.cfg.delta_inact, self.cfg.theta_R)
        self.hmm_detector = HMMDetector()
        self.graph_lstm = GraphLSTM(beta=0.8)
        self.fusion = RiskFusion.from_config()
        
        self.hmm_detector.train_on_profile(self.cfg.profile_name)
        
        self.devices: List[Device] = []
        self.rogue: Optional[Device] = None
        self.epoch_logs: List[Dict[str, Any]] = []

    def setup_simulation(self):
        self.devices = []
        
        # 1. Legitimate Devices
        for u in range(self.cfg.num_users):
            owner_id = f"user_{u}"
            phone = LegitimateDevice(
                device_id=f"{owner_id}_phone",
                owner_id=owner_id,
                name="Android Phone",
                device_type=DeviceType.PRIMARY,
                os_version="Android 14",
                profile_name=self.cfg.profile_name
            )
            self.devices.append(phone)
            
            for d in range(1, self.cfg.devices_per_user):
                laptop = LegitimateDevice(
                    device_id=f"{owner_id}_laptop_{d}",
                    owner_id=owner_id,
                    name=f"Chrome Browser {d}",
                    device_type=DeviceType.LINKED,
                    os_version="Windows 11",
                    profile_name=self.cfg.profile_name
                )
                self.devices.append(laptop)

        # 2. Rogue device setup according to strategy
        if self.strategy == "legacy_mimicry":
            self.rogue = MimicryAttacker(
                device_id="rogue_laptop",
                owner_id="user_0",
                name="Chrome (Linux)",
                device_type=DeviceType.LINKED.value,
                os_version="Linux",
                profile_name=self.cfg.profile_name
            )
        else:
            # Map strategy to AttackStrategy enum
            enum_strategy = AttackStrategy(self.strategy)
            self.rogue = RogueDevice(
                device_id="rogue_laptop",
                owner_id="user_0",
                name="Chrome (Linux)",
                device_type=DeviceType.LINKED,
                os_version="Linux x86_64",
                profile_name=self.cfg.profile_name,
                strategy=enum_strategy
            )

    def run(self) -> Dict[str, Any]:
        self.setup_simulation()
        prev_adj = None
        
        rogue_caught_epoch = None
        false_quarantines = 0
        legit_total = len(self.devices)
        
        # Track history logs
        risk_over_time = []
        trust_over_time = []
        hmm_states_log = []
        
        for epoch in range(1, self.cfg.epochs + 1):
            self.tracker.increment_epoch()
            current_epoch = self.tracker.current_epoch
            
            # Injection check
            if current_epoch == self.cfg.injection_epoch:
                self.devices.append(self.rogue)
                self.rogue.update_key(current_epoch)
                
            # Simulate actions
            for dev in self.devices:
                if dev.is_quarantined:
                    continue
                    
                is_active = (current_epoch % 24) >= 9
                legit_meta = None
                
                # Fetch legitimate phone metadata for mimicry attacker
                if isinstance(self.rogue, MimicryAttacker) or (isinstance(self.rogue, RogueDevice) and self.rogue.strategy == AttackStrategy.MIMIC):
                    legit_devices = [d for d in self.devices if isinstance(d, LegitimateDevice)]
                    if legit_devices:
                        legit_meta = legit_devices[0].get_metadata()
                        
                if isinstance(dev, RogueDevice):
                    dev.simulate_epoch_action(current_epoch, is_active, legit_meta)
                elif isinstance(dev, MimicryAttacker):
                    dev.simulate_epoch_action(current_epoch, is_active, legit_meta)
                else:
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
                    TrustScore.update(dev, evidence, self.cfg.alpha)
                    
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

            # 5. Dual Trigger evaluate
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
                    if dev.device_id == "rogue_laptop":
                        if rogue_caught_epoch is None:
                            rogue_caught_epoch = current_epoch
                    else:
                        false_quarantines += 1
                        
                # Log state metrics
                self.epoch_logs.append({
                    "Epoch": current_epoch,
                    "AttackerStrategy": self.strategy,
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
                    "DetectionEpoch": dev.quarantined_epoch if dev.is_quarantined else None,
                    "DetectionTime": (rogue_caught_epoch - self.cfg.injection_epoch) if rogue_caught_epoch else -1
                })
                
            # Track values for the rogue device
            if self.rogue in self.devices and not self.rogue.is_quarantined:
                risk_over_time.append(self.rogue.final_risk)
                trust_over_time.append(self.rogue.trust_score)
                hmm_states_log.append(self.rogue.hmm_state)

        # 6. Trial statistics calculation
        caught = rogue_caught_epoch is not None
        latency = (rogue_caught_epoch - self.cfg.injection_epoch) if caught else (self.cfg.epochs - self.cfg.injection_epoch)
        evasion_duration = latency
        
        # Binary confusion matrix
        tp = 1 if caught else 0
        fn = 0 if caught else 1
        fp = false_quarantines
        tn = legit_total - false_quarantines
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "caught": caught,
            "latency": latency,
            "evasion_duration": evasion_duration,
            "fpr": fpr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "risk_over_time": risk_over_time,
            "trust_over_time": trust_over_time,
            "hmm_states": hmm_states_log
        }

def run_experiment() -> Dict[str, Any]:
    """Runs RQ3 adversarial mimicry evaluation comparing multiple strategies."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "mimicry.yaml"
    )
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    cfg = MimicryConfig.from_yaml(config_path)
    strategies = config_data.get("attack", {}).get("strategies", ["STEALTH", "MIMIC", "BURST", "RANDOM"])
    # Append legacy_mimicry to compare both new and legacy classes
    strategies.append("legacy_mimicry")
    
    print("==================================================")
    print("Running Experiment: Adversarial Mimicry Attack (RQ3)")
    print("==================================================")
    
    all_results = {}
    logs_to_export = []
    
    for strategy in strategies:
        trial_metrics = []
        print(f"Evaluating strategy: {strategy}...")
        
        for trial in range(cfg.num_trials):
            runner = MimicryTrialRunner(cfg, strategy)
            metrics = runner.run()
            trial_metrics.append(metrics)
            logs_to_export.extend(runner.epoch_logs)
            
        # Statistical aggregations
        agg = {}
        for key in ["latency", "evasion_duration", "fpr", "precision", "recall", "f1"]:
            vals = [m[key] for m in trial_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            
        all_results[strategy] = agg
        
        print(f"  [{strategy}] Mean Detection Latency:    {agg['latency_mean']:.2f} epochs")
        print(f"  [{strategy}] Mean Evasion Duration:     {agg['evasion_duration_mean']:.2f} epochs")
        print(f"  [{strategy}] Mean False Positive Rate:  {agg['fpr_mean']:.2%}")
        print(f"  [{strategy}] Mean F1-Score:             {agg['f1_mean']:.4f}")
        print("--------------------------------------------------")
        
    # Export logs
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "mimicry_results.csv")
    json_path = os.path.join(data_dir, "mimicry_results.json")
    
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
