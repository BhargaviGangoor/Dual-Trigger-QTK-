import os
import sys
# Resolve python paths relative to backend root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import yaml
import random
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

# Simulator imports
from simulator.device import Device, DeviceType, HMMState, TrustState
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice, AttackStrategy
from simulator.silent_device import SilentDevice, HeartbeatMode
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

class AblationType(str, Enum):
    HMM_ONLY = "hmm_only"
    TEMPORAL_ONLY = "temporal_only"
    GRAPH_LSTM_ONLY = "graph_lstm_only"
    FULL = "full"

@dataclass
class AblationConfig:
    """Dataclass holding ablation configuration parameters."""
    epochs: int = 40
    delta_inact: int = 5
    theta_R: float = 0.65
    alpha: float = 0.8
    profile_name: str = "Student"
    num_users: int = 1
    devices_per_user: int = 2
    attack_scenario: str = "late_enrollment"
    attack_strategy: AttackStrategy = AttackStrategy.STEALTH
    injection_epoch: int = 10
    rogue_count: int = 1
    num_trials: int = 5  # Statistical runs

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        sim = data.get("simulation", {})
        att = data.get("attack", {})
        
        return cls(
            epochs=sim.get("epochs", 40),
            delta_inact=sim.get("delta_inact", 5),
            theta_R=sim.get("theta_R", 0.65),
            alpha=sim.get("alpha", 0.8),
            profile_name=sim.get("profile_name", "Student"),
            num_users=sim.get("num_users", 1),
            devices_per_user=sim.get("devices_per_user", 2),
            attack_scenario=att.get("scenario", "late_enrollment"),
            attack_strategy=AttackStrategy(att.get("strategy", "STEALTH")),
            injection_epoch=att.get("injection_epoch", 10),
            rogue_count=att.get("rogue_count", 1),
            num_trials=sim.get("num_trials", 5)
        )

class AblationTrialRunner:
    """Runs a single simulation trial for a specific ablation configuration."""
    def __init__(self, config: AblationConfig, ablation_type: AblationType):
        self.cfg = config
        self.ablation_type = ablation_type
        
        self.tracker = EpochTracker()
        self.inact_trigger = InactivityTrigger(self.cfg.delta_inact)
        self.dual_trigger = DualTrigger(self.cfg.delta_inact, self.cfg.theta_R)
        self.hmm_detector = HMMDetector()
        self.graph_lstm = GraphLSTM(beta=0.8)
        self.fusion = RiskFusion.from_config()
        
        self.hmm_detector.train_on_profile(self.cfg.profile_name)
        
        self.devices: List[Device] = []
        self.rogues: List[RogueDevice] = []
        self.epoch_logs: List[Dict[str, Any]] = []

    def setup_simulation(self):
        self.devices = []
        self.rogues = []
        
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
                
        # Forgotten Tablet (Silent device)
        silent_dev = SilentDevice(
            device_id="silent_forgotten_tablet",
            owner_id="user_0",
            name="Forgotten Tablet",
            device_type=DeviceType.LINKED,
            os_version="Android Tablet",
            heartbeat_mode=HeartbeatMode.VERY_RARE
        )
        self.devices.append(silent_dev)
        
        # 2. Rogue Devices
        for r in range(self.cfg.rogue_count):
            rogue = RogueDevice(
                device_id=f"rogue_client_{r}",
                owner_id="user_0",
                name="Chrome (Linux)",
                device_type=DeviceType.LINKED,
                os_version="Linux",
                profile_name=self.cfg.profile_name,
                strategy=self.cfg.attack_strategy
            )
            self.rogues.append(rogue)
            
        if self.cfg.attack_scenario == "immediate_enrollment":
            for r in self.rogues:
                self.devices.append(r)
                r.update_key(0)

    def run(self) -> Dict[str, Any]:
        self.setup_simulation()
        
        prev_adj = None
        injection_epoch = self.cfg.injection_epoch
        
        for epoch in range(1, self.cfg.epochs + 1):
            self.tracker.increment_epoch()
            current_epoch = self.tracker.current_epoch
            
            # Rogue injection check
            if self.cfg.attack_scenario == "late_enrollment" and current_epoch == injection_epoch:
                for r in self.rogues:
                    if r not in self.devices:
                        self.devices.append(r)
                        r.update_key(current_epoch)
                        
            # Simulate Epoch actions
            for dev in self.devices:
                is_active = (current_epoch % 24) >= 9
                legit_meta = None
                if isinstance(dev, RogueDevice) and dev.strategy == AttackStrategy.MIMIC:
                    legit_devices = [d for d in self.devices if isinstance(d, LegitimateDevice)]
                    if legit_devices:
                        legit_meta = legit_devices[0].get_metadata()
                        
                if isinstance(dev, RogueDevice):
                    dev.simulate_epoch_action(current_epoch, is_active, legit_meta)
                else:
                    dev.simulate_epoch_action(current_epoch, is_active)
                    
            # 1. HMM Behavioral Prediction
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    if self.ablation_type == AblationType.GRAPH_LSTM_ONLY:
                        # Graph-LSTM Only: Disable HMM behavior risk
                        dev.update_hmm_state(HMMState.NORMAL)
                        dev.update_behavioral_risk(0.0)
                    else:
                        self.hmm_detector.predict(dev)
                        
            # 2. Relational Graph Prediction
            u0_devs = [d for d in self.devices if d.owner_id == "user_0"]
            if all(len(d.telemetry_history) >= 2 for d in u0_devs) and len(u0_devs) >= 2:
                u0_histories = [d.telemetry_history for d in u0_devs]
                
                # Check for temporal configuration (Identity Adjacency)
                if self.ablation_type == AblationType.TEMPORAL_ONLY:
                    identity_adj = np.eye(len(u0_devs))
                    adj, rel_scores = self.graph_lstm.evaluate_devices(u0_histories, identity_adj)
                else:
                    adj, rel_scores = self.graph_lstm.evaluate_devices(u0_histories, prev_adj)
                    
                prev_adj = adj
                for idx, dev in enumerate(u0_devs):
                    dev.update_graph_risk(rel_scores[idx])
                    
            # Apply GNN overrides if HMM Only configuration
            if self.ablation_type == AblationType.HMM_ONLY:
                for dev in self.devices:
                    dev.update_graph_risk(0.0)
                    
            # 3. Trust updates
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    hmm_st = dev.hmm_state
                    if hmm_st == HMMState.NORMAL:
                        evidence = 1.0
                    elif hmm_st == HMMState.IDLE:
                        evidence = 0.90
                    elif hmm_st == HMMState.SUSPICIOUS:
                        evidence = 0.60
                    else:
                        evidence = 0.25
                        
                    # GNN penalty unless HMM Only
                    if self.ablation_type != AblationType.HMM_ONLY:
                        if dev.graph_risk > 0.4:
                            evidence = max(0.0, evidence - (0.5 * dev.graph_risk))
                            
                    TrustScore.update(dev, evidence, self.cfg.alpha)
                    
            # 4. Risk Fusion Layer
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    # Ablation modulations in prediction weights
                    if self.ablation_type == AblationType.HMM_ONLY:
                        dev.update_graph_risk(0.0)
                    elif self.ablation_type == AblationType.GRAPH_LSTM_ONLY:
                        dev.update_behavioral_risk(0.0)
                        
                    self.fusion.predict(dev)
                    
            # 5. Dual Trigger evaluate
            for dev in self.devices:
                if dev.is_quarantined or dev.current_trust_state == TrustState.REVOKED:
                    continue
                    
                triggered, reason = self.dual_trigger.evaluate(current_epoch, dev)
                if triggered:
                    dev.quarantine(current_epoch)
                    
                self.epoch_logs.append({
                    "Epoch": current_epoch,
                    "DeviceID": dev.device_id,
                    "TrustScore": round(dev.trust_score, 4),
                    "GraphRisk": round(dev.graph_risk, 4),
                    "FinalRisk": round(dev.final_risk, 4),
                    "Triggered": triggered,
                    "Quarantined": dev.is_quarantined,
                    "Configuration": self.ablation_type.value
                })
                
        return self.compute_metrics()

    def compute_metrics(self) -> Dict[str, Any]:
        tp, fp, fn, tn = 0, 0, 0, 0
        latency_sum = 0
        quarantined_rogues = 0
        rogue_ids = [r.device_id for r in self.rogues]
        evasion_durations = []
        
        for dev in self.devices:
            is_rogue = dev.device_id in rogue_ids
            is_silent = isinstance(dev, SilentDevice)
            
            if is_rogue:
                if dev.is_quarantined:
                    tp += 1
                    quarantined_rogues += 1
                    lat = dev.quarantined_epoch - self.cfg.injection_epoch
                    latency_sum += max(0, lat)
                    evasion_durations.append(max(0, lat))
                else:
                    fn += 1
                    evasion_durations.append(self.cfg.epochs - self.cfg.injection_epoch)
            else:
                if not is_silent:
                    if dev.is_quarantined:
                        fp += 1
                    else:
                        tn += 1
                        
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_quarantine_rate = fp / (self.cfg.num_users * self.cfg.devices_per_user) if (self.cfg.num_users * self.cfg.devices_per_user) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = detection_rate
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_latency = latency_sum / quarantined_rogues if quarantined_rogues > 0 else -1.0
        
        return {
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "false_quarantine_rate": false_quarantine_rate,
            "avg_latency": avg_latency,
            "evasion_duration": sum(evasion_durations) / len(evasion_durations) if evasion_durations else 0.0,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

def run_experiment(config: Optional[AblationConfig] = None) -> Dict[str, Any]:
    """Runs a multi-trial ablation study for HMM Only, Temporal Only, Graph-LSTM Only, and Full models."""
    if config is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "experiment.yaml"
        )
        config = AblationConfig.from_yaml(config_path)
        
    print(f"\nRunning RQ2 Ablation Study ({config.num_trials} trials per configuration)...")
    ablation_types = [AblationType.HMM_ONLY, AblationType.TEMPORAL_ONLY, AblationType.GRAPH_LSTM_ONLY, AblationType.FULL]
    
    results = {}
    logs_to_export = []
    
    for ab_type in ablation_types:
        trial_metrics = []
        print(f"Evaluating ablation configuration: {ab_type.value}...")
        
        for trial in range(config.num_trials):
            runner = AblationTrialRunner(config, ab_type)
            metrics = runner.run()
            trial_metrics.append(metrics)
            logs_to_export.extend(runner.epoch_logs)
            
        # Compute statistical aggregations
        agg_results = {}
        for key in trial_metrics[0].keys():
            values = [m[key] for m in trial_metrics if m[key] != -1.0]
            if values:
                agg_results[f"{key}_mean"] = float(np.mean(values))
                agg_results[f"{key}_std"] = float(np.std(values))
            else:
                agg_results[f"{key}_mean"] = -1.0
                agg_results[f"{key}_std"] = 0.0
                
        results[ab_type.value] = agg_results
        
        print(f"  [{ab_type.value}] Mean Detection Rate: {agg_results['detection_rate_mean']:.2%}")
        print(f"  [{ab_type.value}] Mean False Positive Rate: {agg_results['false_positive_rate_mean']:.2%}")
        print(f"  [{ab_type.value}] Mean Latency: {agg_results['avg_latency_mean']:.2f} epochs")
        print(f"  [{ab_type.value}] Mean F1-Score: {agg_results['f1_score_mean']:.4f}")
        print("--------------------------------------------------")
        
    # Export logs to CSV/JSON
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "ablation_results.csv")
    json_path = os.path.join(data_dir, "ablation_results.json")
    
    if logs_to_export:
        keys = logs_to_export[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(logs_to_export)
            
    with open(json_path, "w") as f:
        json.dump(logs_to_export, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_experiment()
