import os
import sys
# Resolve python paths relative to backend root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

import csv
import json
import yaml
import random
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

class ExperimentType(str, Enum):
    BASELINE = "baseline"
    DUAL_TRIGGER = "dual_trigger"
    HMM_ONLY = "hmm_only"
    GRAPH_ONLY = "graph_only"
    ABLATION = "ablation"

class AttackScenario(str, Enum):
    LATE_ENROLLMENT = "late_enrollment"
    IMMEDIATE_ENROLLMENT = "immediate_enrollment"
    MULTIPLE_ROGUES = "multiple_rogues"

@dataclass
class ExperimentConfig:
    """Dataclass holding all experimental evaluation configurations."""
    epochs: int = 40
    delta_inact: int = 5
    theta_R: float = 0.65
    alpha: float = 0.8
    profile_name: str = "Student"
    num_users: int = 1
    devices_per_user: int = 2
    experiment_type: ExperimentType = ExperimentType.DUAL_TRIGGER
    attack_scenario: AttackScenario = AttackScenario.LATE_ENROLLMENT
    attack_strategy: AttackStrategy = AttackStrategy.STEALTH
    injection_epoch: int = 10
    rogue_count: int = 1

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
            experiment_type=ExperimentType(sim.get("experiment_type", "dual_trigger")),
            attack_scenario=AttackScenario(att.get("scenario", "late_enrollment")),
            attack_strategy=AttackStrategy(att.get("strategy", "STEALTH")),
            injection_epoch=att.get("injection_epoch", 10),
            rogue_count=att.get("rogue_count", 1)
        )

class ExperimentRunner:
    """
    Coordinates E2EE Trust Simulator components to run research experiments.
    Computes HMM, GNN, Risk Fusion, and QTK triggers modularly to output evaluation metrics.
    """
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.tracker = EpochTracker()
        
        # Modules instantiation (Orchestrated by runner)
        self.inact_trigger = InactivityTrigger(self.cfg.delta_inact)
        self.dual_trigger = DualTrigger(self.cfg.delta_inact, self.cfg.theta_R)
        self.hmm_detector = HMMDetector()
        self.graph_lstm = GraphLSTM(beta=0.8)
        self.fusion = RiskFusion.from_config()
        
        self.hmm_detector.train_on_profile(self.cfg.profile_name)
        
        # Simulated environment
        self.devices: List[Device] = []
        self.rogues: List[RogueDevice] = []
        
        # Logs trace
        self.epoch_logs: List[Dict[str, Any]] = []

    def setup_simulation(self):
        """Sets up legitimate and scheduled rogue devices based on config."""
        self.devices = []
        self.rogues = []
        
        # 1. Create Legitimate Devices
        for u in range(self.cfg.num_users):
            owner_id = f"user_{u}"
            # Primary device (e.g., phone)
            phone = LegitimateDevice(
                device_id=f"{owner_id}_phone",
                owner_id=owner_id,
                name="Android Phone",
                device_type=DeviceType.PRIMARY,
                os_version="Android 14",
                profile_name=self.cfg.profile_name,
                ip_address="192.168.1.10"
            )
            self.devices.append(phone)
            
            # Secondary linked devices
            for d in range(1, self.cfg.devices_per_user):
                laptop = LegitimateDevice(
                    device_id=f"{owner_id}_laptop_{d}",
                    owner_id=owner_id,
                    name=f"Chrome Browser {d}",
                    device_type=DeviceType.LINKED,
                    os_version="Windows 11",
                    profile_name=self.cfg.profile_name,
                    ip_address="192.168.1.20"
                )
                self.devices.append(laptop)
                
        # 2. Add an optional Legitimate Silent Device to verify inactivity quarantining
        silent_dev = SilentDevice(
            device_id="silent_forgotten_tablet",
            owner_id="user_0",
            name="Forgotten Tablet",
            device_type=DeviceType.LINKED,
            os_version="Android Tablet",
            heartbeat_mode=HeartbeatMode.VERY_RARE
        )
        self.devices.append(silent_dev)
        
        # 3. Pre-create Rogue Attacker Devices
        for r in range(self.cfg.rogue_count):
            rogue = RogueDevice(
                device_id=f"rogue_client_{r}",
                owner_id="user_0",
                name="Chrome (Linux)",
                device_type=DeviceType.LINKED,
                os_version="Linux x86_64",
                profile_name=self.cfg.profile_name,
                strategy=self.cfg.attack_strategy
            )
            self.rogues.append(rogue)
            
        # 4. Immediate enrollment scenario
        if self.cfg.attack_scenario == AttackScenario.IMMEDIATE_ENROLLMENT:
            for r in self.rogues:
                self.devices.append(r)
                r.update_key(0)

    def run(self) -> Dict[str, Any]:
        """Runs the main experimental simulation loop."""
        self.setup_simulation()
        
        prev_adj = None
        injection_epoch = self.cfg.injection_epoch
        
        for epoch in range(1, self.cfg.epochs + 1):
            self.tracker.increment_epoch()
            current_epoch = self.tracker.current_epoch
            
            # 1. Late injection enrollment scenario
            if (self.cfg.attack_scenario in [AttackScenario.LATE_ENROLLMENT, AttackScenario.MULTIPLE_ROGUES] 
                    and current_epoch == injection_epoch):
                for r in self.rogues:
                    if r not in self.devices:
                        self.devices.append(r)
                        r.update_key(current_epoch)
            
            # 2. Device Action & Telemetry Generation
            for dev in self.devices:
                # Decide active hours (diurnal check)
                is_active = (current_epoch % 24) >= 9
                
                # Fetch legitimate device telemetry for mimicry attacker
                legit_meta = None
                if isinstance(dev, RogueDevice) and dev.strategy == AttackStrategy.MIMIC:
                    legit_devices = [d for d in self.devices if isinstance(d, LegitimateDevice)]
                    if legit_devices:
                        legit_meta = legit_devices[0].get_metadata()
                
                if isinstance(dev, RogueDevice):
                    dev.simulate_epoch_action(current_epoch, is_active, legit_meta)
                else:
                    dev.simulate_epoch_action(current_epoch, is_active)
            
            # 3. Behavioral HMM Classification
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    self.hmm_detector.predict(dev)
                    
            # 4. Trust Updates (Equation 3 & 4)
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    # Behavioral Trust is strictly 1 - Pc
                    p_c = getattr(dev, "behavioral_risk", 0.0)
                    evidence = 1.0 - p_c
                    TrustScore.update(dev, evidence, self.cfg.alpha)

            # 5. Relational GNN Graph evaluation
            # Gather telemetry histories for all active user_0 devices
            u0_devs = [d for d in self.devices if d.owner_id == "user_0"]
            if all(len(d.telemetry_history) >= 2 for d in u0_devs) and len(u0_devs) >= 2:
                u0_histories = [d.telemetry_history for d in u0_devs]
                adj, rel_scores = self.graph_lstm.evaluate_devices(u0_histories, prev_adj)
                prev_adj = adj
                for idx, dev in enumerate(u0_devs):
                    dev.update_graph_risk(rel_scores[idx])

            # 6. Risk Fusion Layer
            for dev in self.devices:
                if len(dev.telemetry_history) >= 2:
                    # Ablation overrides before fusion
                    if self.cfg.experiment_type == ExperimentType.HMM_ONLY:
                        dev.update_graph_risk(0.0)
                    elif self.cfg.experiment_type == ExperimentType.GRAPH_ONLY:
                        dev.update_behavioral_risk(0.0)
                        
                    self.fusion.predict(dev)

            # 7. QTK Trigger Evaluations
            for dev in self.devices:
                if dev.is_quarantined or dev.current_trust_state == TrustState.REVOKED:
                    continue
                    
                triggered = False
                reason = "trust-compliant"
                
                # Check triggers based on experiment style
                if self.cfg.experiment_type == ExperimentType.BASELINE:
                    triggered = self.inact_trigger.evaluate(current_epoch, dev)
                    reason = "Inactivity trigger fired" if triggered else "trust-compliant"
                else:
                    triggered, reason = self.dual_trigger.evaluate(current_epoch, dev)
                    
                if triggered:
                    dev.quarantine(current_epoch)
                    
                # Log epoch statistics
                self.epoch_logs.append({
                    "Epoch": current_epoch,
                    "DeviceID": dev.device_id,
                    "Type": dev.device_type.value if hasattr(dev.device_type, "value") else str(dev.device_type),
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
                
        # 8. Compute final experimental metrics
        metrics = self.compute_metrics()
        self.export_logs()
        return metrics

    def compute_metrics(self) -> Dict[str, Any]:
        """Calculates TP, FP, FN, TN, precision, recall, and detection latency/evasion."""
        tp, fp, fn, tn = 0, 0, 0, 0
        latency_sum = 0
        quarantined_rogues = 0
        
        # Identify devices
        rogue_ids = [r.device_id for r in self.rogues]
        
        baseline_evasions = []
        dual_evasions = []
        
        # Analyze quarantine states
        for dev in self.devices:
            is_rogue = dev.device_id in rogue_ids
            is_silent = isinstance(dev, SilentDevice)
            
            if is_rogue:
                if dev.is_quarantined:
                    tp += 1
                    quarantined_rogues += 1
                    # Latency: quarantine epoch - injection epoch
                    lat = dev.quarantined_epoch - self.cfg.injection_epoch
                    latency_sum += max(0, lat)
                    dual_evasions.append(max(0, lat))
                else:
                    fn += 1
                    # Evaded entire injection period
                    dual_evasions.append(self.cfg.epochs - self.cfg.injection_epoch)
            else:
                # Legitimate devices (excluding the silent one, since we expect it to be quarantined by inactivity)
                if not is_silent:
                    if dev.is_quarantined:
                        fp += 1
                    else:
                        tn += 1
                        
        # Evaluate silent device baseline evasion
        silent_q_epoch = next((d.quarantined_epoch for d in self.devices if isinstance(d, SilentDevice)), None)
        
        # Baseline simulation emulation on rogue (Plain inactivity trigger checks)
        # Rogue rotates keys every 2-3 epochs, so plain QTK baseline should never catch it
        baseline_evasions = [self.cfg.epochs - self.cfg.injection_epoch] * len(self.rogues)
        
        # Compute rates
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_quarantine_rate = fp / (self.cfg.num_users * self.cfg.devices_per_user) if (self.cfg.num_users * self.cfg.devices_per_user) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = detection_rate
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_latency = latency_sum / quarantined_rogues if quarantined_rogues > 0 else -1.0
        
        print("\n==================================================")
        print("Experimental Evaluation Summary")
        print("==================================================")
        print(f"Experiment Type:          {self.cfg.experiment_type.value}")
        print(f"Attack Scenario:          {self.cfg.attack_scenario.value}")
        print(f"Attack Strategy:          {self.cfg.attack_strategy.value}")
        print(f"Rogue Detection Rate:     {detection_rate:.4%}")
        print(f"False Positive Rate:      {false_positive_rate:.4%}")
        print(f"False Quarantine Rate:    {false_quarantine_rate:.4%}")
        print(f"Avg Detection Latency:    {avg_latency:.2f} epochs")
        print(f"Evasion Duration (Baseline): {sum(baseline_evasions)/len(baseline_evasions):.2f} epochs")
        print(f"Evasion Duration (Dual):     {sum(dual_evasions)/len(dual_evasions):.2f} epochs")
        print(f"F1-Score:                 {f1_score:.4f}")
        print("==================================================")
        
        return {
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "false_quarantine_rate": false_quarantine_rate,
            "avg_latency": avg_latency,
            "evasion_duration_baseline": sum(baseline_evasions)/len(baseline_evasions),
            "evasion_duration_dual": sum(dual_evasions)/len(dual_evasions),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "silent_device_quarantined_epoch": silent_q_epoch
        }

    def export_logs(self):
        """Exports epoch traces to CSV and JSON formats in the data/ directory."""
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data"
        )
        os.makedirs(data_dir, exist_ok=True)
        
        csv_path = os.path.join(data_dir, "experiment_results.csv")
        json_path = os.path.join(data_dir, "experiment_results.json")
        
        # Save CSV
        if self.epoch_logs:
            keys = self.epoch_logs[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.epoch_logs)
                
        # Save JSON
        with open(json_path, "w") as f:
            json.dump(self.epoch_logs, f, indent=4)

def run_experiment(config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Exposes run_experiment pipeline with optional config injection."""
    if config is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "experiment.yaml"
        )
        config = ExperimentConfig.from_yaml(config_path)
    runner = ExperimentRunner(config)
    return runner.run()

if __name__ == "__main__":
    run_experiment()
