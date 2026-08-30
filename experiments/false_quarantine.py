import os
import sys
import json
import csv
import random
import numpy as np
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.irregular_legitimate import IrregularLegitimateDevice
from simulator.silent_device import SilentDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.quarantine_state import QuarantineManager
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from evaluation.confidence import aggregate_run_metrics

def run_false_quarantine_trial(
    scenario_name: str,
    run_id: int,
    seed: int,
    epochs: int = 30,
    delta_inact: int = 5,
    theta_R: float = 0.65,
    alpha: float = 0.8,
    beta: float = 0.8,
    hmm: Optional[HMMDetector] = None,
    graph_lstm: Optional[GraphLSTM] = None,
    fusion: Optional[RiskFusion] = None
) -> Dict[str, Any]:
    """
    Evaluates availability and false quarantine on legitimate-only groups under specific stress scenarios:
    - normal_legitimate: 3 standard legitimate devices
    - irregular_legitimate: traveling with timezone / network switches
    - long_idle_legitimate: dormant for 3-4 epochs (below delta_inact)
    - network_changing_legitimate: frequent WiFi / Cellular hopping
    """
    rng = random.Random(seed)
    tracker = EpochTracker()
    dual_trigger = DualTrigger(delta_inact=delta_inact, theta_R=theta_R)

    phone = LegitimateDevice(
        device_id="phone", owner_id="user_0", name="Primary Phone",
        device_type="primary", profile_name="Student", ip_address="172.16.23.10"
    )

    if scenario_name == "normal_legitimate":
        laptop = LegitimateDevice(
            device_id="laptop", owner_id="user_0", name="MacBook Pro",
            device_type="linked", profile_name="Student", ip_address="172.16.23.20"
        )
        tablet = LegitimateDevice(
            device_id="tablet", owner_id="user_0", name="iPad Air",
            device_type="linked", profile_name="Student", ip_address="172.16.23.30"
        )
    elif scenario_name == "irregular_legitimate":
        laptop = IrregularLegitimateDevice(
            device_id="laptop", owner_id="user_0", name="Travel Laptop",
            device_type="linked", profile_name="Traveler", irregularity_type="travel_network_switch"
        )
        tablet = LegitimateDevice(
            device_id="tablet", owner_id="user_0", name="iPad Air",
            device_type="linked", profile_name="Student", ip_address="172.16.23.30"
        )
    elif scenario_name == "long_idle_legitimate":
        laptop = IrregularLegitimateDevice(
            device_id="laptop", owner_id="user_0", name="Idle Laptop",
            device_type="linked", profile_name="Student", irregularity_type="sporadic_idle"
        )
        tablet = LegitimateDevice(
            device_id="tablet", owner_id="user_0", name="iPad Air",
            device_type="linked", profile_name="Student", ip_address="172.16.23.30"
        )
    else: # network_changing_legitimate
        laptop = IrregularLegitimateDevice(
            device_id="laptop", owner_id="user_0", name="Roaming Laptop",
            device_type="linked", profile_name="Traveler", irregularity_type="travel_network_switch"
        )
        tablet = IrregularLegitimateDevice(
            device_id="tablet", owner_id="user_0", name="Roaming Tablet",
            device_type="linked", profile_name="Traveler", irregularity_type="travel_network_switch"
        )

    devices = [phone, laptop, tablet]
    prev_adj = None
    quarantine_events = 0
    recovery_events = 0
    total_loss_epochs = 0
    quarantine_epochs = {}

    for epoch in range(1, epochs + 1):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch

        primary_meta = phone.get_latest_telemetry()
        for dev in devices:
            dev.simulate_epoch(current_epoch, primary_meta if dev != phone else None, rng=rng)

        # HMM + Trust
        for dev in devices:
            hmm.predict(dev)
            TrustScore.update(dev, dev.behavioral_risk, alpha=alpha)

        # Relational Graph
        hists = [d.telemetry_history for d in devices if d.telemetry_history]
        if len(hists) >= 2:
            adj, scores = graph_lstm.evaluate_devices(hists, prev_adj)
            prev_adj = adj
            for i, dev in enumerate(devices):
                if i < len(scores):
                    dev.graph_risk = scores[i]

        # Fusion
        for dev in devices:
            fusion.predict(dev)

        # Quarantine & Recovery simulation
        for dev in devices:
            if not dev.is_quarantined:
                triggered, reason, _ = dual_trigger.dual_trigger_decision(dev, current_epoch)
                if triggered:
                    dev.quarantine(current_epoch, reason.value)
                    quarantine_events += 1
                    quarantine_epochs[dev.device_id] = current_epoch
                    other_devs = [d for d in devices if d != dev]
                    QuarantineManager.quarantine_device(dev, other_devs)
            else:
                # Test recovery mechanism: after 2 epochs, if trust recovers and reporters agree
                if (current_epoch - dev.quarantined_epoch) >= 2:
                    active_reps = [d.device_id for d in devices if not d.is_quarantined]
                    success, secret, msg = QuarantineManager.recover_device(dev, active_reps)
                    if success:
                        recovery_events += 1

        for dev in devices:
            if dev.is_quarantined:
                total_loss_epochs += 1

    total_avail_epochs = len(devices) * epochs
    availability_loss = float(total_loss_epochs / max(1, total_avail_epochs))
    false_quarantine_rate = float(quarantine_events / len(devices))
    fpr = false_quarantine_rate  # All evaluated devices are strictly legitimate

    mttf = float(epochs)
    if quarantine_epochs:
        mttf = float(np.mean(list(quarantine_epochs.values())))

    return {
        "scenario": scenario_name,
        "run_id": run_id,
        "seed": seed,
        "false_quarantine_rate": round(false_quarantine_rate, 4),
        "false_positive_rate": round(fpr, 4),
        "availability_loss": round(availability_loss, 4),
        "quarantined_devices_count": quarantine_events,
        "recovery_events_count": recovery_events,
        "mttf_epoch": round(mttf, 2)
    }

def run_experiment(
    num_runs: int = 20,
    epochs: int = 30,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs RQ4 False-Quarantine and Availability Analysis.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("==================================================")
    print("Running False-Quarantine & Availability Experiment (RQ4)")
    print("==================================================")

    hmm = HMMDetector()
    graph_lstm = GraphLSTM(beta=0.8, seed=base_seed)
    fusion = RiskFusion()

    train_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "train.jsonl"
    )
    val_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "val.jsonl"
    )
    if os.path.exists(train_path):
        train_records = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_records.append(json.loads(line))
        hmm.fit_from_dataset(train_records)
        fusion.fit_from_dataset(train_records, hmm, graph_lstm, epochs=80, lr=0.05)

    theta_R = 0.65
    if os.path.exists(val_path):
        val_records = []
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_records.append(json.loads(line))
        theta_R = fusion.calibrate_threshold(val_records, hmm, graph_lstm, max_fpr=0.10, default_theta=0.65)
        print(f"Calibrated theta_R on VALIDATION split: {theta_R:.3f}")

    scenarios = [
        "normal_legitimate",
        "irregular_legitimate",
        "long_idle_legitimate",
        "network_changing_legitimate"
    ]
    all_fq_results = {}
    flat_rows = []

    for sc in scenarios:
        print(f"Evaluating scenario: {sc} across {num_runs} runs...")
        runs_list = []
        for r in range(num_runs):
            seed = base_seed + 5000 + r
            graph_lstm.reset_norm_stats()
            res = run_false_quarantine_trial(
                scenario_name=sc,
                run_id=r,
                seed=seed,
                epochs=epochs,
                theta_R=theta_R,
                hmm=hmm,
                graph_lstm=graph_lstm,
                fusion=fusion
            )
            runs_list.append(res)
            flat_rows.append(res)

        agg = aggregate_run_metrics(runs_list)
        all_fq_results[sc] = {
            "summary": agg,
            "runs": runs_list
        }

        print(f"  [{sc}] False Quarantine Rate: {agg['false_quarantine_rate']['mean']:.2%} ± {agg['false_quarantine_rate']['ci_margin']:.2%}")
        print(f"  [{sc}] Availability Loss:      {agg['availability_loss']['mean']:.2%} ± {agg['availability_loss']['ci_margin']:.2%}")
        print(f"  [{sc}] Recovery Events:        {agg['recovery_events_count']['mean']:.2f}")
        print(f"  [{sc}] MTTF:                   {agg['mttf_epoch']['mean']:.2f} epochs")
        print("--------------------------------------------------")

    json_path = os.path.join(raw_dir, "false_quarantine.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_fq_results, f, indent=4)

    csv_path = os.path.join(raw_dir, "false_quarantine_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)

    return all_fq_results

if __name__ == "__main__":
    run_experiment()
