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
from simulator.rogue_device import RogueDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from evaluation.metrics import calculate_classification_metrics, calculate_qtk_system_metrics
from evaluation.confidence import aggregate_run_metrics

def run_sensitivity_sweep(
    num_runs_per_point: int = 10,
    epochs: int = 30,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes parameter sweeps over:
    - delta_inact in [3, 5, 7, 10]
    - theta_R in [0.45, 0.55, 0.65, 0.75, 0.85]
    - alpha in [0.6, 0.7, 0.8, 0.9]
    - beta in [0.6, 0.7, 0.8, 0.9]
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("==================================================")
    print("Running Parameter Sensitivity Sweeps (RQ5)")
    print("==================================================")

    hmm = HMMDetector()
    train_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "train.jsonl"
    )
    if os.path.exists(train_path):
        train_records = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_records.append(json.loads(line))
        hmm.fit_from_dataset(train_records)

    delta_sweep = [3, 5, 7, 10]
    theta_sweep = [0.45, 0.55, 0.65, 0.75, 0.85]
    alpha_sweep = [0.6, 0.7, 0.8, 0.9]
    beta_sweep = [0.6, 0.7, 0.8, 0.9]

    sweep_results = []
    
    # 1. 2D Joint Sweep: (delta_inact, theta_R) at base alpha=0.8, beta=0.8
    print("Sweeping (delta_inact x theta_R)...")
    for delta in delta_sweep:
        for theta in theta_sweep:
            point_runs = []
            for r in range(num_runs_per_point):
                seed = base_seed + 6000 + int(delta * 100 + theta * 1000 + r)
                rng = random.Random(seed)
                tracker = EpochTracker()
                dual_trigger = DualTrigger(delta_inact=delta, theta_R=theta)
                graph_lstm = GraphLSTM(beta=0.8, seed=seed)
                fusion = RiskFusion()

                phone = LegitimateDevice(device_id="phone", owner_id="user_0", name="Phone", device_type="primary", profile_name="Student")
                laptop = LegitimateDevice(device_id="laptop", owner_id="user_0", name="Laptop", device_type="linked", profile_name="Student")
                rogue = RogueDevice(device_id="rogue", owner_id="user_0", name="Rogue", device_type="linked", attack_mode="stealth_burst")

                active = [phone, laptop]
                prev_adj = None
                q_status = {d.device_id: False for d in [phone, laptop, rogue]}
                q_epoch = {d.device_id: None for d in [phone, laptop, rogue]}

                for ep in range(1, epochs + 1):
                    tracker.increment_epoch()
                    if ep == 10:
                        active.append(rogue)
                        rogue.update_key(ep)

                    p_meta = phone.get_latest_telemetry()
                    for d in active:
                        d.simulate_epoch(ep, p_meta if d != phone else None, rng=rng)

                    for d in active:
                        hmm.predict(d)
                        TrustScore.update(d, d.behavioral_risk, alpha=0.8)

                    if len(active) >= 2:
                        hists = [d.telemetry_history for d in active if d.telemetry_history]
                        adj, scs = graph_lstm.evaluate_devices(hists, prev_adj)
                        prev_adj = adj
                        for i, d in enumerate(active):
                            if i < len(scs):
                                d.graph_risk = scs[i]

                    for d in active:
                        fusion.predict(d)

                    for d in active:
                        if q_status[d.device_id]:
                            continue
                        trig, reason, _ = dual_trigger.dual_trigger_decision(d, ep)
                        if trig:
                            q_status[d.device_id] = True
                            q_epoch[d.device_id] = ep

                r_caught = q_status[rogue.device_id]
                r_ep = q_epoch[rogue.device_id]
                fp = sum(1 for d in [phone, laptop] if q_status[d.device_id])
                tn = sum(1 for d in [phone, laptop] if not q_status[d.device_id])
                tp = 1 if r_caught else 0
                fn = 0 if r_caught else 1

                clf = calculate_classification_metrics(tp, fp, fn, tn)
                sys_m = calculate_qtk_system_metrics([r_ep], 10, epochs, fp, 2)
                point_runs.append({**clf, **sys_m})

            agg = aggregate_run_metrics(point_runs)
            sweep_results.append({
                "parameter_group": "delta_theta",
                "delta_inact": delta,
                "theta_R": theta,
                "alpha": 0.8,
                "beta": 0.8,
                "detection_rate_mean": agg["detection_rate"]["mean"],
                "detection_rate_ci": agg["detection_rate"]["ci_margin"],
                "fpr_mean": agg["false_positive_rate"]["mean"],
                "fpr_ci": agg["false_positive_rate"]["ci_margin"],
                "latency_mean": agg["avg_detection_latency"]["mean"],
                "latency_ci": agg["avg_detection_latency"]["ci_margin"],
                "f1_mean": agg["f1_score"]["mean"],
                "f1_ci": agg["f1_score"]["ci_margin"]
            })

    # 2. Alpha sweep (Trust decay rate)
    print("Sweeping alpha (Trust decay)...")
    for alpha in alpha_sweep:
        point_runs = []
        for r in range(num_runs_per_point):
            seed = base_seed + 7000 + int(alpha * 100 + r)
            rng = random.Random(seed)
            tracker = EpochTracker()
            dual_trigger = DualTrigger(delta_inact=5, theta_R=0.65)
            graph_lstm = GraphLSTM(beta=0.8, seed=seed)
            fusion = RiskFusion()

            phone = LegitimateDevice(device_id="phone", owner_id="user_0", name="Phone", device_type="primary", profile_name="Student")
            laptop = LegitimateDevice(device_id="laptop", owner_id="user_0", name="Laptop", device_type="linked", profile_name="Student")
            rogue = RogueDevice(device_id="rogue", owner_id="user_0", name="Rogue", device_type="linked", attack_mode="stealth_burst")

            active = [phone, laptop]
            prev_adj = None
            q_status = {d.device_id: False for d in [phone, laptop, rogue]}
            q_epoch = {d.device_id: None for d in [phone, laptop, rogue]}

            for ep in range(1, epochs + 1):
                tracker.increment_epoch()
                if ep == 10:
                    active.append(rogue)
                    rogue.update_key(ep)

                p_meta = phone.get_latest_telemetry()
                for d in active:
                    d.simulate_epoch(ep, p_meta if d != phone else None, rng=rng)

                for d in active:
                    hmm.predict(d)
                    TrustScore.update(d, d.behavioral_risk, alpha=alpha)

                if len(active) >= 2:
                    hists = [d.telemetry_history for d in active if d.telemetry_history]
                    adj, scs = graph_lstm.evaluate_devices(hists, prev_adj)
                    prev_adj = adj
                    for i, d in enumerate(active):
                        if i < len(scs):
                            d.graph_risk = scs[i]

                for d in active:
                    fusion.predict(d)

                for d in active:
                    if q_status[d.device_id]:
                        continue
                    trig, reason, _ = dual_trigger.dual_trigger_decision(d, ep)
                    if trig:
                        q_status[d.device_id] = True
                        q_epoch[d.device_id] = ep

            r_caught = q_status[rogue.device_id]
            r_ep = q_epoch[rogue.device_id]
            fp = sum(1 for d in [phone, laptop] if q_status[d.device_id])
            tn = sum(1 for d in [phone, laptop] if not q_status[d.device_id])
            tp = 1 if r_caught else 0
            fn = 0 if r_caught else 1

            clf = calculate_classification_metrics(tp, fp, fn, tn)
            sys_m = calculate_qtk_system_metrics([r_ep], 10, epochs, fp, 2)
            point_runs.append({**clf, **sys_m})

        agg = aggregate_run_metrics(point_runs)
        sweep_results.append({
            "parameter_group": "alpha_sweep",
            "delta_inact": 5,
            "theta_R": 0.65,
            "alpha": alpha,
            "beta": 0.8,
            "detection_rate_mean": agg["detection_rate"]["mean"],
            "detection_rate_ci": agg["detection_rate"]["ci_margin"],
            "fpr_mean": agg["false_positive_rate"]["mean"],
            "fpr_ci": agg["false_positive_rate"]["ci_margin"],
            "latency_mean": agg["avg_detection_latency"]["mean"],
            "latency_ci": agg["avg_detection_latency"]["ci_margin"],
            "f1_mean": agg["f1_score"]["mean"],
            "f1_ci": agg["f1_score"]["ci_margin"]
        })

    # Save to JSON and CSV
    json_path = os.path.join(raw_dir, "sensitivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)

    csv_path = os.path.join(raw_dir, "sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_results)

    print(f"Sensitivity sweeps completed ({len(sweep_results)} configurations evaluated).")
    return {"sensitivity_results": sweep_results}

if __name__ == "__main__":
    run_sensitivity_sweep()
