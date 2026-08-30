import os
import sys
import json
import csv
import random
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice
from simulator.rogue_device import RogueDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger, TriggerReason
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from evaluation.metrics import calculate_classification_metrics, calculate_qtk_system_metrics
from evaluation.confidence import compute_confidence_intervals, aggregate_run_metrics

def run_single_simulation(
    run_id: int,
    seed: int,
    epochs: int = 30,
    delta_inact: int = 5,
    theta_R: float = 0.65,
    alpha: float = 0.8,
    beta: float = 0.8,
    injection_epoch: int = 10,
    mode: str = "dual_trigger",  # "baseline" or "dual_trigger"
    hmm: Optional[HMMDetector] = None,
    graph_lstm: Optional[GraphLSTM] = None,
    fusion: Optional[RiskFusion] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes a single independent simulation run for either QTK Baseline or Dual-Trigger QTK.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    tracker = EpochTracker()
    inact_trigger = InactivityTrigger(delta_inact=delta_inact)
    dual_trigger = DualTrigger(delta_inact=delta_inact, theta_R=theta_R)

    if hmm is None:
        hmm = HMMDetector()
    if graph_lstm is None:
        graph_lstm = GraphLSTM(beta=beta, seed=seed)
    if fusion is None:
        fusion = RiskFusion()

    # 1. Setup Devices: 2 Normal Legitimate, 1 Silent Device, 1 Active Rogue (injected at epoch 10)
    phone = LegitimateDevice(
        device_id="phone_primary",
        owner_id="user_0",
        name="Android Phone",
        device_type="primary",
        profile_name="Student",
        ip_address="172.16.23.10"
    )
    laptop = LegitimateDevice(
        device_id="laptop_linked",
        owner_id="user_0",
        name="MacBook Pro",
        device_type="linked",
        profile_name="Student",
        ip_address="172.16.23.20"
    )
    silent_dev = SilentDevice(
        device_id="silent_tablet",
        owner_id="user_0",
        name="Forgotten Tablet",
        device_type="linked"
    )
    rogue = RogueDevice(
        device_id="rogue_terminal",
        owner_id="user_0",
        name="Rogue Terminal",
        device_type="linked",
        attack_mode="stealth_burst"
    )

    active_devices = [phone, laptop, silent_dev]
    epoch_logs = []
    prev_adj = None

    for epoch in range(1, epochs + 1):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch

        # Rogue injection
        if current_epoch == injection_epoch:
            active_devices.append(rogue)
            rogue.update_key(current_epoch)

        # Telemetry generation
        primary_peer_meta = phone.get_latest_telemetry()
        for dev in active_devices:
            dev.simulate_epoch(
                current_epoch=current_epoch,
                peer_telemetry=primary_peer_meta if dev != phone else None,
                rng=rng
            )

        # Individual HMM Inference & Trust Tracking
        for dev in active_devices:
            if len(dev.telemetry_history) >= 1:
                state, p_c = hmm.predict(dev)
                TrustScore.update(dev, p_c, alpha=alpha)

        # Relational Dynamic Graph & GNN/Graph-LSTM Inference
        valid_devs = [d for d in active_devices if len(d.telemetry_history) >= 1]
        if len(valid_devs) >= 2:
            histories = [d.telemetry_history for d in valid_devs]
            adj, rel_scores = graph_lstm.evaluate_devices(histories, prev_adj)
            prev_adj = adj
            for idx, dev in enumerate(valid_devs):
                dev.graph_risk = rel_scores[idx]

        # Risk Fusion Layer
        for dev in active_devices:
            fusion.predict(dev)

        # Protocol Quarantine Evaluation
        for dev in active_devices:
            if dev.is_quarantined:
                continue

            if mode == "baseline":
                triggered = inact_trigger.check(dev, current_epoch)
                reason = TriggerReason.INACTIVITY if triggered else TriggerReason.NONE
                detail = f"Inactivity trigger (age: {current_epoch - dev.epoch_last_key_update})" if triggered else "Compliant"
            else:
                triggered, reason, detail = dual_trigger.dual_trigger_decision(dev, current_epoch)

            if triggered:
                dev.quarantine(current_epoch, detail)

            epoch_logs.append({
                "run_id": run_id,
                "seed": seed,
                "mode": mode,
                "epoch": current_epoch,
                "device_id": dev.device_id,
                "is_rogue": isinstance(dev, RogueDevice),
                "is_silent": isinstance(dev, SilentDevice),
                "key_age": current_epoch - dev.epoch_last_key_update,
                "p_c": round(dev.behavioral_risk, 4),
                "s_graph": round(dev.graph_risk, 4),
                "trust_score": round(dev.trust_score, 4),
                "final_risk": round(dev.final_risk, 4),
                "is_quarantined": dev.is_quarantined,
                "trigger_reason": reason.value if hasattr(reason, "value") else str(reason),
                "quarantine_epoch": dev.quarantined_epoch
            })

    # Compute Run Metrics
    rogue_caught = rogue.is_quarantined
    rogue_epoch = rogue.quarantined_epoch if rogue_caught else None

    # Silent device should be quarantined by inactivity in both
    silent_caught = silent_dev.is_quarantined
    silent_epoch = silent_dev.quarantined_epoch

    # False positive on non-silent legitimate devices (phone, laptop)
    false_positives = sum(1 for d in [phone, laptop] if d.is_quarantined)
    true_negatives = sum(1 for d in [phone, laptop] if not d.is_quarantined)

    tp = 1 if rogue_caught else 0
    fn = 0 if rogue_caught else 1
    fp = false_positives
    tn = true_negatives

    clf_metrics = calculate_classification_metrics(tp, fp, fn, tn)
    sys_metrics = calculate_qtk_system_metrics(
        quarantined_rogue_epochs=[rogue_epoch],
        injection_epoch=injection_epoch,
        total_epochs=epochs,
        false_quarantined_legit_count=false_positives,
        total_legitimate_devices=2
    )

    run_summary = {
        "run_id": run_id,
        "seed": seed,
        "mode": mode,
        **clf_metrics,
        **sys_metrics,
        "silent_caught": silent_caught,
        "silent_quarantine_epoch": silent_epoch,
        "rogue_caught": rogue_caught,
        "rogue_quarantine_epoch": rogue_epoch
    }
    return run_summary, epoch_logs

def run_experiment(
    num_runs: int = 20,
    epochs: int = 30,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs Main RQ1 Experiment comparing Baseline QTK vs Dual-Trigger QTK over independent runs.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("==================================================")
    print("Running Main Experiment: Original QTK vs Dual-Trigger QTK (RQ1)")
    print("==================================================")

    # 1. Pre-train models on training data
    train_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "train.jsonl"
    )
    val_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "val.jsonl"
    )
    hmm = HMMDetector()
    graph_lstm = GraphLSTM(beta=0.8, seed=base_seed)
    fusion = RiskFusion()

    train_records = []
    if os.path.exists(train_data_path):
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_records.append(json.loads(line))
        hmm.fit_from_dataset(train_records)
        fusion_fit_summary = fusion.fit_from_dataset(train_records, hmm, graph_lstm, epochs=80, lr=0.05)
        print(f"RiskFusion trained on TRAIN split ({len(train_records)} records). Final Loss: {fusion_fit_summary.get('final_loss', 0.0):.4f}")

    # 2. Calibrate theta_R on validation data
    val_records = []
    theta_R = 0.65
    if os.path.exists(val_data_path):
        with open(val_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_records.append(json.loads(line))
        theta_R = fusion.calibrate_threshold(val_records, hmm, graph_lstm, max_fpr=0.10, default_theta=0.65)
        print(f"Calibrated theta_R on VALIDATION split ({len(val_records)} records): {theta_R:.3f}")

    baseline_runs = []
    dual_runs = []
    all_logs = []

    for r in range(num_runs):
        seed = base_seed + 1000 + r
        graph_lstm.reset_norm_stats()

        # Run Baseline
        b_summary, b_logs = run_single_simulation(
            run_id=r,
            seed=seed,
            epochs=epochs,
            mode="baseline",
            theta_R=theta_R,
            hmm=hmm,
            graph_lstm=graph_lstm,
            fusion=fusion
        )
        baseline_runs.append(b_summary)
        all_logs.extend(b_logs)

        # Run Dual-Trigger
        graph_lstm.reset_norm_stats()
        d_summary, d_logs = run_single_simulation(
            run_id=r,
            seed=seed,
            epochs=epochs,
            mode="dual_trigger",
            theta_R=theta_R,
            hmm=hmm,
            graph_lstm=graph_lstm,
            fusion=fusion
        )
        dual_runs.append(d_summary)
        all_logs.extend(d_logs)

    # Statistical Aggregations with 95% Confidence Intervals
    b_agg = aggregate_run_metrics(baseline_runs)
    d_agg = aggregate_run_metrics(dual_runs)

    print("\n--- Empirical Results Summary (Mean ± 95% CI) ---")
    print(f"Baseline QTK:  Detection Rate = {b_agg['detection_rate']['mean']:.2%} ± {b_agg['detection_rate']['ci_margin']:.2%}")
    print(f"               Evasion Duration = {b_agg['avg_evasion_duration']['mean']:.2f} ± {b_agg['avg_evasion_duration']['ci_margin']:.2f} epochs")
    print(f"               FPR = {b_agg['false_positive_rate']['mean']:.2%} ± {b_agg['false_positive_rate']['ci_margin']:.2%}")
    print(f"               F1-Score = {b_agg['f1_score']['mean']:.4f} ± {b_agg['f1_score']['ci_margin']:.4f}")

    print(f"\nDual-Trigger:  Detection Rate = {d_agg['detection_rate']['mean']:.2%} ± {d_agg['detection_rate']['ci_margin']:.2%}")
    print(f"               Detection Latency = {d_agg['avg_detection_latency']['mean']:.2f} ± {d_agg['avg_detection_latency']['ci_margin']:.2f} epochs")
    print(f"               Evasion Duration = {d_agg['avg_evasion_duration']['mean']:.2f} ± {d_agg['avg_evasion_duration']['ci_margin']:.2f} epochs")
    print(f"               FPR = {d_agg['false_positive_rate']['mean']:.2%} ± {d_agg['false_positive_rate']['ci_margin']:.2%}")
    print(f"               F1-Score = {d_agg['f1_score']['mean']:.4f} ± {d_agg['f1_score']['ci_margin']:.4f}")
    print("==================================================")

    results = {
        "experiment_name": "baseline_vs_dual",
        "num_runs": num_runs,
        "epochs": epochs,
        "base_seed": base_seed,
        "baseline_summary": b_agg,
        "dual_summary": d_agg,
        "baseline_runs": baseline_runs,
        "dual_runs": dual_runs
    }

    # Save to JSON
    json_path = os.path.join(raw_dir, "baseline_vs_dual.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Save run summaries to CSV
    csv_path = os.path.join(raw_dir, "baseline_vs_dual_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=baseline_runs[0].keys())
        writer.writeheader()
        writer.writerows(baseline_runs)
        writer.writerows(dual_runs)

    return results

if __name__ == "__main__":
    run_experiment()
