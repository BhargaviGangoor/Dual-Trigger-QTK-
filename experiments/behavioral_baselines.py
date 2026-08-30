import os
import sys
import json
import csv
import random
import numpy as np
from typing import Dict, Any, List, Tuple

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice
from simulator.rogue_device import RogueDevice
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from qtk.inactivity_trigger import InactivityTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from baselines.qtk_baseline import QTKBaseline
from baselines.threshold_detector import ThresholdDetector
from baselines.hmm_baseline import HMMBaseline
from baselines.isolation_forest import IsolationForestBaseline
from baselines.lstm_baseline import LSTMBaseline
from evaluation.metrics import calculate_classification_metrics, calculate_qtk_system_metrics
from evaluation.confidence import aggregate_run_metrics

def evaluate_baseline_on_run(
    baseline_name: str,
    run_id: int,
    seed: int,
    epochs: int = 30,
    injection_epoch: int = 10,
    detector_obj: Any = None,
    hmm: Optional[HMMDetector] = None,
    graph_lstm: Optional[GraphLSTM] = None,
    fusion: Optional[RiskFusion] = None,
    dual_trigger: Optional[DualTrigger] = None
) -> Dict[str, Any]:
    """
    Evaluates a specific baseline detector on an independent simulation run.
    """
    rng = random.Random(seed)
    tracker = EpochTracker()

    phone = LegitimateDevice(
        device_id="phone_primary", owner_id="user_0", name="Android Phone",
        device_type="primary", profile_name="Student", ip_address="172.16.23.10"
    )
    laptop = LegitimateDevice(
        device_id="laptop_linked", owner_id="user_0", name="MacBook Pro",
        device_type="linked", profile_name="Student", ip_address="172.16.23.20"
    )
    rogue = RogueDevice(
        device_id="rogue_terminal", owner_id="user_0", name="Rogue Terminal",
        device_type="linked", attack_mode="stealth_burst"
    )

    active_devices = [phone, laptop]
    prev_adj = None
    quarantine_status = {d.device_id: False for d in [phone, laptop, rogue]}
    quarantine_epoch = {d.device_id: None for d in [phone, laptop, rogue]}

    for epoch in range(1, epochs + 1):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch

        if current_epoch == injection_epoch:
            active_devices.append(rogue)
            rogue.update_key(current_epoch)

        primary_meta = phone.get_latest_telemetry()
        for dev in active_devices:
            dev.simulate_epoch(current_epoch, primary_meta if dev != phone else None, rng=rng)

        # Baseline decision
        for dev in active_devices:
            if quarantine_status[dev.device_id]:
                continue

            triggered = False
            if baseline_name == "qtk_baseline":
                triggered = detector_obj.evaluate_device(dev, current_epoch)
            elif baseline_name == "threshold_detector":
                triggered = detector_obj.evaluate_device(dev, current_epoch)
            elif baseline_name == "hmm_baseline":
                triggered = detector_obj.evaluate_device(dev, current_epoch)
            elif baseline_name == "isolation_forest":
                triggered = detector_obj.evaluate_device(dev, current_epoch)
            elif baseline_name == "lstm_baseline":
                triggered = detector_obj.evaluate_device(dev, current_epoch)
            elif baseline_name == "dual_trigger_qtk":
                # Run full pipeline
                hmm.predict(dev)
                TrustScore.update(dev, dev.behavioral_risk, alpha=0.8)
                if len(active_devices) >= 2:
                    hists = [d.telemetry_history for d in active_devices if d.telemetry_history]
                    adj, scores = graph_lstm.evaluate_devices(hists, prev_adj)
                    prev_adj = adj
                    for i, d in enumerate(active_devices):
                        if i < len(scores):
                            d.graph_risk = scores[i]
                fusion.predict(dev)
                triggered, reason, _ = dual_trigger.dual_trigger_decision(dev, current_epoch)

            if triggered:
                quarantine_status[dev.device_id] = True
                quarantine_epoch[dev.device_id] = current_epoch

    rogue_caught = quarantine_status[rogue.device_id]
    r_epoch = quarantine_epoch[rogue.device_id]

    false_positives = sum(1 for d in [phone, laptop] if quarantine_status[d.device_id])
    true_negatives = sum(1 for d in [phone, laptop] if not quarantine_status[d.device_id])

    tp = 1 if rogue_caught else 0
    fn = 0 if rogue_caught else 1
    fp = false_positives
    tn = true_negatives

    clf_metrics = calculate_classification_metrics(tp, fp, fn, tn)
    sys_metrics = calculate_qtk_system_metrics(
        quarantined_rogue_epochs=[r_epoch],
        injection_epoch=injection_epoch,
        total_epochs=epochs,
        false_quarantined_legit_count=false_positives,
        total_legitimate_devices=2
    )

    return {
        "baseline_name": baseline_name,
        "run_id": run_id,
        "seed": seed,
        **clf_metrics,
        **sys_metrics
    }

def run_experiment(
    num_runs: int = 20,
    epochs: int = 30,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates all 5 baselines + Dual-Trigger QTK across independent simulation runs.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("==================================================")
    print("Running Comparative Behavioral Baselines Experiment")
    print("==================================================")

    # Initialize detectors
    qtk_base = QTKBaseline(delta_inact=5)
    thresh_base = ThresholdDetector(sync_threshold=12.0, session_threshold=400.0, anomaly_threshold=0.50)
    hmm_detector = HMMDetector()
    hmm_base = HMMBaseline(theta_R=0.65, hmm_detector=hmm_detector)
    iforest_base = IsolationForestBaseline(contamination=0.08, random_state=base_seed)
    lstm_base = LSTMBaseline(feature_dim=5, hidden_dim=16, seq_len=12, seed=base_seed)

    graph_lstm = GraphLSTM(beta=0.8, seed=base_seed)
    fusion = RiskFusion()
    dual_trigger = DualTrigger(delta_inact=5, theta_R=0.65)

    # Pre-train on training dataset
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
        hmm_detector.fit_from_dataset(train_records)
        normal_telemetries = [r["context_telemetry"] for r in train_records if r.get("ground_truth_label") == 0]
        iforest_base.fit_on_normal(normal_telemetries)
        lstm_base.fit_on_normal(normal_telemetries)
        fusion.fit_from_dataset(train_records, hmm_detector, graph_lstm, epochs=80, lr=0.05)

    # Calibrate theta_R on validation dataset
    theta_R = 0.65
    if os.path.exists(val_path):
        val_records = []
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_records.append(json.loads(line))
        theta_R = fusion.calibrate_threshold(val_records, hmm_detector, graph_lstm, max_fpr=0.10, default_theta=0.65)
        print(f"Calibrated theta_R on VALIDATION split: {theta_R:.3f}")

    hmm_base.theta_R = theta_R
    dual_trigger.theta_R = theta_R

    baselines_to_test = [
        ("qtk_baseline", qtk_base),
        ("threshold_detector", thresh_base),
        ("hmm_baseline", hmm_base),
        ("isolation_forest", iforest_base),
        ("lstm_baseline", lstm_base),
        ("dual_trigger_qtk", dual_trigger)
    ]

    all_baseline_results = {}
    flat_rows = []

    for b_name, b_obj in baselines_to_test:
        print(f"Evaluating {b_name} across {num_runs} independent test runs...")
        runs_list = []
        for r in range(num_runs):
            seed = base_seed + 2000 + r
            run_res = evaluate_baseline_on_run(
                baseline_name=b_name,
                run_id=r,
                seed=seed,
                epochs=epochs,
                detector_obj=b_obj,
                hmm=hmm_detector,
                graph_lstm=graph_lstm,
                fusion=fusion,
                dual_trigger=dual_trigger
            )
            runs_list.append(run_res)
            flat_rows.append(run_res)

        agg = aggregate_run_metrics(runs_list)
        all_baseline_results[b_name] = {
            "summary": agg,
            "runs": runs_list
        }

        print(f"  [{b_name}] Detection Rate: {agg['detection_rate']['mean']:.2%} ± {agg['detection_rate']['ci_margin']:.2%}")
        print(f"  [{b_name}] FPR:            {agg['false_positive_rate']['mean']:.2%} ± {agg['false_positive_rate']['ci_margin']:.2%}")
        print(f"  [{b_name}] F1-Score:       {agg['f1_score']['mean']:.4f} ± {agg['f1_score']['ci_margin']:.4f}")
        print(f"  [{b_name}] Latency:        {agg['avg_detection_latency']['mean']:.2f} epochs")
        print("--------------------------------------------------")

    # Save to JSON and CSV
    json_path = os.path.join(raw_dir, "behavioral_baselines.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_baseline_results, f, indent=4)

    csv_path = os.path.join(raw_dir, "behavioral_baselines_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)

    return all_baseline_results

if __name__ == "__main__":
    run_experiment()
