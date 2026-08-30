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
from simulator.mimicry_attacker import MimicryAttacker
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from evaluation.metrics import calculate_classification_metrics, calculate_qtk_system_metrics
from evaluation.confidence import aggregate_run_metrics

def run_mimicry_trial(
    strategy_name: str,
    run_id: int,
    seed: int,
    epochs: int = 30,
    injection_epoch: int = 10,
    delta_inact: int = 5,
    theta_R: float = 0.65,
    alpha: float = 0.8,
    beta: float = 0.8,
    hmm: Optional[HMMDetector] = None,
    graph_lstm: Optional[GraphLSTM] = None,
    fusion: Optional[RiskFusion] = None
) -> Dict[str, Any]:
    """
    Executes a single simulation trial against an adaptive mimicry attacker:
    - naive_rogue: low adaptation
    - moderate_mimicry: matches timing & message frequency
    - strong_mimicry: near-identical feature distributions
    """
    rng = random.Random(seed)
    tracker = EpochTracker()
    dual_trigger = DualTrigger(delta_inact=delta_inact, theta_R=theta_R)

    phone = LegitimateDevice(
        device_id="phone_primary", owner_id="user_0", name="Android Phone",
        device_type="primary", profile_name="Student", ip_address="172.16.23.10"
    )
    laptop = LegitimateDevice(
        device_id="laptop_linked", owner_id="user_0", name="MacBook Pro",
        device_type="linked", profile_name="Student", ip_address="172.16.23.20"
    )
    mimic_rogue = MimicryAttacker(
        device_id="mimic_rogue_dev", owner_id="user_0", name="Mimicking Rogue",
        device_type="linked", profile_name="Student", mimicry_strength=strategy_name
    )

    active_devices = [phone, laptop]
    prev_adj = None
    quarantine_status = {d.device_id: False for d in [phone, laptop, mimic_rogue]}
    quarantine_epoch = {d.device_id: None for d in [phone, laptop, mimic_rogue]}

    for epoch in range(1, epochs + 1):
        tracker.increment_epoch()
        current_epoch = tracker.current_epoch

        if current_epoch == injection_epoch:
            active_devices.append(mimic_rogue)
            mimic_rogue.update_key(current_epoch)

        primary_meta = phone.get_latest_telemetry()
        for dev in active_devices:
            dev.simulate_epoch(current_epoch, primary_meta if dev != phone else None, rng=rng)

        # HMM + Trust
        for dev in active_devices:
            hmm.predict(dev)
            TrustScore.update(dev, dev.behavioral_risk, alpha=alpha)

        # Dynamic Graph & GNN/Graph-LSTM
        if len(active_devices) >= 2:
            hists = [d.telemetry_history for d in active_devices if d.telemetry_history]
            adj, scores = graph_lstm.evaluate_devices(hists, prev_adj)
            prev_adj = adj
            for i, dev in enumerate(active_devices):
                if i < len(scores):
                    dev.graph_risk = scores[i]

        # Risk Fusion
        for dev in active_devices:
            fusion.predict(dev)

        # Dual-Trigger Quarantine Evaluation
        for dev in active_devices:
            if quarantine_status[dev.device_id]:
                continue
            triggered, reason, _ = dual_trigger.dual_trigger_decision(dev, current_epoch)
            if triggered:
                quarantine_status[dev.device_id] = True
                quarantine_epoch[dev.device_id] = current_epoch

    rogue_caught = quarantine_status[mimic_rogue.device_id]
    r_epoch = quarantine_epoch[mimic_rogue.device_id]

    false_positives = sum(1 for d in [phone, laptop] if quarantine_status[d.device_id])
    true_negatives = sum(1 for d in [phone, laptop] if not quarantine_status[d.device_id])

    tp = 1 if rogue_caught else 0
    fn = 0 if rogue_caught else 1
    fp = false_positives
    tn = true_negatives

    clf = calculate_classification_metrics(tp, fp, fn, tn)
    sys_m = calculate_qtk_system_metrics(
        quarantined_rogue_epochs=[r_epoch],
        injection_epoch=injection_epoch,
        total_epochs=epochs,
        false_quarantined_legit_count=false_positives,
        total_legitimate_devices=2
    )

    return {
        "strategy": strategy_name,
        "run_id": run_id,
        "seed": seed,
        **clf,
        **sys_m
    }

def run_experiment(
    num_runs: int = 20,
    epochs: int = 30,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs RQ3 Adversarial Mimicry Experiment across varying attacker strengths.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("==================================================")
    print("Running Adversarial Mimicry Attack Experiment (RQ3)")
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

    strategies = ["naive_rogue", "moderate_mimicry", "strong_mimicry"]
    all_mimicry_results = {}
    flat_rows = []

    for strat in strategies:
        print(f"Evaluating strategy: {strat} across {num_runs} runs...")
        runs_list = []
        for r in range(num_runs):
            seed = base_seed + 4000 + r
            graph_lstm.reset_norm_stats()
            res = run_mimicry_trial(
                strategy_name=strat,
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
        all_mimicry_results[strat] = {
            "summary": agg,
            "runs": runs_list
        }

        print(f"  [{strat}] Detection Rate: {agg['detection_rate']['mean']:.2%} ± {agg['detection_rate']['ci_margin']:.2%}")
        print(f"  [{strat}] Evasion Dur:   {agg['avg_evasion_duration']['mean']:.2f} ± {agg['avg_evasion_duration']['ci_margin']:.2f} epochs")
        print(f"  [{strat}] F1-Score:       {agg['f1_score']['mean']:.4f} ± {agg['f1_score']['ci_margin']:.4f}")
        print(f"  [{strat}] Latency:        {agg['avg_detection_latency']['mean']:.2f} epochs")
        print("--------------------------------------------------")

    json_path = os.path.join(raw_dir, "mimicry.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_mimicry_results, f, indent=4)

    csv_path = os.path.join(raw_dir, "mimicry_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)

    return all_mimicry_results

if __name__ == "__main__":
    run_experiment()
