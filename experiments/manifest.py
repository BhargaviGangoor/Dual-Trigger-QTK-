import os
import sys
import json
import time
import platform
import datetime
from typing import Dict, Any, Optional

def create_experiment_manifest(
    experiment_name: str = "Dual-Trigger-QTK-Master",
    base_seed: int = 42,
    num_runs: int = 20,
    epochs_per_run: int = 30,
    results_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs and persists a comprehensive experiment manifest containing environment,
    hyperparameter configuration, random seed schedules, and dataset split metadata.
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    os.makedirs(results_dir, exist_ok=True)

    manifest = {
        "manifest_version": "1.0.0",
        "experiment_name": experiment_name,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine()
        },
        "hyperparameters": {
            "qtk": {
                "delta_inact": 5,
                "theta_R": 0.65,
                "shamir_prime": 2147483647
            },
            "trust": {
                "alpha_decay": 0.80,
                "initial_trust": 1.0
            },
            "dynamic_graph": {
                "beta_decay": 0.80,
                "similarity_weights": [0.25, 0.25, 0.25, 0.25]
            },
            "weighted_gnn": {
                "feature_dim": 5,
                "hidden_dim": 16,
                "normalization": "symmetric_laplacian"
            },
            "graph_lstm": {
                "seq_len": 12,
                "hidden_dim": 16,
                "learning_rate": 0.01,
                "training_epochs": 30,
                "regularization_l2": 1e-4,
                "training_assumption": "normal_trajectories_only"
            },
            "risk_fusion": {
                "input_dim": 3,
                "weights_initial": [1.8, 2.2, 1.2],
                "bias_initial": -1.0,
                "loss": "binary_cross_entropy"
            }
        },
        "dataset_and_seeds": {
            "base_seed": base_seed,
            "total_runs": num_runs,
            "epochs_per_run": epochs_per_run,
            "train_runs": int(num_runs * 0.70),
            "val_runs": int(num_runs * 0.15),
            "test_runs": num_runs - int(num_runs * 0.70) - int(num_runs * 0.15),
            "train_seed_range": [base_seed, base_seed + int(num_runs * 0.70) - 1],
            "val_seed_range": [base_seed + int(num_runs * 0.70), base_seed + int(num_runs * 0.85) - 1],
            "test_seed_range": [base_seed + int(num_runs * 0.85), base_seed + num_runs - 1]
        },
        "deliverables_traceability_matrix": {
            "Table 1 (Baseline vs Dual-Trigger Performance)": {
                "raw_file": "results/raw/baseline_vs_dual.json",
                "csv_file": "results/tables/table1_baseline_vs_dual.csv",
                "latex_file": "results/tables/table1_baseline_vs_dual.tex",
                "seed_offset": "base_seed + 1000 + r (Seeds 1042-1061)",
                "config": {"delta_inact": 5, "theta_R": 0.65, "injection_epoch": 10, "epochs": 30}
            },
            "Table 2 (Comparative Behavioral Baselines)": {
                "raw_file": "results/raw/behavioral_baselines.json",
                "csv_file": "results/tables/table2_behavioral_baselines.csv",
                "latex_file": "results/tables/table2_behavioral_baselines.tex",
                "seed_offset": "base_seed + 2000 + r (Seeds 2042-2061)",
                "config": {"baselines": ["qtk_baseline", "threshold_detector", "hmm_baseline", "isolation_forest", "lstm_baseline", "dual_trigger_qtk"]}
            },
            "Table 3 (Component Ablation Study)": {
                "raw_file": "results/raw/ablation.json",
                "csv_file": "results/tables/table3_ablation_study.csv",
                "latex_file": "results/tables/table3_ablation_study.tex",
                "seed_offset": "base_seed + 3000 + r (Seeds 3042-3061)",
                "config": {"variants": ["hmm_only", "hmm_plus_trust", "hmm_plus_graph", "full_model"]}
            },
            "Table 4 (Adversarial Mimicry Analysis)": {
                "raw_file": "results/raw/mimicry.json",
                "csv_file": "results/tables/table4_adversarial_mimicry.csv",
                "latex_file": "results/tables/table4_adversarial_mimicry.tex",
                "seed_offset": "base_seed + 4000 + r (Seeds 4042-4061)",
                "config": {"strategies": ["naive_rogue", "moderate_mimicry", "strong_mimicry"]}
            },
            "Table 5 (False-Quarantine & Availability)": {
                "raw_file": "results/raw/false_quarantine.json",
                "csv_file": "results/tables/table5_false_quarantine_availability.csv",
                "latex_file": "results/tables/table5_false_quarantine_availability.tex",
                "seed_offset": "base_seed + 5000 + r (Seeds 5042-5061)",
                "config": {"scenarios": ["normal_legitimate", "irregular_legitimate", "long_idle_legitimate", "network_changing_legitimate"]}
            },
            "Table 6 (Computational Scalability Benchmark)": {
                "raw_file": "results/raw/scalability.json",
                "csv_file": "results/tables/table6_scalability_benchmark.csv",
                "latex_file": "results/tables/table6_scalability_benchmark.tex",
                "config": {"device_counts": [4, 8, 16, 32, 64], "trials_per_count": 5}
            },
            "Figure 1 (Baseline vs Dual-Trigger Detection Curve)": {
                "source": "results/raw/baseline_vs_dual.json",
                "pdf_file": "results/figures/fig1_baseline_vs_dual.pdf",
                "png_file": "results/figures/fig1_baseline_vs_dual.png"
            },
            "Figure 2 (Component Ablation F1 and Latency)": {
                "source": "results/raw/ablation.json",
                "pdf_file": "results/figures/fig2_ablation_f1_latency.pdf",
                "png_file": "results/figures/fig2_ablation_f1_latency.png"
            },
            "Figure 3 (Adversarial Mimicry Evasion Durations)": {
                "source": "results/raw/mimicry.json",
                "pdf_file": "results/figures/fig3_mimicry_evasion.pdf",
                "png_file": "results/figures/fig3_mimicry_evasion.png"
            },
            "Figure 4 (False Quarantine Availability Loss)": {
                "source": "results/raw/false_quarantine.json",
                "pdf_file": "results/figures/fig4_availability_loss.pdf",
                "png_file": "results/figures/fig4_availability_loss.png"
            },
            "Figure 5 (Parameter Sensitivity Heatmap)": {
                "source": "results/raw/sensitivity.json",
                "pdf_file": "results/figures/fig5_sensitivity_heatmap.pdf",
                "png_file": "results/figures/fig5_sensitivity_heatmap.png"
            },
            "Figure 6 (Computational Scalability Curve)": {
                "source": "results/raw/scalability.json",
                "pdf_file": "results/figures/fig6_scalability_runtime.pdf",
                "png_file": "results/figures/fig6_scalability_runtime.png"
            }
        },
        "target_venue": "ICOCO 2026",
        "scientific_integrity_guarantee": "Empirical runs only; all metrics generated dynamically from raw experiment logs."
    }

    manifest_path = os.path.join(results_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return manifest

if __name__ == "__main__":
    create_experiment_manifest()
