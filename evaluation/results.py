import os
import sys
import json
import csv
import pandas as pd
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.reproducibility import verify_and_run_all
from evaluation.plots import generate_all_plots

def generate_publication_tables(results_dir: Optional[str] = None):
    """
    Parses empirical raw output files from results/raw/ and generates
    formatted CSV and LaTeX (.tex) summary tables in results/tables/.
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    raw_dir = os.path.join(results_dir, "raw")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    print("Generating IEEE Camera-Ready Tables strictly from empirical raw outputs...")

    # 1. Simulation Parameters Table
    params = [
        {"Parameter": "Simulation Epochs per Run", "Value": "30"},
        {"Parameter": "Legitimate Devices per Group", "Value": "3 (Phone primary, Laptop linked, Tablet linked)"},
        {"Parameter": "Silent Device Inactivity Age", "Value": "Configurable (Monotonically increments)"},
        {"Parameter": "Rogue Device Key Update Interval", "Value": "2 epochs (Rotates before delta_inact)"},
        {"Parameter": "Rogue Injection Epoch", "Value": "Epoch 10"},
        {"Parameter": "HMM Sequence Length (L)", "Value": "12 epochs"},
        {"Parameter": "HMM Behavioral States", "Value": "4 (Normal, Idle, Suspicious, High-Risk)"},
        {"Parameter": "Trust Update Rate (alpha)", "Value": "0.80"},
        {"Parameter": "Dynamic Graph Decay (beta)", "Value": "0.80"},
        {"Parameter": "Behavioral Risk Threshold (theta_R)", "Value": "0.65"},
        {"Parameter": "Inactivity Threshold (delta_inact)", "Value": "5 epochs"},
        {"Parameter": "Independent Repetitions", "Value": "20 runs (Deterministic Seeds 42-61)"}
    ]
    df_params = pd.DataFrame(params)
    df_params.to_csv(os.path.join(tables_dir, "table_simulation_parameters.csv"), index=False)
    with open(os.path.join(tables_dir, "table_simulation_parameters.tex"), "w", encoding="utf-8") as f:
        f.write(df_params.to_latex(index=False))

    # 2. Table 1: Baseline vs Dual-Trigger Performance (RQ1)
    bvd_path = os.path.join(raw_dir, "baseline_vs_dual.json")
    if os.path.exists(bvd_path):
        with open(bvd_path, "r", encoding="utf-8") as f:
            bvd_data = json.load(f)
        b_sum = bvd_data.get("baseline_summary", {})
        d_sum = bvd_data.get("dual_summary", {})

        t1_rows = [
            {
                "Framework": "Original QTK (Baseline)",
                "Detection Rate (%)": f"{b_sum['detection_rate']['mean']*100:.2f} ± {b_sum['detection_rate']['ci_margin']*100:.2f}",
                "FPR (%)": f"{b_sum['false_positive_rate']['mean']*100:.2f} ± {b_sum['false_positive_rate']['ci_margin']*100:.2f}",
                "Detection Latency (ep)": "N/A (Evaded)",
                "Evasion Duration (ep)": f"{b_sum['avg_evasion_duration']['mean']:.2f} ± {b_sum['avg_evasion_duration']['ci_margin']:.2f}",
                "F1-Score": f"{b_sum['f1_score']['mean']:.4f} ± {b_sum['f1_score']['ci_margin']:.4f}"
            },
            {
                "Framework": "Dual-Trigger QTK (Ours)",
                "Detection Rate (%)": f"{d_sum['detection_rate']['mean']*100:.2f} ± {d_sum['detection_rate']['ci_margin']*100:.2f}",
                "FPR (%)": f"{d_sum['false_positive_rate']['mean']*100:.2f} ± {d_sum['false_positive_rate']['ci_margin']*100:.2f}",
                "Detection Latency (ep)": f"{d_sum['avg_detection_latency']['mean']:.2f} ± {d_sum['avg_detection_latency']['ci_margin']:.2f}",
                "Evasion Duration (ep)": f"{d_sum['avg_evasion_duration']['mean']:.2f} ± {d_sum['avg_evasion_duration']['ci_margin']:.2f}",
                "F1-Score": f"{d_sum['f1_score']['mean']:.4f} ± {d_sum['f1_score']['ci_margin']:.4f}"
            }
        ]
        df_t1 = pd.DataFrame(t1_rows)
        df_t1.to_csv(os.path.join(tables_dir, "table1_baseline_vs_dual.csv"), index=False)
        with open(os.path.join(tables_dir, "table1_baseline_vs_dual.tex"), "w", encoding="utf-8") as f:
            f.write(df_t1.to_latex(index=False))

    # 3. Table 2: Comparative Behavioral Baselines
    base_path = os.path.join(raw_dir, "behavioral_baselines.json")
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        t2_rows = []
        for m, data in base_data.items():
            s = data["summary"]
            t2_rows.append({
                "Detector": m.replace("_", " ").title(),
                "Detection Rate (%)": f"{s['detection_rate']['mean']*100:.2f} ± {s['detection_rate']['ci_margin']*100:.2f}",
                "FPR (%)": f"{s['false_positive_rate']['mean']*100:.2f} ± {s['false_positive_rate']['ci_margin']*100:.2f}",
                "Latency (epochs)": f"{s['avg_detection_latency']['mean']:.2f} ± {s['avg_detection_latency']['ci_margin']:.2f}",
                "Precision": f"{s['precision']['mean']:.4f} ± {s['precision']['ci_margin']:.4f}",
                "F1-Score": f"{s['f1_score']['mean']:.4f} ± {s['f1_score']['ci_margin']:.4f}"
            })
        df_t2 = pd.DataFrame(t2_rows)
        df_t2.to_csv(os.path.join(tables_dir, "table2_behavioral_baselines.csv"), index=False)
        with open(os.path.join(tables_dir, "table2_behavioral_baselines.tex"), "w", encoding="utf-8") as f:
            f.write(df_t2.to_latex(index=False))

    # 4. Table 3: Component Ablation Study (RQ2)
    abl_path = os.path.join(raw_dir, "ablation.json")
    if os.path.exists(abl_path):
        with open(abl_path, "r", encoding="utf-8") as f:
            abl_data = json.load(f)

        cfg_name_map = {
            "hmm_only": "HMM Only",
            "hmm_plus_trust": "HMM + Trust Decay",
            "hmm_plus_graph": "HMM + Dynamic Graph",
            "full_model": "Full Dual-Trigger Model"
        }
        t3_rows = []
        for c, data in abl_data.items():
            s = data["summary"]
            t3_rows.append({
                "Ablation Configuration": cfg_name_map.get(c, c),
                "Detection Rate (%)": f"{s['detection_rate']['mean']*100:.2f} ± {s['detection_rate']['ci_margin']*100:.2f}",
                "FPR (%)": f"{s['false_positive_rate']['mean']*100:.2f} ± {s['false_positive_rate']['ci_margin']*100:.2f}",
                "Latency (epochs)": f"{s['avg_detection_latency']['mean']:.2f} ± {s['avg_detection_latency']['ci_margin']:.2f}",
                "F1-Score": f"{s['f1_score']['mean']:.4f} ± {s['f1_score']['ci_margin']:.4f}"
            })
        df_t3 = pd.DataFrame(t3_rows)
        df_t3.to_csv(os.path.join(tables_dir, "table3_ablation_study.csv"), index=False)
        with open(os.path.join(tables_dir, "table3_ablation_study.tex"), "w", encoding="utf-8") as f:
            f.write(df_t3.to_latex(index=False))

    # 5. Table 4: Adversarial Mimicry Analysis (RQ3)
    mim_path = os.path.join(raw_dir, "mimicry.json")
    if os.path.exists(mim_path):
        with open(mim_path, "r", encoding="utf-8") as f:
            mim_data = json.load(f)

        t4_rows = []
        for s_name, data in mim_data.items():
            s = data["summary"]
            t4_rows.append({
                "Attacker Strategy": s_name.replace("_", " ").title(),
                "Detection Rate (%)": f"{s['detection_rate']['mean']*100:.2f} ± {s['detection_rate']['ci_margin']*100:.2f}",
                "Detection Latency (ep)": f"{s['avg_detection_latency']['mean']:.2f} ± {s['avg_detection_latency']['ci_margin']:.2f}",
                "Evasion Duration (ep)": f"{s['avg_evasion_duration']['mean']:.2f} ± {s['avg_evasion_duration']['ci_margin']:.2f}",
                "FPR (%)": f"{s['false_positive_rate']['mean']*100:.2f} ± {s['false_positive_rate']['ci_margin']*100:.2f}",
                "F1-Score": f"{s['f1_score']['mean']:.4f} ± {s['f1_score']['ci_margin']:.4f}"
            })
        df_t4 = pd.DataFrame(t4_rows)
        df_t4.to_csv(os.path.join(tables_dir, "table4_adversarial_mimicry.csv"), index=False)
        with open(os.path.join(tables_dir, "table4_adversarial_mimicry.tex"), "w", encoding="utf-8") as f:
            f.write(df_t4.to_latex(index=False))

    # 6. Table 5: False-Quarantine & Availability Analysis (RQ4)
    fq_path = os.path.join(raw_dir, "false_quarantine.json")
    if os.path.exists(fq_path):
        with open(fq_path, "r", encoding="utf-8") as f:
            fq_data = json.load(f)

        t5_rows = []
        for sc, data in fq_data.items():
            s = data["summary"]
            t5_rows.append({
                "Legitimate Scenario": sc.replace("_", " ").title(),
                "FQ Frequency (events/dev)": f"{s['false_quarantine_rate']['mean']:.2f} ± {s['false_quarantine_rate']['ci_margin']:.2f}",
                "Availability Loss (%)": f"{s['availability_loss']['mean']*100:.2f} ± {s['availability_loss']['ci_margin']*100:.2f}",
                "Mean Time to FQ (epochs)": f"{s['mttf_epoch']['mean']:.2f} ± {s['mttf_epoch']['ci_margin']:.2f}",
                "Recovery Events": f"{s['recovery_events_count']['mean']:.2f}"
            })
        df_t5 = pd.DataFrame(t5_rows)
        df_t5.to_csv(os.path.join(tables_dir, "table5_false_quarantine_availability.csv"), index=False)
        with open(os.path.join(tables_dir, "table5_false_quarantine_availability.tex"), "w", encoding="utf-8") as f:
            f.write(df_t5.to_latex(index=False))

    # 7. Table 6: Scalability Benchmark
    scal_path = os.path.join(raw_dir, "scalability.json")
    if os.path.exists(scal_path):
        with open(scal_path, "r", encoding="utf-8") as f:
            scal_data = json.load(f)

        t6_rows = []
        for entry in scal_data:
            t6_rows.append({
                "Group Size (N)": entry["num_devices"],
                "Dynamic Graph (ms)": f"{entry['graph_evolution_time_ms']:.2f} ± {entry['graph_evolution_std_ms']:.2f}",
                "Weighted GNN (ms)": f"{entry['gnn_forward_time_ms']:.2f} ± {entry['gnn_forward_std_ms']:.2f}",
                "Graph-LSTM (ms)": f"{entry['graph_lstm_time_ms']:.2f} ± {entry['graph_lstm_std_ms']:.2f}",
                "Total Per-Epoch (ms)": f"{entry['total_epoch_time_ms']:.2f} ± {entry['total_epoch_std_ms']:.2f}",
                "Peak Memory (KB)": f"{entry['peak_memory_kb']:.1f}"
            })
        df_t6 = pd.DataFrame(t6_rows)
        df_t6.to_csv(os.path.join(tables_dir, "table6_scalability_benchmark.csv"), index=False)
        with open(os.path.join(tables_dir, "table6_scalability_benchmark.tex"), "w", encoding="utf-8") as f:
            f.write(df_t6.to_latex(index=False))

    print(f"All publication tables successfully written to {tables_dir}")

def main():
    print("==================================================")
    print("      Dual-Trigger QTK Evaluation Runner         ")
    print("   (Automated Camera-Ready Experiment Engine)    ")
    print("==================================================")

    # 1. Execute all experiments cleanly from scratch
    verify_and_run_all(num_runs=20, base_seed=42)

    # 2. Build LaTeX and CSV Tables from generated outputs
    generate_publication_tables()

    # 3. Generate 300 DPI Figures from generated outputs
    generate_all_plots()

    print("\n==================================================")
    print("EVALUATION & DELIVERABLES GENERATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
