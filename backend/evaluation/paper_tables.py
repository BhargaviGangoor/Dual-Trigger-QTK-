import os
import json
import csv
import pandas as pd
from typing import Dict, Any

def generate_simulation_parameters_table(output_dir: str):
    """Generates the static Simulation Parameters Table."""
    params = [
        {"Parameter": "Epochs", "Value": "100 (varies by exp)"},
        {"Parameter": "Number of legitimate devices", "Value": "10+"},
        {"Parameter": "Silent devices", "Value": "Configurable"},
        {"Parameter": "Rogue devices", "Value": "1-5"},
        {"Parameter": "Injection epoch", "Value": "10"},
        {"Parameter": "Sequence length", "Value": "2+"},
        {"Parameter": "HMM states", "Value": "Normal, Idle, Suspicious, HighRisk"},
        {"Parameter": "α (Trust update rate)", "Value": "0.8"},
        {"Parameter": "β (Graph decay)", "Value": "0.8"},
        {"Parameter": "θ_R (Risk Threshold)", "Value": "0.65"},
        {"Parameter": "δ_inact (Inactivity Threshold)", "Value": "5"},
        {"Parameter": "Simulation duration", "Value": "Varies by experiment"}
    ]
    df = pd.DataFrame(params)
    df.to_csv(os.path.join(output_dir, "simulation_parameters.csv"), index=False)
    with open(os.path.join(output_dir, "simulation_parameters.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))

def generate_evaluation_metrics_table(all_results: Dict[str, Any], output_dir: str):
    """Generates Table 1: Baseline vs Dual-Trigger Performance."""
    try:
        dr = all_results.get("baseline_vs_dual", {}).get("dr_dual", 0.0)
        fpr = all_results.get("baseline_vs_dual", {}).get("fpr_dual", 0.0)
        lat = all_results.get("baseline_vs_dual", {}).get("avg_latency_dual", 0.0)
        f1 = all_results.get("baseline_vs_dual", {}).get("f1_dual", 0.0)
    except Exception:
        dr, fpr, lat, f1 = 0, 0, 0, 0
        
    data = [
        {"Metric": "Detection Rate (DR)", "Value": f"{dr:.2f}%"},
        {"Metric": "False Positive Rate (FPR)", "Value": f"{fpr:.2f}%"},
        {"Metric": "Detection Latency", "Value": f"{lat:.2f} epochs"},
        {"Metric": "F1-Score", "Value": f"{f1:.4f}"}
    ]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
    with open(os.path.join(output_dir, "evaluation_metrics.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))

def generate_ablation_table(all_results: Dict[str, Any], output_dir: str):
    """Generates the Ablation Study Table."""
    ablation_res = all_results.get("ablation", {})
    rows = []
    for config, metrics in ablation_res.items():
        if isinstance(metrics, dict):
            rows.append({
                "Configuration": config.replace("_", " ").title(),
                "Detection Rate": f"{metrics.get('dr_mean', 0.0):.2f}%",
                "FPR": f"{metrics.get('fpr_mean', 0.0):.2f}%",
                "Latency": f"{metrics.get('avg_latency_mean', 0.0):.2f}",
                "F1-Score": f"{metrics.get('f1_mean', 0.0):.4f}"
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "ablation_study.csv"), index=False)
        with open(os.path.join(output_dir, "ablation_study.tex"), "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False))

def generate_case_study_table(data_dir: str, output_dir: str):
    """Generates the Case Study Table from JSON."""
    case_path = os.path.join(data_dir, "case_study.json")
    if os.path.exists(case_path):
        with open(case_path, "r") as f:
            case_data = json.load(f)
        rows = case_data.get("case_study", [])
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(output_dir, "case_study.csv"), index=False)
            with open(os.path.join(output_dir, "case_study.tex"), "w", encoding="utf-8") as f:
                f.write(df.to_latex(index=False))

def generate_all_tables(data_dir: str, results_dir: str):
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # Load primary results
    results_path = os.path.join(data_dir, "processed", "results.json")
    if not os.path.exists(results_path):
        results_path = os.path.join(data_dir, "results.json")
        
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    generate_simulation_parameters_table(tables_dir)
    generate_evaluation_metrics_table(all_results, tables_dir)
    generate_ablation_table(all_results, tables_dir)
    generate_case_study_table(data_dir, tables_dir)
    print(f"Tables successfully generated in {tables_dir}")
