"""
Evaluation pipeline runner that orchestrates and executes all experiments defined in Section VI,
as well as newly added experiments for IEEE paper generation.
"""

import sys
import os
import json

# Adjust path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import (
    baseline_vs_dual, ablation, mimicry, false_quarantine, sensitivity,
    roc_analysis, confusion_matrix, timeline, case_study
)
from evaluation.paper_plots import generate_all_plots
from evaluation.paper_tables import generate_all_tables
from evaluation.workflow_fig import generate_workflow_fig

def main():
    print("==================================================")
    print("      Dual-Trigger QTK Evaluation Runner         ")
    print("      (IEEE Conference Paper Framework)          ")
    print("==================================================")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    output_dir = os.path.join(data_dir, "processed")
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Run Core Evaluation Experiments
    res_baseline = baseline_vs_dual.run_experiment()
    print("--------------------------------------------------")
    res_ablation = ablation.run_experiment()
    print("--------------------------------------------------")
    res_mimicry = mimicry.run_experiment()
    print("--------------------------------------------------")
    res_false_q = false_quarantine.run_experiment()
    print("--------------------------------------------------")
    res_sens = sensitivity.run_experiment()
    print("--------------------------------------------------")
    
    # Combine results
    all_results = {
        "baseline_vs_dual": res_baseline,
        "ablation": res_ablation,
        "mimicry": res_mimicry,
        "false_quarantine": res_false_q,
        "sensitivity": res_sens
    }
    
    # Save numerical results to JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Numerical results successfully written to: {json_path}")
    
    # 2. Run Missing Paper Experiments
    print("--------------------------------------------------")
    roc_analysis.run_experiment()
    print("--------------------------------------------------")
    confusion_matrix.run_experiment()
    print("--------------------------------------------------")
    timeline.run_experiment()
    print("--------------------------------------------------")
    case_study.run_experiment()
    print("--------------------------------------------------")
    
    # 3. Generate Paper Tables & Figures
    print("Generating IEEE Publication Tables...")
    generate_all_tables(data_dir, results_dir)
    
    print("Generating IEEE Publication Figures (300 DPI)...")
    generate_all_plots(data_dir, results_dir)
    
    print("Generating Experimental Workflow Diagram...")
    generate_workflow_fig(os.path.join(results_dir, "figures"))
    
    print("==================================================")
    print("               Evaluation Complete!              ")
    print(f"      Check {results_dir} for output assets.     ")
    print("==================================================")

if __name__ == "__main__":
    main()
