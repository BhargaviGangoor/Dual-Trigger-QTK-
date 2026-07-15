import os
import json
from experiments import baseline_vs_dual, ablation, mimicry, false_quarantine, sensitivity
from evaluation.plots import save_plots

def main():
    print("==================================================")
    print("      Dual-Trigger QTK Evaluation Runner         ")
    print("==================================================")
    
    # Create required data directories
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run experiments
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
    
    # Generate and save plots
    save_plots(res_baseline, res_ablation, output_dir)
    print("==================================================")
    print("               Evaluation Complete!              ")
    print("==================================================")

if __name__ == "__main__":
    main()
