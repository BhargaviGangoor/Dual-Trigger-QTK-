import os
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def save_plots(baseline_vs_dual_results: Dict[str, Any], ablation_results: Dict[str, Any], output_dir: str):
    """Generates and saves performance charts to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Baseline vs Dual-Trigger Evasion Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['Plain QTK (Baseline)', 'Dual-Trigger QTK']
    evasion_durations = [
        baseline_vs_dual_results.get("evasion_duration_baseline", 30),
        baseline_vs_dual_results.get("evasion_duration_dual", 0)
    ]
    colors = ['#f44336', '#4caf50']
    
    bars = ax.bar(categories, evasion_durations, color=colors, width=0.5)
    ax.set_ylabel('Rogue Evasion Duration (epochs)', fontsize=12)
    ax.set_title('Rogue-Device Access Persistence Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(evasion_durations) + 5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} epochs',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qtk_evasion_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Ablation Detection Latency Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    configs = list(ablation_results.keys())
    latencies = [ablation_results[cfg]['latency_epochs'] for cfg in configs]
    
    # Capitalize for labels
    labels = [cfg.replace('_', ' ').title() for cfg in configs]
    
    bars = ax.bar(labels, latencies, color='#2196f3', width=0.5)
    ax.set_ylabel('Detection Latency (epochs)', fontsize=12)
    ax.set_title('Ablation Study: Anomaly Detection Latency', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 40)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_latency.png'), dpi=150)
    plt.close()
    
    print(f"Evaluation charts saved successfully to: {output_dir}")
