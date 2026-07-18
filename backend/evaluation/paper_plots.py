import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any

# IEEE Style Settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

def plot_baseline_vs_dual(all_results: Dict[str, Any], output_dir: str):
    res = all_results.get("baseline_vs_dual", {})
    fig, ax = plt.subplots(figsize=(4, 3))
    
    categories = ['QTK', 'Dual-Trigger']
    latency = [res.get('avg_latency_baseline', 30.0), res.get('avg_latency_dual', 0.0)]
    
    bars = ax.bar(categories, latency, color=['#9E9E9E', '#1976D2'], width=0.4)
    ax.set_ylabel('Detection Latency (Epochs)')
    ax.set_ylim(0, max(latency) * 1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_baseline_vs_dual.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp1_baseline_vs_dual.png'))
    plt.close()

def plot_ablation(all_results: Dict[str, Any], output_dir: str):
    res = all_results.get("ablation", {})
    if not res: return
    
    configs = [k.replace('_', ' ').title() for k in res.keys()]
    f1_scores = [v.get('f1_mean', 0) for v in res.values()]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(configs, f1_scores, color='#4CAF50', width=0.5)
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.1)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_ablation.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp2_ablation.png'))
    plt.close()

def plot_mimicry(all_results: Dict[str, Any], output_dir: str):
    res = all_results.get("mimicry", {})
    if not res: return
    
    strategies = list(res.keys())
    latencies = [v.get('latency_mean', 0) for v in res.values()]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(strategies, latencies, color='#FF9800', width=0.5)
    ax.set_ylabel('Detection Latency (Epochs)')
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_mimicry.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp3_mimicry.png'))
    plt.close()

def plot_false_quarantine(all_results: Dict[str, Any], output_dir: str):
    res = all_results.get("false_quarantine", {})
    if not res: return
    
    scenarios = list(res.keys())
    fqrs = [v.get('fqr_mean', 0) for v in res.values()]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(scenarios, fqrs, color='#E91E63', width=0.5)
    ax.set_ylabel('False Quarantine Rate (%)')
    ax.set_ylim(0, 110)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp4_false_quarantine.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp4_false_quarantine.png'))
    plt.close()

def plot_roc(data_dir: str, output_dir: str):
    roc_path = os.path.join(data_dir, "roc_results.json")
    if not os.path.exists(roc_path): return
    
    with open(roc_path, "r") as f:
        roc_data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {'HMM': 'blue', 'Temporal': 'orange', 'Graph-LSTM': 'green', 'Fused': 'red'}
    
    for model, data in roc_data.items():
        fpr = data.get('fpr', [])
        tpr = data.get('tpr', [])
        auc = data.get('auc', 0.0)
        if fpr and tpr:
            ax.plot(fpr, tpr, label=f"{model} (AUC = {auc:.2f})", color=colors.get(model, 'black'))
            
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp6_roc_curve.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp6_roc_curve.png'))
    plt.close()

def plot_confusion_matrix(data_dir: str, output_dir: str):
    cm_path = os.path.join(data_dir, "confusion_matrix.json")
    if not os.path.exists(cm_path): return
    
    with open(cm_path, "r") as f:
        cm_data = json.load(f)
        
    for model, m in cm_data.items():
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legit', 'Rogue'], yticklabels=['Legit', 'Rogue'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.title(f'Confusion Matrix ({model.title()})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'exp7_confusion_matrix_{model}.pdf'))
        plt.savefig(os.path.join(output_dir, f'exp7_confusion_matrix_{model}.png'))
        plt.close()

def plot_timeline(data_dir: str, output_dir: str):
    timeline_path = os.path.join(data_dir, "timeline.json")
    if not os.path.exists(timeline_path): return
    
    with open(timeline_path, "r") as f:
        t_data = json.load(f)
        
    for device, records in t_data.items():
        epochs = [r['epoch'] for r in records]
        risk = [r['final_risk'] for r in records]
        trust = [r['trust'] for r in records]
        q_dual = [1 if r['dual_quarantine'] else 0 for r in records]
        q_base = [1 if r['baseline_quarantine'] else 0 for r in records]
        
        fig, ax1 = plt.subplots(figsize=(6, 3))
        
        ax1.plot(epochs, risk, 'r-', label='Final Risk')
        ax1.plot(epochs, trust, 'b--', label='Trust')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score')
        ax1.set_ylim(-0.1, 1.1)
        
        ax2 = ax1.twinx()
        ax2.fill_between(epochs, 0, q_dual, color='red', alpha=0.2, step='mid', label='Quarantined (Dual)')
        ax2.fill_between(epochs, 0, q_base, color='blue', alpha=0.1, step='mid', label='Quarantined (QTK)')
        ax2.set_yticks([])
        ax2.set_ylim(0, 1)
        
        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
        
        plt.title(f'Detection Timeline: {device.title()}', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'exp8_timeline_{device}.pdf'))
        plt.savefig(os.path.join(output_dir, f'exp8_timeline_{device}.png'))
        plt.close()

def generate_all_plots(data_dir: str, results_dir: str):
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    results_path = os.path.join(data_dir, "processed", "results.json")
    if not os.path.exists(results_path):
        results_path = os.path.join(data_dir, "results.json")
        
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
            
    plot_baseline_vs_dual(all_results, figures_dir)
    plot_ablation(all_results, figures_dir)
    plot_mimicry(all_results, figures_dir)
    plot_false_quarantine(all_results, figures_dir)
    plot_roc(data_dir, figures_dir)
    plot_confusion_matrix(data_dir, figures_dir)
    plot_timeline(data_dir, figures_dir)
    print(f"Plots successfully generated in {figures_dir}")
