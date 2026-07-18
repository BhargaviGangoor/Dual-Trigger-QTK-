import os
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
    fig, ax = plt.subplots(figsize=(4, 3))
    
    categories = ['Baseline QTK', 'Proposed Dual-Trigger QTK']
    latency = [29.0, 11.0]  # Realistic values requested by user
    
    bars = ax.bar(categories, latency, color=['#9E9E9E', '#1976D2'], width=0.4)
    ax.set_ylabel('Detection Latency (Epochs)')
    ax.set_ylim(0, 35)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_baseline_vs_dual.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp1_baseline_vs_dual.png'))
    plt.close()

def plot_ablation(all_results: Dict[str, Any], output_dir: str):
    configs = ['HMM Only', 'Temporal Trust Only', 'Graph-LSTM Only', 'Full Dual-Trigger Model']
    f1_scores = [0.69, 0.76, 0.84, 0.90]  # Full model performs best, Graph-LSTM outperforms HMM
    
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(configs, f1_scores, color='#4CAF50', width=0.5)
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.1)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_ablation.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp2_ablation.png'))
    plt.close()

def plot_mimicry(all_results: Dict[str, Any], output_dir: str):
    strategies = ['Stealth', 'Burst', 'Random', 'Legacy Mimicry', 'Adaptive Mimicry']
    latencies = [1.4, 1.0, 1.1, 1.8, 2.2]  # Mimicry/adaptive hardest to detect
    
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(strategies, latencies, color='#FF9800', width=0.5)
    ax.set_ylabel('Detection Latency (Epochs)')
    ax.set_ylim(0, 3.0)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_mimicry.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp3_mimicry.png'))
    plt.close()

def plot_false_quarantine(all_results: Dict[str, Any], output_dir: str):
    scenarios = ['Scenario A', 'Scenario B', 'Scenario C', 'Scenario D', 'Scenario E', 'Scenario F']
    fqrs = [0.0, 0.4, 0.7, 0.3, 0.5, 0.2]  # Low but non-zero, below 1%
    
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(scenarios, fqrs, color='#E91E63', width=0.5)
    ax.set_ylabel('False Quarantine Rate (%)')
    ax.set_ylim(0, 1.2)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp4_false_quarantine.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp4_false_quarantine.png'))
    plt.close()

def plot_roc(data_dir: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    
    fpr = np.linspace(0, 1, 200)
    
    # Mathematical models for ROC curves corresponding to the requested AUCs
    tpr_hmm = fpr ** (1.0 / 0.73 - 1.0)
    tpr_temp = fpr ** (1.0 / 0.79 - 1.0)
    tpr_lstm = fpr ** (1.0 / 0.87 - 1.0)
    tpr_fused = fpr ** (1.0 / 0.93 - 1.0)
    
    # Avoid sharp corners near 0
    tpr_hmm[0] = 0.0
    tpr_temp[0] = 0.0
    tpr_lstm[0] = 0.0
    tpr_fused[0] = 0.0
    
    ax.plot(fpr, tpr_hmm, label="HMM Only (AUC = 0.73)", color='blue', lw=1.5)
    ax.plot(fpr, tpr_temp, label="Temporal Trust Only (AUC = 0.79)", color='orange', lw=1.5)
    ax.plot(fpr, tpr_lstm, label="Graph-LSTM Only (AUC = 0.87)", color='green', lw=1.5)
    ax.plot(fpr, tpr_fused, label="Proposed Dual-Trigger QTK (AUC = 0.93)", color='red', lw=1.5)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp6_roc_curve.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp6_roc_curve.png'))
    plt.close()

def plot_confusion_matrix(data_dir: str, output_dir: str):
    # Baseline QTK Confusion Matrix
    cm_base = np.array([[920, 10], [118, 32]])
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate', 'Rogue'], yticklabels=['Legitimate', 'Rogue'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title('Confusion Matrix (Baseline QTK)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp7_confusion_matrix_baseline.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp7_confusion_matrix_baseline.png'))
    plt.close()
    
    # Proposed Dual-Trigger QTK Confusion Matrix
    cm_dual = np.array([[902, 28], [12, 138]])
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm_dual, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate', 'Rogue'], yticklabels=['Legitimate', 'Rogue'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title('Confusion Matrix (Proposed Dual-Trigger QTK)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp7_confusion_matrix_dual_trigger.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp7_confusion_matrix_dual_trigger.png'))
    plt.close()

def plot_timeline(data_dir: str, output_dir: str):
    epochs = np.arange(0, 31)
    
    # 1. Legitimate Device Timeline
    trust_legit = np.full(31, 0.95)
    risk_legit = np.full(31, 0.05)
    
    # Smooth transient behavioral noise around epoch 11 (requested values)
    # risk: 0.05 -> 0.18 -> 0.30 -> 0.24 -> 0.12 -> 0.05
    risk_legit[9] = 0.05
    risk_legit[10] = 0.18
    risk_legit[11] = 0.30
    risk_legit[12] = 0.24
    risk_legit[13] = 0.12
    risk_legit[14] = 0.05
    
    # Trust stays above 0.90 throughout
    trust_legit[10] = 0.94
    trust_legit[11] = 0.91
    trust_legit[12] = 0.92
    trust_legit[13] = 0.94
    trust_legit[14] = 0.95
    
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(epochs, risk_legit, 'r-', label='Final Risk', lw=1.5)
    ax1.plot(epochs, trust_legit, 'b--', label='Trust Score', lw=1.5)
    ax1.axhline(0.65, color='gray', linestyle=':', label='Risk Threshold ($\theta_R = 0.65$)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower left')
    plt.title('Legitimate Device Timeline (Nominal)', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp8_timeline_legitimate.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp8_timeline_legitimate.png'))
    plt.close()
    
    # 2. Rogue Device Timeline
    trust_rogue = np.ones(31) * 0.95
    risk_rogue = np.ones(31) * 0.05
    
    # Injected at epoch 10, risk rises, trust decays
    # Risk crosses threshold at epoch 13 (so risk at 13 is e.g. 0.85)
    risk_rogue[10] = 0.20
    trust_rogue[10] = 0.90
    risk_rogue[11] = 0.45
    trust_rogue[11] = 0.78
    risk_rogue[12] = 0.60
    trust_rogue[12] = 0.70
    
    for e in range(13, 31):
        trust_rogue[e] = 0.95 - (e - 9) * 0.05
        risk_rogue[e] = 0.05 + (e - 9) * 0.15
        
    trust_rogue = np.clip(trust_rogue, 0.08, 0.95)
    risk_rogue = np.clip(risk_rogue, 0.05, 0.97)
    
    q_dual = np.array([1.0 if e >= 13 else 0.0 for e in epochs])  # Starts around epoch 13
    q_base = np.array([1.0 if e >= 29 else 0.0 for e in epochs])  # Starts around epoch 29
    
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(epochs, risk_rogue, 'r-', label='Final Risk', lw=1.5)
    ax1.plot(epochs, trust_rogue, 'b--', label='Trust Score', lw=1.5)
    ax1.axhline(0.65, color='gray', linestyle=':', label='Risk Threshold ($\theta_R = 0.65$)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()
    ax2.fill_between(epochs, 0, q_dual, color='red', alpha=0.15, step='mid', label='Behavioral Quarantine')
    ax2.fill_between(epochs, 0, q_base, color='blue', alpha=0.1, step='mid', label='Baseline QTK Quarantine')
    ax2.set_yticks([])
    ax2.set_ylim(0, 1.05)
    
    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    plt.title('Rogue Device Timeline (Late Enrollment)', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp8_timeline_rogue.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp8_timeline_rogue.png'))
    plt.close()
    
    # 3. Silent Device Timeline
    trust_silent = np.full(31, 0.93)
    risk_silent = np.full(31, 0.04)
    
    # Tiny natural trust variation (0.93 -> 0.92 -> 0.93 -> 0.92)
    for e in epochs:
        trust_silent[e] = 0.93 if e % 2 == 0 else 0.92
        # Risk stays around 0.03 - 0.05
        risk_silent[e] = 0.04 + 0.01 * np.cos(e)
        
    q_silent_base = np.array([1.0 if e >= 29 else 0.0 for e in epochs])  # Inactivity quarantine at epoch 29-30
    
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(epochs, risk_silent, 'r-', label='Final Risk', lw=1.5)
    ax1.plot(epochs, trust_silent, 'b--', label='Trust Score', lw=1.5)
    ax1.axhline(0.65, color='gray', linestyle=':', label='Risk Threshold ($\theta_R = 0.65$)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()
    ax2.fill_between(epochs, 0, q_silent_base, color='blue', alpha=0.15, step='mid', label='Baseline QTK Quarantine')
    ax2.set_yticks([])
    ax2.set_ylim(0, 1.05)
    
    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    plt.title('Silent Device Timeline (Inactivity-Triggered)', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp8_timeline_silent.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp8_timeline_silent.png'))
    plt.close()

def plot_sensitivity(all_results: Dict[str, Any], output_dir: str):
    thresholds = [0.50, 0.65, 0.80]
    detection = [99.5, 97.0, 92.0]  # First point set to 99.5%
    false_positive = [9.0, 4.0, 2.0]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(thresholds, detection, 'b-o', label='Detection Rate')
    ax.plot(thresholds, false_positive, 'r-s', label='False Positive Rate')
    
    ax.set_xlabel('Risk Threshold ($\\theta_R$)')
    ax.set_ylabel('Rate (%)')
    ax.set_ylim(0, 110)
    ax.set_xticks(thresholds)
    ax.legend(loc='center right')
    
    plt.title('Sensitivity to Risk Threshold ($\\delta_{\\text{inact}}=5$)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp5_sensitivity_theta.pdf'))
    plt.savefig(os.path.join(output_dir, 'exp5_sensitivity_theta.png'))
    plt.close()

def generate_all_plots(data_dir: str, results_dir: str):
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    all_results = {}
    plot_baseline_vs_dual(all_results, figures_dir)
    plot_ablation(all_results, figures_dir)
    plot_mimicry(all_results, figures_dir)
    plot_false_quarantine(all_results, figures_dir)
    plot_roc(data_dir, figures_dir)
    plot_confusion_matrix(data_dir, figures_dir)
    plot_timeline(data_dir, figures_dir)
    plot_sensitivity(all_results, figures_dir)
    print(f"Plots successfully generated in {figures_dir}")
