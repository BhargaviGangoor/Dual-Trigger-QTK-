# Paper Claim Audit

This document audits all empirical, methodological, and security claims intended for the Dual-Trigger QTK paper against the codebase, experimental results, and theoretical boundaries.

## 1. Quantitative Empirical Claims

| Paper Claim | Evidence Source | Exact Value in Repo | Supported? | Needed Revision / Notes |
|-------------|-----------------|---------------------|------------|-------------------------|
| **Dual-Trigger QTK eliminates active-rogue evasion** | Table 1 (`results/tables/table1_baseline_vs_dual.tex`), `results/raw/baseline_vs_dual_runs.csv` | Detection Rate: $100.00\% \pm 0.00\%$, Latency: $0.05 \pm 0.10$ ep | **YES** | Contrast with Original QTK ($0.00\%$ detection, $20.00$ ep evasion). |
| **Original QTK completely fails under active rogue key updates** | Table 1, `tests/test_qtk.py:test_inactivity_trigger_active_rogue_evasion` | Detection Rate: $0.00\% \pm 0.00\%$, Evasion: $20.00 \pm 0.00$ ep | **YES** | Highlight that inactivity age never exceeds $\delta_{\text{inact}} = 5$. |
| **Comparative baseline performance** | Table 2 (`results/tables/table2_behavioral_baselines.tex`), `results/raw/behavioral_baselines_runs.csv` | F1: Threshold ($1.0000$), IForest ($0.9500$), HMM ($0.9333$), Dual-QTK ($0.8833$), LSTM ($0.6584$) | **YES** | Transparently state that heuristic threshold performs well in simple synthetic data, while relational models are designed for cross-device mimicry. |
| **Component Ablation** | Table 3 (`results/tables/table3_ablation_study.tex`), `results/raw/ablation_runs.csv` | F1: HMM ($0.9167$), HMM+Trust ($0.9667$), HMM+Graph ($0.9500$), Full ($0.8833$). Latency: Full ($0.00$ ep), HMM+Trust ($1.20$ ep) | **YES** | Frame full model as optimizing for minimum detection latency ($0.00$ ep) vs F1-score trade-off. |
| **Adversarial Mimicry Dynamics** | Table 4 (`results/tables/table4_adversarial_mimicry.tex`), `results/raw/mimicry_runs.csv` | Evasion Duration: Naive ($0.00$ ep), Moderate ($6.30$ ep), Strong ($0.85$ ep) | **YES** | Explain non-monotonic behavior: strong cross-device mirroring creates relational graph anomalies. |
| **Availability Impact on Legitimate Devices** | Table 5 (`results/tables/table5_false_quarantine_availability.tex`), `results/raw/false_quarantine_runs.csv` | Availability Loss: Normal ($6.00\%$), Traveler ($21.89\%$), Idle ($2.00\%$), Network-Changing ($51.22\%$) | **YES** | Rename metric from "FQR" to "False Quarantine Frequency (events/device)". Disclose network-changing availability loss as a limitation. |
| **Computational Scalability** | Table 6 (`results/tables/table6_scalability_benchmark.tex`), `results/raw/scalability.json` | $N=64$: Total per-epoch $= 749.79 \pm 8.27$ ms ($< 0.75$ s), Memory $= 957.8$ KB | **YES** | Accurately describe as sub-second per-epoch runtime for groups up to 64 devices. |
| **Parameter Stability** | `results/raw/sensitivity.json` | Detection remains $100\%$ across $\theta_R \in [0.45, 0.85]$, $\delta_{\text{inact}} \in [3, 10]$ | **YES** | Support the selection of operating point $(\theta_R = 0.65, \delta_{\text{inact}} = 5)$. |

---

## 2. Methodological Claims

| Methodology Topic | Claim in Draft / Abstract | Actual Implementation | Supported? | Needed Text Alignment |
|-------------------|--------------------------|-----------------------|------------|-----------------------|
| **HMM Parameter Estimation** | HMM models state transitions | Supervised moment estimation from labeled profile distributions (`train.jsonl`); fixed discrete states | **YES** | Clarify in paper that profile moment estimation is used rather than unsupervised Baum-Welch. |
| **Graph-LSTM Training** | Graph-LSTM detects relational anomalies | 1-layer normalized GCN + LSTM autoencoder trained on normal reconstruction loss with $L_2$ regularization | **YES** | Describe autoencoder reconstruction error objective accurately. |
| **Risk Fusion** | Fusion layer combines multi-signal evidence | Linear combination with sigmoid activation; $z = [P_c, S_{\text{graph}}, 1 - T]^T$, $W_f = [1.8, 2.2, 1.2], b = -1.0$ | **YES** | Document exact fusion formula and parameters. |
| **Trust Decay & Distrust** | Exponential trust decay with penalty factor | $T_t = \alpha T_{t-1} + (1-\alpha)(1-R)$, with quadratic distrust penalty when risk exceeds threshold | **YES** | Fully implemented and verified in `models/trust_score.py`. |
| **Evaluation Integrity** | Independent evaluation | 20 independent simulation runs per condition; no data leakage across runs or splits | **YES** | Emphasize rigorous run-level independence. |

---

## 3. Security Boundary & Cryptographic Claims

| Claim Type | Intended Claim | Implementation Evidence | Classification | Allowed Paper Phrasing |
|------------|----------------|-------------------------|----------------|------------------------|
| **MLS Cryptography Isolation** | Behavioral layer operates without violating MLS key security | All MLS key schedules (HKDF, HPKE, epoch secrets, path secrets) remain strictly inside the MLS group state; behavioral layer receives only non-secret telemetry | **SUPPORTED BY IMPLEMENTATION** | "The behavioral detection layer consumes only non-secret telemetry and maintains strict cryptographic isolation from MLS key schedules." |
| **Formal Security Proof** | Formal proof of MLS compositional security or game-based reduction | No formal game-based or reductionist security proof exists in the repository | **NOT SUPPORTED** | **DO NOT CLAIM** formal security proof. Describe as "engineering feasibility demonstration with strict architectural isolation". |
| **Forward Secrecy & PCS** | Forward secrecy and Post-Compromise Security are preserved | OpenMLS/TreeKEM performs standard Add/Remove/Update commit processing and epoch transitions upon quarantine | **SUPPORTED AS ARCHITECTURAL ARGUMENT** | "Standard MLS forward secrecy and post-compromise security guarantees are preserved because quarantine triggers standard MLS membership removal and epoch rekeying." |
| **Authorization Model** | ML model does not directly alter group cryptographic state | Quarantine decision outputs a policy recommendation; authorized group administrator/creator initiates standard MLS Remove proposal | **SUPPORTED BY IMPLEMENTATION** | "Quarantine is executed via authorized MLS membership management operations rather than direct ML manipulation of group state." |
