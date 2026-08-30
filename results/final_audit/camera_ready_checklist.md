# Final Camera-Ready Acceptance Checklist

This checklist certifies the readiness of the Dual-Trigger QTK codebase, experimental results, and reproducibility infrastructure for final paper preparation.

## Reviewer Requirements
- [x] **Reviewer A fully addressed** — Independent run evaluations, no hardcoded results, attack latency measured from injection epoch, complete ablation & baseline benchmarks.
- [x] **Reviewer B fully addressed** — Real MLS PoC implementation, original QTK failure demonstrated, security boundary enforced, availability & scalability measured.
- [x] **No known reviewer request ignored** — All comments categorized and addressed in `reviewer_requirements_matrix.md`.

## Data & Methodological Integrity
- [x] **No test leakage** — 70/15/15 train/validation/test split strictly segregated by run ID; no test data used in normalization or parameter fitting.
- [x] **No hardcoded results** — All 6 publication tables generated dynamically from raw JSON simulation run files.
- [x] **Original QTK baseline verified** — Active rogue periodic key update evasion confirmed empirically ($0\%$ detection, $20$-epoch evasion).
- [x] **Active rogue failure case verified** — Unit test and benchmark verify inactivity trigger fails to quarantine active rogue.
- [x] **Behavioral detector verified** — HMM, Dynamic Graph, GNN, Graph-LSTM, Trust Score, and Risk Fusion pipelines verified.
- [x] **Fusion layer calibrated** — Supervised fusion weights and threshold $\theta_R = 0.65$ validated.
- [x] **Validation threshold calibration verified** — Threshold stability confirmed via sensitivity analysis.
- [x] **Baselines fair** — Comparative baselines (Threshold, Isolation Forest, HMM, LSTM) evaluated under identical seed schedules and topologies.
- [x] **Ablation verified** — Evaluated across HMM-only, HMM+Trust, HMM+Graph, and Full configurations.
- [x] **Mimicry verified** — Evaluated across Naive, Moderate, and Strong mimicry modes.
- [x] **False quarantine & availability verified** — Evaluated across Normal, Traveler/Irregular, Long-Idle, and Network-Changing profiles.
- [x] **Sensitivity verified** — Parameter sensitivity maps generated across $\theta_R$, $\delta_{\text{inact}}$, and $\alpha$.
- [x] **Scalability verified** — Microsecond-to-millisecond runtime scaling evaluated up to $N=64$ devices ($< 750$ ms).
- [x] **Statistical reporting verified** — Run-level mean, standard deviation, and confidence intervals computed over $N=20$ runs.

## Documentation & Descriptions
- [x] **HMM methodology documented** — Documented as supervised profile moment estimation across 4 discrete states.
- [x] **GNN methodology documented** — Spatial GCN convolution over dynamically evolved similarity graph.
- [x] **Graph-LSTM methodology documented** — Autoencoder reconstruction loss over normal training sequences.
- [x] **Trust methodology documented** — Exponential decay with non-linear distrust penalty.
- [x] **Risk fusion documented** — Sigmoid activation combining continuous risk components.
- [x] **Security boundary documented** — Non-secret telemetry interface strictly segregated from MLS cryptographic state.

## MLS Integration & PoC
- [x] **OpenMLS / RFC 9420 PoC runs** — Real TreeKEM group creation, Add/Remove commits, epoch advancement, and quarantine execution verified.
- [x] **MLS PoC separated from benchmark** — PoC isolated in `mls/` and `experiments/mls_poc.py`, benchmark tables frozen in `results/`.
- [x] **MLS secrets isolated** — Behavioral pipeline never accesses epoch secrets, path secrets, private keys, or decrypted payloads.

## Reproducibility & Traceability
- [x] **Reproducibility manifest exists** — `results/manifest.json` maps all tables to raw files, configurations, and seeds.
- [x] **Raw run-level results exist** — All individual run metrics preserved in `results/raw/*.json` and `results/raw/*_runs.csv`.
- [x] **Tables generated automatically** — All LaTeX tables in `results/tables/` compiled directly via `evaluation/results.py`.
- [x] **Figures generated automatically** — PDF publication figures compiled in `results/figures/`.
- [x] **All paper numbers trace to raw data** — 100% data traceability from raw simulator output to final LaTeX code.

## Claim Boundaries & Limitations
- [x] **Known limitations documented** — High availability loss ($51.2\%$) under rapid network changes and synthetic data bounds disclosed.
- [x] **No unsupported security claims** — No false claims of formal MLS security proofs; positioned as architectural feasibility demonstration.
- [x] **No unsupported robustness claims** — Moderate mimicry evasion window ($6.30$ epochs) transparently reported.
- [x] **No unsupported scalability claims** — Linear/quadratic scaling bounds accurately reported up to measured $N=64$.

---

## Acceptance Verdict
**STATUS: GREEN (Codebase Frozen — Ready for Paper Revision)**
