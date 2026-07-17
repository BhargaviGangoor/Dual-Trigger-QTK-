# E2EE Trust Simulator Codebase Structure

This document provides a comprehensive explanation of the folder layout, architectural components, and individual files in the **Dual-Trigger QTK / E2EE Multi-Device Trust Simulator** project.

---

## 📂 High-Level Repository Layout

```text
e2ee-trust-simulator/
├── backend/                   # Python Backend & Standalone Simulation Framework
│   ├── app/                   # FastAPI Web Server Application
│   ├── configs/               # YAML Configuration files for Simulation & Thresholds
│   ├── evaluation/            # Scripts to compute experimental metrics and draw plots
│   ├── experiments/           # Standalone Python scripts for research evaluation (RQs)
│   ├── models/                # Core standalone Machine Learning & Trust Models
│   ├── qtk/                   # Standalone QTK decision & Shamir Secret Sharing logic
│   ├── simulator/             # Standalone telemetry & device action simulators
│   ├── tests/                 # Unit & Integration tests for the core models
│   ├── requirements.txt       # Python package dependencies
│   ├── run.py                 # FastAPI server start script
│   └── test_rebuilt.py        # Verification diagnostic script
│
├── frontend/                  # Vite + React + TypeScript Frontend
│   ├── public/                # Static assets for browser delivery
│   ├── src/                   # React source code (components, styles, entry points)
│   │   ├── assets/            # App images and logos
│   │   ├── components/        # Frontend interactive dashboards
│   │   ├── App.tsx            # Main controller, speed controls, WebSocket client
│   │   ├── App.css            # Styles specific to main controller layout
│   │   ├── index.css          # Global CSS (Tailwind variables and rules)
│   │   └── main.tsx           # React DOM bootstrapping
│   ├── package.json           # npm scripts and dependency definitions
│   └── tsconfig.json          # TypeScript compiler configuration
│
└── README.md                  # Project overview, research background, and roadmap
```

---

## ⚙️ Backend Architecture

The backend consists of two key parts:
1. **Interactive Web Application Layer (`backend/app/`)**: Integrates with a database (SQLAlchemy models) to run, save, and stream interactive simulations via a WebSockets connection to the React frontend.
2. **Standalone Research Testbed (`backend/simulator/`, `backend/models/`, `backend/experiments/`, etc.)**: Standalone Python scripts that do not require a database, allowing researchers to evaluate the mathematical models and answer the research questions (RQ1–RQ5) using CLI scripts.

### 🖥️ 1. Interactive Application Web Server (`backend/app/`)

This directory houses the FastAPI application code, database adapters, and models tailored for the interactive UI.

*   **`database.py`**  
    *Location:* [database.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/app/database.py)  
    Defines the database connection engine and manages SQLite/PostgreSQL sessions using SQLAlchemy. Provides the `get_db()` dependency.
*   **`models.py`**  
    *Location:* [models.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/app/models.py)  
    Contains ORM models representing the database schema:
    *   `User`: Represents an application-level user with associated behavior profiles.
    *   `Device`: Represents an MLS client device with its security state (`Trusted`, `Quarantined`, `Revoked`), key update metadata, and Shamir shares.
    *   `Message`: Stores messaging history for WhatsApp chat rendering.
    *   `MetadataRecord`: Logs behavioral telemetry snapshots (IP, VPN flags, message counts, battery level) used for anomaly calculations.
    *   `SimulationEvent`: Log entries of security alerts, key rotations, and state transitions.
*   **`schemas.py`**  
    *Location:* [schemas.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/app/schemas.py)  
    Pydantic schemas used for request validation, API output serialization, and WebSocket payload structural consistency.
*   **`main.py`**  
    *Location:* [main.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/app/main.py)  
    The FastAPI entry point defining REST endpoints (`/api/simulate`, `/api/profiles`, `/api/devices`) and a real-time `/ws/simulation` WebSockets channel for broadcasting interactive updates.
*   **`plugins/`**  
    *Location:* [plugins/](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/app/plugins)  
    *   `plugin.py`: Manages the discovery of external attack patterns and behavioral models, allowing developers to extend simulator behavior without touching core server logic.
*   **`simulator/`**  
    *Location:* [app/simulator/](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/app/simulator)  
    Adaptations of simulation rules specifically structured to operate on SQLAlchemy models.
    *   `attacks.py`: Modifies database models to simulate active exploits (Ghost Pairing, Session Hijacking, Read-only Spy, Delayed Sync).
    *   `engine.py`: Manages timeline progression, active hours, and generates database rows representing daily device interactions.
    *   `profiles.py`: Behavioral distributions for different user templates (Student, Employee, Traveler).
*   **`trust/`**  
    *Location:* [app/trust/](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/app/trust)  
    Integrates the mathematical detection models with DB models.
    *   `decay.py`: Implements trust score progression using exponential decay.
    *   `federated.py`: Computes training metrics for Federated Learning vs Centralized curves.
    *   `fsm.py`: FSM state machine transitions for Device Trust lifecycle.
    *   `fusion.py`: Fuses multi-source anomaly scores into a unified risk $R(d,t)$.
    *   `graph.py`: Evaluates relational distance among user devices using a Decay-Weighted GCN.
    *   `hmm.py`: Fits HMM parameters to detect abnormal behavioral phases.
    *   `lstm.py`: Runs sequential autoencoder inference to assess timeline anomalies.
    *   `qtk.py`: Shamir Secret Sharing logic for keys.

---

### 🔬 2. Standalone Research Library & Testbed

This set of folders implements the standalone, file-based simulation framework designed for running experiments via CLI command lines.

#### 🎛️ Configurations (`backend/configs/`)
*   **`simulation.yaml`**  
    *Location:* [simulation.yaml](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/configs/simulation.yaml)  
    Stores detailed profile settings (active hours, average hourly message count, IP prefixes, battery depletion rates, allowed devices) for Student, Corporate Employee, Traveler, and Business Owner.
*   **`thresholds.yaml`**  
    *Location:* [thresholds.yaml](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/configs/thresholds.yaml)  
    Stores numerical thresholds, decay rates, learning parameters, and weight factors (e.g. QTK $\delta_{\text{inact}} = 5$, $\theta_R = 0.65$, base alpha = $0.8$).

#### 📊 Core Standalone Machine Learning Models (`backend/models/`)
These pure-NumPy modules evaluate device and user graph data sequences without DB constraints.
*   **`hmm.py`**  
    *Location:* [models/hmm.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/models/hmm.py)  
    Implements a Gaussian Hidden Markov Model with 4 hidden states (*Normal*, *Hijacked*, *Ghost*, *Network Anomaly*) and Viterbi sequence decoding.
*   **`weighted_gnn.py`**  
    *Location:* [weighted_gnn.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/models/weighted_gnn.py)  
    A pure-NumPy Graph Convolutional Network (GCN) that aggregates device node features across user subnets.
*   **`graph_lstm.py`**  
    *Location:* [graph_lstm.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/models/graph_lstm.py)  
    Sequentially processes GCN-derived graph embeddings over time to trace structural anomalies.
*   **`dynamic_graph.py`**  
    *Location:* [dynamic_graph.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/models/dynamic_graph.py)  
    Integrates similarity score calculations and updates adjacency relationships dynamically.
*   **`trust_score.py`**  
    *Location:* [trust_score.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/models/trust_score.py)  
    Calculates numerical device trust decay and recovery coefficients.
*   **`risk_fusion.py`**  
    *Location:* [risk_fusion.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/models/risk_fusion.py)  
    Blends individual risk features and relational scores using a logistic regression layer to estimate $R(d,t)$.

#### 🧪 QTK Protocol Mechanism (`backend/qtk/`)
*   **`epoch_tracker.py`**  
    *Location:* [epoch_tracker.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/qtk/epoch_tracker.py)  
    Tracks the progression of MLS epochs during experimental simulations.
*   **`inactivity_trigger.py`**  
    *Location:* [inactivity_trigger.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/qtk/inactivity_trigger.py)  
    The original QTK check: triggers quarantine if $(e_i - e_{pk}(d) \ge \delta_{\text{inact}})$.
*   **`dual_trigger.py`**  
    *Location:* [dual_trigger.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/qtk/dual_trigger.py)  
    The extended QTK check: evaluates both inactivity gaps and behavioral risk $(R(d,t) \ge \theta_R)$ parameters.
*   **`quarantine_state.py`**  
    *Location:* [quarantine_state.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/qtk/quarantine_state.py)  
    Simulates key splits using Shamir's Secret Sharing (modular arithmetic, modular inverse, Lagrange interpolation) and quarantine status controls.

#### 🦾 Behavioral Telemetry Simulation (`backend/simulator/`)
*   **`device.py`**  
    *Location:* [simulator/device.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/simulator/device.py)  
    Base class containing identity metrics, key indexes, and historical memory for a simulated device.
*   **`legitimate_device.py`**  
    *Location:* [legitimate_device.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/simulator/legitimate_device.py)  
    Updates MLS keys regularly and generates clean, typical profile activity.
*   **`rogue_device.py`**  
    *Location:* [rogue_device.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/simulator/rogue_device.py)  
    Represents an active rogue attacker. Intentionally rotates keys frequently (every 2 epochs) to avoid inactivity quarantine while exhibiting anomalous behavior.
*   **`mimicry_attacker.py`**  
    *Location:* [mimicry_attacker.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/simulator/mimicry_attacker.py)  
    Represents an adaptive rogue device. It learns legitimate active hours and message counts to bypass standard threshold-based alarms.
*   **`silent_device.py`**  
    *Location:* [silent_device.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/simulator/silent_device.py)  
    Simulates a device that stops responding completely (used to test baseline inactivity detection).
*   **`telemetry_generator.py`**  
    *Location:* [telemetry_generator.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/simulator/telemetry_generator.py)  
    Static helper to sample telemetry values (VPN switches, location bounds, sync speeds) for normal or malicious states.

#### 🧪 Experimental Scenarios (`backend/experiments/`)
*   **`baseline_vs_dual.py`**  
    *Location:* [baseline_vs_dual.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/experiments/baseline_vs_dual.py)  
    Runs RQ1. Assesses whether active rogue devices can successfully be quarantined using the behavioral triggers.
*   **`ablation.py`**  
    *Location:* [ablation.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/experiments/ablation.py)  
    Runs RQ2. Disables GNN, LSTM, or HMM features separately to calculate individual component contribution.
*   **`mimicry.py`**  
    *Location:* [mimicry.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/experiments/mimicry.py)  
    Runs RQ3. Measures evasion latency when the rogue device imitates typical user behaviors.
*   **`false_quarantine.py`**  
    *Location:* [false_quarantine.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/experiments/false_quarantine.py)  
    Runs RQ4. Measures false alarms when a legitimate device changes networks or goes traveling.
*   **`sensitivity.py`**  
    *Location:* [sensitivity.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/experiments/sensitivity.py)  
    Runs RQ5. Jointly varies threshold parameters to plot trade-offs between speed and false quarantine rates.

#### 📊 Evaluation and Results (`backend/evaluation/`)
*   **`metrics.py`**  
    *Location:* [metrics.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/evaluation/metrics.py)  
    Calculates validation statistics: Accuracy, Precision, Recall, F1-score, and False Positive Rate.
*   **`results.py`**  
    *Location:* [results.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/evaluation/results.py)  
    Utility to serialize experimental results to disk in JSON formats.
*   **`plots.py`**  
    *Location:* [plots.py](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/backend/evaluation/plots.py)  
    Generates Matplotlib charts for visual publications.

---

## 🎨 Frontend Application (`frontend/src/`)

The user interface is built in React using TypeScript. It provides a visual, interactive demonstration of the dual-trigger behavior.

*   **`App.tsx`**  
    *Location:* [App.tsx](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/frontend/src/App.tsx)  
    Main component. Establishes the WebSocket connection, manages general simulation settings (speed, active user profile, injected attacks, thresholds), keeps track of message databases and rotation events, and exposes control tabs.
*   **`App.css`**  
    *Location:* [App.css](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/frontend/src/App.css)  
    Component-specific layout overrides for clean sidebar navigation and alignment.
*   **`index.css`**  
    *Location:* [index.css](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/frontend/src/index.css)  
    Global stylesheet defining core styling parameters, scrollbars, and standard utility wrappers.
*   **`components/WhatsApp/WhatsApp.tsx`**  
    *Location:* [WhatsApp.tsx](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/frontend/src/components/WhatsApp/WhatsApp.tsx)  
    Implements the **WhatsApp-style Messaging Sandbox**. Allows researchers to:
    *   Send and receive messages in real time.
    *   Monitor the device roster and check their active trust scores.
    *   Manually inject attacks (e.g. Ghost Pairing, Session Hijacking, Read-only Spy) to witness the live reaction.
    *   Observe security events (e.g., trust scores dropping, devices quarantined, Shamir secret keys split or reconstructed).
*   **`components/Dashboard/Dashboard.tsx`**  
    *Location:* [Dashboard.tsx](file:///c:/Users/Admin/Downloads/e2ee-trust-simulator/frontend/src/components/Dashboard/Dashboard.tsx)  
    Implements the **Research Evaluation Dashboard**. Renders:
    *   **Trust Scores Over Time**: Multi-line chart showing individual device trust status evolution.
    *   **HMM State Likelihood**: Displays probability curves for the hidden states.
    *   **Federated vs. Centralized Learning**: Comparisons of accuracy and convergence speed over training epochs.
    *   **Confusion Matrix**: Highlights true positives, false positives, and overall metrics.
    *   **Parameter Contol Sliders**: Modifies $\delta_{\text{inact}}$, $\theta_R$, and $\alpha$ on the fly to see how the mathematical curves react.
