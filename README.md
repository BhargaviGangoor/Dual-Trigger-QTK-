# Dual-Trigger QTK Experimental Testbed

A research prototype for evaluating whether Quarantined-TreeKEM (QTK) can be extended beyond inactivity-based quarantine to address both silent devices and active rogue devices in multi-device end-to-end encrypted group messaging.

## Research Motivation

Quarantined-TreeKEM (QTK) addresses long-inactive members in MLS-based group communication by placing them into quarantine rather than immediately removing them. Its quarantine decision is based on inactivity: a member is quarantined once the age of its key update exceeds a configured threshold.

This works well for a device that goes silent. It does not directly address a rogue device that remains active enough to stay below the inactivity threshold.

This project studies whether behavioral evidence can provide a second path into the existing QTK quarantine process.

The proposed decision rule is:

[
\operatorname{Quarantine}(d)
\iff
\left(e_i-e_{pk}(d)\geq\delta_{\text{inact}}\right)
\lor
\left(R(d,t)\geq\theta_R\right)
]

The first condition is QTK's original inactivity trigger. The second is a behavior-based risk trigger.

The project does not replace QTK's cryptographic quarantine mechanism. It focuses on extending the decision that determines when quarantine is invoked.

---

## System Model

Each device is modeled as a separate MLS client. Multiple clients may belong to the same application-level user.

```text
User u
|
+-- Phone d1          Legitimate MLS client
+-- Laptop d2         Legitimate MLS client
+-- Tablet d3         Legitimate MLS client
+-- Rogue device dr   Attacker-controlled MLS client
```

The rogue device has been successfully enrolled and admitted to the target MLS group. From the protocol's perspective, it is therefore a valid group member. The challenge is to determine whether its behavior provides enough evidence to invoke quarantine.

---

## Architecture

The system evaluates devices through two independent paths.

```text
                         Device Activity
                               |
              +----------------+----------------+
              |                                 |
        QTK Inactivity                    Behavioral Analysis
              |                                 |
       Key-update age                  +--------+--------+
              |                        |                 |
      delta_inact check               HMM        Dynamic Device Graph
              |                        |                 |
              |                 Individual Risk    Weighted GNN
              |                                          |
              |                                      Graph-LSTM
              |                                          |
              |                                  Relational Risk
              |                                          |
              |                        +-----------------+
              |                        |
              |                   Risk Fusion
              |                        |
              |                     R(d,t)
              |                        |
              +----------- OR --------+
                          |
                    QTK Quarantine
```

---

## 1. QTK Baseline

The baseline models the QTK inactivity decision using MLS epochs.

For each device, the simulator tracks:

* current epoch (e_i)
* epoch of the device's most recent key update (e_{pk}(d))
* inactivity threshold (\delta_{\text{inact}})

The baseline quarantine condition is:

[
e_i-e_{pk}(d)\geq\delta_{\text{inact}}
]

This provides the direct comparison point for all experiments.

---

## 2. Device Behavior Simulation

Each simulated device generates a sequence of behavioral observations.

Example features include:

* synchronization interval
* idle duration
* session activity
* network transitions
* message synchronization patterns
* temporal activity patterns
* coarse location context, where available

The simulator supports five main scenarios.

### Normal Device

A legitimate device with behavior consistent with its established history.

### Silent Device

A legitimate or abandoned device that stops participating and eventually crosses QTK's inactivity threshold.

### Active Rogue Device

An attacker-controlled device that deliberately updates often enough to remain below QTK's inactivity threshold while exhibiting different behavioral patterns.

### Mimicking Rogue Device

An adaptive attacker that attempts to imitate the timing and activity patterns of legitimate devices.

### Irregular Legitimate Device

A legitimate device whose behavior changes because of travel, network changes, unusual working hours, or long idle periods. This scenario is used to measure false quarantine.

---

## 3. Per-Device Behavioral Model

Each device is modeled using a Hidden Markov Model with four behavioral states:

```text
Normal
Idle
Suspicious
High-Risk
```

The HMM estimates a device-level anomaly score from its observed behavioral history.

The model is intended to answer:

> Does this device behave differently from its own established pattern?

---

## 4. Dynamic Device Relationship Graph

Per-device analysis may miss a rogue device whose individual behavior appears plausible.

The system therefore creates a dynamic graph over the devices belonging to the same application-level user.

```text
Phone -------- Laptop
  |               |
  |               |
Tablet -------- Rogue
```

Each device is represented as a node. Edge weights represent behavioral similarity and evolve over time.

The graph is intended to answer:

> Does this device behave consistently with the user's other devices?

A weighted GNN captures the current relational structure, while a Graph-LSTM models how these relationships change over time.

The output is a device-specific relational anomaly score.

---

## 5. Behavioral Risk Fusion

The system combines:

* HMM anomaly score
* graph-based relational anomaly score
* accumulated device trust

into a unified risk score:

[
R(d,t)\in[0,1]
]

This score does not directly replace QTK. It acts as an additional trigger for the QTK quarantine decision.

---

## 6. Dual-Trigger QTK Decision

A device is sent to the QTK quarantine process when either:

1. it exceeds QTK's original inactivity threshold, or
2. its behavioral risk exceeds the configured risk threshold.

```text
QTK inactivity trigger
          |
          +------+
                 |
                 OR ----> QTK Quarantine
                 |
          +------+
          |
Behavioral risk trigger
```

The experimental prototype focuses on the quarantine decision. QTK's cryptographic quarantine and recovery mechanisms are treated as the downstream protocol mechanism and are not redesigned by the behavioral model.

---

# Experimental Questions

The project is designed around the following research questions.

## RQ1: Can the behavioral trigger detect an active rogue device that remains below QTK's inactivity threshold?

This is the primary experiment.

```text
Active Rogue
     |
Updates before delta_inact
     |
QTK trigger = false
     |
Behavioral risk evaluated
     |
Does R(d,t) reach theta_R?
```

The experiment compares:

```text
QTK Baseline
vs.
Dual-Trigger QTK
```

---

## RQ2: Which behavioral component contributes most to detection?

The following configurations are compared:

```text
HMM only
Temporal model only
Graph-LSTM only
Full fused model
```

This ablation study determines whether the graph-based relational component provides useful information beyond simpler per-device models.

---

## RQ3: Can an adaptive rogue device evade behavioral detection?

A mimicry attacker gradually changes its behavior to resemble legitimate devices.

The experiment compares:

```text
Naive Rogue
vs.
Adaptive Mimicking Rogue
```

This tests the limits of the behavioral assumption rather than only evaluating the model against an easy attacker.

---

## RQ4: What is the cost in false quarantine?

Legitimate devices may behave unusually without being compromised.

The system therefore evaluates:

* irregular legitimate devices
* long-idle legitimate devices
* network changes
* temporal routine changes

The goal is to measure whether improved rogue-device detection comes at an unacceptable availability cost.

---

## RQ5: How sensitive is the system to its thresholds?

The experiments jointly vary:

[
\delta_{\text{inact}}
\quad\text{and}\quad
\theta_R
]

and measure the trade-off between detection, false positives, and detection latency.

---

# Evaluation Metrics

The primary metrics are:

* Rogue-device detection rate
* False-positive rate
* False-quarantine rate
* Detection latency
* Precision, recall, and F1-score

A QTK-specific metric is also included:

## QTK Evasion Duration

QTK Evasion Duration measures how long a rogue device retains access while remaining below the inactivity threshold.

The main comparison is:

```text
Evasion duration under QTK
vs.
Evasion duration under Dual-Trigger QTK
```

This directly measures whether the additional behavioral trigger reduces the period during which an active rogue device can remain outside quarantine.

---

# Suggested Project Structure

```text
dual-trigger-qtk/
|
+-- simulator/
|   +-- device.py
|   +-- legitimate_device.py
|   +-- silent_device.py
|   +-- rogue_device.py
|   +-- mimicry_attacker.py
|   +-- telemetry_generator.py
|
+-- qtk/
|   +-- epoch_tracker.py
|   +-- inactivity_trigger.py
|   +-- dual_trigger.py
|   +-- quarantine_state.py
|
+-- models/
|   +-- hmm.py
|   +-- dynamic_graph.py
|   +-- weighted_gnn.py
|   +-- graph_lstm.py
|   +-- trust_score.py
|   +-- risk_fusion.py
|
+-- experiments/
|   +-- baseline_vs_dual.py
|   +-- ablation.py
|   +-- mimicry.py
|   +-- false_quarantine.py
|   +-- sensitivity.py
|
+-- evaluation/
|   +-- metrics.py
|   +-- plots.py
|   +-- results.py
|
+-- configs/
|   +-- simulation.yaml
|   +-- thresholds.yaml
|
+-- data/
|   +-- generated/
|   +-- processed/
|
+-- README.md
+-- requirements.txt
```

---

# Implementation Roadmap

## Phase 1: QTK-Aware Simulator

Implement:

* multiple devices per user
* MLS-style epoch progression
* per-device key-update epochs
* QTK inactivity threshold
* normal, silent, and active-rogue scenarios

At the end of this phase, the project should already demonstrate the motivating case:

```text
Silent device
    -> QTK catches it

Active rogue
    -> remains below delta_inact
    -> QTK does not quarantine it
```

## Phase 2: HMM Baseline

Add per-device behavioral modeling and measure whether the HMM can distinguish normal, silent, irregular, and rogue behavior.

## Phase 3: Relational Model

Build the dynamic device graph and add the weighted GNN and Graph-LSTM.

Compare the relational model against the HMM baseline.

## Phase 4: Risk Fusion

Combine individual anomaly, relational anomaly, and trust history into (R(d,t)).

Integrate the risk threshold with the QTK decision:

```text
inactivity_trigger OR behavioral_trigger
```

## Phase 5: Adversarial Evaluation

Add:

* mimicry attacker
* irregular legitimate behavior
* false-quarantine analysis
* threshold sensitivity analysis

## Phase 6: Optional MLS Integration

If time permits, connect the decision engine to an existing MLS implementation.

This is an optional systems extension. The core research experiment evaluates the QTK quarantine-trigger decision and does not require reimplementing the full MLS or QTK cryptographic protocol from scratch.

---

# Scope

This project is a protocol-aware experimental testbed, not a new cryptographic implementation of QTK.

The core research question is:

> Can behavioral and relational evidence provide a useful additional trigger for QTK quarantine when a rogue device remains active enough to avoid the original inactivity threshold?

The project does not assume that the answer is yes. Its purpose is to test that hypothesis against the QTK baseline, adaptive attackers, and legitimate behavioral variation.
