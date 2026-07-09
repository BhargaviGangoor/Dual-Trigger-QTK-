# Ghost Pairing Paper — Final Upgrade Plan (QTK-Integrated Version)
## PART 1 — LAYMAN VERSION

**The original problem:** Once a device is linked to a messaging account, it's trusted forever. A lost, sold, forgotten, or secretly-added device can silently keep reading your encrypted messages — "ghost pairing."

**Original solution:** Watch device behavior (logins, sync timing, idle time) with an HMM (statistical guesser) and LSTM (pattern-memory neural net), and move devices down a trust ladder (Trusted → Idle → Suspicious → Verification Required → Revoked).

**The upgrades  added, in order:**

1. **Watch the "friend group," not just the individual** — a Graph-based LSTM that looks at how your devices behave *relative to each other* (sync timing correlation, shared network patterns), not just each device alone. A ghost device sticks out because it doesn't "fit" with your other devices, even if its solo behavior looks fine.

2. **Combine signals with a learned fusion layer** — instead of a crude "flag if EITHER signal looks bad" rule, a small learned model decides how much to trust each signal.

3. **Make suspicion cost something real (this is the big one)** — instead of just flagging a device, you found that a real, published cryptographic protocol called **Quarantined-TreeKEM (QTK)**, from a 2024 top security conference (CCS), already does something like this inside real group-messaging encryption (MLS, the modern replacement for ad hoc protocols like Signal's). QTK quarantines "ghost users" (their actual term!) who've been *inactive* for too long, locking their encryption keys behind a system where other members have to cooperate to unlock them. But QTK's trigger is dumb — it only checks "how long since this device was last active," a plain timer. It **cannot** catch a ghost device that's sneaky and stays quietly active while behaving suspiciously.

4. **Your real contribution:** swap QTK's dumb timer for your smart trust score (from #1 and #2 above). Now the system quarantines a device not just because it's been silent too long, but because it's *behaviorally or relationally* suspicious — catching the sneaky ghost devices QTK's timer completely misses.

5. **Prove it, and stress-test it** — run real experiments comparing your smart trigger against QTK's plain timer, test whether a clever attacker could fake "normal" behavior to sneak past you, and check whether your own mechanism could be abused to falsely lock out a real device (a legitimate laptop that's just been idle).

**Why this is a good final-year paper now:** You're no longer proposing something vague. You're making a precise, provable claim against a specific, real, respected paper (QTK, CCS 2024): "your timer misses this threat class, ours catches it — here's the proof." That's exactly the shape of a strong, fundable, publishable contribution.

---

## PART 2 — TECHNICAL VERSION: THE QTK-GROUNDED ARCHITECTURE

### A. What QTK actually does (know this cold — it's your direct comparison point)

- QTK = a TreeKEM-based Continuous Group Key Agreement (CGKA) protocol, fully compatible with the MLS standard (RFC 9420), associated with a **(t,m)-perfect secret sharing scheme**.
- ## PART 2 — TECHNICAL VERSION: THE QTK-GROUNDED ARCHITECTURE

### A. What QTK actually does (know this cold — it's your direct comparison point)

- QTK = a TreeKEM-based Continuous Group Key Agreement (CGKA) protocol, fully compatible with the MLS standard (RFC 9420), associated with a **(t,m)-perfect secret sharing scheme**.
- It implements a **quarantine mechanism for inactive users**, which the paper literally calls **"ghost users."**
- **Trigger condition (the weak point you're targeting):** at each commit, the committer checks whether a user's encryption key age exceeds a fixed parameter δ_inact:
  `e_i − e_pk ≥ δ_inact  →  declare "ghost user"  →  quarantine`
  where e_i = current epoch, e_pk = epoch of that user's last path/key update.

 **Quarantine mechanism:** the ghost user's key material is updated *on their behalf* and locked behind a (t,m) threshold secret-sharing scheme — t of m other members must cooperate to reconstruct it.
- **Recovery:** if the quarantined user reconnects before full expulsion, quarantine lifts and they can recover messages sent during their offline period.
- **The gap:** this trigger is purely time-based. A device that stays technically active (still pinging, still syncing occasionally) but is behaviorally anomalous — the actual ghost-pairing threat model — is invisible to QTK's δ_inact check.

- ### B. Your modified architecture (QTK + your ML trigger)

```
Behavioral Observation Module (unchanged from original paper)
        │
        ├──> HMM (solo, per-device) ──> Pc
        ├──> Device Relationship Graph + Graph-LSTM ──> St_graph
        ├──> Fusion Layer ──> R(d,t)  [unified trust-risk score]
        │
        ▼
Modified QTK Trigger Condition:
   OLD:  (e_i − e_pk) ≥ δ_inact                     → quarantine
   NEW:  (e_i − e_pk) ≥ δ_inact   OR   R(d,t) ≥ θ_R  → quarantine
        │
        ▼
QTK Quarantine Mechanism (UNCHANGED — inherit their proven crypto machinery)
   (t,m)-threshold secret sharing on the flagged device's key material
        │
        ▼
Recovery path: device must clear BOTH (a) reconnect check AND (b) R(d,t) drop

**Key design principle:** you are NOT replacing QTK's crypto. You are ADDING a second, ML-driven trigger condition alongside their original epoch-based one. This is important for your security argument (see below).
   below θ_R before shares are reconstructed and device rejoins normally
```

# Mathematical Formulation

This section presents the mathematical foundation of the proposed **Behavior-Aware Ghost Pairing Detection Framework** integrated with **Quarantined-TreeKEM (QTK)**.

The framework continuously evaluates the trustworthiness of paired devices using behavioral analysis, graph-based relational learning, and a modified quarantine trigger that extends the original QTK protocol.

---

# 1. Device Trust Evolution

Each paired device maintains a dynamic trust score that evolves over time based on historical trust and newly observed behavioral evidence.

$$
T_{t+1}=\alpha T_t+(1-\alpha)B_t
$$

where

| Symbol | Description |
|---------|-------------|
| $T_t$ | Trust score at time $t$ |
| $B_t$ | Behavioral trust score |
| $\alpha$ | Trust retention coefficient ($0 \leq \alpha \leq 1$) |

The behavioral score is generated by the Hidden Markov Model (HMM):

$$
B_t=f_{HMM}(X_t)
$$

where

- $X_t$ represents behavioral observations collected at time $t$.
- $f_{HMM}$ denotes the Hidden Markov Model.

Typical behavioral observations include:

- Login frequency
- Device synchronization intervals
- Network transitions
- Idle duration
- Session activity
- Message synchronization patterns

---

# 2. Dynamic Device Relationship Graph

Instead of evaluating devices independently, the framework models relationships among all trusted devices.

Each device is represented as a node, while the similarity between two devices is represented as a weighted edge.

The edge weight evolves over time according to

$$
w_{ij}^{t+1}
=
\beta w_{ij}^{t}
+
(1-\beta)S_{ij}(t)
$$

where

| Symbol | Description |
|---------|-------------|
| $w_{ij}^{t}$ | Previous edge weight |
| $\beta$ | Edge memory coefficient |
| $S_{ij}(t)$ | Behavioral similarity between devices |

The similarity score is computed as

$$
S_{ij}(t)
=
\lambda_1Sync
+
\lambda_2Network
+
\lambda_3Location
+
\lambda_4Time
$$

where

- **Sync** represents synchronization similarity.
- **Network** represents similarity in network usage.
- **Location** captures geographical consistency.
- **Time** measures temporal behavioral similarity.

This allows trusted relationships to strengthen or weaken naturally over time.

---

# 3. Decay-Weighted Graph Convolution

The dynamic device graph is processed using a **Decay-Weighted Graph Neural Network (DW-GNN)**.

For each node, the embedding is updated as

$$
h_i^{(l+1)}
=
\sigma
\left(
\sum_{j\in N(i)}
\frac{w_{ij}}
{\sqrt{D_iD_j}}
W^{(l)}
h_j^{(l)}
\right)
$$

where

| Symbol | Description |
|---------|-------------|
| $h_i^{(l)}$ | Node embedding at layer $l$ |
| $N(i)$ | Neighboring devices |
| $w_{ij}$ | Dynamic edge weight |
| $D_i$ | Degree of node $i$ |
| $W^{(l)}$ | Trainable weight matrix |
| $\sigma$ | Activation function |

The resulting embeddings capture the behavioral consistency of each device relative to all other trusted devices.

These embeddings are then processed by a Graph-LSTM to produce the relational anomaly score.

---

# 4. Behavioral Risk Fusion

The framework combines multiple sources of evidence into a single behavioral risk score.

The input feature vector is

$$
z=
\begin{bmatrix}
P_c\\
S_{graph}\\
T_t
\end{bmatrix}
$$

where

| Symbol | Description |
|---------|-------------|
| $P_c$ | HMM anomaly probability |
| $S_{graph}$ | Graph-LSTM anomaly score |
| $T_t$ | Current trust score |

The unified behavioral risk is computed using either Logistic Regression or a lightweight Multi-Layer Perceptron (MLP).

### Logistic Regression

$$
R(d,t)
=
\sigma(W_fz+b)
$$

or

### Multi-Layer Perceptron

$$
R(d,t)
=
MLP(z;\theta)
$$

where

| Symbol | Description |
|---------|-------------|
| $R(d,t)$ | Unified behavioral risk score |
| $\theta$ | Learnable model parameters |

The output represents the probability that a paired device behaves anomalously.

---

# 5. Modified Quarantined-TreeKEM (QTK) Trigger

The original QTK protocol quarantines a device **only when it remains inactive for a predefined number of epochs**.

The original condition is

$$
e_i-e_{pk}(d)\ge\delta_{inact}
$$

where

| Symbol | Description |
|---------|-------------|
| $e_i$ | Current epoch |
| $e_{pk}(d)$ | Last key update epoch of device $d$ |
| $\delta_{inact}$ | Inactivity threshold |

This approach cannot detect **behaviorally active ghost-paired devices** that remain synchronized while behaving suspiciously.

To address this limitation, we introduce an additional behavioral trigger.

A device is quarantined if **either**

- it exceeds the inactivity threshold, or
- its behavioral risk exceeds the acceptable threshold.

The modified trigger becomes

$$
Quarantine(d)
\iff
(e_i-e_{pk}(d)\ge\delta_{inact})
\;\lor\;
(R(d,t)\ge\theta_R)
$$

where

| Symbol | Description |
|---------|-------------|
| $R(d,t)$ | Behavioral risk score |
| $\theta_R$ | Risk threshold |

This modification **extends** QTK without altering its cryptographic mechanisms.

The original inactivity-based trigger remains intact, while the proposed behavioral trigger enables earlier detection of stealthy ghost-paired devices.

---

# Complete Computational Pipeline

```text
                 Behavioral Observations (Xt)
                            │
                            ▼
                    Hidden Markov Model
                            │
                            ▼
                  Behavioral Score Bt
                            │
                            ▼
                Device Trust Evolution
       T(t+1) = αT(t) + (1-α)Bt
                            │
                            ▼
              Device Relationship Graph
                            │
                            ▼
             Dynamic Edge Weight Update
                            │
                            ▼
        Decay-Weighted Graph Convolution
                            │
                            ▼
                     Graph-LSTM
                            │
                            ▼
                Relational Anomaly Score
                            │
                            ▼
                 Behavioral Risk Fusion
                            │
                            ▼
                 Unified Risk Score R(d,t)
                            │
                            ▼
              Modified QTK Trigger Decision
                            │
                            ▼
         ┌───────────────────────────────┐
         │ Quarantine      Continue Trust │
         └───────────────────────────────┘
```

---

