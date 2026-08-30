import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from simulator.device import HMMState

class HMMDetector:
    """
    4-State Gaussian Hidden Markov Model for per-device behavioral anomaly detection.
    
    Semantic Hidden States (Fixed, non-permuted mapping):
      - State 0: NORMAL      (Standard diurnal sync/message activity, legitimate local IP)
      - State 1: IDLE        (Standby/nighttime, zero message count, minimal sync)
      - State 2: SUSPICIOUS  (Elevated sync frequency, long session, foreign VPN routing)
      - State 3: HIGH-RISK   (Aggressive exfiltration bursts, rapid sync, known adversary VPN)
    
    Anomaly Score Formulation:
      P_c(d, t) = P(S_t = Suspicious | X_{1:t}) + P(S_t = High-Risk | X_{1:t})
    computed via the Forward Algorithm in log-space.
    
    Training / Estimation:
      Parameters are estimated directly from labeled profile distributions in training runs,
      guaranteeing state semantics are deterministic and never permuted across seeds.
    """
    def __init__(self, n_states: int = 4, feature_dim: int = 5, seq_len: int = 12):
        self.n_states = n_states
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.is_trained = False

        # State priors pi: [Normal, Idle, Suspicious, High-Risk]
        self.startprob_ = np.array([0.85, 0.10, 0.03, 0.02], dtype=np.float64)

        # Transition matrix A: P(S_{t} | S_{t-1})
        self.transmat_ = np.array([
            [0.90, 0.07, 0.02, 0.01],  # From Normal
            [0.15, 0.80, 0.03, 0.02],  # From Idle
            [0.05, 0.05, 0.75, 0.15],  # From Suspicious
            [0.02, 0.03, 0.15, 0.80]   # From High-Risk
        ], dtype=np.float64)

        # Emission Means for [session_dur_norm, sync_freq_norm, msg_count_norm, ip_changed, tz_changed]
        self.means_ = np.array([
            [0.20, 0.20, 0.20, 0.02, 0.01],  # Normal
            [0.01, 0.02, 0.00, 0.01, 0.00],  # Idle
            [0.45, 0.65, 0.35, 0.75, 0.60],  # Suspicious
            [0.90, 0.95, 0.80, 0.95, 0.90]   # High-Risk
        ], dtype=np.float64)

        # Diagonal Covariances (Variances)
        self.covars_ = np.array([
            [0.05, 0.05, 0.05, 0.02, 0.01],  # Normal
            [0.01, 0.01, 0.01, 0.01, 0.01],  # Idle
            [0.10, 0.10, 0.10, 0.08, 0.08],  # Suspicious
            [0.12, 0.12, 0.12, 0.05, 0.05]   # High-Risk
        ], dtype=np.float64)

    def extract_features(self, history_records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts and normalizes observation features from device telemetry history.
        """
        if not history_records:
            return np.zeros((0, self.feature_dim), dtype=np.float64)

        # Window to max seq_len
        records = history_records[-self.seq_len:]
        X = []
        for i, rec in enumerate(records):
            session_dur = float(rec.get("session_duration_sec", 0.0)) / 600.0
            sync_freq = float(rec.get("sync_frequency", 0.0)) / 20.0
            msgs = float(rec.get("message_count_sent", 0.0)) / 50.0

            ip_ch = float(rec.get("ip_changed", 0.0))
            tz_ch = float(rec.get("tz_changed", 0.0))

            if rec.get("is_vpn", 0.0) > 0:
                ip_ch = max(ip_ch, 0.8)

            X.append([session_dur, sync_freq, msgs, ip_ch, tz_ch])

        return np.array(X, dtype=np.float64)

    def log_gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """Computes log likelihood of a multivariate diagonal Gaussian observation."""
        var = np.clip(var, 1e-5, None)
        log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var)
        return float(log_prob)

    def forward_algorithm(self, X: np.ndarray) -> np.ndarray:
        """
        Computes forward variables alpha_t(i) in log-space for numerical stability.
        Returns: normalized posterior probability distribution over states at time T: P(S_T | X_{1:T})
        """
        T = len(X)
        if T == 0:
            return self.startprob_.copy()

        log_start = np.log(np.clip(self.startprob_, 1e-12, None))
        log_trans = np.log(np.clip(self.transmat_, 1e-12, None))

        alpha = np.zeros((T, self.n_states), dtype=np.float64)

        # t = 0
        for i in range(self.n_states):
            log_emit = self.log_gaussian_pdf(X[0], self.means_[i], self.covars_[i])
            alpha[0, i] = log_start[i] + log_emit

        # t = 1 .. T-1
        for t in range(1, T):
            for j in range(self.n_states):
                log_emit = self.log_gaussian_pdf(X[t], self.means_[j], self.covars_[j])
                terms = alpha[t-1, :] + log_trans[:, j]
                max_term = np.max(terms)
                log_sum = max_term + np.log(np.sum(np.exp(terms - max_term)))
                alpha[t, j] = log_sum + log_emit

        # Softmax normalization at final step T
        final_log = alpha[-1, :]
        max_log = np.max(final_log)
        exp_probs = np.exp(final_log - max_log)
        norm_probs = exp_probs / np.sum(exp_probs)
        return norm_probs

    def evaluate_history(self, history_records: List[Dict[str, Any]]) -> Tuple[HMMState, float]:
        """
        Runs the log-space Forward algorithm on device history.
        Returns:
            predicted_state (HMMState): Most likely hidden state at current epoch.
            p_c (float): Anomaly score P_c(d,t) = P(Suspicious) + P(High-Risk) in [0, 1].
        """
        if len(history_records) < 1:
            return HMMState.NORMAL, 0.0

        X = self.extract_features(history_records)
        probs = self.forward_algorithm(X)

        best_state_idx = int(np.argmax(probs))
        state_map = {0: HMMState.NORMAL, 1: HMMState.IDLE, 2: HMMState.SUSPICIOUS, 3: HMMState.HIGH_RISK}
        predicted_state = state_map.get(best_state_idx, HMMState.NORMAL)

        # P_c(d,t) = P(Suspicious) + P(High-Risk)
        p_c = float(np.clip(probs[2] + probs[3], 0.0, 1.0))
        return predicted_state, round(p_c, 4)

    def predict(self, device) -> Tuple[HMMState, float]:
        """
        Infers HMM state and anomaly score for device and updates its attributes.
        """
        history = getattr(device, "telemetry_history", [])
        state, p_c = self.evaluate_history(history)
        device.hmm_state = state
        device.behavioral_risk = p_c
        return state, p_c

    def fit_from_dataset(self, training_records: List[Dict[str, Any]]):
        """
        Estimates Gaussian emission means and variances from labeled training dataset splits.
        """
        normal_samples = []
        idle_samples = []
        suspicious_samples = []
        high_risk_samples = []

        for r in training_records:
            ctx = r.get("context_telemetry", {})
            label = r.get("ground_truth_label", 0)
            feat = [
                float(ctx.get("session_duration_sec", 0.0)) / 600.0,
                float(ctx.get("sync_frequency", 0.0)) / 20.0,
                float(ctx.get("message_count_sent", 0.0)) / 50.0,
                float(ctx.get("ip_changed", 0.0)),
                float(ctx.get("tz_changed", 0.0))
            ]
            if ctx.get("is_vpn", 0.0) > 0:
                feat[3] = max(feat[3], 0.8)

            if label == 0:
                if ctx.get("sync_frequency", 0.0) < 0.1:
                    idle_samples.append(feat)
                else:
                    normal_samples.append(feat)
            else:
                if ctx.get("sync_frequency", 0.0) > 0.8 or ctx.get("is_vpn", 0.0) > 0:
                    high_risk_samples.append(feat)
                else:
                    suspicious_samples.append(feat)

        if normal_samples:
            arr = np.array(normal_samples)
            self.means_[0] = np.mean(arr, axis=0)
            self.covars_[0] = np.clip(np.var(arr, axis=0), 0.01, 0.5)

        if idle_samples:
            arr = np.array(idle_samples)
            self.means_[1] = np.mean(arr, axis=0)
            self.covars_[1] = np.clip(np.var(arr, axis=0), 0.005, 0.2)

        if suspicious_samples:
            arr = np.array(suspicious_samples)
            self.means_[2] = np.mean(arr, axis=0)
            self.covars_[2] = np.clip(np.var(arr, axis=0), 0.02, 0.5)

        if high_risk_samples:
            arr = np.array(high_risk_samples)
            self.means_[3] = np.mean(arr, axis=0)
            self.covars_[3] = np.clip(np.var(arr, axis=0), 0.02, 0.5)

        self.is_trained = True

    def save_weights(self, filepath: str):
        """Saves learned HMM parameters to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            startprob=self.startprob_,
            transmat=self.transmat_,
            means=self.means_,
            covars=self.covars_
        )

    def load_weights(self, filepath: str):
        """Loads learned HMM parameters from disk."""
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.startprob_ = data["startprob"]
            self.transmat_ = data["transmat"]
            self.means_ = data["means"]
            self.covars_ = data["covars"]
            self.is_trained = True
