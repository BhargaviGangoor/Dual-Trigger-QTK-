import numpy as np
from typing import List, Tuple, Dict, Any

class HMMDetector:
    def __init__(self, n_components: int = 4):
        """
        NumPy-based Gaussian Hidden Markov Model (HMM).
        
        4 Hidden States:
        0 - Legitimate / Normal User
        1 - Compromised Session (Session Hijacking)
        2 - Ghost Device (Silent paired device)
        3 - Network Anomaly / Transient state
        """
        self.n_components = n_components
        self.is_trained = True
        
        # Start probabilities
        self.startprob_ = np.array([0.9, 0.02, 0.01, 0.07])
        
        # Transition probabilities
        self.transmat_ = np.array([
            [0.95, 0.01, 0.005, 0.035], # From Legitimate
            [0.10, 0.80, 0.05,  0.05],  # From Compromised
            [0.01, 0.01, 0.95,  0.03],  # From Ghost Device
            [0.40, 0.05, 0.05,  0.50]   # From Network Anomaly
        ])

        # Emission means for [Session Duration (s), Sync Frequency (per hr), Sent Msg Count, IP Shift, TZ Shift]
        self.means_ = np.array([
            [120.0, 4.0,  5.0, 0.05, 0.01],  # Legitimate
            [900.0, 15.0, 1.0, 0.90, 0.85],  # Compromised
            [3600.0, 24.0, 0.0, 0.95, 0.90], # Ghost Device
            [10.0,  0.5,  0.5, 0.40, 0.05]   # Network Anomaly
        ])

        # Covariances for each state's features (diagonal representation, as variances)
        self.covars_ = np.array([
            [50.0**2,  2.0**2,  3.0**2,  0.1**2, 0.05**2], # Legitimate
            [300.0**2, 5.0**2,  1.0**2,  0.2**2, 0.2**2],  # Compromised
            [600.0**2, 6.0**2,  0.1**2,  0.1**2, 0.1**2],  # Ghost Device
            [5.0**2,   0.2**2,  0.5**2,  0.3**2, 0.1**2]   # Network Anomaly
        ])

    def train_on_profile(self, profile_name: str):
        """Adjusts Gaussian emissions slightly to align with the specific behavior profile."""
        if profile_name == "Traveler":
            self.means_[0][3] = 0.40  # Regular IP changes are normal for traveler
            self.means_[0][4] = 0.35  # Regular Timezone changes are normal for traveler
        elif profile_name == "Corporate Employee":
            self.means_[0][3] = 0.02
            self.means_[0][4] = 0.01
        elif profile_name == "VPN User":
            self.means_[0][3] = 0.70  # VPN switches IPs frequently
            self.means_[0][4] = 0.10

    def extract_features(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Translates list of telemetry dicts to NumPy feature vectors."""
        features = []
        for i, r in enumerate(records):
            ip_changed = 1.0 if i > 0 and r.get("network_ip") != records[i-1].get("network_ip") else 0.0
            tz_changed = 1.0 if i > 0 and r.get("active_timezone") != records[i-1].get("active_timezone") else 0.0
            
            features.append([
                float(r.get("session_duration_sec", 0.0)),
                float(r.get("sync_frequency", 0.0)),
                float(r.get("message_count_sent", 0.0)),
                ip_changed,
                tz_changed
            ])
            
        return np.array(features)

    def log_gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """Computes log probability of Gaussian density."""
        var = np.clip(var, 1e-6, None)
        log_p = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
        return float(np.sum(log_p))

    def evaluate_device(self, records: List[Dict[str, Any]]) -> Tuple[int, float]:
        """
        Uses dynamic Viterbi decoding algorithm in log-space to infer the state sequence.
        Returns:
        predicted_state (int): The current state index (0 to 3)
        confidence (float): Normalized likelihood score (0.0 to 1.0)
        """
        if len(records) < 2:
            return 3, 0.5

        try:
            X = self.extract_features(records)
            T = len(X)
            N = self.n_components
            
            # Trellis table to store log-likelihoods
            V = np.zeros((T, N))
            # Backpointers matrix
            B = np.zeros((T, N), dtype=int)
            
            # 1. Initialization (t=0)
            log_start = np.log(np.clip(self.startprob_, 1e-12, None))
            for i in range(N):
                log_emission = self.log_gaussian_pdf(X[0], self.means_[i], self.covars_[i])
                V[0, i] = log_start[i] + log_emission
                
            # 2. Trellis updates (t > 0)
            log_trans = np.log(np.clip(self.transmat_, 1e-12, None))
            for t in range(1, T):
                for j in range(N):
                    log_emission = self.log_gaussian_pdf(X[t], self.means_[j], self.covars_[j])
                    probs = V[t-1, :] + log_trans[:, j]
                    best_prev_state = np.argmax(probs)
                    V[t, j] = probs[best_prev_state] + log_emission
                    B[t, j] = best_prev_state
            
            # 3. Path Backtracking
            best_final_state = np.argmax(V[T-1, :])
            
            # Normalize confidence using softmax over final log-probabilities
            final_probs = V[T-1, :]
            final_probs = final_probs - np.max(final_probs)  # stability
            exp_probs = np.exp(final_probs)
            softmax_probs = exp_probs / np.sum(exp_probs)
            confidence = float(softmax_probs[best_final_state])
            
            return int(best_final_state), round(confidence, 4)
            
        except Exception as e:
            return 3, 0.5

    def get_matrices(self) -> Dict[str, Any]:
        return {
            "transition_matrix": self.transmat_.tolist(),
            "emission_means": self.means_.tolist(),
            "start_probabilities": self.startprob_.tolist()
        }
