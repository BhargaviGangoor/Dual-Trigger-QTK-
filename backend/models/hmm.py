import numpy as np
from typing import List, Tuple, Dict, Any

class HMMDetector:
    """
    Implements the Hidden Markov Model (HMM) behavioral detector described in Section IV-A.
    Uses four hidden states defined in Equation 2:
    S = {Normal, Idle, Suspicious, High-Risk}
    
    Translates observed telemetry histories into anomaly likelihood scores Pc(d, t) using Equation 3:
    Pc(d, t) = P(St = Suspicious | X_1:t) + P(St = High-Risk | X_1:t)
    """
    def __init__(self, n_components: int = 4):
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
            [50.0**2,  2.0**2,  3.0**2,  0.5**2, 0.5**2], # Legitimate (increased std for binary indicators)
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
        Uses the Forward algorithm in log-space to compute the filtered marginal 
        probabilities P(S_t | X_1:t) at the current time step t.
        Returns:
        predicted_state (int): The most likely state index at time t (0 to 3)
        p_c (float): The combined anomaly probability P(S_t=Suspicious | X_1:t) + P(S_t=HighRisk | X_1:t)
        """
        if len(records) < 2:
            return 0, 0.0

        try:
            X = self.extract_features(records)
            T = len(X)
            N = self.n_components
            
            # Trellis table to store forward log-probabilities (alpha)
            alpha = np.zeros((T, N))
            
            # 1. Initialization (t=0)
            log_start = np.log(np.clip(self.startprob_, 1e-12, None))
            for i in range(N):
                log_emission = self.log_gaussian_pdf(X[0], self.means_[i], self.covars_[i])
                alpha[0, i] = log_start[i] + log_emission
                
            # 2. Trellis updates (t > 0)
            log_trans = np.log(np.clip(self.transmat_, 1e-12, None))
            for t in range(1, T):
                for j in range(N):
                    log_emission = self.log_gaussian_pdf(X[t], self.means_[j], self.covars_[j])
                    
                    # Compute log-sum-exp for the transitions
                    terms = alpha[t-1, :] + log_trans[:, j]
                    max_term = np.max(terms)
                    log_sum = max_term + np.log(np.sum(np.exp(terms - max_term)))
                    
                    alpha[t, j] = log_sum + log_emission
            
            # 3. Exact probabilities at the final time step
            final_log_probs = alpha[T-1, :]
            max_log_prob = np.max(final_log_probs)
            norm_probs = np.exp(final_log_probs - max_log_prob)
            norm_probs = norm_probs / np.sum(norm_probs)
            
            # State indices: 0: Normal, 1: Suspicious (Compromised), 2: HighRisk (Ghost), 3: Idle (Network Anomaly)
            best_final_state = int(np.argmax(norm_probs))
            
            # Equation 2: Pc(d,t) = P(St = Suspicious | X_1:t) + P(St = High-Risk | X_1:t)
            p_c = float(norm_probs[1] + norm_probs[2])
            
            return best_final_state, round(p_c, 4)
            
        except Exception as e:
            return 0, 0.0

    def get_matrices(self) -> Dict[str, Any]:
        return {
            "transition_matrix": self.transmat_.tolist(),
            "emission_means": self.means_.tolist(),
            "start_probabilities": self.startprob_.tolist()
        }

    def predict(self, device) -> Tuple[str, float]:
        """
        Runs the Forward algorithm on the device's telemetry history to compute
        the exact mathematical probability of anomaly (P_c), and updates the 
        device's hmm_state and behavioral_risk.
        """
        from simulator.device import HMMState
        
        records = getattr(device, "telemetry_history", [])
        state_idx, p_c = self.evaluate_device(records)
        
        # Map most likely state index to HMMState enum for logging purposes
        if state_idx == 0:
            hmm_state = HMMState.NORMAL
        elif state_idx == 3:
            hmm_state = HMMState.IDLE
        elif state_idx == 1:
            hmm_state = HMMState.SUSPICIOUS
        else: # 2
            hmm_state = HMMState.HIGH_RISK
            
        # Guarantee mathematical bounds
        p_c = max(0.0, min(1.0, float(round(p_c, 4))))
        
        if hasattr(device, "update_hmm_state"):
            device.update_hmm_state(hmm_state)
        else:
            device.hmm_state = hmm_state
            
        if hasattr(device, "update_behavioral_risk"):
            device.update_behavioral_risk(p_c)
        else:
            device.behavioral_risk = p_c
            
        return hmm_state, p_c
