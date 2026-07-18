import numpy as np
from typing import List, Tuple, Dict, Any

class HMMDetector:
    def __init__(self, n_components: int = 4):
        """
        Pure NumPy implementation of a Gaussian Hidden Markov Model (HMM) 
        to avoid compiled Cython C++ installation dependencies on Windows.
        
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

        # Covariances for each state's features (diagonal representation, represented as variances)
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

    def extract_features(self, records: List[Any]) -> np.ndarray:
        """Translates database MetadataRecord objects to features."""
        features = []
        for i, r in enumerate(records):
            ip_changed = 1.0 if i > 0 and r.network_ip != records[i-1].network_ip else 0.0
            tz_changed = 1.0 if i > 0 and r.active_timezone != records[i-1].active_timezone else 0.0
            
            features.append([
                float(r.session_duration_sec),
                float(r.sync_frequency),
                float(r.message_count_sent),
                ip_changed,
                tz_changed
            ])
            
        return np.array(features)

    def log_gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Computes log probability of Gaussian density."""
        # Avoid zero variance division
        var = np.clip(var, 1e-6, None)
        
        # log p(x) = -0.5 * log(2 * pi * var) - (x - mean)^2 / (2 * var)
        log_p = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
        return np.sum(log_p)

    def evaluate_device(self, records: List[Any]) -> Tuple[int, float]:
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
        """Returns HMM matrices for visualization."""
        return {
            "transition_matrix": self.transmat_.tolist(),
            "emission_means": self.means_.tolist(),
            "start_probabilities": self.startprob_.tolist()
        }
