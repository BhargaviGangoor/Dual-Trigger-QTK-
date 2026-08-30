import numpy as np
from typing import List, Dict, Any, Optional

class LSTMBaseline:
    """
    Temporal Baseline: Pure LSTM Autoencoder (No Graph / Spatial Component).
    Processes individual device temporal feature trajectories X_{t-L+1:t} over window L=12
    and calculates sequence reconstruction error.
    """
    def __init__(self, feature_dim: int = 5, hidden_dim: int = 16, seq_len: int = 12, seed: int = 42):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.seed = seed

        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (feature_dim + hidden_dim))

        self.W_i = rng.uniform(-limit, limit, (hidden_dim, feature_dim + hidden_dim)).astype(np.float64)
        self.W_f = rng.uniform(-limit, limit, (hidden_dim, feature_dim + hidden_dim)).astype(np.float64)
        self.W_c = rng.uniform(-limit, limit, (hidden_dim, feature_dim + hidden_dim)).astype(np.float64)
        self.W_o = rng.uniform(-limit, limit, (hidden_dim, feature_dim + hidden_dim)).astype(np.float64)

        self.b_i = np.zeros((hidden_dim, 1), dtype=np.float64)
        self.b_f = np.ones((hidden_dim, 1), dtype=np.float64)
        self.b_c = np.zeros((hidden_dim, 1), dtype=np.float64)
        self.b_o = np.zeros((hidden_dim, 1), dtype=np.float64)

        self.W_dec = rng.uniform(-limit, limit, (feature_dim, hidden_dim)).astype(np.float64)
        self.b_dec = np.zeros((feature_dim, 1), dtype=np.float64)

        self.norm_mean = 0.04
        self.norm_std = 0.03
        self.is_trained = False

    def fit_on_normal(self, normal_telemetries: List[Dict[str, Any]], epochs: int = 5, learning_rate: float = 0.01):
        """
        Calibrates sequence reconstruction normalization parameters (norm_mean, norm_std)
        on clean, normative training telemetries.
        """
        if not normal_telemetries:
            self.is_trained = True
            return

        # Segment telemetries into consecutive sequences of length seq_len
        sequences = []
        for i in range(0, len(normal_telemetries) - self.seq_len + 1, self.seq_len):
            chunk = normal_telemetries[i:i + self.seq_len]
            X_chunk = self.extract_features(chunk)
            if len(X_chunk) == self.seq_len:
                sequences.append(X_chunk)

        if not sequences:
            # Fallback for small chunks
            X_all = self.extract_features(normal_telemetries)
            if len(X_all) > 0:
                sequences = [X_all]

        if not sequences:
            self.is_trained = True
            return

        # Train decoder projection on normal sequences
        for epoch in range(epochs):
            for seq in sequences:
                T = seq.shape[0]
                if T < 2:
                    continue
                h = np.zeros((self.hidden_dim, 1), dtype=np.float64)
                c = np.zeros((self.hidden_dim, 1), dtype=np.float64)
                for t in range(T):
                    x_t = seq[t].reshape(self.feature_dim, 1)
                    concat = np.vstack((h, x_t))
                    i_t = self._sigmoid(self.W_i @ concat + self.b_i)
                    f_t = self._sigmoid(self.W_f @ concat + self.b_f)
                    c_tilde = np.tanh(self.W_c @ concat + self.b_c)
                    c = f_t * c + i_t * c_tilde
                    o_t = self._sigmoid(self.W_o @ concat + self.b_o)
                    h = o_t * np.tanh(c)
                    x_hat = self.W_dec @ h + self.b_dec
                    grad_x_hat = 2.0 * (x_hat - x_t) / (T * self.feature_dim)
                    self.W_dec -= learning_rate * grad_x_hat @ h.T
                    self.b_dec -= learning_rate * grad_x_hat

        # Compute empirical reconstruction error distribution on normal data
        mses = []
        for seq in sequences:
            reconstructed = self.reconstruct_sequence(seq)
            mse = float(np.mean((seq - reconstructed) ** 2))
            mses.append(mse)

        if mses:
            self.norm_mean = float(np.mean(mses))
            self.norm_std = float(np.std(mses))
            if self.norm_std < 1e-4:
                self.norm_std = 0.02
        self.is_trained = True

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def extract_features(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Extracts normalized (T, feature_dim) sequence matrix from telemetry history."""
        X = []
        for rec in records[-self.seq_len:]:
            dur = float(rec.get("session_duration_sec", 0.0)) / 600.0
            sync = float(rec.get("sync_frequency", 0.0)) / 20.0
            msgs = float(rec.get("message_count_sent", 0.0)) / 50.0
            ip_ch = float(rec.get("ip_changed", 0.0))
            tz_ch = float(rec.get("tz_changed", 0.0))

            if rec.get("is_vpn", 0.0) > 0:
                ip_ch = max(ip_ch, 0.8)

            X.append([dur, sync, msgs, ip_ch, tz_ch])
        return np.array(X, dtype=np.float64)

    def reconstruct_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Reconstructs sequence of shape (T, feature_dim)."""
        T = sequence.shape[0]
        h = np.zeros((self.hidden_dim, 1), dtype=np.float64)
        c = np.zeros((self.hidden_dim, 1), dtype=np.float64)

        reconstructions = []
        for t in range(T):
            x_t = sequence[t].reshape(self.feature_dim, 1)
            concat = np.vstack((h, x_t))

            i_t = self._sigmoid(self.W_i @ concat + self.b_i)
            f_t = self._sigmoid(self.W_f @ concat + self.b_f)
            c_tilde = np.tanh(self.W_c @ concat + self.b_c)

            c = f_t * c + i_t * c_tilde
            o_t = self._sigmoid(self.W_o @ concat + self.b_o)
            h = o_t * np.tanh(c)

            x_hat = self.W_dec @ h + self.b_dec
            reconstructions.append(x_hat.flatten())

        return np.array(reconstructions, dtype=np.float64)

    def evaluate_history(self, history_records: List[Dict[str, Any]]) -> float:
        """
        Computes normalized temporal anomaly score in [0, 1] for a single device.
        """
        if len(history_records) < 1:
            return 0.0

        X = self.extract_features(history_records)
        reconstructed = self.reconstruct_sequence(X)
        mse = float(np.mean((X - reconstructed) ** 2))

        z = (mse - self.norm_mean) / max(1e-4, self.norm_std)
        score = float(self._sigmoid(z))
        return round(score, 4)

    def evaluate_device(self, device, current_epoch: int, threshold: float = 0.65) -> bool:
        """
        Returns True if single-device temporal LSTM reconstruction anomaly meets threshold.
        """
        history = getattr(device, "telemetry_history", [])
        if not history:
            return False
        score = self.evaluate_history(history)
        return score >= threshold
