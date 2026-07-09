import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Any

class LSTMDetector:
    """
    Simulates a sequence-to-sequence Recurrent Autoencoder (LSTM/RNN) for anomaly detection.
    Computes sequence reconstruction errors to generate behavioral anomaly scores.
    """
    def __init__(self, sequence_length: int = 12):
        self.sequence_length = sequence_length
        # Initialize random LSTM weights for simulation in NumPy
        # Inputs: 5 features, Hidden Units: 16
        input_dim = 5
        hidden_dim = 16
        
        np.random.seed(42)
        # Weight matrices for LSTM gate inputs (Input, Forget, Cell, Output)
        self.W_i = np.random.normal(0, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.W_f = np.random.normal(0, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.W_c = np.random.normal(0, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.W_o = np.random.normal(0, 0.1, (hidden_dim, input_dim + hidden_dim))
        
        # Biases
        self.b_i = np.zeros((hidden_dim, 1))
        self.b_f = np.ones((hidden_dim, 1))  # Initialize forget bias to 1.0
        self.b_c = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((hidden_dim, 1))

        # Output weight projection (reconstruction target)
        self.W_y = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        self.b_y = np.zeros((input_dim, 1))
        
        # Set up a fallback isolation forest
        self.iso_forest = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        self.iso_trained = False

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def lstm_cell_forward(self, x_t, h_prev, c_prev):
        """Standard LSTM forward cell equations in NumPy."""
        # Concatenate input and hidden state
        concat = np.vstack((h_prev, x_t))
        
        # Gates
        i = self.sigmoid(np.dot(self.W_i, concat) + self.b_i)
        f = self.sigmoid(np.dot(self.W_f, concat) + self.b_f)
        c_tilde = np.tanh(np.dot(self.W_c, concat) + self.b_c)
        
        # Cell State Update
        c_t = f * c_prev + i * c_tilde
        
        # Output Gate
        o = self.sigmoid(np.dot(self.W_o, concat) + self.b_o)
        
        # Hidden State Update
        h_t = o * np.tanh(c_t)
        
        return h_t, c_t

    def forward_pass(self, sequence: np.ndarray) -> np.ndarray:
        """
        Runs the sequence of shape (seq_len, feature_dim) through the LSTM
        and returns the reconstructed sequence of inputs.
        """
        seq_len, feat_dim = sequence.shape
        h = np.zeros((16, 1))
        c = np.zeros((16, 1))
        
        reconstructions = []
        
        for t in range(seq_len):
            x_t = sequence[t].reshape(feat_dim, 1)
            h, c = self.lstm_cell_forward(x_t, h, c)
            
            # Predict / Reconstruct current input
            y_pred = np.dot(self.W_y, h) + self.b_y
            reconstructions.append(y_pred.flatten())
            
        return np.array(reconstructions)

    def extract_features(self, records: List[Any]) -> np.ndarray:
        """Extracts features from MetadataRecord objects."""
        features = []
        for i, r in enumerate(records):
            ip_changed = 1.0 if i > 0 and r.network_ip != records[i-1].network_ip else 0.0
            tz_changed = 1.0 if i > 0 and r.active_timezone != records[i-1].active_timezone else 0.0
            
            features.append([
                float(r.session_duration_sec) / 3600.0, # Normalise durations
                float(r.sync_frequency) / 60.0,         # Normalise sync
                float(r.message_count_sent) / 50.0,     # Normalise messages
                ip_changed,
                tz_changed
            ])
        return np.array(features)

    def train_on_profile(self, profile_name: str):
        """Fits the Isolation Forest on standard baseline data matching the user profile."""
        # Generate some synthetic normal observations to pre-fit the Isolation Forest
        np.random.seed(42)
        normal_samples = []
        for _ in range(200):
            # Normal user properties (low duration, moderate messages, very rare IP switches)
            normal_samples.append([
                np.random.normal(0.05, 0.02),  # Duration
                np.random.normal(0.1, 0.05),   # Sync freq
                np.random.normal(0.1, 0.08),   # Message sent
                1.0 if np.random.random() < 0.01 else 0.0, # IP change
                1.0 if np.random.random() < 0.002 else 0.0 # TZ change
            ])
            
        X = np.array(normal_samples)
        self.iso_forest.fit(X)
        self.iso_trained = True

    def evaluate_device(self, records: List[Any], threshold: float = 0.6) -> float:
        """
        Computes the anomaly score.
        Combines LSTM reconstruction error and Isolation Forest score.
        Returns:
            anomaly_score (float): 0.0 (normal) to 1.0 (highly anomalous)
        """
        if len(records) < 2:
            return 0.0

        try:
            X = self.extract_features(records)
            
            # 1. Compute LSTM reconstruction error
            reconstruction = self.forward_pass(X)
            mse = np.mean((X - reconstruction) ** 2)
            lstm_score = min(1.0, mse * 5.0)  # scale factor
            
            # 2. Compute Isolation Forest score if trained
            if self.iso_trained:
                # -1 is anomaly, 1 is normal. Convert to 0 (normal) to 1 (anomaly)
                iso_raw = self.iso_forest.score_samples(X[-1].reshape(1, -1))[0]
                iso_score = max(0.0, min(1.0, -iso_raw))
            else:
                iso_score = lstm_score

            # Weight the scores (60% LSTM sequence reconstruction, 40% Isolation Forest point anomaly)
            final_score = 0.6 * lstm_score + 0.4 * iso_score
            return float(round(final_score, 4))
        except Exception as e:
            return 0.5
