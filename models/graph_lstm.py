import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .dynamic_graph import DynamicGraph
from .weighted_gnn import WeightedGNN

class GraphLSTM:
    r"""
    Graph-LSTM Temporal-Relational Sequence Model.

    Architecture & Operation:
      1. Relational Topology: Dynamic similarity graph W(t) evolves pairwise edge weights
         w_{ij}(t+1) = beta * w_{ij}(t) + (1 - beta) * S_{ij}(t).
      2. Spatial Embedding: Decay-Weighted GCN layer computes spatial node embeddings
         H_t = ReLU( D^{-1/2} (W(t) + I) D^{-1/2} X_t W_{gcn} ) in R^{hidden_dim}.
      3. Temporal Sequence Autoencoder: LSTM cell processes the sequential embeddings
         H_{t-L+1:t} over a sliding window L=12 and reconstructs normative relational patterns:
         \hat{H}_t = W_{dec} h_t + b_{dec}.

    Training Objective (Normal-Training-Only Assumption):
      Trained strictly on clean, normal/legitimate multi-device relational trajectories (y = 0)
      using Mean Squared Error (MSE) reconstruction loss:
      L = (1 / (T * d_h)) \sum_{t=1}^T || H_t - \hat{H}_t ||_2^2 + \lambda ||W||_2^2.

    Anomaly Scoring:
      Deviations from normative group relational patterns produce elevated reconstruction error:
      S_{graph}(d, t) = \sigma( (MSE_d - \mu_{norm}) / \sigma_{norm} ) in [0, 1].
    """

    def __init__(
        self,
        feature_dim: int = 5,
        hidden_dim: int = 16,
        seq_len: int = 12,
        beta: float = 0.8,
        seed: int = 42
    ):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.beta = beta
        self.seed = seed

        self.dynamic_graph = DynamicGraph(beta=beta)
        self.weighted_gnn = WeightedGNN(feature_dim=feature_dim, hidden_dim=hidden_dim, seed=seed)

        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (hidden_dim + 2 * hidden_dim))

        # LSTM Gate Weights: [W_xi, W_hi] combined
        self.W_i = rng.uniform(-limit, limit, (hidden_dim, 2 * hidden_dim)).astype(np.float64)
        self.W_f = rng.uniform(-limit, limit, (hidden_dim, 2 * hidden_dim)).astype(np.float64)
        self.W_c = rng.uniform(-limit, limit, (hidden_dim, 2 * hidden_dim)).astype(np.float64)
        self.W_o = rng.uniform(-limit, limit, (hidden_dim, 2 * hidden_dim)).astype(np.float64)

        # Biases (forget gate initialized to 1.0 for stability)
        self.b_i = np.zeros((hidden_dim, 1), dtype=np.float64)
        self.b_f = np.ones((hidden_dim, 1), dtype=np.float64)
        self.b_c = np.zeros((hidden_dim, 1), dtype=np.float64)
        self.b_o = np.zeros((hidden_dim, 1), dtype=np.float64)

        # Output / Decoder Projection
        self.W_dec = rng.uniform(-limit, limit, (hidden_dim, hidden_dim)).astype(np.float64)
        self.b_dec = np.zeros((hidden_dim, 1), dtype=np.float64)

        # Normalization statistics for anomaly scoring: S_graph = sigma((MSE - norm_mean) / norm_std).
        # These are static defaults unless train_on_sequences() is called with normal training data.
        # Call reset_norm_stats() at the start of each independent simulation run to ensure
        # normalization state does not accumulate across runs.
        self._initial_norm_mean = 0.05
        self._initial_norm_std = 0.04
        self.norm_mean = self._initial_norm_mean
        self.norm_std = self._initial_norm_std
        self.is_trained = False

    def reset_norm_stats(self):
        """
        Resets anomaly score normalization statistics to their initial values.
        Call this at the start of each independent simulation run to prevent
        normalization state from accumulating across runs and breaking run independence.
        """
        self.norm_mean = self._initial_norm_mean
        self.norm_std = self._initial_norm_std

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def _lstm_forward_step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs one step of LSTM cell forward pass:
        x_t: (hidden_dim, 1), h_prev: (hidden_dim, 1), c_prev: (hidden_dim, 1)
        """
        concat = np.vstack((h_prev, x_t))  # (2*hidden_dim, 1)

        i_t = self._sigmoid(self.W_i @ concat + self.b_i)
        f_t = self._sigmoid(self.W_f @ concat + self.b_f)
        c_tilde = np.tanh(self.W_c @ concat + self.b_c)

        c_t = f_t * c_prev + i_t * c_tilde
        o_t = self._sigmoid(self.W_o @ concat + self.b_o)
        h_t = o_t * np.tanh(c_t)

        x_hat = self.W_dec @ h_t + self.b_dec
        return h_t, c_t, x_hat

    def reconstruct_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Processes a sequence of GCN embeddings of shape (T, hidden_dim).
        Returns reconstructed sequence of shape (T, hidden_dim).
        """
        T = sequence.shape[0]
        h = np.zeros((self.hidden_dim, 1), dtype=np.float64)
        c = np.zeros((self.hidden_dim, 1), dtype=np.float64)

        reconstructions = []
        for t in range(T):
            x_t = sequence[t].reshape(self.hidden_dim, 1)
            h, c, x_hat = self._lstm_forward_step(x_t, h, c)
            reconstructions.append(x_hat.flatten())

        return np.array(reconstructions, dtype=np.float64)

    def evaluate_devices(
        self,
        devices_meta_history: List[List[Dict[str, Any]]],
        prev_adj: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Evaluates multi-device metadata histories:
        1. Evolves the dynamic relationship graph.
        2. Computes Decay-Weighted GCN spatial embeddings across time.
        3. Runs Graph-LSTM temporal autoencoder to compute relational anomaly scores S_graph(d,t).
        
        Returns:
            adj: (num_devices, num_devices) dynamic adjacency matrix
            anomaly_scores: list of float anomaly scores in [0, 1] for each device
        """
        num_devices = len(devices_meta_history)
        if num_devices == 0:
            return np.zeros((0, 0)), []

        min_len = min(len(h) for h in devices_meta_history)
        if min_len < 1:
            return np.eye(num_devices), [0.0] * num_devices

        effective_seq_len = min(min_len, self.seq_len)
        aligned_history = [h[-effective_seq_len:] for h in devices_meta_history]

        # 1. Evolve Graph Adjacency using latest observations
        latest_metas = [h[-1] for h in aligned_history]
        adj = self.dynamic_graph.evolve_adjacency(latest_metas, prev_adj)

        # 2. Build Multi-Device Feature Matrices for each time step t
        # shape: (T, num_devices, feature_dim)
        features_seq = []
        for t in range(effective_seq_len):
            step_feats = []
            for d in range(num_devices):
                rec = aligned_history[d][t]
                dur = float(rec.get("session_duration_sec", 0.0)) / 600.0
                sync = float(rec.get("sync_frequency", 0.0)) / 20.0
                msgs = float(rec.get("message_count_sent", 0.0)) / 50.0
                ip_ch = float(rec.get("ip_changed", 0.0))
                tz_ch = float(rec.get("tz_changed", 0.0))

                if rec.get("is_vpn", 0.0) > 0:
                    ip_ch = max(ip_ch, 0.8)

                step_feats.append([dur, sync, msgs, ip_ch, tz_ch])
            features_seq.append(np.array(step_feats, dtype=np.float64))

        # 3. Compute GCN Node Embeddings for each time step: H_t = GCN(X_t, adj)
        gcn_embeddings = []
        for t in range(effective_seq_len):
            X_t = features_seq[t]  # (num_devices, feature_dim)
            H_t = self.weighted_gnn.forward(X_t, adj)  # (num_devices, hidden_dim)
            gcn_embeddings.append(H_t)

        gcn_arr = np.array(gcn_embeddings)  # (T, num_devices, hidden_dim)

        # 4. Run Graph-LSTM on each device's temporal trajectory and compute MSE
        anomaly_scores = []
        for d in range(num_devices):
            dev_seq = gcn_arr[:, d, :]  # (T, hidden_dim)
            reconstructed = self.reconstruct_sequence(dev_seq)  # (T, hidden_dim)
            mse = float(np.mean((dev_seq - reconstructed) ** 2))

            # Normalize MSE into sigmoidal relational anomaly score S_graph in [0, 1]
            z = (mse - self.norm_mean) / max(1e-4, self.norm_std)
            rel_score = float(self._sigmoid(z))
            anomaly_scores.append(round(rel_score, 4))

        return adj, anomaly_scores

    def train_on_sequences(
        self,
        normal_sequences: List[np.ndarray],
        learning_rate: float = 0.01,
        epochs: int = 30,
        l2_reg: float = 1e-4
    ):
        """
        Fits LSTM autoencoder weights on normative training sequences using gradient descent.
        """
        if not normal_sequences:
            return

        all_mses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for seq in normal_sequences:
                T = seq.shape[0]
                if T < 2:
                    continue

                reconstructed = self.reconstruct_sequence(seq)
                err = reconstructed - seq
                loss = np.mean(err ** 2)
                epoch_loss += loss

                # Gradient step on output projection
                h_last = np.zeros((self.hidden_dim, 1))
                c_last = np.zeros((self.hidden_dim, 1))
                for t in range(T):
                    x_t = seq[t].reshape(self.hidden_dim, 1)
                    h_last, c_last, x_hat = self._lstm_forward_step(x_t, h_last, c_last)
                    grad_x_hat = 2.0 * (x_hat - x_t) / (T * self.hidden_dim)
                    self.W_dec -= learning_rate * (grad_x_hat @ h_last.T + l2_reg * self.W_dec)
                    self.b_dec -= learning_rate * grad_x_hat

        # Compute baseline MSE distribution for score normalization
        for seq in normal_sequences:
            reconstructed = self.reconstruct_sequence(seq)
            mse = float(np.mean((seq - reconstructed) ** 2))
            all_mses.append(mse)

        if all_mses:
            self.norm_mean = float(np.mean(all_mses))
            self.norm_std = float(np.std(all_mses))
            if self.norm_std < 1e-4:
                self.norm_std = 0.02
        self.is_trained = True

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            W_i=self.W_i, W_f=self.W_f, W_c=self.W_c, W_o=self.W_o,
            b_i=self.b_i, b_f=self.b_f, b_c=self.b_c, b_o=self.b_o,
            W_dec=self.W_dec, b_dec=self.b_dec,
            norm_mean=self.norm_mean, norm_std=self.norm_std
        )

    def load_weights(self, filepath: str):
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.W_i = data["W_i"]
            self.W_f = data["W_f"]
            self.W_c = data["W_c"]
            self.W_o = data["W_o"]
            self.b_i = data["b_i"]
            self.b_f = data["b_f"]
            self.b_c = data["b_c"]
            self.b_o = data["b_o"]
            self.W_dec = data["W_dec"]
            self.b_dec = data["b_dec"]
            self.norm_mean = float(data["norm_mean"])
            self.norm_std = float(data["norm_std"])
            self.is_trained = True
