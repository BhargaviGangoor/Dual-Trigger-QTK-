import os
import numpy as np

class WeightedGNN:
    """
    Decay-Weighted Graph Convolutional Network (DW-GNN) layer.
    Propagates multi-device relational features through normalized adjacency weights:
    H = ReLU( D^{-1/2} * W_tilde * D^{-1/2} * X * W_gcn )
    """
    def __init__(self, feature_dim: int = 5, hidden_dim: int = 16, seed: int = 42):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seed = seed

        rng = np.random.RandomState(seed)
        # Xavier / Glorot Initialization
        limit = np.sqrt(6.0 / (feature_dim + hidden_dim))
        self.W_gcn = rng.uniform(-limit, limit, (feature_dim, hidden_dim)).astype(np.float64)

    def forward(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Runs one Decay-Weighted Graph Convolution layer:
        Args:
            X (np.ndarray): Node feature matrix of shape (num_devices, feature_dim)
            adj (np.ndarray): Dynamic adjacency matrix of shape (num_devices, num_devices)
        Returns:
            H (np.ndarray): Node embedding matrix of shape (num_devices, hidden_dim)
        """
        n = X.shape[0]
        if n == 0:
            return np.zeros((0, self.hidden_dim), dtype=np.float64)

        # 1. Add self-loops: W_tilde = adj + I
        W_tilde = adj + np.eye(n, dtype=np.float64)

        # 2. Degree Matrix D: D_ii = sum_j W_tilde_ij
        d = np.sum(W_tilde, axis=1)
        d = np.clip(d, 1e-6, None)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        D_inv_sqrt = np.diag(d_inv_sqrt)

        # 3. Symmetric Normalization: A_norm = D^{-1/2} * W_tilde * D^{-1/2}
        A_norm = D_inv_sqrt @ W_tilde @ D_inv_sqrt

        # 4. Message Propagation and Projection: H = A_norm * X * W_gcn
        Z = A_norm @ X
        H = Z @ self.W_gcn

        # 5. Non-linear Activation (ReLU)
        return np.maximum(0.0, H)

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W_gcn=self.W_gcn)

    def load_weights(self, filepath: str):
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.W_gcn = data["W_gcn"]
