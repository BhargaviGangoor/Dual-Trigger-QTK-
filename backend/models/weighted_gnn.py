import numpy as np

class WeightedGNN:
    def __init__(self, feature_dim: int = 5, hidden_dim: int = 16, seed: int = 42):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # GNN weights (NumPy initialized)
        np.random.seed(seed)
        self.W_gcn = np.random.normal(0, 0.1, (self.feature_dim, self.hidden_dim))

    def run_gcn(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Runs one layer of Decay-Weighted Graph Convolution:
        H = relu(D^-1/2 * W_tilde * D^-1/2 * X * W_gcn)
        """
        # Add self-loops
        W_tilde = adj + np.eye(adj.shape[0])
        
        # Compute degree values
        d_val = np.sum(W_tilde, axis=1)
        d_val = np.clip(d_val, 1e-5, None)
        
        # D^-1/2 matrix
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d_val))
        
        # Normalized Adjacency: A_norm = D^-1/2 * W_tilde * D^-1/2
        A_norm = np.dot(np.dot(D_inv_sqrt, W_tilde), D_inv_sqrt)
        
        # Message propagation: Z = A_norm * X
        prop = np.dot(A_norm, X)
        
        # Projection: H = prop * W_gcn
        H = np.dot(prop, self.W_gcn)
        
        # Activation (ReLU)
        H_relu = np.maximum(0, H)
        return H_relu
