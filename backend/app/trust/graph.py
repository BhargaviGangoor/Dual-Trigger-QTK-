import numpy as np
from typing import List, Dict, Any, Tuple

class DeviceRelationshipGraph:
    """
    Manages the dynamic device relationship graph for a user's devices.
    Calculates pairwise device similarities and evolves edge weights over time.
    Processes the graph using a Decay-Weighted Graph Convolutional Network (DW-GCN)
    and a Graph-LSTM sequence model to compute relational anomaly scores.
    """
    def __init__(self, feature_dim: int = 5, hidden_dim: int = 16):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Memory coefficient for edge weight evolution
        self.beta = 0.8
        
        # GNN weights (NumPy initialized)
        np.random.seed(42)
        self.W_gcn = np.random.normal(0, 0.1, (self.feature_dim, self.hidden_dim))
        
        # Graph-LSTM weights
        self.W_i = np.random.normal(0, 0.1, (self.hidden_dim, self.hidden_dim + self.hidden_dim))
        self.W_f = np.random.normal(0, 0.1, (self.hidden_dim, self.hidden_dim + self.hidden_dim))
        self.W_c = np.random.normal(0, 0.1, (self.hidden_dim, self.hidden_dim + self.hidden_dim))
        self.W_o = np.random.normal(0, 0.1, (self.hidden_dim, self.hidden_dim + self.hidden_dim))
        
        self.b_i = np.zeros((self.hidden_dim, 1))
        self.b_f = np.ones((self.hidden_dim, 1))
        self.b_c = np.zeros((self.hidden_dim, 1))
        self.b_o = np.zeros((self.hidden_dim, 1))
        
        # Reconstruct projected GCN embedding
        self.W_y = np.random.normal(0, 0.1, (self.hidden_dim, self.hidden_dim))
        self.b_y = np.zeros((self.hidden_dim, 1))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def calculate_similarity(self, dev_i: Dict[str, Any], dev_j: Dict[str, Any]) -> float:
        """
        Computes S_ij(t) = lambda_1*Sync + lambda_2*Network + lambda_3*Location + lambda_4*Time
        """
        # 1. Sync similarity
        sync_i = dev_i.get("sync_frequency", 1.0)
        sync_j = dev_j.get("sync_frequency", 1.0)
        sync_sim = np.exp(-abs(sync_i - sync_j) / 5.0)
        
        # 2. Network similarity
        net_i = dev_i.get("network_type", "WiFi")
        net_j = dev_j.get("network_type", "WiFi")
        ip_i = dev_i.get("network_ip", "127.0.0.1")
        ip_j = dev_j.get("network_ip", "127.0.0.1")
        
        net_sim = 0.5 if net_i == net_j else 0.1
        # Check subnet prefix (first three octets)
        prefix_i = ".".join(ip_i.split(".")[:3]) if ip_i else ""
        prefix_j = ".".join(ip_j.split(".")[:3]) if ip_j else ""
        if prefix_i and prefix_i == prefix_j:
            net_sim += 0.5
            
        # 3. Location similarity
        loc_i = dev_i.get("location_country", "United States")
        loc_j = dev_j.get("location_country", "United States")
        loc_sim = 1.0 if loc_i == loc_j else 0.0
        
        # 4. Time similarity
        tz_i = dev_i.get("active_timezone", "UTC")
        tz_j = dev_j.get("active_timezone", "UTC")
        time_sim = 1.0 if tz_i == tz_j else 0.0
        
        # Uniform weights
        S_ij = 0.25 * sync_sim + 0.25 * net_sim + 0.25 * loc_sim + 0.25 * time_sim
        return float(S_ij)

    def compute_edge_weights(self, devices: List[Dict[str, Any]], prev_adj: np.ndarray) -> np.ndarray:
        """
        Evolves adjacency matrix edge weights:
        w_ij^{t+1} = beta * w_ij^t + (1 - beta) * S_ij(t)
        """
        n = len(devices)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, j] = 1.0
                else:
                    s_ij = self.calculate_similarity(devices[i], devices[j])
                    prev_w = prev_adj[i, j] if (prev_adj is not None and i < prev_adj.shape[0] and j < prev_adj.shape[1]) else 0.8
                    adj[i, j] = self.beta * prev_w + (1.0 - self.beta) * s_ij
        return adj

    def run_gcn(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Runs one layer of Decay-Weighted Graph Convolution:
        H = relu(D^-1/2 * W_adj * D^-1/2 * X * W_gcn)
        """
        # Add self-loops
        W_tilde = adj + np.eye(adj.shape[0])
        
        # Calculate degree matrix
        D_tilde = np.diag(np.sum(W_tilde, axis=1))
        
        # D_tilde^-1/2
        d_val = np.diag(D_tilde)
        d_inv_sqrt = np.where(d_val > 0, 1.0 / np.sqrt(d_val), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Symmetric normalization
        A_norm = np.dot(np.dot(D_inv_sqrt, W_tilde), D_inv_sqrt)
        
        # GCN projection
        proj = np.dot(X, self.W_gcn) # (N, hidden_dim)
        H = np.dot(A_norm, proj) # (N, hidden_dim)
        
        # ReLU activation
        return np.maximum(0, H)

    def lstm_forward(self, sequence: np.ndarray) -> np.ndarray:
        """Runs GCN embeddings over time through Graph-LSTM."""
        seq_len, hidden_dim = sequence.shape
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        
        reconstructions = []
        for t in range(seq_len):
            x_t = sequence[t].reshape(self.hidden_dim, 1)
            concat = np.vstack((h, x_t))
            
            i = self.sigmoid(np.dot(self.W_i, concat) + self.b_i)
            f = self.sigmoid(np.dot(self.W_f, concat) + self.b_f)
            c_tilde = np.tanh(np.dot(self.W_c, concat) + self.b_c)
            
            c = f * c + i * c_tilde
            o = self.sigmoid(np.dot(self.W_o, concat) + self.b_o)
            h = o * np.tanh(c)
            
            y_pred = np.dot(self.W_y, h) + self.b_y
            reconstructions.append(y_pred.flatten())
            
        return np.array(reconstructions)

    def evaluate_devices(self, devices_meta_history: List[List[Any]], prev_adj: np.ndarray = None) -> Tuple[np.ndarray, List[float]]:
        """
        Processes multi-device metadata history to calculate:
        1. Evolved adjacency matrix (edge weights).
        2. Relational anomaly scores for each device.
        Args:
            devices_meta_history: List of shape (num_devices, seq_len) containing dictionaries or objects of device metadata.
        Returns:
            adj: Dynamic adjacency matrix
            anomaly_scores: Relational anomaly score for each device
        """
        num_devices = len(devices_meta_history)
        if num_devices == 0:
            return np.array([]), []
            
        seq_len = len(devices_meta_history[0])
        
        # Build features for each time step t
        # shape: (seq_len, num_devices, feature_dim)
        features_history = []
        for t in range(seq_len):
            step_feats = []
            for d in range(num_devices):
                rec = devices_meta_history[d][t]
                # Check if object has attributes or dict keys
                if hasattr(rec, "session_duration_sec"):
                    session_duration = getattr(rec, "session_duration_sec", 0.0)
                    sync_frequency = getattr(rec, "sync_frequency", 0.0)
                    message_count_sent = getattr(rec, "message_count_sent", 0)
                    network_ip = getattr(rec, "network_ip", "")
                    network_type = getattr(rec, "network_type", "")
                    active_timezone = getattr(rec, "active_timezone", "")
                    location_country = getattr(rec, "location_country", "")
                else:
                    session_duration = rec.get("session_duration_sec", 0.0)
                    sync_frequency = rec.get("sync_frequency", 0.0)
                    message_count_sent = rec.get("message_count_sent", 0)
                    network_ip = rec.get("network_ip", "")
                    network_type = rec.get("network_type", "")
                    active_timezone = rec.get("active_timezone", "")
                    location_country = rec.get("location_country", "")
                
                # Check for IP or timezone change from previous step
                ip_changed = 0.0
                tz_changed = 0.0
                if t > 0:
                    prev_rec = devices_meta_history[d][t-1]
                    prev_ip = getattr(prev_rec, "network_ip", "") if hasattr(prev_rec, "network_ip") else prev_rec.get("network_ip", "")
                    prev_tz = getattr(prev_rec, "active_timezone", "") if hasattr(prev_rec, "active_timezone") else prev_rec.get("active_timezone", "")
                    if network_ip != prev_ip: ip_changed = 1.0
                    if active_timezone != prev_tz: tz_changed = 1.0

                step_feats.append([
                    float(session_duration) / 3600.0,
                    float(sync_frequency) / 60.0,
                    float(message_count_sent) / 50.0,
                    ip_changed,
                    tz_changed
                ])
            features_history.append(np.array(step_feats))
            
        # Evolve edge weights using final step metadata
        last_step_devices = []
        for d in range(num_devices):
            rec = devices_meta_history[d][-1]
            if hasattr(rec, "sync_frequency"):
                last_step_devices.append({
                    "sync_frequency": getattr(rec, "sync_frequency", 0.0),
                    "network_type": getattr(rec, "network_type", ""),
                    "network_ip": getattr(rec, "network_ip", ""),
                    "location_country": getattr(rec, "location_country", ""),
                    "active_timezone": getattr(rec, "active_timezone", "")
                })
            else:
                last_step_devices.append(rec)
                
        adj = self.compute_edge_weights(last_step_devices, prev_adj)
        
        # Run DW-GCN for each time step in sequence
        gcn_embeddings_history = []
        for t in range(seq_len):
            X_t = features_history[t]
            H_t = self.run_gcn(X_t, adj)
            gcn_embeddings_history.append(H_t)
            
        # Run Graph-LSTM per device across time sequence
        anomaly_scores = []
        for d in range(num_devices):
            device_seq = np.array([gcn_embeddings_history[t][d] for t in range(seq_len)])
            reconstruction = self.lstm_forward(device_seq)
            
            mse = np.mean((device_seq - reconstruction) ** 2)
            relational_score = min(1.0, mse * 8.0)
            anomaly_scores.append(float(round(relational_score, 4)))
            
        return adj, anomaly_scores
