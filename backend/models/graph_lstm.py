import numpy as np
from typing import List, Dict, Any, Tuple
from .dynamic_graph import DynamicGraph
from .weighted_gnn import WeightedGNN

class GraphLSTM:
    """
    Implements the Graph-LSTM temporal-relational model described in Section IV-C of the research paper.
    Tracks the evolution of normalized GCN device embeddings over time, computing reconstruction
    errors to evaluate how well each device fits the collective relationship structure.
    """
    def __init__(self, feature_dim: int = 5, hidden_dim: int = 16, beta: float = 0.8, seed: int = 42):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dynamic_graph = DynamicGraph(beta)
        self.weighted_gnn = WeightedGNN(feature_dim, hidden_dim, seed)
        
        # LSTM weight matrices
        np.random.seed(seed)
        self.W_i = np.random.normal(0, 0.1, (hidden_dim, hidden_dim + hidden_dim))
        self.W_f = np.random.normal(0, 0.1, (hidden_dim, hidden_dim + hidden_dim))
        self.W_c = np.random.normal(0, 0.1, (hidden_dim, hidden_dim + hidden_dim))
        self.W_o = np.random.normal(0, 0.1, (hidden_dim, hidden_dim + hidden_dim))
        
        self.b_i = np.zeros((hidden_dim, 1))
        self.b_f = np.ones((hidden_dim, 1))  # forget gate bias
        self.b_c = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((hidden_dim, 1))
        
        # Reconstruction projection layers
        self.W_y = np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
        self.b_y = np.zeros((hidden_dim, 1))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

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

    def evaluate_devices(self, devices_meta_history: List[List[Dict[str, Any]]], 
                         prev_adj: np.ndarray = None) -> Tuple[np.ndarray, List[float]]:
        """
        Processes multi-device metadata history to calculate:
        1. Evolved adjacency matrix (edge weights).
        2. Relational anomaly scores for each device.
        Args:
            devices_meta_history: List of shape (num_devices, seq_len) containing dictionaries of device metadata.
        Returns:
            adj: Dynamic adjacency matrix
            anomaly_scores: Relational anomaly score for each device (0.0 to 1.0)
        """
        num_devices = len(devices_meta_history)
        if num_devices == 0:
            return np.array([]), []
            
        # Determine the minimum history length across all devices to align them
        seq_len = min(len(h) for h in devices_meta_history) if devices_meta_history else 0
        if seq_len == 0:
            return np.array([]), [0.0] * num_devices
        
        # Truncate histories to seq_len from the end
        aligned_meta_history = [h[-seq_len:] for h in devices_meta_history]
        
        # Build features for each time step t
        # shape: (seq_len, num_devices, feature_dim)
        features_history = []
        for t in range(seq_len):
            step_feats = []
            for d in range(num_devices):
                rec = aligned_meta_history[d][t]
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
                    prev_rec = aligned_meta_history[d][t-1]
                    prev_ip = prev_rec.get("network_ip", "")
                    prev_tz = prev_rec.get("active_timezone", "")
                    if network_ip != prev_ip: 
                        ip_changed = 1.0
                    if active_timezone != prev_tz: 
                        tz_changed = 1.0

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
            last_step_devices.append(aligned_meta_history[d][-1])
                
        adj = self.dynamic_graph.evolve_adjacency(last_step_devices, prev_adj)
        
        # Run DW-GCN for each time step in sequence
        gcn_embeddings_history = []
        for t in range(seq_len):
            X_t = features_history[t]
            H_t = self.weighted_gnn.run_gcn(X_t, adj)
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
