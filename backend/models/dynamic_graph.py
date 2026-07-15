import numpy as np
from typing import List, Dict, Any

class DynamicGraph:
    def __init__(self, beta: float = 0.8):
        """
        Manages the dynamic device relationship graph.
        beta: Memory coefficient for edge weight evolution:
              w_ij^{t+1} = beta * w_ij^t + (1 - beta) * S_ij(t)
        """
        self.beta = beta

    def calculate_similarity(self, dev_i: Dict[str, Any], dev_j: Dict[str, Any]) -> float:
        """
        Computes similarity S_ij(t) based on features: Sync, Network, Location, Timezone.
        dev_i and dev_j are dicts of device metadata.
        """
        # 1. Sync frequency similarity
        sync_i = dev_i.get("sync_frequency", 1.0)
        sync_j = dev_j.get("sync_frequency", 1.0)
        sync_sim = np.exp(-abs(sync_i - sync_j) / 5.0)
        
        # 2. Network similarity
        net_i = dev_i.get("network_type", "WiFi")
        net_j = dev_j.get("network_type", "WiFi")
        ip_i = dev_i.get("network_ip", "127.0.0.1")
        ip_j = dev_j.get("network_ip", "127.0.0.1")
        
        net_sim = 0.5 if net_i == net_j else 0.1
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
        
        # Uniform weights fusion
        S_ij = 0.25 * sync_sim + 0.25 * net_sim + 0.25 * loc_sim + 0.25 * time_sim
        return float(S_ij)

    def evolve_adjacency(self, current_metadata: List[Dict[str, Any]], prev_adj: np.ndarray) -> np.ndarray:
        """
        Evolves adjacency matrix edge weights:
        w_ij^{t+1} = beta * w_ij^t + (1 - beta) * S_ij(t)
        """
        n = len(current_metadata)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, j] = 1.0
                else:
                    s_ij = self.calculate_similarity(current_metadata[i], current_metadata[j])
                    prev_w = prev_adj[i, j] if (prev_adj is not None and i < prev_adj.shape[0] and j < prev_adj.shape[1]) else 0.8
                    adj[i, j] = self.beta * prev_w + (1.0 - self.beta) * s_ij
        return adj
