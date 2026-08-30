import numpy as np
from typing import List, Dict, Any, Optional

class DynamicGraph:
    """
    Manages the dynamic device relationship graph for user group devices.
    Edge weights represent behavioral similarity and evolve dynamically over time:
    w_ij(t+1) = beta * w_ij(t) + (1 - beta) * S_ij(t)
    """
    def __init__(self, beta: float = 0.8):
        self.beta = beta

    def calculate_similarity(self, dev_i: Dict[str, Any], dev_j: Dict[str, Any]) -> float:
        """
        Computes the pairwise behavioral similarity S_ij(t) between two devices:
        Evaluates sync rate, network prefix, location country, and active timezone.
        """
        # 1. Synchronization frequency similarity
        sync_i = float(dev_i.get("sync_frequency", 1.0))
        sync_j = float(dev_j.get("sync_frequency", 1.0))
        sync_sim = np.exp(-abs(sync_i - sync_j) / 5.0)

        # 2. Network type and IP prefix similarity
        net_i = dev_i.get("network_type", "WiFi")
        net_j = dev_j.get("network_type", "WiFi")
        ip_i = dev_i.get("network_ip", "127.0.0.1")
        ip_j = dev_j.get("network_ip", "127.0.0.1")

        net_sim = 0.5 if net_i == net_j else 0.1
        prefix_i = ".".join(ip_i.split(".")[:3]) if ip_i else ""
        prefix_j = ".".join(ip_j.split(".")[:3]) if ip_j else ""
        if prefix_i and prefix_i == prefix_j:
            net_sim += 0.5

        # Penalize if either device is routing via foreign VPN while the other is not
        if dev_i.get("is_vpn", 0.0) != dev_j.get("is_vpn", 0.0):
            net_sim = max(0.0, net_sim - 0.5)

        # 3. Location country similarity
        loc_i = dev_i.get("location_country", "United States")
        loc_j = dev_j.get("location_country", "United States")
        loc_sim = 1.0 if loc_i == loc_j else 0.0

        # 4. Active timezone similarity
        tz_i = dev_i.get("active_timezone", "UTC")
        tz_j = dev_j.get("active_timezone", "UTC")
        tz_sim = 1.0 if tz_i == tz_j else 0.0

        # Weighted combination with lambda = [0.25, 0.25, 0.25, 0.25]
        S_ij = 0.25 * sync_sim + 0.25 * net_sim + 0.25 * loc_sim + 0.25 * tz_sim
        return float(np.clip(S_ij, 0.0, 1.0))

    def evolve_adjacency(
        self,
        current_telemetries: List[Dict[str, Any]],
        prev_adj: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evolves the dynamic adjacency matrix:
        w_ij(t+1) = beta * w_ij(t) + (1 - beta) * S_ij(t)
        """
        n = len(current_telemetries)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, j] = 1.0
                else:
                    s_ij = self.calculate_similarity(current_telemetries[i], current_telemetries[j])
                    prev_w = 0.8
                    if prev_adj is not None and i < prev_adj.shape[0] and j < prev_adj.shape[1]:
                        prev_w = prev_adj[i, j]
                    adj[i, j] = self.beta * prev_w + (1.0 - self.beta) * s_ij

        return adj
