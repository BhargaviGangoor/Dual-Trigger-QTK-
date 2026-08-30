import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hmm import HMMDetector
from models.dynamic_graph import DynamicGraph
from models.weighted_gnn import WeightedGNN
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from simulator.device import HMMState

class TestModels(unittest.TestCase):
    def test_hmm_detector_bounds(self):
        hmm = HMMDetector()
        mock_normal_history = [
            {"session_duration_sec": 120.0, "sync_frequency": 4.0, "message_count_sent": 6, "ip_changed": 0.0, "tz_changed": 0.0},
            {"session_duration_sec": 130.0, "sync_frequency": 4.2, "message_count_sent": 8, "ip_changed": 0.0, "tz_changed": 0.0},
            {"session_duration_sec": 110.0, "sync_frequency": 3.8, "message_count_sent": 5, "ip_changed": 0.0, "tz_changed": 0.0}
        ]
        state, p_c = hmm.evaluate_history(mock_normal_history)
        self.assertIn(state, [HMMState.NORMAL, HMMState.IDLE])
        self.assertGreaterEqual(p_c, 0.0)
        self.assertLessEqual(p_c, 1.0)
        self.assertLess(p_c, 0.50)

        # Anomaly case
        mock_anom_history = [
            {"session_duration_sec": 600.0, "sync_frequency": 22.0, "message_count_sent": 40, "is_vpn": 1.0, "ip_changed": 1.0, "tz_changed": 1.0},
            {"session_duration_sec": 700.0, "sync_frequency": 25.0, "message_count_sent": 50, "is_vpn": 1.0, "ip_changed": 1.0, "tz_changed": 1.0}
        ]
        state_anom, p_c_anom = hmm.evaluate_history(mock_anom_history)
        self.assertIn(state_anom, [HMMState.SUSPICIOUS, HMMState.HIGH_RISK])
        self.assertGreater(p_c_anom, 0.50)

    def test_dynamic_graph_evolution(self):
        graph = DynamicGraph(beta=0.8)
        dev1 = {"sync_frequency": 4.0, "network_type": "WiFi", "network_ip": "192.168.1.10", "location_country": "US", "active_timezone": "UTC"}
        dev2 = {"sync_frequency": 4.2, "network_type": "WiFi", "network_ip": "192.168.1.20", "location_country": "US", "active_timezone": "UTC"}
        
        sim = graph.calculate_similarity(dev1, dev2)
        self.assertGreater(sim, 0.70)

        adj = graph.evolve_adjacency([dev1, dev2])
        self.assertEqual(adj.shape, (2, 2))
        self.assertEqual(adj[0, 0], 1.0)
        self.assertEqual(adj[1, 1], 1.0)
        self.assertGreater(adj[0, 1], 0.0)

    def test_weighted_gnn_convolution(self):
        gnn = WeightedGNN(feature_dim=5, hidden_dim=16, seed=42)
        X = np.random.normal(0, 1, (3, 5))
        adj = np.eye(3)
        H = gnn.forward(X, adj)
        self.assertEqual(H.shape, (3, 16))
        # Non-negative (ReLU output)
        self.assertTrue((H >= 0.0).all())

    def test_graph_lstm_evaluation(self):
        graph_lstm = GraphLSTM(beta=0.8, seed=42)
        history = [
            [{"session_duration_sec": 120.0, "sync_frequency": 4.0, "message_count_sent": 5} for _ in range(5)],
            [{"session_duration_sec": 130.0, "sync_frequency": 4.2, "message_count_sent": 6} for _ in range(5)]
        ]
        adj, scores = graph_lstm.evaluate_devices(history)
        self.assertEqual(adj.shape, (2, 2))
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_trust_decay_and_distrust(self):
        prev_trust = 1.0
        p_c = 0.8  # anomalous evidence
        evidence_trust = 1.0 - p_c  # 0.2
        # T_1 = 0.8 * 1.0 + 0.2 * 0.2 = 0.84
        new_trust = TrustScore.calculate_decay(prev_trust, evidence_trust, alpha=0.8)
        self.assertEqual(new_trust, 0.84)
        distrust = TrustScore.compute_distrust(new_trust)
        self.assertEqual(distrust, 0.16)

    def test_risk_fusion_layer(self):
        fusion = RiskFusion()
        # Clean inputs: p_c=0.05, s_graph=0.05, trust=0.98 -> R_dt should be low (< 0.5)
        clean_risk = fusion.fuse(p_c=0.05, s_graph=0.05, trust_score=0.98)
        self.assertLess(clean_risk, 0.50)

        # High risk inputs: p_c=0.9, s_graph=0.85, trust=0.20 -> R_dt should be high (>= 0.65)
        high_risk = fusion.fuse(p_c=0.90, s_graph=0.85, trust_score=0.20)
        self.assertGreaterEqual(high_risk, 0.65)

if __name__ == "__main__":
    unittest.main()
