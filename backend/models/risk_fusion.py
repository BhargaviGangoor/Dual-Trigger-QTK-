import os
import yaml
import numpy as np
from typing import Dict, Any, List

class RiskFusion:
    def __init__(self, weights: List[float] = None, bias: float = 0.2):
        self.weights = np.array(weights if weights else [1.5, 2.0, -1.0])
        self.bias = bias

    @classmethod
    def from_config(cls):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "thresholds.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            model_cfg = cfg.get("models", {})
            return cls(
                weights=model_cfg.get("fusion_weights", [1.5, 2.0, -1.0]),
                bias=model_cfg.get("fusion_bias", 0.2)
            )
        return cls()

    def fuse(self, hmm_state: int, hmm_confidence: float, 
             relational_anomaly_score: float, trust_score: float) -> float:
        """
        Fuses HMM state, GNN/Graph-LSTM relational anomaly, and trust history
        into a unified behavioral risk score R(d,t).
        Formula:
        R(d,t) = sigmoid( W_f . z + b )
        z = [P_c, S_graph, T_t]
        P_c = hmm_confidence if hmm_state in [1, 2] else 0.0 (Session Compromise or Ghost Device)
        S_graph = relational_anomaly_score
        T_t = trust_score
        """
        P_c = hmm_confidence if hmm_state in [1, 2] else 0.0
        S_graph = relational_anomaly_score
        T_t = trust_score

        z = np.array([P_c, S_graph, T_t])
        
        # Compute R_d_t
        logit = np.dot(self.weights, z) + self.bias
        R_d_t = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        
        return float(round(R_d_t, 4))
