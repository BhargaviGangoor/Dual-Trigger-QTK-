import os
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class RiskFusion:
    """
    Trainable Behavioral Risk Fusion Layer.
    Fuses individual HMM anomaly score P_c(d,t), relational anomaly score S_graph(d,t),
    and accumulated distrust (1 - T_t(d)) into a unified risk score R(d,t) in [0, 1]:
    
    z(d, t) = [P_c(d, t), S_graph(d, t), 1 - T_t(d)]^T
    R(d, t) = sigmoid( W_f . z(d, t) + b )
    
    Trained via Binary Cross-Entropy on labeled training data.
    """
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        bias: float = -1.0,
        learning_rate: float = 0.05
    ):
        self.weights = np.array(weights if weights is not None else [1.8, 2.2, 1.2], dtype=np.float64)
        self.bias = float(bias)
        self.learning_rate = learning_rate
        self.is_trained = False

    @classmethod
    def from_config(cls, config_path: Optional[str] = None):
        """Initializes RiskFusion from YAML configuration."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "model.yaml"
            )
        weights = [1.8, 2.2, 1.2]
        bias = -1.0
        lr = 0.05
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            rf_cfg = cfg.get("risk_fusion", {})
            weights = rf_cfg.get("initial_weights", weights)
            bias = rf_cfg.get("initial_bias", bias)
            lr = rf_cfg.get("learning_rate", lr)
        return cls(weights=weights, bias=bias, learning_rate=lr)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))

    def fuse(self, p_c: float, s_graph: float, trust_score: float) -> float:
        """
        Computes fused risk R(d,t) from component scores:
        Args:
            p_c: Individual HMM anomaly probability in [0, 1]
            s_graph: Relational anomaly score in [0, 1]
            trust_score: Accumulated trust score T_t(d) in [0, 1]
        Returns:
            R_dt: Unified behavioral risk score in [0, 1]
        """
        p_c = max(0.0, min(1.0, float(p_c)))
        s_graph = max(0.0, min(1.0, float(s_graph)))
        distrust = max(0.0, min(1.0, 1.0 - float(trust_score)))

        z = np.array([p_c, s_graph, distrust], dtype=np.float64)
        logit = float(np.dot(self.weights, z) + self.bias)
        R_dt = self._sigmoid(logit)
        return round(float(R_dt), 4)

    def predict(self, device) -> float:
        """
        Computes final risk R(d,t) using device's current attributes
        and updates device.final_risk.
        """
        p_c = getattr(device, "behavioral_risk", 0.0)
        s_graph = getattr(device, "graph_risk", 0.0)
        t_t = getattr(device, "trust_score", 1.0)

        r_dt = self.fuse(p_c, s_graph, t_t)
        device.final_risk = r_dt
        return r_dt

    def fit(
        self,
        X_samples: np.ndarray,
        y_labels: np.ndarray,
        epochs: int = 80,
        lr: Optional[float] = None,
        l2_reg: float = 1e-4
    ) -> List[float]:
        """
        Trains weights (W_f, b) using Binary Cross-Entropy loss via gradient descent.
        Args:
            X_samples: (N, 3) feature matrix [p_c, s_graph, 1 - T_t]
            y_labels: (N,) binary ground truth vector (0 for legitimate, 1 for rogue/mimicry)
        """
        if lr is None:
            lr = self.learning_rate

        N = len(y_labels)
        if N == 0:
            return []

        losses = []
        for epoch in range(epochs):
            logits = X_samples @ self.weights + self.bias
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

            # BCE Loss
            loss = -np.mean(y_labels * np.log(np.clip(preds, 1e-12, 1.0)) + 
                            (1.0 - y_labels) * np.log(np.clip(1.0 - preds, 1e-12, 1.0)))
            losses.append(float(loss))

            # Gradients
            err = preds - y_labels
            grad_w = (X_samples.T @ err) / N + l2_reg * self.weights
            grad_b = np.mean(err)

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

        self.is_trained = True
        return losses

    def fit_from_dataset(
        self,
        train_records: List[Dict[str, Any]],
        hmm_detector: Any,
        graph_lstm: Any,
        epochs: int = 80,
        lr: float = 0.05
    ) -> Dict[str, Any]:
        """
        Extracts multi-component features [p_c, s_graph, distrust] from training dataset observations
        and trains RiskFusion parameters (W_f, b) via Binary Cross-Entropy.
        """
        if not train_records:
            return {}

        init_w = self.weights.copy()
        init_b = self.bias

        X_list = []
        y_list = []

        # Group records by run and device to construct temporal histories
        runs_devices: Dict[Any, List[Dict[str, Any]]] = {}
        for r in train_records:
            key = (r.get("run_id", 0), r.get("device_id", "dev"))
            if key not in runs_devices:
                runs_devices[key] = []
            runs_devices[key].append(r)

        for (run_id, dev_id), recs in runs_devices.items():
            recs_sorted = sorted(recs, key=lambda x: x.get("epoch", 0))
            hist = []
            trust = 1.0
            for r in recs_sorted:
                ctx = r.get("context_telemetry", {})
                hist.append(ctx)
                label = r.get("ground_truth_label", 0)

                # Compute individual HMM score P_c
                _, p_c = hmm_detector.evaluate_history(hist)
                # Compute relational graph score S_graph
                _, s_g_scores = graph_lstm.evaluate_devices([hist, hist])
                s_g = s_g_scores[0] if s_g_scores else 0.35

                # Update trust
                b_t = max(0.0, min(1.0, 1.0 - p_c))
                trust = 0.8 * trust + 0.2 * b_t
                distrust = max(0.0, min(1.0, 1.0 - trust))

                X_list.append([p_c, s_g, distrust])
                y_list.append(label)

        X_arr = np.array(X_list, dtype=np.float64)
        y_arr = np.array(y_list, dtype=np.float64)

        losses = self.fit(X_arr, y_arr, epochs=epochs, lr=lr)

        fit_summary = {
            "initial_weights": init_w.tolist(),
            "final_weights": self.weights.tolist(),
            "initial_bias": float(init_b),
            "final_bias": float(self.bias),
            "initial_loss": losses[0] if losses else 0.0,
            "final_loss": losses[-1] if losses else 0.0,
            "num_epochs": epochs,
            "num_samples": len(y_arr)
        }
        return fit_summary

    def calibrate_threshold(
        self,
        val_records: List[Dict[str, Any]],
        hmm_detector: Any,
        graph_lstm: Any,
        max_fpr: float = 0.10,
        default_theta: float = 0.65
    ) -> float:
        """
        Calibrates optimal risk threshold theta_R strictly on the VALIDATION split.
        Maximizes F1-score subject to FPR <= max_fpr.
        """
        if not val_records:
            return default_theta

        runs_devices: Dict[Any, List[Dict[str, Any]]] = {}
        for r in val_records:
            key = (r.get("run_id", 0), r.get("device_id", "dev"))
            if key not in runs_devices:
                runs_devices[key] = []
            runs_devices[key].append(r)

        fused_scores = []
        labels = []

        for (run_id, dev_id), recs in runs_devices.items():
            recs_sorted = sorted(recs, key=lambda x: x.get("epoch", 0))
            hist = []
            trust = 1.0
            for r in recs_sorted:
                ctx = r.get("context_telemetry", {})
                hist.append(ctx)
                label = r.get("ground_truth_label", 0)

                _, p_c = hmm_detector.evaluate_history(hist)
                _, s_g_scores = graph_lstm.evaluate_devices([hist, hist])
                s_g = s_g_scores[0] if s_g_scores else 0.35

                b_t = max(0.0, min(1.0, 1.0 - p_c))
                trust = 0.8 * trust + 0.2 * b_t

                r_dt = self.fuse(p_c, s_g, trust)
                fused_scores.append(r_dt)
                labels.append(label)

        scores_arr = np.array(fused_scores, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.int32)

        best_f1 = -1.0
        best_theta = default_theta

        for theta in np.arange(0.35, 0.90, 0.01):
            preds = (scores_arr >= theta).astype(np.int32)
            tp = int(np.sum((preds == 1) & (labels_arr == 1)))
            fp = int(np.sum((preds == 1) & (labels_arr == 0)))
            fn = int(np.sum((preds == 0) & (labels_arr == 1)))
            tn = int(np.sum((preds == 0) & (labels_arr == 0)))

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            if fpr <= max_fpr and f1 > best_f1:
                best_f1 = f1
                best_theta = float(round(theta, 3))

        return best_theta

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, weights=self.weights, bias=self.bias)

    def load_weights(self, filepath: str):
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.weights = data["weights"]
            self.bias = float(data["bias"])
            self.is_trained = True
