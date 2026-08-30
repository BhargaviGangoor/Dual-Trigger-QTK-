import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Optional

class IsolationForestBaseline:
    """
    Machine Learning Baseline: Isolation Forest Anomaly Detector.
    Fits an ensemble of isolation trees on normative feature vectors [session_dur, sync_freq, msgs, ip_ch, tz_ch].
    Flags out-of-distribution behavioral observations.
    """
    def __init__(self, contamination: float = 0.08, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state
        )
        self.is_trained = False

    def extract_features(self, telemetry: Dict[str, Any]) -> np.ndarray:
        """Extracts normalized 5D feature vector from context telemetry."""
        dur = float(telemetry.get("session_duration_sec", 0.0)) / 600.0
        sync = float(telemetry.get("sync_frequency", 0.0)) / 20.0
        msgs = float(telemetry.get("message_count_sent", 0.0)) / 50.0
        ip_ch = float(telemetry.get("ip_changed", 0.0))
        tz_ch = float(telemetry.get("tz_changed", 0.0))

        if telemetry.get("is_vpn", 0.0) > 0:
            ip_ch = max(ip_ch, 0.8)

        return np.array([dur, sync, msgs, ip_ch, tz_ch], dtype=np.float64)

    def fit_on_normal(self, normal_telemetries: List[Dict[str, Any]]):
        """Fits the Isolation Forest on clean training telemetries."""
        if not normal_telemetries:
            # Fallback synthetic fitting
            synthetic = np.random.normal([0.2, 0.2, 0.2, 0.0, 0.0], 0.05, (100, 5))
            self.model.fit(synthetic)
            self.is_trained = True
            return

        X = np.array([self.extract_features(t) for t in normal_telemetries])
        self.model.fit(X)
        self.is_trained = True

    def evaluate_telemetry(self, telemetry: Dict[str, Any]) -> float:
        """
        Returns an anomaly score in [0, 1] (higher = more anomalous).
        """
        if not self.is_trained:
            self.fit_on_normal([])

        X = self.extract_features(telemetry).reshape(1, -1)
        # decision_function returns negative values for anomalies
        raw_score = float(self.model.decision_function(X)[0])
        # Map raw score to [0, 1] range
        prob = 1.0 / (1.0 + np.exp(raw_score * 5.0))
        return round(float(prob), 4)

    def evaluate_device(self, device, current_epoch: int, threshold: float = 0.60) -> bool:
        """
        Returns True if latest telemetry is classified as anomalous.
        """
        history = getattr(device, "telemetry_history", [])
        if not history:
            return False
        score = self.evaluate_telemetry(history[-1])
        return score >= threshold
