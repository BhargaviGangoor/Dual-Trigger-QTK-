from typing import Dict, Any, Optional

class ThresholdDetector:
    """
    Simple Heuristic Baseline: Threshold-Based Anomaly Detector.
    Evaluates rule-based static threshold violations on raw telemetry features:
    - High sync polling (> 12 syncs/hour)
    - VPN / Proxy routing (is_vpn == 1.0)
    - Excessive session length (> 400 sec)
    """
    def __init__(
        self,
        sync_threshold: float = 12.0,
        session_threshold: float = 400.0,
        anomaly_threshold: float = 0.50
    ):
        self.sync_threshold = sync_threshold
        self.session_threshold = session_threshold
        self.anomaly_threshold = anomaly_threshold

    def evaluate_telemetry(self, context_telemetry: Dict[str, Any]) -> float:
        """
        Computes heuristic anomaly score in [0, 1] based on threshold violations.
        """
        score = 0.0
        sync = float(context_telemetry.get("sync_frequency", 0.0))
        dur = float(context_telemetry.get("session_duration_sec", 0.0))
        is_vpn = float(context_telemetry.get("is_vpn", 0.0))
        ip_changed = float(context_telemetry.get("ip_changed", 0.0))

        if is_vpn > 0:
            score += 0.45
        if sync > self.sync_threshold:
            score += 0.35
        if dur > self.session_threshold:
            score += 0.20
        if ip_changed > 0:
            score += 0.10

        return min(1.0, score)

    def evaluate_device(self, device, current_epoch: int) -> bool:
        """
        Returns True if device violates static threshold bounds.
        """
        history = getattr(device, "telemetry_history", [])
        if not history:
            return False
        latest = history[-1]
        score = self.evaluate_telemetry(latest)
        return score >= self.anomaly_threshold
