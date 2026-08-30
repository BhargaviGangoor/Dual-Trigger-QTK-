from models.hmm import HMMDetector
from typing import Optional

class HMMBaseline:
    """
    Behavioral Baseline: HMM-Only Anomaly Detector.
    Evaluates individual per-device behavioral sequences without relational graph or trust decay.
    Flags anomaly when P_c(d, t) >= theta_R.
    """
    def __init__(self, theta_R: float = 0.65, hmm_detector: Optional[HMMDetector] = None):
        self.theta_R = theta_R
        self.hmm = hmm_detector if hmm_detector is not None else HMMDetector()

    def evaluate_device(self, device, current_epoch: int) -> bool:
        """
        Returns True if HMM individual anomaly score meets or exceeds theta_R.
        """
        history = getattr(device, "telemetry_history", [])
        if len(history) < 1:
            return False
        state, p_c = self.hmm.evaluate_history(history)
        return p_c >= self.theta_R
