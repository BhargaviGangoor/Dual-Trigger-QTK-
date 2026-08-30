from enum import Enum
from typing import Tuple, Optional
import os
import yaml
from .inactivity_trigger import InactivityTrigger

class TriggerReason(str, Enum):
    """Reason code returned by the Dual-Trigger QTK decision engine."""
    INACTIVITY = "INACTIVITY"
    BEHAVIORAL = "BEHAVIORAL"
    BOTH = "BOTH"
    NONE = "NONE"

class DualTrigger:
    """
    Implements the Dual-Trigger Quarantined-TreeKEM decision rule:
    Quarantine(d) <==> (e_i - e_pk(d) >= delta_inact) OR (R(d,t) >= theta_R)
    """
    def __init__(self, delta_inact: int = 5, theta_R: float = 0.65):
        self.delta_inact = delta_inact
        self.theta_R = theta_R
        self.inactivity_trigger = InactivityTrigger(delta_inact)

    @classmethod
    def from_config(cls, config_path: Optional[str] = None):
        """Loads threshold parameters from YAML config."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "thresholds.yaml"
            )
        delta_inact = 5
        theta_R = 0.65
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            qtk_cfg = cfg.get("qtk", {})
            delta_inact = qtk_cfg.get("delta_inact", 5)
            theta_R = qtk_cfg.get("theta_R", 0.65)
        return cls(delta_inact=delta_inact, theta_R=theta_R)

    def check_inactivity(self, device, current_epoch: int) -> bool:
        """
        Evaluates the protocol inactivity condition: (e_i - e_pk(d)) >= delta_inact.
        """
        return self.inactivity_trigger.check(device, current_epoch)

    def check_behavioral_risk(self, device, R_dt: Optional[float] = None) -> bool:
        """
        Evaluates the behavioral risk condition: R(d,t) >= theta_R.
        """
        risk = R_dt if R_dt is not None else getattr(device, "final_risk", 0.0)
        return risk >= self.theta_R

    def dual_trigger_decision(
        self,
        device,
        current_epoch: int,
        R_dt: Optional[float] = None
    ) -> Tuple[bool, TriggerReason, str]:
        """
        Evaluates the dual trigger rule:
        Returns:
            should_quarantine (bool): True if quarantine should be invoked.
            reason (TriggerReason): INACTIVITY, BEHAVIORAL, BOTH, or NONE.
            detail (str): Human-readable explanatory string.
        """
        inact_fired = self.check_inactivity(device, current_epoch)
        behav_fired = self.check_behavioral_risk(device, R_dt)

        age = current_epoch - getattr(device, "epoch_last_key_update", 0)
        risk = R_dt if R_dt is not None else getattr(device, "final_risk", 0.0)

        if inact_fired and behav_fired:
            return True, TriggerReason.BOTH, f"Both triggers fired (Key Age: {age} >= {self.delta_inact}, Risk: {risk:.3f} >= {self.theta_R})"
        elif inact_fired:
            return True, TriggerReason.INACTIVITY, f"Inactivity threshold exceeded (Key Age: {age} >= {self.delta_inact})"
        elif behav_fired:
            return True, TriggerReason.BEHAVIORAL, f"Behavioral risk threshold exceeded (Risk: {risk:.3f} >= {self.theta_R})"
        else:
            return False, TriggerReason.NONE, "Device is trust-compliant (no quarantine)."

    def evaluate(self, current_epoch: int, device, R_dt: Optional[float] = None) -> Tuple[bool, str]:
        """Convenience method returning (triggered, explanation)."""
        triggered, reason, detail = self.dual_trigger_decision(device, current_epoch, R_dt)
        return triggered, detail
