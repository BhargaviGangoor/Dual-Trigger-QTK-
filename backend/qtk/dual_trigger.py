import os
import yaml
from typing import Tuple, Optional
from simulator.device import Device
from .inactivity_trigger import InactivityTrigger

class DualTrigger:
    """
    Implements the dual-trigger condition from Equation 11 of the research paper:
    Quarantine(d) <=> (e_i - e_pk(d) >= delta_inact) OR (R(d,t) >= theta_R)
    
    This evaluates both inactivity-based triggers and behavioral risk triggers,
    and quarantines the device if either condition is met.
    """
    def __init__(self, delta_inact: int = 5, theta_R: float = 0.65):
        self.delta_inact = delta_inact
        self.theta_R = theta_R
        self.inactivity_trigger = InactivityTrigger(delta_inact)

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
            qtk_cfg = cfg.get("qtk", {})
            return cls(
                delta_inact=qtk_cfg.get("delta_inact", 5),
                theta_R=qtk_cfg.get("theta_R", 0.65)
            )
        return cls()

    def evaluate(self, current_epoch: int, device: Device, R_dt: Optional[float] = None) -> Tuple[bool, str]:
        """
        Evaluates the dual-trigger condition:
        Quarantine(d) <=> (InactivityTrigger == True) OR (R(d,t) >= theta_R)
        """
        if R_dt is None:
            R_dt = getattr(device, "final_risk", 0.0)
            
        inactivity_fired = self.inactivity_trigger.evaluate(current_epoch, device)
        behavior_fired = R_dt >= self.theta_R
        epoch_gap = current_epoch - device.epoch_last_key_update

        if inactivity_fired and behavior_fired:
            return True, f"Both triggers fired (Epoch Gap: {epoch_gap} >= {self.delta_inact}, Risk: {R_dt:.2f} >= {self.theta_R})"
        elif inactivity_fired:
            return True, f"Inactivity timer expired (Epoch Gap: {epoch_gap} >= {self.delta_inact})"
        elif behavior_fired:
            return True, f"Behavioral anomaly detected (Risk: {R_dt:.2f} >= {self.theta_R})"
            
        return False, "Device is trust-compliant."
