from qtk.inactivity_trigger import InactivityTrigger

class QTKBaseline:
    """
    Protocol Baseline: Original Quarantined-TreeKEM (QTK).
    Quarantines devices solely based on inactivity when key update age >= delta_inact.
    Has no behavioral detection capability.
    """
    def __init__(self, delta_inact: int = 5):
        self.delta_inact = delta_inact
        self.trigger = InactivityTrigger(delta_inact)

    def evaluate_device(self, device, current_epoch: int) -> bool:
        """
        Returns True if device should be quarantined due to key inactivity.
        """
        return self.trigger.check(device, current_epoch)
