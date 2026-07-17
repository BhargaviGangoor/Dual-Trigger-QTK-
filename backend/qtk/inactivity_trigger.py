from simulator.device import Device

class InactivityTrigger:
    """
    Implements the baseline inactivity-based trigger condition from Equation 1 of the research paper:
    e_i - e_pk(d) >= delta_inact
    
    Quarantines devices that go inactive/silent for too long (Ghost devices).
    """
    def __init__(self, delta_inact: int = 5):
        self.delta_inact = delta_inact

    def evaluate(self, current_epoch: int, device: Device) -> bool:
        """
        Evaluates the plain QTK inactivity rule:
        Quarantine if (current_epoch - epoch_last_key_update) >= delta_inact (Equation 1)
        """
        epoch_gap = current_epoch - device.epoch_last_key_update
        return epoch_gap >= self.delta_inact
