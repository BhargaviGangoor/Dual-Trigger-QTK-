from simulator.device import Device

class InactivityTrigger:
    def __init__(self, delta_inact: int = 5):
        self.delta_inact = delta_inact

    def evaluate(self, current_epoch: int, device: Device) -> bool:
        """
        Evaluates the plain QTK inactivity rule:
        Quarantine if (current_epoch - epoch_last_key_update) >= delta_inact
        """
        epoch_gap = current_epoch - device.epoch_last_key_update
        return epoch_gap >= self.delta_inact
