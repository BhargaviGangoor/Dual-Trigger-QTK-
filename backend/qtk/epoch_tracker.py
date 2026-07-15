from simulator.device import Device

class EpochTracker:
    def __init__(self, initial_epoch: int = 0):
        self.current_epoch = initial_epoch

    def increment_epoch(self):
        self.current_epoch += 1

    def sync_device(self, device: Device):
        """Updates the device's key-update epoch to match the current global epoch."""
        device.update_key(self.current_epoch)
