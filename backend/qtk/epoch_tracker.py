from simulator.device import Device

class EpochTracker:
    """
    Tracks the group epochs (e_i) and manages updating a device's last key update epoch (e_pk(d))
    as described in the research paper.
    """
    def __init__(self, initial_epoch: int = 0):
        self.current_epoch = initial_epoch

    def increment_epoch(self):
        """Increments the current epoch e_i."""
        self.current_epoch += 1

    def sync_device(self, device: Device):
        """
        Updates the device's key-update epoch (e_pk(d)) to match the current global epoch (e_i).
        This resets its inactivity timer gap.
        """
        device.update_key(self.current_epoch)
