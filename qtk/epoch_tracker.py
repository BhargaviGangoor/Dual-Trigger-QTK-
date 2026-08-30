from typing import Dict, List, Optional

class EpochTracker:
    """
    Manages MLS group epoch progression (e_i) and tracks key update history (e_pk(d))
    for all participating client devices.
    """
    def __init__(self, initial_epoch: int = 0):
        self.current_epoch: int = initial_epoch
        self.epoch_history: List[int] = [initial_epoch]

    def increment_epoch(self) -> int:
        """Advances the MLS global group epoch e_i -> e_{i+1}."""
        self.current_epoch += 1
        self.epoch_history.append(self.current_epoch)
        return self.current_epoch

    def sync_device_key(self, device) -> int:
        """
        Records a key update commit from the device at the current epoch.
        Resets its inactivity age: e_i - e_pk(d) = 0.
        """
        device.update_key(self.current_epoch)
        return self.current_epoch

    def get_key_age(self, device) -> int:
        """Computes key update age: e_i - e_pk(d)."""
        return self.current_epoch - getattr(device, "epoch_last_key_update", 0)

    def reset(self, initial_epoch: int = 0):
        """Resets the epoch tracker to initial state."""
        self.current_epoch = initial_epoch
        self.epoch_history = [initial_epoch]
