class InactivityTrigger:
    """
    Implements the original QTK inactivity-based quarantine rule:
    Quarantine(d) <==> (e_i - e_pk(d)) >= delta_inact
    """
    def __init__(self, delta_inact: int = 5):
        self.delta_inact = delta_inact

    def check(self, device, current_epoch: int) -> bool:
        """
        Evaluates whether the key update age of device meets or exceeds delta_inact.
        """
        last_key_epoch = getattr(device, "epoch_last_key_update", 0)
        age = current_epoch - last_key_epoch
        return age >= self.delta_inact

    def evaluate(self, current_epoch: int, device) -> bool:
        """Compatibility alias for check()."""
        return self.check(device, current_epoch)
