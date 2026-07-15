from .device import Device

class SilentDevice(Device):
    def simulate_epoch_action(self, current_epoch: int, is_active_hour: bool):
        # A silent device represents an offline/lost/abandoned device.
        # It performs no key updates and sends no telemetry.
        pass
