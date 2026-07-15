from .device import Device
from .telemetry_generator import TelemetryGenerator

class LegitimateDevice(Device):
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: str, 
                 os_version: str, profile_name: str, battery_level: int = 100, 
                 ip_address: str = "127.0.0.1", network_type: str = "WiFi", 
                 country: str = "US", timezone: str = "UTC"):
        super().__init__(device_id, owner_id, name, device_type, os_version, 
                         battery_level, ip_address, network_type, country, timezone)
        self.profile_name = profile_name

    def simulate_epoch_action(self, current_epoch: int, is_active_hour: bool):
        """Simulates device activity for the epoch. Legitimate device updates keys regularly."""
        # Key update logic: update key if active and epoch gap is >= 3 (well before delta_inact = 5)
        epoch_gap = current_epoch - self.epoch_last_key_update
        if is_active_hour and epoch_gap >= 3:
            self.update_key(current_epoch)

        # Generate telemetry
        if is_active_hour:
            telemetry = TelemetryGenerator.generate_normal_telemetry(
                self.profile_name, 
                current_epoch % 24, 
                self.ip_address
            )
            # Synchronize device properties with generated values
            self.ip_address = telemetry["network_ip"]
            self.network_type = telemetry["network_type"]
            self.country = telemetry["location_country"]
            self.timezone = telemetry["active_timezone"]
            
            self.add_telemetry(telemetry)
