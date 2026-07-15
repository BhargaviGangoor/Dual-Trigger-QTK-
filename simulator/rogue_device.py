from .device import Device
from .telemetry_generator import TelemetryGenerator

class RogueDevice(Device):
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: str, 
                 os_version: str, profile_name: str, battery_level: int = 100, 
                 ip_address: str = "185.220.101.5", network_type: str = "VPN", 
                 country: str = "Russia", timezone: str = "Europe/Moscow"):
        super().__init__(device_id, owner_id, name, device_type, os_version, 
                         battery_level, ip_address, network_type, country, timezone)
        self.profile_name = profile_name

    def simulate_epoch_action(self, current_epoch: int, is_active_hour: bool):
        # Rogue device is active. It updates keys frequently (every 2 epochs)
        # to ensure it always avoids the inactivity quarantine (delta_inact = 5).
        epoch_gap = current_epoch - self.epoch_last_key_update
        if epoch_gap >= 2:
            self.update_key(current_epoch)

        # Generates telemetry that has rogue/compromised/ghost characteristics
        normal_telemetry = TelemetryGenerator.generate_normal_telemetry(
            self.profile_name, 
            current_epoch % 24, 
            self.ip_address
        )
        anomalous_telemetry = TelemetryGenerator.generate_ghost_anomaly(normal_telemetry)
        
        # Synchronize properties
        self.ip_address = anomalous_telemetry["network_ip"]
        self.network_type = anomalous_telemetry["network_type"]
        self.country = anomalous_telemetry["location_country"]
        self.timezone = anomalous_telemetry["active_timezone"]
        
        self.add_telemetry(anomalous_telemetry)
