import random
from .device import Device
from .telemetry_generator import TelemetryGenerator

class MimicryAttacker(Device):
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: str, 
                 os_version: str, profile_name: str, battery_level: int = 100, 
                 ip_address: str = "185.220.101.9", network_type: str = "VPN", 
                 country: str = "Russia", timezone: str = "Europe/Moscow"):
        super().__init__(device_id, owner_id, name, device_type, os_version, 
                         battery_level, ip_address, network_type, country, timezone)
        self.profile_name = profile_name

    def simulate_epoch_action(self, current_epoch: int, is_active_hour: bool, normal_device_metadata: dict = None):
        # Adaptive mimicry attacker. It does key updates similarly to normal devices (every 3 epochs)
        epoch_gap = current_epoch - self.epoch_last_key_update
        if epoch_gap >= 3:
            self.update_key(current_epoch)

        # It mimics the legitimate device's telemetry features (like sync frequency, session duration)
        # to try to blend in, but maintains a VPN connection and foreign location.
        normal_telemetry = TelemetryGenerator.generate_normal_telemetry(
            self.profile_name, 
            current_epoch % 24, 
            self.ip_address
        )
        
        # Mimicry: copy key telemetry features from the normal device's history to bypass edge filters
        if normal_device_metadata:
            normal_telemetry["sync_frequency"] = max(
                0.1, 
                round(normal_device_metadata.get("sync_frequency", 4.0) + random.uniform(-0.5, 0.5), 2)
            )
            normal_telemetry["session_duration_sec"] = max(
                5.0, 
                round(normal_device_metadata.get("session_duration_sec", 120.0) + random.uniform(-10.0, 10.0), 2)
            )
            normal_telemetry["message_count_sent"] = max(
                0, 
                int(normal_device_metadata.get("message_count_sent", 5) * random.uniform(0.8, 1.2))
            )

        # But it still has foreign/VPN network footprint representing physical proxy routing
        normal_telemetry["network_ip"] = "185.220.101." + str(random.randint(2, 254))
        normal_telemetry["network_type"] = "VPN"
        normal_telemetry["location_country"] = "Russia"
        normal_telemetry["active_timezone"] = "Europe/Moscow"
        normal_telemetry["idle_time_sec"] = 0.0

        self.ip_address = normal_telemetry["network_ip"]
        self.network_type = normal_telemetry["network_type"]
        self.country = normal_telemetry["location_country"]
        self.timezone = normal_telemetry["active_timezone"]
        
        self.add_telemetry(normal_telemetry)
