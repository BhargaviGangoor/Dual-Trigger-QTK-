import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType
from .telemetry_generator import TelemetryGenerator

class SilentDevice(Device):
    """
    Simulates a legitimate device that has gone silent / powered off / lost.
    Ceases MLS key update commits so that key_update_age >= delta_inact,
    testing that original QTK inactivity-based quarantine properly triggers.
    """
    def __init__(
        self,
        device_id: str,
        owner_id: str,
        name: str = "Silent Tablet",
        device_type: DeviceType | str = DeviceType.LINKED,
        os_version: str = "Android 13",
        battery_level: int = 20,
        ip_address: str = "192.168.1.50",
        network_type: str = "WiFi",
        country: str = "United States",
        timezone: str = "America/New_York",
        initial_epoch: int = 0,
        emit_dormant_heartbeat: bool = False
    ):
        super().__init__(
            device_id=device_id,
            owner_id=owner_id,
            name=name,
            device_type=device_type,
            os_version=os_version,
            battery_level=battery_level,
            ip_address=ip_address,
            network_type=network_type,
            country=country,
            timezone=timezone,
            initial_epoch=initial_epoch
        )
        self.emit_dormant_heartbeat = emit_dormant_heartbeat

    def simulate_epoch(
        self,
        current_epoch: int,
        peer_telemetry: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """
        Simulates one epoch of silence. Performs NO key updates.
        """
        r = rng if rng is not None else random

        # 1. No key updates performed
        performed_key_update = False

        # 2. Minimal context telemetry
        self.battery_level = max(0, self.battery_level - r.randint(0, 1))

        if self.emit_dormant_heartbeat and r.random() < 0.05:
            # Sporadic offline background check
            context = {
                "session_duration_sec": 0.0,
                "sync_frequency": 0.05,
                "sync_interval_sec": 72000.0,
                "message_count_sent": 0,
                "message_count_received": 0,
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "location_country": self.country,
                "active_timezone": self.timezone,
                "battery_level": self.battery_level,
                "login_frequency": 0.0,
                "is_vpn": 0.0,
                "ip_changed": 0.0,
                "tz_changed": 0.0
            }
        else:
            # Completely dormant
            context = {
                "session_duration_sec": 0.0,
                "sync_frequency": 0.0,
                "sync_interval_sec": 86400.0,
                "message_count_sent": 0,
                "message_count_received": 0,
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "location_country": self.country,
                "active_timezone": self.timezone,
                "battery_level": self.battery_level,
                "login_frequency": 0.0,
                "is_vpn": 0.0,
                "ip_changed": 0.0,
                "tz_changed": 0.0
            }

        # 3. Protocol Telemetry
        protocol = {
            "current_epoch": current_epoch,
            "epoch_last_key_update": self.epoch_last_key_update,
            "key_update_age": current_epoch - self.epoch_last_key_update,
            "performed_key_update": performed_key_update,
            "is_quarantined": self.is_quarantined
        }

        observation = TelemetryGenerator.build_observation(
            run_id=0,
            seed=0,
            device_id=self.device_id,
            epoch=current_epoch,
            device_type=self.device_type.value,
            protocol_telemetry=protocol,
            context_telemetry=context,
            ground_truth_label=0
        )
        self.add_telemetry(context)
        return observation
