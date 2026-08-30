import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType
from .telemetry_generator import TelemetryGenerator

class LegitimateDevice(Device):
    """
    Simulates a legitimate MLS client device.
    Regularly updates MLS keys (every 2-3 epochs) to stay well within delta_inact,
    and exhibits standard diurnal activity matching behavioral profiles.
    """
    def __init__(
        self,
        device_id: str,
        owner_id: str,
        name: str = "Legitimate Device",
        device_type: DeviceType | str = DeviceType.PRIMARY,
        os_version: str = "Android 14",
        profile_name: str = "Student",
        battery_level: int = 100,
        ip_address: str = "192.168.1.10",
        network_type: str = "WiFi",
        country: str = "United States",
        timezone: str = "America/New_York",
        initial_epoch: int = 0
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
        self.profile_name = profile_name

    def simulate_epoch(
        self,
        current_epoch: int,
        peer_telemetry: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """
        Simulates legitimate device protocol actions and generates epoch telemetry.
        """
        r = rng if rng is not None else random

        # 1. Key Rotation: Legitimate devices rotate keys before delta_inact (every 2-3 epochs)
        epoch_gap = current_epoch - self.epoch_last_key_update
        performed_key_update = False
        if epoch_gap >= 3 or (epoch_gap >= 2 and r.random() < 0.60):
            self.update_key(current_epoch)
            performed_key_update = True

        # 2. Generate Context Telemetry
        prev_meta = self.get_latest_telemetry()
        context = TelemetryGenerator.generate_normal_context(
            profile_name=self.profile_name,
            epoch=current_epoch,
            prev_telemetry=prev_meta,
            peer_telemetry=peer_telemetry,
            rng=r
        )

        # Update device state from generated context
        self.ip_address = context["network_ip"]
        self.network_type = context["network_type"]
        self.country = context["location_country"]
        self.timezone = context["active_timezone"]

        # Battery progression
        self.battery_level = max(5, self.battery_level - r.randint(0, 2))
        context["battery_level"] = self.battery_level

        # 3. Protocol Telemetry
        protocol = {
            "current_epoch": current_epoch,
            "epoch_last_key_update": self.epoch_last_key_update,
            "key_update_age": current_epoch - self.epoch_last_key_update,
            "performed_key_update": performed_key_update,
            "is_quarantined": self.is_quarantined
        }

        # Store observation in history
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
