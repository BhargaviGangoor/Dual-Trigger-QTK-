import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType
from .telemetry_generator import TelemetryGenerator

class IrregularLegitimateDevice(Device):
    """
    Simulates a legitimate device that exhibits unusual but benign behavior patterns:
    - Business travel with country/timezone shifts
    - Network switching between Cellular, Hotel WiFi, and Work VPNs
    - Sporadic idle periods (up to 3-4 epochs, below delta_inact)
    - Low power / battery saver states
    Used to evaluate RQ4 (False Quarantine Rate & Availability Impact).
    Must NOT be labeled malicious (ground_truth_label = 0).
    """
    def __init__(
        self,
        device_id: str,
        owner_id: str,
        name: str = "Traveling Laptop",
        device_type: DeviceType | str = DeviceType.LINKED,
        os_version: str = "Windows 11",
        profile_name: str = "Traveler",
        battery_level: int = 80,
        ip_address: str = "192.168.1.25",
        network_type: str = "WiFi",
        country: str = "United States",
        timezone: str = "America/New_York",
        initial_epoch: int = 0,
        irregularity_type: str = "travel_network_switch"
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
        self.irregularity_type = irregularity_type
        self.is_traveling = False
        self.idle_epochs_remaining = 0

    def simulate_epoch(
        self,
        current_epoch: int,
        peer_telemetry: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """
        Simulates irregular legitimate device operations and telemetry generation.
        """
        r = rng if rng is not None else random

        # 1. Key Rotation: Still updates keys before delta_inact (typically every 3-4 epochs)
        epoch_gap = current_epoch - self.epoch_last_key_update
        performed_key_update = False
        if epoch_gap >= 4 or (epoch_gap >= 3 and r.random() < 0.80):
            self.update_key(current_epoch)
            performed_key_update = True

        # 2. Dynamic Irregular Context
        prev_ip = self.ip_address
        prev_tz = self.timezone

        if self.irregularity_type == "travel_network_switch":
            # Occasional country and network hopping (e.g. flight/conference)
            if current_epoch % 8 == 0:
                self.is_traveling = not self.is_traveling

            if self.is_traveling:
                self.country = "Germany"
                self.timezone = "Europe/Berlin"
                self.network_type = r.choice(["Cellular", "WiFi", "VPN"])
                self.ip_address = f"82.102.15.{r.randint(2, 254)}"
            else:
                self.country = "United States"
                self.timezone = "America/New_York"
                self.network_type = r.choice(["WiFi", "Cellular"])
                self.ip_address = f"172.16.23.{r.randint(2, 254)}"

            sync_freq = round(max(0.5, r.gauss(3.5, 1.5)), 2)
            session_dur = round(max(10.0, r.gauss(90.0, 35.0)), 2)
            msgs = max(0, int(r.gauss(6.0, 3.0)))

        elif self.irregularity_type == "sporadic_idle":
            # Idle for a couple of epochs (e.g. during flight / long meeting), then active burst
            if current_epoch % 6 in [1, 2]:
                sync_freq = 0.5
                session_dur = 5.0
                msgs = 0
            else:
                sync_freq = round(max(1.0, r.gauss(5.0, 2.0)), 2)
                session_dur = round(max(20.0, r.gauss(150.0, 40.0)), 2)
                msgs = max(1, int(r.gauss(10.0, 4.0)))

        else: # low_power_battery
            self.battery_level = max(5, self.battery_level - r.randint(2, 5))
            sync_freq = 1.0
            session_dur = 15.0
            msgs = max(0, int(r.gauss(2.0, 1.0)))

        sync_int = round(3600.0 / max(0.1, sync_freq), 2)
        idle_time = max(0.0, 3600.0 - session_dur)

        ip_changed = 1.0 if (prev_ip != self.ip_address) else 0.0
        tz_changed = 1.0 if (prev_tz != self.timezone) else 0.0

        context = {
            "session_duration_sec": session_dur,
            "sync_frequency": sync_freq,
            "sync_interval_sec": sync_int,
            "message_count_sent": msgs,
            "message_count_received": int(msgs * r.uniform(1.0, 2.0)),
            "network_type": self.network_type,
            "network_ip": self.ip_address,
            "location_country": self.country,
            "active_timezone": self.timezone,
            "battery_level": self.battery_level,
            "login_frequency": round(max(0.2, r.gauss(0.8, 0.3)), 2),
            "is_vpn": 1.0 if self.network_type == "VPN" else 0.0,
            "ip_changed": ip_changed,
            "tz_changed": tz_changed
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
            ground_truth_label=0  # Legitimate Device
        )
        self.add_telemetry(context)
        return observation
