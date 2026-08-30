import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType
from .telemetry_generator import TelemetryGenerator

class MimicryAttacker(Device):
    """
    Simulates an adaptive adversary that attempts to imitate legitimate device telemetry
    to evade both inactivity triggers and behavioral anomaly detectors.
    Evaluates RQ3 (Mimicry Robustness).
    """
    def __init__(
        self,
        device_id: str,
        owner_id: str,
        name: str = "Mimicking Rogue",
        device_type: DeviceType | str = DeviceType.LINKED,
        os_version: str = "Linux x86_64",
        profile_name: str = "Student",
        battery_level: int = 90,
        ip_address: str = "185.220.101.9",
        network_type: str = "VPN",
        country: str = "Netherlands",
        timezone: str = "Europe/Amsterdam",
        initial_epoch: int = 0,
        mimicry_strength: str = "moderate_mimicry"  # naive_rogue, moderate_mimicry, strong_mimicry
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
        self.mimicry_strength = mimicry_strength

    def simulate_epoch(
        self,
        current_epoch: int,
        peer_telemetry: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """
        Simulates adaptive mimicry attack action and generates telemetry.
        """
        r = rng if rng is not None else random

        # 1. Key Rotation: Updates keys every 2-3 epochs to stay under delta_inact
        epoch_gap = current_epoch - self.epoch_last_key_update
        performed_key_update = False
        if epoch_gap >= 3 or (epoch_gap >= 2 and r.random() < 0.50):
            self.update_key(current_epoch)
            performed_key_update = True

        # 2. Adaptive Telemetry Generation based on Mimicry Strength
        prev_ip = self.ip_address
        prev_tz = self.timezone

        if self.mimicry_strength == "strong_mimicry":
            # Strongly mimics peer legitimate device features
            if peer_telemetry:
                target_sync = peer_telemetry.get("sync_frequency", 4.0) + r.uniform(-0.3, 0.3)
                target_dur = peer_telemetry.get("session_duration_sec", 120.0) + r.uniform(-10.0, 10.0)
                target_msgs = max(0, int(peer_telemetry.get("message_count_sent", 8) * r.uniform(0.9, 1.1)))
            else:
                target_sync = r.uniform(3.5, 5.0)
                target_dur = r.uniform(100.0, 140.0)
                target_msgs = r.randint(5, 12)

            # Strong mimicry attempts IP prefix imitation or stealth residential proxy
            self.network_type = "WiFi" if r.random() < 0.60 else "VPN"
            self.ip_address = f"172.16.23.{r.randint(150, 250)}" if self.network_type == "WiFi" else f"185.220.101.{r.randint(2, 254)}"
            self.country = "United States" if self.network_type == "WiFi" else "Netherlands"
            self.timezone = "America/New_York" if self.network_type == "WiFi" else "Europe/Amsterdam"

        elif self.mimicry_strength == "moderate_mimicry":
            # Moderately mimics timing and active hours, but maintains VPN routing
            if peer_telemetry:
                target_sync = peer_telemetry.get("sync_frequency", 4.0) + r.uniform(-1.0, 1.0)
                target_dur = peer_telemetry.get("session_duration_sec", 120.0) + r.uniform(-30.0, 30.0)
                target_msgs = max(0, int(peer_telemetry.get("message_count_sent", 8) * r.uniform(0.7, 1.3)))
            else:
                target_sync = r.uniform(4.0, 7.0)
                target_dur = r.uniform(80.0, 180.0)
                target_msgs = r.randint(4, 15)

            self.network_type = "VPN"
            self.ip_address = f"185.220.101.{r.randint(2, 254)}"
            self.country = "Netherlands"
            self.timezone = "Europe/Amsterdam"

        else: # naive_rogue
            target_sync = r.uniform(7.0, 14.0)
            target_dur = r.uniform(250.0, 500.0)
            target_msgs = r.randint(15, 35)
            self.network_type = "VPN"
            self.ip_address = f"185.220.101.{r.randint(2, 254)}"
            self.country = "Netherlands"
            self.timezone = "Europe/Amsterdam"

        sync_int = round(3600.0 / max(0.1, target_sync), 2)
        idle_time = max(0.0, 3600.0 - target_dur)

        ip_changed = 1.0 if (prev_ip != self.ip_address) else 0.0
        tz_changed = 1.0 if (prev_tz != self.timezone) else 0.0

        context = {
            "session_duration_sec": round(target_dur, 2),
            "sync_frequency": round(target_sync, 2),
            "sync_interval_sec": sync_int,
            "message_count_sent": target_msgs,
            "message_count_received": int(target_msgs * r.uniform(1.2, 2.5)),
            "network_type": self.network_type,
            "network_ip": self.ip_address,
            "location_country": self.country,
            "active_timezone": self.timezone,
            "battery_level": self.battery_level,
            "login_frequency": round(r.uniform(0.8, 1.8), 2),
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
            ground_truth_label=1  # Malicious / Mimicry Attacker
        )
        self.add_telemetry(context)
        return observation
