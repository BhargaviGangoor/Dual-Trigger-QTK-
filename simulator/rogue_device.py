import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType
from .telemetry_generator import TelemetryGenerator

class RogueDevice(Device):
    """
    Simulates an active attacker-controlled device (Active Rogue).
    Deliberately performs key updates every 2 epochs to stay below delta_inact,
    evading inactivity-only QTK while exhibiting detectable behavioral deviations.
    """
    def __init__(
        self,
        device_id: str,
        owner_id: str,
        name: str = "Compromised Terminal",
        device_type: DeviceType | str = DeviceType.LINKED,
        os_version: str = "Linux x86_64",
        battery_level: int = 95,
        ip_address: str = "185.220.101.5",
        network_type: str = "VPN",
        country: str = "Netherlands",
        timezone: str = "Europe/Amsterdam",
        initial_epoch: int = 0,
        attack_mode: str = "stealth_burst"
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
        self.attack_mode = attack_mode

    def simulate_epoch(
        self,
        current_epoch: int,
        peer_telemetry: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """
        Simulates active rogue device operations:
        - Rotates keys every 2 epochs (age is never >= delta_inact).
        - Generates anomalous telemetry footprints.
        """
        r = rng if rng is not None else random

        # 1. Key Rotation: ALWAYS rotates keys frequently to evade delta_inact
        epoch_gap = current_epoch - self.epoch_last_key_update
        performed_key_update = False
        if epoch_gap >= 2:
            self.update_key(current_epoch)
            performed_key_update = True

        # 2. Contextual Telemetry with Behavioral Deviations
        prev_ip = self.ip_address
        prev_tz = self.timezone

        # Overlapping telemetry features with temporal/relational desynchronization
        hour_of_day = current_epoch % 24
        is_peer_sleeping = (hour_of_day < 8 or hour_of_day >= 22)

        # Active Rogue exhibits anomalous activity especially during peer idle/sleep epochs,
        # plus relational network disconnect, while keeping marginal feature distributions overlapping.
        if is_peer_sleeping:
            # Nocturnal activity anomaly: rogue continues active exfiltration while peers sleep
            sync_freq = round(r.uniform(5.0, 9.5), 2)
            session_dur = round(r.uniform(110.0, 240.0), 2)
            msgs_sent = r.randint(6, 18)
        else:
            # Daytime activity: overlaps with legitimate active ranges
            if r.random() < 0.35:
                # Moderate burst
                sync_freq = round(r.uniform(7.5, 11.0), 2)
                session_dur = round(r.uniform(140.0, 280.0), 2)
                msgs_sent = r.randint(10, 24)
            else:
                sync_freq = round(r.uniform(3.5, 7.0), 2)
                session_dur = round(r.uniform(70.0, 160.0), 2)
                msgs_sent = r.randint(2, 10)

        # Realistic network routing: 30% chance VPN, 40% cloud/foreign ISP, 30% cellular roaming
        net_roll = r.random()
        if net_roll < 0.30:
            self.network_type = "VPN"
            is_vpn = 1.0
            self.ip_address = f"185.220.101.{r.randint(2, 254)}"
            self.country = "Netherlands"
            self.timezone = "Europe/Amsterdam"
        elif net_roll < 0.70:
            self.network_type = "Cellular"
            is_vpn = 0.0
            self.ip_address = f"198.51.100.{r.randint(2, 254)}"
            self.country = "United States"
            self.timezone = "America/Chicago"  # Timezone mismatch with peer
        else:
            self.network_type = "WiFi"
            is_vpn = 0.0
            self.ip_address = f"203.0.113.{r.randint(2, 254)}"
            self.country = "United States"
            self.timezone = "America/New_York"

        # Occasional IP maintenance (doesn't change every single epoch)
        if r.random() < 0.40 and prev_ip:
            self.ip_address = prev_ip
            self.timezone = prev_tz

        sync_int = round(3600.0 / max(0.1, sync_freq), 2)
        idle_time = max(0.0, 3600.0 - session_dur)

        ip_changed = 1.0 if (prev_ip != self.ip_address) else 0.0
        tz_changed = 1.0 if (prev_tz != self.timezone) else 0.0

        context = {
            "session_duration_sec": session_dur,
            "sync_frequency": sync_freq,
            "sync_interval_sec": sync_int,
            "message_count_sent": msgs_sent,
            "message_count_received": int(msgs_sent * r.uniform(1.5, 3.0)),
            "network_type": self.network_type,
            "network_ip": self.ip_address,
            "location_country": self.country,
            "active_timezone": self.timezone,
            "battery_level": self.battery_level,
            "login_frequency": round(r.uniform(1.0, 3.5), 2),
            "is_vpn": is_vpn,
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
            ground_truth_label=1  # Malicious / Rogue
        )
        self.add_telemetry(context)
        return observation
