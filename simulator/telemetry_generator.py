import random
import os
import yaml
from typing import Dict, Any, List

class TelemetryGenerator:
    _profiles = None

    @staticmethod
    def _load_profiles():
        if TelemetryGenerator._profiles is not None:
            return TelemetryGenerator._profiles

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "simulation.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                TelemetryGenerator._profiles = yaml.safe_load(f)
        else:
            # Fallback profiles if config not found
            TelemetryGenerator._profiles = {
                "Student": {
                    "active_hours": list(range(9, 24)),
                    "avg_messages_per_hour": 15,
                    "std_messages_per_hour": 6,
                    "networks": ["WiFi", "Cellular"],
                    "countries": ["United States"],
                    "timezones": ["America/New_York"],
                    "ip_prefix": "172.16.23.",
                    "max_battery_drain": 5,
                    "avg_sync_delay_sec": 0.8
                }
            }
        return TelemetryGenerator._profiles

    @staticmethod
    def get_profile(profile_name: str) -> Dict[str, Any]:
        profiles = TelemetryGenerator._load_profiles()
        return profiles.get(profile_name, list(profiles.values())[0])

    @staticmethod
    def generate_normal_telemetry(profile_name: str, hour: int, prev_ip: str = None) -> Dict[str, Any]:
        """Generates typical telemetry features according to the behavioral profile."""
        prof = TelemetryGenerator.get_profile(profile_name)
        
        # Sample network parameters
        networks = prof.get("networks", ["WiFi"])
        countries = prof.get("countries", ["United States"])
        timezones = prof.get("timezones", ["America/New_York"])
        ip_prefix = prof.get("ip_prefix", "192.168.1.")
        
        net_type = random.choice(networks)
        country = random.choice(countries)
        timezone = random.choice(timezones)
        
        # Decide if IP switches or stays consistent (mostly consistent)
        if prev_ip and random.random() < 0.90:
            ip = prev_ip
        else:
            ip = ip_prefix + str(random.randint(2, 254))

        # Sample session metadata
        session_duration = max(5.0, random.normalvariate(180, 60))
        sync_freq = max(1.0, random.normalvariate(prof.get("avg_sync_delay_sec", 1.0) * 5, 2.0))
        
        # Sample messaging
        msg_count = max(0, int(random.normalvariate(
            prof.get("avg_messages_per_hour", 5),
            prof.get("std_messages_per_hour", 2)
        )))

        return {
            "network_type": net_type,
            "network_ip": ip,
            "active_timezone": timezone,
            "location_country": country,
            "session_duration_sec": round(session_duration, 2),
            "sync_frequency": round(sync_freq, 2),
            "message_count_sent": msg_count,
            "login_frequency": 1.0,
            "idle_time_sec": round(random.uniform(0, 300), 2)
        }

    @staticmethod
    def generate_hijack_anomaly(normal_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Modifies normal metadata to simulate token theft / session hijack."""
        anomalous = normal_meta.copy()
        # Proxy/Hacker IP
        anomalous["network_ip"] = "45.89.230." + str(random.randint(2, 254))
        anomalous["network_type"] = "VPN"
        anomalous["location_country"] = "Netherlands"
        anomalous["active_timezone"] = "Europe/Amsterdam"
        anomalous["session_duration_sec"] = round(random.uniform(300, 1800), 2)
        anomalous["login_frequency"] = normal_meta.get("login_frequency", 1.0) + 3.0
        anomalous["sync_frequency"] = round(normal_meta.get("sync_frequency", 4.0) + 8.0, 2)
        anomalous["idle_time_sec"] = 0.0
        return anomalous

    @staticmethod
    def generate_ghost_anomaly(normal_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Modifies normal metadata to simulate a silent rogue paired device (ghost monitoring)."""
        anomalous = normal_meta.copy()
        # VPN/Tor IP and distant location
        anomalous["network_ip"] = "185.220.101." + str(random.randint(2, 254))
        anomalous["network_type"] = "VPN"
        anomalous["location_country"] = "Russia"
        anomalous["active_timezone"] = "Europe/Moscow"
        anomalous["session_duration_sec"] = round(random.uniform(1200, 3600), 2)
        anomalous["sync_frequency"] = round(random.uniform(20.0, 30.0), 2)
        anomalous["message_count_sent"] = 0  # Read-only spy device
        anomalous["idle_time_sec"] = 0.0
        return anomalous
