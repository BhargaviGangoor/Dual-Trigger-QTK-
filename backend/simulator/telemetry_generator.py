from enum import Enum
import random
import os
import datetime
import yaml
from typing import Dict, Any, List, Optional

class NetworkType(str, Enum):
    """Network connection types used in client telemetry."""
    WIFI = "WiFi"
    CELLULAR = "Cellular"
    ETHERNET = "Ethernet"
    VPN = "VPN"

class TelemetryGenerator:
    """
    Generates realistic, temporally consistent telemetry records (X_t) based on behavioral profiles.
    Provides fine-grained perturbation primitives for simulating attacks and anomalies.
    """
    _profiles: Optional[Dict[str, Any]] = None

    @staticmethod
    def _load_profiles() -> Dict[str, Any]:
        """Loads behavior profiles from configs/simulation.yaml or returns defaults."""
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
                    "avg_messages_per_hour": 5.0,
                    "std_messages_per_hour": 2.0,
                    "networks": ["WiFi", "Cellular"],
                    "countries": ["United States"],
                    "timezones": ["America/New_York"],
                    "ip_prefix": "172.16.23.",
                    "max_battery_drain": 5,
                    "avg_sync_delay_sec": 900.0
                }
            }
        return TelemetryGenerator._profiles

    @staticmethod
    def get_profile(profile_name: str) -> Dict[str, Any]:
        """Retrieves config dictionary for the given profile name."""
        profiles = TelemetryGenerator._load_profiles()
        return profiles.get(profile_name, list(profiles.values())[0])

    @staticmethod
    def generate_normal_telemetry(profile_name: str, hour: int, prev_ip: Optional[str] = None,
                                  prev_network: Optional[str] = None, weekday: bool = True,
                                  correlation_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates typical telemetry features matching the paper's X_t feature vector.
        Maintains temporal consistency and graph correlation when peer data is supplied.
        """
        prof = TelemetryGenerator.get_profile(profile_name)
        
        # 1. Graph correlation / subnets sharing
        if correlation_meta and random.random() < 0.80:
            country = correlation_meta.get("country") or correlation_meta.get("location_country") or "United States"
            timezone = correlation_meta.get("timezone") or correlation_meta.get("active_timezone") or "America/New_York"
            net_type = correlation_meta.get("network_type") or NetworkType.WIFI
            
            # Shared subnet prefix
            if "network_ip" in correlation_meta:
                ip_parts = correlation_meta["network_ip"].split(".")
                if len(ip_parts) == 4:
                    ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.{random.randint(2, 254)}"
                else:
                    ip = correlation_meta["network_ip"]
            else:
                ip = prof.get("ip_prefix", "192.168.1.") + str(random.randint(2, 254))
        else:
            # Standard generation
            networks = prof.get("networks", ["WiFi", "Cellular"])
            countries = prof.get("countries", ["United States"])
            timezones = prof.get("timezones", ["America/New_York"])
            ip_prefix = prof.get("ip_prefix", "192.168.1.")
            
            # Temporal realism: network sticks with high probability
            if prev_network and random.random() < 0.95:
                net_type = prev_network
            else:
                net_type = random.choice(networks)
                
            country = random.choice(countries)
            timezone = random.choice(timezones)
            
            # IP changes stick with high probability
            if prev_ip and random.random() < 0.92:
                ip = prev_ip
            else:
                ip = ip_prefix + str(random.randint(2, 254))

        # 2. Activity / messaging parameters adjusted for weekdays vs weekends
        base_messages = prof.get("avg_messages_per_hour", 5)
        if not weekday:
            base_messages = max(1, int(base_messages * 0.6))  # 40% reduction on weekends
            
        msg_count = max(0, int(random.normalvariate(
            base_messages,
            prof.get("std_messages_per_hour", 2)
        )))

        # 3. Synchronisation parameters
        avg_delay = prof.get("avg_sync_delay_sec", 900.0)
        target_freq = 3600.0 / avg_delay if avg_delay > 0 else 4.0
        sync_frequency = round(max(0.5, random.normalvariate(target_freq, 1.5)), 2)
        sync_interval = round(3600.0 / sync_frequency, 2)
        
        # 4. Session details
        session_duration = max(5.0, random.normalvariate(120, 40))
        idle_time = round(random.uniform(0, 300), 2)
        
        telemetry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "network_type": net_type,
            "network_ip": ip,
            "timezone": timezone,
            "active_timezone": timezone,
            "country": country,
            "location_country": country,
            "session_duration": round(session_duration, 2),
            "session_duration_sec": round(session_duration, 2),
            "idle_time": idle_time,
            "idle_time_sec": idle_time,
            "sync_interval": round(sync_interval, 2),
            "sync_frequency": sync_frequency,
            "messages_sent": msg_count,
            "message_count_sent": msg_count,
            "messages_received": random.randint(msg_count, msg_count * 2 + 3),
            "message_count_received": random.randint(msg_count, msg_count * 2 + 3),
            "login_frequency": 1.0,
            "battery_level": 100
        }
        
        # Map additional indicators if useful
        telemetry["network_change"] = prev_network is not None and prev_network != net_type
        telemetry["ip_change"] = prev_ip is not None and prev_ip != ip
        telemetry["device_online"] = True
        
        return telemetry

    # --- Reusable Fine-Grained Perturbation Primitives (Matches Paper Modularity) ---

    @staticmethod
    def apply_vpn(telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Modifies telemetry to simulate VPN tunnel routing."""
        perturbed = telemetry.copy()
        perturbed["network_type"] = NetworkType.VPN
        perturbed["network_ip"] = "185.220.101." + str(random.randint(2, 254))
        perturbed["country"] = "Netherlands"
        perturbed["location_country"] = "Netherlands"
        perturbed["timezone"] = "Europe/Amsterdam"
        perturbed["active_timezone"] = "Europe/Amsterdam"
        perturbed["network_change"] = True
        perturbed["ip_change"] = True
        return perturbed

    @staticmethod
    def apply_network_hopping(telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Alternates network connections (WiFi vs. Cellular) between updates."""
        perturbed = telemetry.copy()
        current_net = telemetry.get("network_type", NetworkType.WIFI)
        perturbed["network_type"] = NetworkType.CELLULAR if current_net == NetworkType.WIFI else NetworkType.WIFI
        perturbed["network_change"] = True
        return perturbed

    @staticmethod
    def apply_timezone_shift(telemetry: Dict[str, Any], country: str = "Russia", timezone: str = "Europe/Moscow") -> Dict[str, Any]:
        """Simulates timezone shifts due to proxy routing or physical shifts."""
        perturbed = telemetry.copy()
        perturbed["country"] = country
        perturbed["location_country"] = country
        perturbed["timezone"] = timezone
        perturbed["active_timezone"] = timezone
        return perturbed

    @staticmethod
    def apply_low_activity(telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Restricts telemetry values to minimal dormant indicators."""
        perturbed = telemetry.copy()
        perturbed["session_duration"] = 0.0
        perturbed["session_duration_sec"] = 0.0
        perturbed["idle_time"] = 3600.0
        perturbed["idle_time_sec"] = 3600.0
        perturbed["messages_sent"] = 0
        perturbed["message_count_sent"] = 0
        perturbed["messages_received"] = 0
        perturbed["message_count_received"] = 0
        perturbed["login_frequency"] = 0.0
        return perturbed

    @staticmethod
    def apply_high_sync_delay(telemetry: Dict[str, Any], multiplier: float = 100.0) -> Dict[str, Any]:
        """Delays synchronisation intervals, reducing frequency."""
        perturbed = telemetry.copy()
        sync_int = telemetry.get("sync_interval", 1.0) * multiplier
        perturbed["sync_interval"] = round(sync_int, 2)
        perturbed["sync_frequency"] = round(3600.0 / sync_int, 2)
        return perturbed

    @staticmethod
    def apply_message_suppression(telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Sets outgoing message counts to zero (read-only/spy eavesdropping)."""
        perturbed = telemetry.copy()
        perturbed["messages_sent"] = 0
        perturbed["message_count_sent"] = 0
        return perturbed

    @staticmethod
    def apply_ip_instability(telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates rapid IP address modifications."""
        perturbed = telemetry.copy()
        ip_parts = telemetry.get("network_ip", "192.168.1.1").split(".")
        if len(ip_parts) == 4:
            perturbed["network_ip"] = f"{ip_parts[0]}.{ip_parts[1]}.{random.randint(1, 254)}.{random.randint(2, 254)}"
        else:
            perturbed["network_ip"] = "192.168." + f"{random.randint(1, 254)}.{random.randint(2, 254)}"
        perturbed["ip_change"] = True
        return perturbed

    @staticmethod
    def apply_session_extension(telemetry: Dict[str, Any], duration_sec: float = 3600.0) -> Dict[str, Any]:
        """Modifies session lengths to represent persistent background operations."""
        perturbed = telemetry.copy()
        perturbed["session_duration"] = round(duration_sec, 2)
        perturbed["session_duration_sec"] = round(duration_sec, 2)
        return perturbed
