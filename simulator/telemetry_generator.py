import os
import yaml
import random
from typing import Dict, Any, List, Optional
import numpy as np

class TelemetryGenerator:
    """
    Generates structured per-device, per-epoch telemetry observations.
    Explicitly separates MLS/protocol-derived telemetry from contextual/device telemetry.
    """
    _profiles: Optional[Dict[str, Any]] = None

    @classmethod
    def load_profiles(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        if cls._profiles is not None and config_path is None:
            return cls._profiles

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "simulation.yaml"
            )

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            cls._profiles = data.get("profiles", {})
        else:
            # Safe fallback profiles
            cls._profiles = {
                "Student": {
                    "active_hours": list(range(9, 24)),
                    "avg_messages_per_hour": 12.0,
                    "std_messages_per_hour": 4.0,
                    "avg_session_duration_sec": 120.0,
                    "std_session_duration_sec": 35.0,
                    "avg_sync_interval_sec": 900.0,
                    "networks": ["WiFi", "Cellular"],
                    "countries": ["United States"],
                    "timezones": ["America/New_York"],
                    "ip_prefix": "172.16.23.",
                    "battery_drain_rate": 2
                }
            }
        return cls._profiles

    @classmethod
    def get_profile(cls, profile_name: str) -> Dict[str, Any]:
        profiles = cls.load_profiles()
        return profiles.get(profile_name, list(profiles.values())[0])

    @staticmethod
    def generate_normal_context(
        profile_name: str,
        epoch: int,
        prev_telemetry: Optional[Dict[str, Any]] = None,
        peer_telemetry: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """
        Generates legitimate device contextual telemetry aligned with behavioral profiles.
        """
        r = rng if rng is not None else random
        prof = TelemetryGenerator.get_profile(profile_name)

        active_hours = prof.get("active_hours", list(range(8, 22)))
        hour_of_day = epoch % 24
        is_active_hour = hour_of_day in active_hours

        # 1. Network & Location Consistency
        is_vpn = 0.0
        if r.random() < 0.08:  # 8% chance legitimate device uses a secure corporate or privacy VPN
            is_vpn = 1.0
            net_type = "VPN"
            country = prof.get("countries", ["United States"])[0]
            timezone = prof.get("timezones", ["America/New_York"])[0]
            ip = f"198.51.100.{r.randint(2, 254)}"
        elif peer_telemetry and r.random() < 0.75:
            # Correlate with group peer (same subnet / location)
            country = peer_telemetry.get("location_country", "United States")
            timezone = peer_telemetry.get("active_timezone", "America/New_York")
            net_type = peer_telemetry.get("network_type", "WiFi")
            ip = peer_telemetry.get("network_ip", "192.168.1.10")
            parts = ip.split(".")
            if len(parts) == 4:
                ip = f"{parts[0]}.{parts[1]}.{parts[2]}.{r.randint(2, 254)}"
        elif prev_telemetry and r.random() < 0.85:
            # Maintain previous context
            country = prev_telemetry.get("location_country", "United States")
            timezone = prev_telemetry.get("active_timezone", "America/New_York")
            net_type = prev_telemetry.get("network_type", "WiFi")
            ip = prev_telemetry.get("network_ip", prof.get("ip_prefix", "192.168.1.") + "10")
        else:
            # Sample from profile distributions
            countries = prof.get("countries", ["United States"])
            timezones = prof.get("timezones", ["America/New_York"])
            networks = prof.get("networks", ["WiFi", "Cellular"])
            ip_prefix = prof.get("ip_prefix", "192.168.1.")

            country = r.choice(countries)
            timezone = r.choice(timezones)
            net_type = r.choice(networks)
            ip = ip_prefix + str(r.randint(2, 254))

        # 2. Activity / Messages / Timing
        if is_active_hour:
            base_msgs = prof.get("avg_messages_per_hour", 10.0)
            std_msgs = prof.get("std_messages_per_hour", 3.0)
            # Occasional daytime burst
            if r.random() < 0.15:
                base_msgs *= 1.8
            msgs = max(0, int(r.gauss(base_msgs, std_msgs)))
            
            avg_dur = prof.get("avg_session_duration_sec", 120.0)
            std_dur = prof.get("std_session_duration_sec", 30.0)
            session_dur = max(5.0, r.gauss(avg_dur, std_dur))
            
            avg_sync = prof.get("avg_sync_interval_sec", 900.0)
            sync_int = max(30.0, r.gauss(avg_sync, avg_sync * 0.2))
            sync_freq = round(3600.0 / sync_int, 2)
            idle_time = max(0.0, 3600.0 - (session_dur * max(1, msgs // 3)))
            login_freq = round(max(0.2, r.gauss(1.0, 0.3)), 2)
        else:
            # Inactive night/sleep hours
            msgs = max(0, int(r.gauss(1.0, 1.0)))
            session_dur = max(0.0, r.gauss(15.0, 10.0))
            sync_int = max(1800.0, r.gauss(3600.0, 600.0))
            sync_freq = round(3600.0 / sync_int, 2)
            idle_time = max(0.0, 3600.0 - session_dur)
            login_freq = 0.1

        prev_ip = prev_telemetry.get("network_ip") if prev_telemetry else None
        prev_tz = prev_telemetry.get("active_timezone") if prev_telemetry else None

        ip_changed = 1.0 if (prev_ip is not None and prev_ip != ip) else 0.0
        tz_changed = 1.0 if (prev_tz is not None and prev_tz != timezone) else 0.0

        return {
            "session_duration_sec": round(session_dur, 2),
            "sync_frequency": sync_freq,
            "sync_interval_sec": round(sync_int, 2),
            "message_count_sent": msgs,
            "message_count_received": int(msgs * r.uniform(1.0, 2.5)),
            "network_type": net_type,
            "network_ip": ip,
            "location_country": country,
            "active_timezone": timezone,
            "battery_level": 100,
            "login_frequency": login_freq,
            "is_vpn": is_vpn,
            "ip_changed": ip_changed,
            "tz_changed": tz_changed
        }

    @staticmethod
    def extract_feature_vector(context_telemetry: Dict[str, Any]) -> np.ndarray:
        """
        Normalizes contextual telemetry into a 5D feature vector for HMM, GNN, and baselines:
        [session_duration_norm, sync_frequency_norm, message_count_norm, ip_changed, tz_changed]
        """
        dur = float(context_telemetry.get("session_duration_sec", 0.0)) / 600.0  # normalized to ~10 min scale
        sync = float(context_telemetry.get("sync_frequency", 0.0)) / 20.0       # normalized to ~20/hr scale
        msgs = float(context_telemetry.get("message_count_sent", 0.0)) / 50.0   # normalized to ~50 msgs scale
        ip_ch = float(context_telemetry.get("ip_changed", 0.0))
        tz_ch = float(context_telemetry.get("tz_changed", 0.0))
        
        # If VPN is used, it also flags high network deviation
        if context_telemetry.get("is_vpn", 0.0) > 0:
            ip_ch = max(ip_ch, 0.8)

        return np.array([dur, sync, msgs, ip_ch, tz_ch], dtype=np.float32)

    @staticmethod
    def build_observation(
        run_id: int,
        seed: int,
        device_id: str,
        epoch: int,
        device_type: str,
        protocol_telemetry: Dict[str, Any],
        context_telemetry: Dict[str, Any],
        ground_truth_label: int
    ) -> Dict[str, Any]:
        """
        Builds a complete, standard observation dictionary with separated protocol/context data.
        """
        features = TelemetryGenerator.extract_feature_vector(context_telemetry).tolist()
        return {
            "run_id": run_id,
            "seed": seed,
            "device_id": device_id,
            "epoch": epoch,
            "device_type": device_type,
            "protocol_telemetry": protocol_telemetry,
            "context_telemetry": context_telemetry,
            "features": features,
            "ground_truth_label": ground_truth_label  # 0: Legitimate/Irregular/Silent, 1: Rogue/Mimicry
        }
