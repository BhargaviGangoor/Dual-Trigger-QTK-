import uuid
import datetime
import random
from typing import Dict, Any, Tuple

class AttackSimulator:
    @staticmethod
    def inject_ghost_pairing(user_id: int, user_timezone: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulates an attacker pairing a rogue secondary device (Ghost Pairing).
        The attacker does this silently, so the device is linked but the user is unaware.
        Returns:
            device_data: The rogue device properties.
            event_data: The pairing event.
        """
        device_id = f"ghost-{uuid.uuid4().hex[:8]}"
        pub_key = f"key_{uuid.uuid4().hex[:12]}"
        fingerprint = f"fp_{uuid.uuid4().hex[:16]}"
        
        # Attacker's device profile (usually different network, location, and timezone)
        attacker_ip = "185.220.101." + str(random.randint(2, 254))  # Tor/VPN Exit node
        attacker_countries = ["Russia", "Ukraine", "Romania", "China", "Unknown"]
        attacker_country = random.choice(attacker_countries)
        attacker_timezone = "Europe/Moscow" if attacker_country == "Russia" else "Asia/Shanghai"
        
        device_data = {
            "id": device_id,
            "user_id": user_id,
            "name": "Chrome (Linux)",  # Common spoofed device
            "device_type": "linked",
            "public_key": pub_key,
            "fingerprint": fingerprint,
            "ip_address": attacker_ip,
            "network_type": "VPN",
            "country": attacker_country,
            "timezone": attacker_timezone,
            "os_version": "Linux x86_64",
            "battery_level": 100,
            "trust_score": 0.5,  # Initializes with low confidence
            "current_trust_state": "Suspicious"
        }
        
        event_data = {
            "event_type": "attack_trigger",
            "device_id": device_id,
            "description": f"Attack Injected: Ghost Pairing. Rogue device linked as {device_data['name']} from {attacker_country}.",
            "reason": "Unauthorized QR pairing / session hijacking.",
            "trust_score_before": 1.0,
            "trust_score_after": 0.5,
            "fsm_state_before": "Trusted",
            "fsm_state_after": "Suspicious"
        }
        
        return device_data, event_data

    @staticmethod
    def apply_hijack_anomaly(normal_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modifies a normal metadata record to simulate active session hijacking / token theft.
        An attacker is using the session credentials from a different IP and network context.
        """
        anomalous_meta = normal_meta.copy()
        
        # Inject hijack signs
        anomalous_meta["network_ip"] = "45.89.230." + str(random.randint(2, 254))  # Hacker proxy IP
        anomalous_meta["network_type"] = "VPN"
        anomalous_meta["location_country"] = "Netherlands"
        anomalous_meta["active_timezone"] = "Europe/Amsterdam"
        anomalous_meta["session_duration_sec"] = random.uniform(300, 1800)  # Prolonged active sessions
        anomalous_meta["login_frequency"] = normal_meta["login_frequency"] + 3.0  # Increased concurrent logins
        anomalous_meta["sync_frequency"] = normal_meta["sync_frequency"] + 8.0   # Fast synchronization rate
        anomalous_meta["idle_time_sec"] = 0.0  # Active usage
        
        return anomalous_meta

    @staticmethod
    def apply_delayed_sync_anomaly(normal_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates an attacker trying to hide activity by intentionally postponing device synchronization.
        """
        anomalous_meta = normal_meta.copy()
        anomalous_meta["sync_frequency"] = random.uniform(0.01, 0.1)  # Very low sync
        anomalous_meta["session_duration_sec"] = random.uniform(5.0, 30.0) # Keeps failing / reconnecting
        anomalous_meta["delivery_count"] = int(normal_meta["message_count_sent"] * 0.1)  # Failed deliveries
        return anomalous_meta

    @staticmethod
    def apply_read_only_spy_anomaly(normal_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates an attacker silently monitoring messages.
        High message read counts, but zero messages sent.
        """
        anomalous_meta = normal_meta.copy()
        anomalous_meta["message_count_sent"] = 0
        anomalous_meta["message_count_received"] = normal_meta["message_count_received"]
        anomalous_meta["read_count"] = normal_meta["message_count_received"] + random.randint(5, 20)
        anomalous_meta["session_duration_sec"] = random.uniform(600, 3600)  # Always open
        return anomalous_meta

    @staticmethod
    def simulate_location_spoof(normal_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates a location spoofing attack (simulated GPS/IP coordinates do not match expected router).
        """
        anomalous_meta = normal_meta.copy()
        anomalous_meta["network_ip"] = "190.2.143." + str(random.randint(2, 254))
        anomalous_meta["location_country"] = "Brazil"
        anomalous_meta["active_timezone"] = "America/Sao_Paulo"
        return anomalous_meta
