from typing import List, Dict, Any

class Device:
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: str, 
                 os_version: str, battery_level: int = 100, ip_address: str = "127.0.0.1", 
                 network_type: str = "WiFi", country: str = "US", timezone: str = "UTC"):
        self.device_id = device_id
        self.owner_id = owner_id
        self.name = name
        self.device_type = device_type  # "primary" or "linked"
        self.os_version = os_version
        self.battery_level = battery_level
        self.ip_address = ip_address
        self.network_type = network_type
        self.country = country
        self.timezone = timezone
        
        self.epoch_last_key_update = 0
        self.is_quarantined = False
        self.quarantined_epoch = None
        self.trust_score = 1.0
        self.current_trust_state = "Trusted"  # "Trusted", "Suspicious", "Quarantined", "Revoked"
        self.telemetry_history: List[Dict[str, Any]] = []
        self.qtk_shares: Dict[str, Any] = {}

    def update_key(self, current_epoch: int):
        self.epoch_last_key_update = current_epoch

    def add_telemetry(self, record: Dict[str, Any]):
        self.telemetry_history.append(record)
        if len(self.telemetry_history) > 100:  # Keep a decent window of historical telemetry
            self.telemetry_history.pop(0)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns standard metadata for the latest observation."""
        if not self.telemetry_history:
            return {
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "active_timezone": self.timezone,
                "location_country": self.country,
                "session_duration_sec": 0.0,
                "sync_frequency": 0.0,
                "message_count_sent": 0,
                "login_frequency": 1.0,
                "idle_time_sec": 0.0
            }
        return self.telemetry_history[-1]
