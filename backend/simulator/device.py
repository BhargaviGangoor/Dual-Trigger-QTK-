from typing import List, Dict, Any

from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import datetime

class DeviceType(str, Enum):
    """Types of devices supported by the MLS protocol."""
    PRIMARY = "primary"
    LINKED = "linked"

class HMMState(str, Enum):
    """Behavioral states estimated by the Hidden Markov Model."""
    NORMAL = "Normal"
    IDLE = "Idle"
    SUSPICIOUS = "Suspicious"
    HIGH_RISK = "High-Risk"

class TrustState(str, Enum):
    """Protocol-level trust states for the device lifecycle."""
    TRUSTED = "Trusted"
    SUSPICIOUS = "Suspicious"
    QUARANTINED = "Quarantined"
    REVOKED = "Revoked"

class Device:
    """
    Represents one MLS client/device owned by a user.
    Maintains telemetry history, trust scoring states, and secret sharing keys.
    """
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: DeviceType | str, 
                 os_version: str, battery_level: int = 100, ip_address: str = "127.0.0.1", 
                 network_type: str = "WiFi", country: str = "US", timezone: str = "UTC"):
        self.device_id: str = device_id
        self.owner_id: str = owner_id
        self.name: str = name
        self.device_type: DeviceType = DeviceType(device_type) if isinstance(device_type, str) else device_type
        self.os_version: str = os_version
        self.battery_level: int = battery_level
        self.ip_address: str = ip_address
        self.network_type: str = network_type
        self.country: str = country
        self.timezone: str = timezone
        
        # Protocol & Trust State
        self.epoch_last_key_update: int = 0
        self.is_quarantined: bool = False
        self.quarantined_epoch: Optional[int] = None
        self.trust_score: float = 1.0  # T_t(d)
        self.current_trust_state: TrustState = TrustState.TRUSTED
        
        # Behavioral & Risk variables from the paper
        self.behavioral_risk: float = 0.0   # P_c(d,t)
        self.graph_risk: float = 0.0        # S_graph(d,t)
        self.final_risk: float = 0.0        # R(d,t)
        self.hmm_state: HMMState = HMMState.NORMAL
        
        # Data & Cryptographic memory
        self.telemetry_history: List[Dict[str, Any]] = []
        self.secret_shares: Optional[Dict[str, Tuple[int, int]]] = None

    @property
    def qtk_shares(self) -> Dict[str, Tuple[int, int]]:
        """Backwards compatibility helper for existing code referencing qtk_shares."""
        if self.secret_shares is None:
            return {}
        return self.secret_shares

    @qtk_shares.setter
    def qtk_shares(self, value: Optional[Dict[str, Tuple[int, int]]]):
        """Backwards compatibility helper for setting shares."""
        if not value:
            self.secret_shares = None
        else:
            self.secret_shares = value

    def update_key(self, current_epoch: int):
        """Updates the last epoch index where the device performed a key update."""
        self.epoch_last_key_update = current_epoch

    def add_telemetry(self, record: Dict[str, Any]):
        """
        Appends a telemetry record, gracefully normalising alias fields
        and filling in any missing fields with defaults or device states.
        """
        normalized = record.copy()
        
        # Gracefully handle missing timestamp
        if "timestamp" not in normalized:
            normalized["timestamp"] = datetime.datetime.utcnow().isoformat()
            
        # Basic environmental attributes (fall back to device parameters)
        normalized["network_type"] = normalized.get("network_type") or self.network_type
        normalized["network_ip"] = normalized.get("network_ip") or normalized.get("ip_address") or self.ip_address
        normalized["timezone"] = normalized.get("timezone") or normalized.get("active_timezone") or self.timezone
        normalized["active_timezone"] = normalized.get("active_timezone") or normalized.get("timezone") or self.timezone
        normalized["country"] = normalized.get("country") or normalized.get("location_country") or self.country
        normalized["location_country"] = normalized.get("location_country") or normalized.get("country") or self.country
        normalized["battery_level"] = normalized.get("battery_level") if normalized.get("battery_level") is not None else self.battery_level
        
        # Temporal & Activity Metrics
        normalized["session_duration"] = normalized.get("session_duration") or normalized.get("session_duration_sec") or 0.0
        normalized["session_duration_sec"] = normalized.get("session_duration_sec") or normalized.get("session_duration") or 0.0
        
        normalized["idle_time"] = normalized.get("idle_time") or normalized.get("idle_time_sec") or 0.0
        normalized["idle_time_sec"] = normalized.get("idle_time_sec") or normalized.get("idle_time") or 0.0
        
        # Synchronization frequency mappings
        normalized["sync_interval"] = normalized.get("sync_interval") or normalized.get("sync_frequency") or 0.0
        normalized["sync_frequency"] = normalized.get("sync_frequency") or normalized.get("sync_interval") or 0.0
        
        # Message count mappings
        normalized["messages_sent"] = normalized.get("messages_sent") if normalized.get("messages_sent") is not None else normalized.get("message_count_sent", 0)
        normalized["message_count_sent"] = normalized.get("message_count_sent") if normalized.get("message_count_sent") is not None else normalized.get("messages_sent", 0)
        
        normalized["messages_received"] = normalized.get("messages_received") if normalized.get("messages_received") is not None else normalized.get("message_count_received", 0)
        normalized["message_count_received"] = normalized.get("message_count_received") if normalized.get("message_count_received") is not None else normalized.get("messages_received", 0)
        
        normalized["login_frequency"] = normalized.get("login_frequency") if normalized.get("login_frequency") is not None else 1.0
        
        self.telemetry_history.append(normalized)
        if len(self.telemetry_history) > 100:
            self.telemetry_history.pop(0)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the latest telemetry record or a default schema dictionary."""
        if not self.telemetry_history:
            return {
                "timestamp": None,
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "timezone": self.timezone,
                "active_timezone": self.timezone,
                "country": self.country,
                "location_country": self.country,
                "session_duration": 0.0,
                "session_duration_sec": 0.0,
                "idle_time": 0.0,
                "idle_time_sec": 0.0,
                "sync_interval": 0.0,
                "sync_frequency": 0.0,
                "messages_sent": 0,
                "message_count_sent": 0,
                "messages_received": 0,
                "message_count_received": 0,
                "login_frequency": 1.0,
                "battery_level": self.battery_level
            }
        return self.telemetry_history[-1]

    # --- Trust, Behavioral Risk, and Protocol State Helpers ---
    
    def update_trust(self, trust_score: float):
        """Updates the dynamic trust score T_t(d) computed from behavioral feedback."""
        self.trust_score = max(0.0, min(1.0, float(trust_score)))

    def update_behavioral_risk(self, risk: float):
        """Updates the estimated device-level behavioral anomaly score P_c(d,t)."""
        self.behavioral_risk = max(0.0, min(1.0, float(risk)))

    def update_graph_risk(self, risk: float):
        """Updates the relational anomaly score S_graph(d,t) from GCN/LSTM node analysis."""
        self.graph_risk = max(0.0, min(1.0, float(risk)))

    def update_final_risk(self, risk: float):
        """Updates the final fused behavioral risk score R(d,t)."""
        self.final_risk = max(0.0, min(1.0, float(risk)))

    def update_hmm_state(self, state: HMMState | str):
        """Updates the HMM behavioral classification state."""
        self.hmm_state = HMMState(state) if isinstance(state, str) else state

    def update_trust_state(self, state: TrustState | str):
        """Updates the protocol trust lifecycle state."""
        self.current_trust_state = TrustState(state) if isinstance(state, str) else state

    def quarantine(self, epoch: int):
        """Invokes the QTK containment state at the specified epoch."""
        self.is_quarantined = True
        self.quarantined_epoch = epoch
        self.current_trust_state = TrustState.QUARANTINED

    def recover(self):
        """Lifts quarantine and restores the device to a Trusted protocol state."""
        self.is_quarantined = False
        self.quarantined_epoch = None
        self.current_trust_state = TrustState.TRUSTED

    def revoke(self):
        """Expels/revokes the device from the group session."""
        self.is_quarantined = False
        self.quarantined_epoch = None
        self.current_trust_state = TrustState.REVOKED
