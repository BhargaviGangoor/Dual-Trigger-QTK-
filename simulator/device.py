from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import datetime

class DeviceType(str, Enum):
    """Types of MLS client devices."""
    PRIMARY = "primary"
    LINKED = "linked"

class TrustState(str, Enum):
    """Protocol trust states in the device lifecycle."""
    TRUSTED = "Trusted"
    SUSPICIOUS = "Suspicious"
    QUARANTINED = "Quarantined"
    REVOKED = "Revoked"

class HMMState(str, Enum):
    """Hidden Markov Model behavioral states."""
    NORMAL = "Normal"
    IDLE = "Idle"
    SUSPICIOUS = "Suspicious"
    HIGH_RISK = "High-Risk"

class Device:
    """
    Base representation of an MLS client device within an encrypted messaging group.
    Maintains protocol epoch tracking, cryptographic quarantine state, and telemetry history.
    """
    def __init__(
        self,
        device_id: str,
        owner_id: str,
        name: str,
        device_type: DeviceType | str = DeviceType.PRIMARY,
        os_version: str = "Android 14",
        battery_level: int = 100,
        ip_address: str = "192.168.1.10",
        network_type: str = "WiFi",
        country: str = "United States",
        timezone: str = "America/New_York",
        initial_epoch: int = 0
    ):
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

        # Protocol Tracking
        self.epoch_last_key_update: int = initial_epoch
        self.is_quarantined: bool = False
        self.quarantined_epoch: Optional[int] = None
        self.quarantine_reason: Optional[str] = None
        self.current_trust_state: TrustState = TrustState.TRUSTED

        # Evaluated Scores (Computed externally by models, NOT internally mocked)
        self.trust_score: float = 1.0       # T_t(d)
        self.behavioral_risk: float = 0.0   # P_c(d,t) from HMM
        self.graph_risk: float = 0.0        # S_graph(d,t) from GNN/Graph-LSTM
        self.final_risk: float = 0.0        # R(d,t) from Risk Fusion
        self.hmm_state: HMMState = HMMState.NORMAL

        # Memory & Cryptographic Shares
        self.telemetry_history: List[Dict[str, Any]] = []
        self.secret_shares: Optional[Dict[str, Tuple[int, int]]] = None

    @property
    def key_age(self) -> int:
        """Helper to get current key age relative to last known epoch."""
        return 0

    def get_key_age(self, current_epoch: int) -> int:
        """Returns key update age: current_epoch - epoch_last_key_update."""
        return current_epoch - self.epoch_last_key_update

    def update_key(self, current_epoch: int):
        """Records an MLS key update commit at the given epoch."""
        self.epoch_last_key_update = current_epoch

    def add_telemetry(self, record: Dict[str, Any]):
        """Appends a telemetry observation dictionary to history."""
        self.telemetry_history.append(record)
        if len(self.telemetry_history) > 100:
            self.telemetry_history.pop(0)

    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent telemetry observation."""
        return self.telemetry_history[-1] if self.telemetry_history else None

    def quarantine(self, epoch: int, reason: str = "Unknown"):
        """Transitions device to Quarantined protocol state."""
        self.is_quarantined = True
        self.quarantined_epoch = epoch
        self.quarantine_reason = reason
        self.current_trust_state = TrustState.QUARANTINED

    def recover(self):
        """Lifts quarantine containment upon successful key recovery."""
        self.is_quarantined = False
        self.quarantined_epoch = None
        self.quarantine_reason = None
        self.current_trust_state = TrustState.TRUSTED

    def revoke(self):
        """Permanently revokes/expels the device from group session."""
        self.is_quarantined = False
        self.quarantined_epoch = None
        self.current_trust_state = TrustState.REVOKED

    def __repr__(self) -> str:
        return f"<Device id={self.device_id} type={self.device_type.value} state={self.current_trust_state.value} last_key_ep={self.epoch_last_key_update}>"
