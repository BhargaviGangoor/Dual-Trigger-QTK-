from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Pydantic Schemas ---

class MessageBase(BaseModel):
    id: str
    sender_name: str
    receiver_name: str
    content_type: str
    size_bytes: int
    direction: str
    reply_to_id: Optional[str] = None
    is_starred: bool = False
    is_pinned: bool = False
    is_archived: bool = False
    status: str
    reactions: List[Dict[str, Any]] = []

class MessageCreate(MessageBase):
    user_id: int
    sender_device_id: Optional[str] = None

class MessageSchema(MessageBase):
    timestamp: datetime
    sender_device_id: Optional[str] = None

    class Config:
        from_attributes = True

# --- Metadata Record ---
class MetadataRecordBase(BaseModel):
    login_frequency: float
    sync_frequency: float
    session_duration_sec: float
    message_count_sent: int
    message_count_received: int
    read_count: int
    delivery_count: int
    network_ip: str
    network_type: str
    active_timezone: str
    location_country: str
    idle_time_sec: float
    battery_level: int
    os_version: str

class MetadataRecordCreate(MetadataRecordBase):
    device_id: str

class MetadataRecordSchema(MetadataRecordBase):
    id: int
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool

    class Config:
        from_attributes = True

# --- Simulation Event ---
class SimulationEventBase(BaseModel):
    event_type: str
    description: str
    trust_score_before: Optional[float] = None
    trust_score_after: Optional[float] = None
    fsm_state_before: Optional[str] = None
    fsm_state_after: Optional[str] = None
    reason: Optional[str] = None

class SimulationEventCreate(SimulationEventBase):
    device_id: Optional[str] = None

class SimulationEventSchema(SimulationEventBase):
    id: int
    timestamp: datetime
    device_id: Optional[str] = None

    class Config:
        from_attributes = True

# --- Device ---
class DeviceBase(BaseModel):
    id: str
    name: str
    device_type: str
    public_key: str
    fingerprint: str
    ip_address: str
    network_type: str
    country: str
    timezone: str
    os_version: str
    battery_level: int

class DeviceCreate(DeviceBase):
    user_id: int

class DeviceSchema(DeviceBase):
    pairing_timestamp: datetime
    last_active: datetime
    trust_score: float
    current_trust_state: str
    is_active: bool
    events: List[SimulationEventSchema] = []

    class Config:
        from_attributes = True

# --- User ---
class UserBase(BaseModel):
    name: str
    avatar: Optional[str] = None
    behavior_profile: str

class UserCreate(UserBase):
    pass

class UserSchema(UserBase):
    id: int
    created_at: datetime
    devices: List[DeviceSchema] = []

    class Config:
        from_attributes = True

# --- Simulation Run ---
class SimulationRunBase(BaseModel):
    name: str
    behavior_profile: str
    attack_type: str
    duration_sim_days: int
    ml_algorithm: str
    config_json: Dict[str, Any] = {}

class SimulationRunCreate(SimulationRunBase):
    pass

class SimulationRunSchema(SimulationRunBase):
    id: int
    timestamp: datetime
    total_events: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    detection_latency_sec: float
    false_positives: int
    false_negatives: int

    class Config:
        from_attributes = True

# --- Configuration for Simulation Trigger ---
class SimulationConfig(BaseModel):
    user_profile: str = "Casual User"
    attack_type: str = "Ghost Pairing"
    attack_day: int = 10
    duration_days: int = 30
    noise_level: float = 0.05
    ml_algorithms: List[str] = ["HMM", "Isolation Forest"]
    alpha: float = 0.8
    detection_threshold: float = 0.6
    seed: int = 42
