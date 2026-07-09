import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    avatar = Column(String, nullable=True)
    behavior_profile = Column(String, default="Casual User")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")

class Device(Base):
    __tablename__ = "devices"
    
    id = Column(String, primary_key=True, index=True)  # unique device identifier (e.g. fingerprint or random uuid)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)  # e.g., "Android Phone", "Chrome Browser"
    device_type = Column(String, default="linked")  # "primary" or "linked"
    public_key = Column(String, nullable=False)  # simulated identity key
    fingerprint = Column(String, nullable=False)  # simulated key fingerprint
    pairing_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    last_active = Column(DateTime, default=datetime.datetime.utcnow)
    ip_address = Column(String, default="127.0.0.1")
    network_type = Column(String, default="WiFi")  # WiFi, Cellular, Ethernet, VPN
    country = Column(String, default="United States")
    timezone = Column(String, default="UTC")
    os_version = Column(String, default="Windows 11")
    battery_level = Column(Integer, default=100)
    trust_score = Column(Float, default=1.0)
    current_trust_state = Column(String, default="Trusted")  # Trusted, Idle, Suspicious, Verification Required, Revoked
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="devices")
    metadata_records = relationship("MetadataRecord", back_populates="device", cascade="all, delete-orphan")
    events = relationship("SimulationEvent", back_populates="device", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    sender_device_id = Column(String, ForeignKey("devices.id", ondelete="SET NULL"), nullable=True)
    sender_name = Column(String, nullable=False)
    receiver_name = Column(String, nullable=False)
    content_type = Column(String, default="text")  # text, image, video, document, audio, sticker
    size_bytes = Column(Integer, default=100)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    direction = Column(String, default="outgoing")  # incoming or outgoing
    reply_to_id = Column(String, nullable=True)
    is_starred = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    status = Column(String, default="read")  # sent, delivered, read
    reactions = Column(JSON, default=list)  # list of reactions: [{"user": "Alice", "emoji": "👍"}]
    
    user = relationship("User", back_populates="messages")

class MetadataRecord(Base):
    __tablename__ = "metadata_records"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, ForeignKey("devices.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Behavioral signals
    login_frequency = Column(Float, default=0.0)      # logins per day
    sync_frequency = Column(Float, default=0.0)       # sync events per hour
    session_duration_sec = Column(Float, default=0.0)
    message_count_sent = Column(Integer, default=0)
    message_count_received = Column(Integer, default=0)
    read_count = Column(Integer, default=0)
    delivery_count = Column(Integer, default=0)
    
    # Environment signals
    network_ip = Column(String, nullable=False)
    network_type = Column(String, nullable=False)
    active_timezone = Column(String, nullable=False)
    location_country = Column(String, nullable=False)
    idle_time_sec = Column(Float, default=0.0)
    battery_level = Column(Integer, default=100)
    os_version = Column(String, default="Windows 11")
    
    # Anomaly scores
    anomaly_score = Column(Float, default=0.0)
    is_anomaly = Column(Boolean, default=False)
    
    device = relationship("Device", back_populates="metadata_records")

class SimulationEvent(Base):
    __tablename__ = "simulation_events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    event_type = Column(String, nullable=False)  # login, logout, sync, ip_change, attack_trigger, trust_decay, verification, revocation
    device_id = Column(String, ForeignKey("devices.id", ondelete="CASCADE"), nullable=True)
    description = Column(String, nullable=False)
    
    # Trust tracking snapshot
    trust_score_before = Column(Float, nullable=True)
    trust_score_after = Column(Float, nullable=True)
    fsm_state_before = Column(String, nullable=True)
    fsm_state_after = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    
    device = relationship("Device", back_populates="events")

class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    name = Column(String, nullable=False)
    behavior_profile = Column(String, nullable=False)
    attack_type = Column(String, nullable=False)
    duration_sim_days = Column(Integer, default=30)
    ml_algorithm = Column(String, default="HMM + LSTM")
    total_events = Column(Integer, default=0)
    
    # Metrics
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    detection_latency_sec = Column(Float, default=0.0)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    
    config_json = Column(JSON, default=dict)
