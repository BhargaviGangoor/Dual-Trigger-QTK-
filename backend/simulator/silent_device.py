from enum import Enum
import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType, HMMState, TrustState

class HeartbeatMode(str, Enum):
    """Modes representing different styles of inactivity/offlineness."""
    NONE = "NONE"             # Completely silent (e.g., powered off, dead battery)
    VERY_RARE = "VERY_RARE"   # Sends rare telemetry heartbeat (e.g., forgotten tablet on standby)
    OFFLINE = "OFFLINE"       # Disconnected (e.g., abandoned device with sporadic offline checks)

class SilentDevice(Device):
    """
    Represents a legitimate device that has become inactive (lost, sold, forgotten, or powered off).
    It stops updating keys and generating normal user activity, allowing its key-update age
    to naturally exceed delta_inact so that the QTK baseline inactivity trigger can quarantine it.
    """
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: DeviceType | str, 
                 os_version: str, battery_level: int = 100, ip_address: str = "127.0.0.1", 
                 network_type: str = "WiFi", country: str = "US", timezone: str = "UTC",
                 heartbeat_mode: HeartbeatMode | str = HeartbeatMode.NONE):
        super().__init__(device_id, owner_id, name, device_type, os_version, 
                         battery_level, ip_address, network_type, country, timezone)
        self.heartbeat_mode: HeartbeatMode = HeartbeatMode(heartbeat_mode) if isinstance(heartbeat_mode, str) else heartbeat_mode

    def change_heartbeat_mode(self, mode: HeartbeatMode | str):
        """Allows dynamically changing the heartbeat mode of the silent device."""
        self.heartbeat_mode = HeartbeatMode(mode) if isinstance(mode, str) else mode

    def simulate_epoch_action(self, current_epoch: int, is_active_hour: Optional[bool] = None):
        """
        Simulates one epoch of inactivity.
        Performs NO key updates, optionally generates rare heartbeat telemetry, and keeps risks low.
        """
        # 1. NO Key Updates: To allow epoch_gap = current_epoch - epoch_last_key_update to grow and cross delta_inact
        
        # 2. Telemetry generation based on heartbeat mode
        import datetime
        emit_heartbeat = False
        
        if self.heartbeat_mode == HeartbeatMode.VERY_RARE:
            # ~2% probability per epoch (approx. once every two days)
            emit_heartbeat = random.random() < 0.02
        elif self.heartbeat_mode == HeartbeatMode.OFFLINE:
            # ~5% probability per epoch (approx. once a day)
            emit_heartbeat = random.random() < 0.05
            
        if emit_heartbeat:
            # Slowly deplete battery
            self.battery_level = max(0, self.battery_level - random.choice([0, 1]))
            
            telemetry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "timezone": self.timezone,
                "country": self.country,
                "battery_level": self.battery_level,
                "session_duration": 0.0,
                "idle_time": 3600.0,
                # Extremely long sync intervals representing dormant behavior
                "sync_interval": 86400.0 if self.heartbeat_mode == HeartbeatMode.VERY_RARE else 172800.0,
                "messages_sent": 0,
                "messages_received": 0,
                "login_frequency": 0.01 if self.heartbeat_mode == HeartbeatMode.VERY_RARE else 0.0
            }
            self.add_telemetry(telemetry)
            
        # 3. Maintain HMM state to IDLE
        self.update_hmm_state(HMMState.IDLE)
        
        # 4. Decay trust score gently to reflect inactivity (not compromise)
        # drifts to 0.90 representing a clean, inactive state
        new_trust = 0.95 * self.trust_score + 0.05 * 0.90
        self.update_trust(new_trust)
        
        # 5. Keep risks low (non-malicious)
        b_risk = random.uniform(0.01, 0.08)
        g_risk = random.uniform(0.01, 0.05)
        f_risk = 0.6 * b_risk + 0.4 * g_risk
        
        self.update_behavioral_risk(b_risk)
        self.update_graph_risk(g_risk)
        self.update_final_risk(f_risk)
