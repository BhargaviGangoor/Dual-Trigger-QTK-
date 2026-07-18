from enum import Enum
import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType, HMMState, TrustState
from .telemetry_generator import TelemetryGenerator

class AttackStrategy(str, Enum):
    """Attack strategies for the rogue device to compromise groups while evading detection."""
    STEALTH = "STEALTH"
    MIMIC = "MIMIC"
    BURST = "BURST"
    RANDOM = "RANDOM"

class RogueDevice(Device):
    """
    Simulates a rogue/compromised device operating under the control of an attacker.
    Tries to remain undetected inside the group by rotating keys before inactivity timeouts
    and executing strategic telemetry spoofing (Stealth, Mimic, Burst, Random).
    """
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: DeviceType | str, 
                 os_version: str, profile_name: str, battery_level: int = 100, 
                 ip_address: str = "185.220.101.5", network_type: str = "VPN", 
                 country: str = "Russia", timezone: str = "Europe/Moscow",
                 strategy: AttackStrategy | str = AttackStrategy.STEALTH):
        super().__init__(device_id, owner_id, name, device_type, os_version, 
                         battery_level, ip_address, network_type, country, timezone)
        self.profile_name: str = profile_name
        self.strategy: AttackStrategy = AttackStrategy(strategy) if isinstance(strategy, str) else strategy

    def change_strategy(self, strategy: AttackStrategy | str):
        """Allows dynamically switching the device's attack behavior."""
        self.strategy = AttackStrategy(strategy) if isinstance(strategy, str) else strategy

    def simulate_epoch_action(self, current_epoch: int, is_active_hour: Optional[bool] = None,
                              normal_device_metadata: Optional[Dict[str, Any]] = None):
        """
        Simulates rogue device actions during the current epoch.
        Updates key rotations to stay below delta_inact, spoof telemetries, and sets simulated states/risks.
        """
        # 1. Update keys frequently to stay below inactivity threshold delta_inact = 5
        epoch_gap = current_epoch - self.epoch_last_key_update
        should_update = False
        
        if self.strategy == AttackStrategy.BURST:
            # Hold key updates until close to detection gap
            if epoch_gap >= 4:
                should_update = True
        elif self.strategy == AttackStrategy.RANDOM:
            if epoch_gap >= random.randint(1, 4):
                should_update = True
        else: # STEALTH or MIMIC
            # Update key frequently (every 2 or 3 epochs) to blend in
            if epoch_gap >= random.choice([2, 3]):
                should_update = True
                
        if should_update:
            self.update_key(current_epoch)

        # 2. Parameterize and construct telemetry observations based on attack strategy
        import datetime
        
        base_meta = TelemetryGenerator.generate_normal_telemetry(
            self.profile_name, 
            current_epoch % 24, 
            self.ip_address
        )
        
        telemetry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "network_type": self.network_type,
            "network_ip": self.ip_address,
            "timezone": self.timezone,
            "country": self.country,
            "battery_level": self.battery_level
        }
        
        is_burst_active = False
        
        if self.strategy == AttackStrategy.STEALTH:
            # Subtle deviations. Try to stay within Normal HMM profile but drift over time.
            meta = normal_device_metadata if normal_device_metadata else base_meta
            
            # Modify IP slightly 20% of the time (subnet scanning)
            ip_parts = meta.get("network_ip", "192.168.1.1").split(".")
            ip_prefix = ".".join(ip_parts[:3]) + "." if len(ip_parts) == 4 else "192.168.1."
            ip = ip_prefix + str(random.randint(2, 254)) if random.random() < 0.20 else meta.get("network_ip", "192.168.1.10")
            
            # Slowly drift sync_frequency and message_count over time
            time_factor = min(1.0, current_epoch / 30.0)
            drift_sync_freq = meta.get("sync_frequency", 4.0) * (1.0 + time_factor * 1.5)
            drift_msgs = meta.get("message_count_sent", 5.0) + int(time_factor * 20)
            
            telemetry.update({
                "network_type": self.network_type,
                "network_ip": ip,
                "session_duration": meta.get("session_duration_sec", 120.0) * random.uniform(0.9, 1.2),
                "session_duration_sec": meta.get("session_duration_sec", 120.0) * random.uniform(0.9, 1.2),
                "idle_time": meta.get("idle_time_sec", 300.0),
                "idle_time_sec": meta.get("idle_time_sec", 300.0),
                "sync_interval": round(3600.0 / max(0.1, drift_sync_freq), 2),
                "sync_frequency": round(drift_sync_freq, 2),
                "messages_sent": int(drift_msgs),
                "message_count_sent": int(drift_msgs),
                "messages_received": int(drift_msgs),
                "message_count_received": int(drift_msgs),
                "login_frequency": meta.get("login_frequency", 1.0)
            })
            self.ip_address = ip
                
        elif self.strategy == AttackStrategy.MIMIC:
            # Mimic legitimate telemetry features but routing via VPN
            vpn_use = random.random() < 0.50
            self.network_type = "VPN" if vpn_use else "WiFi"
            self.ip_address = "185.220.101." + str(random.randint(2, 254)) if vpn_use else base_meta["network_ip"]
            
            if normal_device_metadata:
                sync_interval = round(3600.0 / max(0.1, normal_device_metadata.get("sync_frequency", 4.0)), 2)
                messages_sent = normal_device_metadata.get("message_count_sent", 5)
                session_dur = normal_device_metadata.get("session_duration_sec", 120.0)
                idle = normal_device_metadata.get("idle_time_sec", 300.0)
            else:
                sync_interval = round(3600.0 / max(0.1, base_meta["sync_frequency"]), 2)
                messages_sent = base_meta["message_count_sent"]
                session_dur = base_meta["session_duration_sec"]
                idle = base_meta["idle_time_sec"]
                
            telemetry.update({
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "session_duration": round(session_dur * random.uniform(0.95, 1.05), 2),
                "session_duration_sec": round(session_dur * random.uniform(0.95, 1.05), 2),
                "idle_time": round(idle * random.uniform(0.95, 1.05), 2),
                "idle_time_sec": round(idle * random.uniform(0.95, 1.05), 2),
                "sync_interval": round(sync_interval * random.uniform(0.95, 1.05), 2),
                "sync_frequency": round(3600.0 / max(0.1, sync_interval * random.uniform(0.95, 1.05)), 2),
                "messages_sent": max(0, int(messages_sent * random.uniform(0.8, 1.2))),
                "message_count_sent": max(0, int(messages_sent * random.uniform(0.8, 1.2))),
                "messages_received": random.randint(1, 10),
                "login_frequency": 1.0
            })
            
        elif self.strategy == AttackStrategy.BURST:
            # Multi-epoch idle states followed by sudden active bursts
            is_burst_active = random.random() < 0.15
            
            if is_burst_active:
                self.network_type = "VPN"
                self.ip_address = "185.220.101." + str(random.randint(2, 254))
                telemetry.update({
                    "network_type": "VPN",
                    "network_ip": self.ip_address,
                    "session_duration": round(random.uniform(900.0, 1800.0), 2),
                    "session_duration_sec": round(random.uniform(900.0, 1800.0), 2),
                    "idle_time": 0.0,
                    "idle_time_sec": 0.0,
                    "sync_interval": round(random.uniform(2.0, 10.0), 2),
                    "sync_frequency": round(3600.0 / random.uniform(2.0, 10.0), 2),
                    "messages_sent": random.randint(10, 30),
                    "message_count_sent": random.randint(10, 30),
                    "messages_received": random.randint(50, 150),
                    "login_frequency": 5.0
                })
            else:
                telemetry.update({
                    "session_duration": 0.0,
                    "session_duration_sec": 0.0,
                    "idle_time": 3600.0,
                    "idle_time_sec": 3600.0,
                    "sync_interval": 3600.0,
                    "sync_frequency": 1.0,
                    "messages_sent": 0,
                    "message_count_sent": 0,
                    "messages_received": 0,
                    "login_frequency": 0.0
                })
                
        else: # RANDOM strategy
            self.network_type = random.choice(["VPN", "Cellular", "WiFi"])
            self.ip_address = f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            self.timezone = random.choice(["Asia/Shanghai", "Europe/Moscow", "America/Chicago", "UTC"])
            self.country = random.choice(["China", "Russia", "Netherlands", "United States"])
            
            telemetry.update({
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "timezone": self.timezone,
                "country": self.country,
                "session_duration": round(random.uniform(1.0, 7200.0), 2),
                "session_duration_sec": round(random.uniform(1.0, 7200.0), 2),
                "idle_time": round(random.uniform(0.0, 3600.0), 2),
                "idle_time_sec": round(random.uniform(0.0, 3600.0), 2),
                "sync_interval": round(random.uniform(1.0, 3600.0), 2),
                "sync_frequency": round(3600.0 / random.uniform(1.0, 3600.0), 2),
                "messages_sent": random.randint(0, 100),
                "message_count_sent": random.randint(0, 100),
                "messages_received": random.randint(0, 200),
                "login_frequency": round(random.uniform(0.1, 10.0), 1)
            })
            
        self.battery_level = max(5, self.battery_level - random.randint(0, 2))
        telemetry["battery_level"] = self.battery_level
        self.add_telemetry(telemetry)

        # 3. Simulate HMM behavioral state updates
        if self.strategy == AttackStrategy.STEALTH:
            hmm_state = random.choices(
                [HMMState.NORMAL, HMMState.IDLE, HMMState.SUSPICIOUS, HMMState.HIGH_RISK],
                weights=[0.75, 0.20, 0.048, 0.002]
            )[0]
        elif self.strategy == AttackStrategy.MIMIC:
            hmm_state = random.choices(
                [HMMState.NORMAL, HMMState.IDLE, HMMState.SUSPICIOUS, HMMState.HIGH_RISK],
                weights=[0.60, 0.30, 0.09, 0.01]
            )[0]
        elif self.strategy == AttackStrategy.BURST:
            if is_burst_active:
                hmm_state = random.choices(
                    [HMMState.HIGH_RISK, HMMState.SUSPICIOUS, HMMState.IDLE, HMMState.NORMAL],
                    weights=[0.70, 0.25, 0.04, 0.01]
                )[0]
            else:
                hmm_state = HMMState.IDLE
        else: # RANDOM
            hmm_state = random.choices(
                [HMMState.HIGH_RISK, HMMState.SUSPICIOUS, HMMState.IDLE, HMMState.NORMAL],
                weights=[0.45, 0.35, 0.10, 0.10]
            )[0]
            
        self.update_hmm_state(hmm_state)

        # 4. Simulate trust score decay/recovery based on HMM state
        if hmm_state == HMMState.NORMAL:
            evidence = 1.0
        elif hmm_state == HMMState.IDLE:
            evidence = 0.90
        elif hmm_state == HMMState.SUSPICIOUS:
            evidence = 0.55
        else: # HIGH_RISK
            evidence = 0.20
            
        alpha = 0.8
        new_trust = alpha * self.trust_score + (1.0 - alpha) * evidence
        self.update_trust(new_trust)
        
        # 5. Simulate risk levels (behavioral, graph, final risks)
        if self.strategy == AttackStrategy.STEALTH:
            b_risk = random.uniform(0.05, 0.18)
            g_risk = random.uniform(0.04, 0.15)
        elif self.strategy == AttackStrategy.MIMIC:
            b_risk = random.uniform(0.12, 0.28)
            g_risk = random.uniform(0.10, 0.22)
        elif self.strategy == AttackStrategy.BURST:
            if is_burst_active:
                b_risk = random.uniform(0.60, 0.85)
                g_risk = random.uniform(0.55, 0.80)
            else:
                b_risk = random.uniform(0.04, 0.12)
                g_risk = random.uniform(0.05, 0.15)
        else: # RANDOM
            b_risk = random.uniform(0.50, 0.88)
            g_risk = random.uniform(0.45, 0.82)
            
        f_risk = 0.65 * b_risk + 0.35 * g_risk
        
        self.update_behavioral_risk(b_risk)
        self.update_graph_risk(g_risk)
        self.update_final_risk(f_risk)
