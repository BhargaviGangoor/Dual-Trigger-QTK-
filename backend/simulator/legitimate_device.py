from enum import Enum
import random
from typing import Optional, Dict, Any
from .device import Device, DeviceType, HMMState, TrustState
from .telemetry_generator import TelemetryGenerator

class ActivityState(str, Enum):
    """Realistic activity profiles/states for a user's device."""
    ACTIVE = "active"
    IDLE = "idle"
    SLEEPING = "sleeping"
    TRAVELLING = "travelling"
    LOW_USAGE = "low usage"

class LegitimateDevice(Device):
    """
    Simulates a legitimate user's device with realistic, non-uniform activity.
    Handles key updates, timezone transitions, battery drain, and mock HMM / Risk updates.
    """
    def __init__(self, device_id: str, owner_id: str, name: str, device_type: DeviceType | str, 
                 os_version: str, profile_name: str, battery_level: int = 100, 
                 ip_address: str = "127.0.0.1", network_type: str = "WiFi", 
                 country: str = "US", timezone: str = "UTC"):
        super().__init__(device_id, owner_id, name, device_type, os_version, 
                         battery_level, ip_address, network_type, country, timezone)
        self.profile_name: str = profile_name
        self.is_travelling: bool = False
        self.travel_duration: int = 0
        self.inactivity_duration: int = 0

    def simulate_epoch_action(self, current_epoch: int, is_active_hour: Optional[bool] = None):
        """
        Simulates device operations during the current epoch.
        Updates location states, key rotations, telemetries, HMM states, trust scores, and risk rates.
        """
        # 1. Determine Dynamic Activity State (active, idle, sleeping, travelling, low usage)
        if self.inactivity_duration > 0:
            self.inactivity_duration -= 1
            activity_state = ActivityState.IDLE
        else:
            # 0.5% chance to transition into a multi-epoch inactive/powered off state (vacation/power off)
            if random.random() < 0.005:
                self.inactivity_duration = random.randint(6, 15)
                activity_state = ActivityState.IDLE
            else:
                # Travel state tracking
                if self.is_travelling:
                    self.travel_duration -= 1
                    if self.travel_duration <= 0:
                        self.is_travelling = False
                else:
                    # 1% chance to start a travel period lasting 12 to 36 epochs
                    if random.random() < 0.01:
                        self.is_travelling = True
                        self.travel_duration = random.randint(12, 36)
                
                # Check timezone hour of day (epoch based)
                hour_of_day = current_epoch % 24
                
                if self.is_travelling:
                    activity_state = ActivityState.TRAVELLING
                elif 0 <= hour_of_day < 7: # Sleep hours
                    activity_state = ActivityState.SLEEPING
                else:
                    # Active hours: distribute states dynamically
                    if is_active_hour is False:
                        activity_state = random.choices(
                            [ActivityState.IDLE, ActivityState.LOW_USAGE, ActivityState.ACTIVE],
                            weights=[0.70, 0.25, 0.05]
                        )[0]
                    else:
                        activity_state = random.choices(
                            [ActivityState.ACTIVE, ActivityState.LOW_USAGE, ActivityState.IDLE],
                            weights=[0.65, 0.20, 0.15]
                        )[0]

        # 2. Key Update Rotation Logic (QTK-compatible)
        # Legitimate devices rotate keys before δ_inact (typically every 3 epochs).
        # We allow occasional missed updates when sleeping, traveling, or inactive.
        epoch_gap = current_epoch - self.epoch_last_key_update
        if activity_state in [ActivityState.ACTIVE, ActivityState.LOW_USAGE, ActivityState.TRAVELLING]:
            if epoch_gap >= 3:
                # 90% chance to update immediately, or delay slightly (simulating normal delay/vacation)
                if epoch_gap >= 4 or random.random() < 0.90:
                    self.update_key(current_epoch)

        # 3. Telemetry Observation Generation
        import datetime
        telemetry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "network_type": self.network_type,
            "network_ip": self.ip_address,
            "timezone": self.timezone,
            "country": self.country,
            "battery_level": self.battery_level
        }
        
        if activity_state == ActivityState.SLEEPING:
            telemetry.update({
                "session_duration": 0.0,
                "idle_time": 3600.0,
                "sync_interval": round(random.uniform(1800, 3600), 2),
                "messages_sent": 0,
                "messages_received": 0,
                "login_frequency": 0.0
            })
            self.battery_level = min(100, self.battery_level + random.choice([0, 1, 2]))  # Charging
            
        elif activity_state == ActivityState.IDLE:
            telemetry.update({
                "session_duration": 0.0,
                "idle_time": round(random.uniform(1800, 3600), 2),
                "sync_interval": round(random.uniform(900, 1800), 2),
                "messages_sent": 0,
                "messages_received": 0,
                "login_frequency": 0.1
            })
            self.battery_level = max(5, self.battery_level - random.choice([0, 1]))
            
        elif activity_state == ActivityState.LOW_USAGE:
            telemetry.update({
                "session_duration": round(random.uniform(5.0, 45.0), 2),
                "idle_time": round(random.uniform(600, 1800), 2),
                "sync_interval": round(random.uniform(60, 300), 2),
                "messages_sent": random.randint(0, 1),
                "messages_received": random.randint(0, 3),
                "login_frequency": 0.5
            })
            self.battery_level = max(5, self.battery_level - random.randint(0, 2))
            
        elif activity_state == ActivityState.TRAVELLING:
            # Shift environment dynamically to simulate physical location change
            travel_ip = "82.102.15." + str(random.randint(2, 254))
            telemetry.update({
                "network_type": "Cellular",
                "network_ip": travel_ip,
                "timezone": "Europe/Berlin",
                "country": "Germany",
                "session_duration": round(random.uniform(10.0, 90.0), 2),
                "idle_time": round(random.uniform(1200, 2400), 2),
                "sync_interval": round(random.uniform(180, 600), 2),
                "messages_sent": random.randint(0, 3),
                "messages_received": random.randint(1, 8),
                "login_frequency": 0.8
            })
            self.ip_address = travel_ip
            self.network_type = "Cellular"
            self.country = "Germany"
            self.timezone = "Europe/Berlin"
            self.battery_level = max(5, self.battery_level - random.randint(1, 3))
            
        else:  # ACTIVE state
            base_meta = TelemetryGenerator.generate_normal_telemetry(
                self.profile_name, 
                current_epoch % 24, 
                self.ip_address
            )
            self.ip_address = base_meta["network_ip"]
            self.network_type = base_meta["network_type"]
            self.country = base_meta["location_country"]
            self.timezone = base_meta["active_timezone"]
            
            telemetry.update({
                "network_type": self.network_type,
                "network_ip": self.ip_address,
                "timezone": self.timezone,
                "country": self.country,
                "session_duration": base_meta["session_duration_sec"],
                "idle_time": base_meta["idle_time_sec"],
                "sync_interval": round(3600.0 / max(0.1, base_meta["sync_frequency"]), 2),
                "messages_sent": base_meta["message_count_sent"],
                "messages_received": random.randint(base_meta["message_count_sent"], base_meta["message_count_sent"] * 3 + 2),
                "login_frequency": base_meta["login_frequency"]
            })
            self.battery_level = max(5, self.battery_level - random.randint(1, 3))

        # Commit battery level changes and append to device telemetry history
        telemetry["battery_level"] = self.battery_level
        self.add_telemetry(telemetry)

        # 4. Simulate HMM behavioral state updates
        if activity_state == ActivityState.SLEEPING:
            hmm_state = random.choices(
                [HMMState.IDLE, HMMState.NORMAL, HMMState.SUSPICIOUS, HMMState.HIGH_RISK],
                weights=[0.85, 0.13, 0.019, 0.001]
            )[0]
        elif activity_state == ActivityState.IDLE:
            hmm_state = random.choices(
                [HMMState.IDLE, HMMState.NORMAL, HMMState.SUSPICIOUS, HMMState.HIGH_RISK],
                weights=[0.75, 0.20, 0.048, 0.002]
            )[0]
        elif activity_state == ActivityState.TRAVELLING:
            hmm_state = random.choices(
                [HMMState.NORMAL, HMMState.SUSPICIOUS, HMMState.IDLE, HMMState.HIGH_RISK],
                weights=[0.60, 0.34, 0.057, 0.003]
            )[0]
        else: # ACTIVE or LOW_USAGE
            hmm_state = random.choices(
                [HMMState.NORMAL, HMMState.IDLE, HMMState.SUSPICIOUS, HMMState.HIGH_RISK],
                weights=[0.90, 0.07, 0.029, 0.001]
            )[0]
        
        self.update_hmm_state(hmm_state)

        # 5. Simulate HMM-driven Trust Decay & Recovery Math
        if hmm_state == HMMState.NORMAL:
            evidence = 1.0
        elif hmm_state == HMMState.IDLE:
            evidence = 0.90
        elif hmm_state == HMMState.SUSPICIOUS:
            evidence = 0.60
        else: # HIGH_RISK
            evidence = 0.25
            
        alpha = 0.8
        new_trust = alpha * self.trust_score + (1.0 - alpha) * evidence
        self.update_trust(new_trust)

        # 6. Simulate Risk Updates (behavioral, graph, final risks)
        if hmm_state == HMMState.NORMAL:
            b_risk = random.uniform(0.01, 0.08)
        elif hmm_state == HMMState.IDLE:
            b_risk = random.uniform(0.05, 0.15)
        elif hmm_state == HMMState.SUSPICIOUS:
            b_risk = random.uniform(0.35, 0.55)
        else: # HIGH_RISK
            b_risk = random.uniform(0.70, 0.90)
            
        if activity_state == ActivityState.TRAVELLING:
            g_risk = random.uniform(0.08, 0.25)
        else:
            g_risk = random.uniform(0.01, 0.08)
            
        f_risk = 0.6 * b_risk + 0.4 * g_risk
        
        self.update_behavioral_risk(b_risk)
        self.update_graph_risk(g_risk)
        self.update_final_risk(f_risk)
