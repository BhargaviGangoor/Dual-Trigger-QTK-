import random
from typing import Dict, Any, List

class BehaviorProfile:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    def sample_active_hours(self, hour: int) -> bool:
        """Determines if the user is typically active at this hour."""
        active_hours = self.config.get("active_hours", range(8, 22))
        prob = self.config.get("activity_prob_by_hour", {})
        
        base_prob = 0.8 if hour in active_hours else 0.05
        # If there's an explicit override probability for this hour, use it
        return random.random() < prob.get(hour, base_prob)

    def sample_message_count(self) -> int:
        """Sample number of messages sent in an active hour."""
        return max(0, int(random.normalvariate(
            self.config.get("avg_messages_per_hour", 5),
            self.config.get("std_messages_per_hour", 2)
        )))

    def sample_network(self) -> Dict[str, str]:
        """Samples network configurations, IPs and locations."""
        networks = self.config.get("networks", ["WiFi", "Cellular"])
        countries = self.config.get("countries", ["United States"])
        timezones = self.config.get("timezones", ["America/New_York"])
        
        network_type = random.choice(networks)
        country = random.choice(countries)
        timezone = random.choice(timezones)
        
        # IP prefix logic
        ip_prefix = self.config.get("ip_prefix", "192.168.1.")
        ip_address = ip_prefix + str(random.randint(2, 254))
        
        return {
            "network_type": network_type,
            "ip_address": ip_address,
            "country": country,
            "timezone": timezone
        }

    def sample_battery_drain(self, current: int) -> int:
        """Simulates battery depletion during activity."""
        drain = random.randint(1, self.config.get("max_battery_drain", 3))
        return max(5, current - drain)

    def sample_sync_delay(self) -> float:
        """Simulates synchronization delay in seconds."""
        return max(0.1, random.exponential(self.config.get("avg_sync_delay_sec", 1.0)))


PROFILES: Dict[str, Dict[str, Any]] = {
    "Student": {
        "active_hours": range(9, 24),  # Active late
        "activity_prob_by_hour": {0: 0.4, 1: 0.2, 2: 0.1, 8: 0.2},
        "avg_messages_per_hour": 15,
        "std_messages_per_hour": 6,
        "networks": ["WiFi", "Cellular"],
        "countries": ["United States"],
        "timezones": ["America/New_York"],
        "ip_prefix": "172.16.23.",
        "max_battery_drain": 5,
        "avg_sync_delay_sec": 0.8,
        "allowed_devices": ["Macbook", "Android Phone", "iPad"]
    },
    "Corporate Employee": {
        "active_hours": range(8, 18),  # Office hours
        "activity_prob_by_hour": {12: 0.9, 13: 0.5, 18: 0.2, 20: 0.1, 21: 0.05, 22: 0.02},
        "avg_messages_per_hour": 6,
        "std_messages_per_hour": 2,
        "networks": ["WiFi", "Ethernet"],
        "countries": ["United States"],
        "timezones": ["America/Chicago"],
        "ip_prefix": "10.200.45.",
        "max_battery_drain": 2,
        "avg_sync_delay_sec": 0.3,
        "allowed_devices": ["Windows Laptop", "iPhone"]
    },
    "Traveler": {
        "active_hours": range(7, 23),
        "avg_messages_per_hour": 8,
        "std_messages_per_hour": 4,
        "networks": ["Cellular", "WiFi"],  # Rarely ethernet
        "countries": ["Germany", "France", "United Kingdom", "Italy"],  # Travels often
        "timezones": ["Europe/Berlin", "Europe/London", "Europe/Rome"],
        "ip_prefix": "82.102.15.",
        "max_battery_drain": 6,
        "avg_sync_delay_sec": 2.5,  # Poorer sync due to travel networks
        "allowed_devices": ["iPhone", "Android Tablet"]
    },
    "Business Owner": {
        "active_hours": range(7, 21),  # Always-on business hours
        "activity_prob_by_hour": {6: 0.5, 21: 0.3, 22: 0.2},
        "avg_messages_per_hour": 25,  # Very active communicator
        "std_messages_per_hour": 8,
        "networks": ["WiFi", "Ethernet", "Cellular"],
        "countries": ["Canada"],
        "timezones": ["America/Toronto"],
        "ip_prefix": "24.53.111.",
        "max_battery_drain": 4,
        "avg_sync_delay_sec": 0.4,
        "allowed_devices": ["Windows PC", "Android Phone", "Macbook"]
    },
    "Heavy User": {
        "active_hours": range(6, 24),
        "activity_prob_by_hour": {0: 0.6, 1: 0.4, 2: 0.2},
        "avg_messages_per_hour": 35,
        "std_messages_per_hour": 10,
        "networks": ["WiFi", "Cellular"],
        "countries": ["United States"],
        "timezones": ["America/Los_Angeles"],
        "ip_prefix": "68.4.150.",
        "max_battery_drain": 7,
        "avg_sync_delay_sec": 0.2,
        "allowed_devices": ["iPhone", "Macbook", "iPad", "Windows PC", "Android Phone"]
    },
    "Casual User": {
        "active_hours": range(9, 21),
        "activity_prob_by_hour": {12: 0.6, 18: 0.7, 8: 0.1},
        "avg_messages_per_hour": 3,
        "std_messages_per_hour": 1,
        "networks": ["WiFi", "Cellular"],
        "countries": ["United States"],
        "timezones": ["America/New_York"],
        "ip_prefix": "98.139.180.",
        "max_battery_drain": 2,
        "avg_sync_delay_sec": 1.2,
        "allowed_devices": ["Android Phone", "Chrome Browser"]
    },
    "Night Owl": {
        "active_hours": range(0, 6),  # Sleep during day
        "activity_prob_by_hour": {12: 0.05, 13: 0.05, 18: 0.1, 20: 0.3, 22: 0.7, 23: 0.9, 0: 0.9, 1: 0.9, 2: 0.9, 3: 0.8, 4: 0.6, 5: 0.4},
        "avg_messages_per_hour": 12,
        "std_messages_per_hour": 5,
        "networks": ["WiFi", "Cellular"],
        "countries": ["United States"],
        "timezones": ["America/New_York"],
        "ip_prefix": "75.12.98.",
        "max_battery_drain": 4,
        "avg_sync_delay_sec": 0.5,
        "allowed_devices": ["Macbook", "iPhone"]
    },
    "VPN User": {
        "active_hours": range(8, 23),
        "avg_messages_per_hour": 8,
        "std_messages_per_hour": 3,
        "networks": ["VPN", "Cellular"],
        "countries": ["Switzerland", "Sweden", "Iceland"],  # Rotates VPN nodes
        "timezones": ["Europe/Zurich", "Europe/Stockholm"],
        "ip_prefix": "45.155.205.",  # VPN pool
        "max_battery_drain": 4,
        "avg_sync_delay_sec": 1.5,
        "allowed_devices": ["Linux Laptop", "Android Phone"]
    }
}

def get_profile(name: str) -> BehaviorProfile:
    config = PROFILES.get(name, PROFILES["Casual User"])
    return BehaviorProfile(name, config)

def get_all_profiles() -> List[str]:
    return list(PROFILES.keys())
