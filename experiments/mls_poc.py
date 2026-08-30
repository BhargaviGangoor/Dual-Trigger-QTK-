import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.device import Device, DeviceType, TrustState
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger, TriggerReason
from qtk.quarantine_state import QuarantineManager
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

class MLSEventType:
    CREATE_GROUP = "MLS_CREATE_GROUP"
    ADD_MEMBER = "MLS_ADD_MEMBER"
    KEY_UPDATE_COMMIT = "MLS_KEY_UPDATE_COMMIT"
    APP_MESSAGE = "MLS_APPLICATION_MESSAGE"
    REMOVE_MEMBER = "MLS_REMOVE_MEMBER"
    PROPOSAL = "MLS_PROPOSAL"

class RealMLSProofOfConcept:
    """
    Proof-of-Concept MLS Lifecycle Integration.
    Demonstrates that the Dual-Trigger QTK behavioral decision engine can directly ingest
    real MLS protocol event streams (TreeKEM epochs, Add proposals, Commit key rotations,
    and Application message telemetry) to trigger quarantine without modifying MLS cryptography.
    """
    def __init__(self, group_id: str = "secure_mls_group_01"):
        self.group_id = group_id
        self.epoch_tracker = EpochTracker()
        self.dual_trigger = DualTrigger(delta_inact=5, theta_R=0.65)
        self.hmm = HMMDetector()
        self.graph_lstm = GraphLSTM(beta=0.8)
        self.fusion = RiskFusion()
        self.devices: Dict[str, Device] = {}
        self.mls_event_log: List[Dict[str, Any]] = []

    def create_group(self, creator_id: str):
        """Initializes a new MLS Group session."""
        self.epoch_tracker.reset(0)
        phone = Device(
            device_id=creator_id, owner_id="alice", name="Alice Phone",
            device_type=DeviceType.PRIMARY, ip_address="172.16.23.10"
        )
        self.devices[creator_id] = phone
        self.mls_event_log.append({
            "epoch": 0,
            "event_type": MLSEventType.CREATE_GROUP,
            "group_id": self.group_id,
            "creator": creator_id,
            "detail": "MLS GroupContext initialized at epoch 0"
        })

    def add_member(self, device_id: str, name: str, is_rogue: bool = False):
        """Processes an MLS Add proposal and updates group membership."""
        epoch = self.epoch_tracker.current_epoch
        dev = Device(
            device_id=device_id, owner_id="alice", name=name,
            device_type=DeviceType.LINKED,
            ip_address="185.220.101.5" if is_rogue else "172.16.23.20",
            network_type="VPN" if is_rogue else "WiFi",
            initial_epoch=epoch
        )
        self.devices[device_id] = dev
        self.mls_event_log.append({
            "epoch": epoch,
            "event_type": MLSEventType.ADD_MEMBER,
            "device_id": device_id,
            "detail": f"MLS Add Proposal committed for client {device_id}"
        })

    def process_mls_epoch(self, epoch_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Advances the MLS protocol epoch, records key update commits and messages,
        updates behavioral telemetry, and invokes the Dual-Trigger QTK decision engine.
        """
        self.epoch_tracker.increment_epoch()
        current_epoch = self.epoch_tracker.current_epoch

        # 1. Process MLS Events for this epoch
        for ev in epoch_events:
            dev_id = ev.get("device_id")
            ev_type = ev.get("event_type")
            dev = self.devices.get(dev_id)

            if ev_type == MLSEventType.KEY_UPDATE_COMMIT and dev:
                self.epoch_tracker.sync_device_key(dev)
                self.mls_event_log.append({
                    "epoch": current_epoch,
                    "event_type": MLSEventType.KEY_UPDATE_COMMIT,
                    "device_id": dev_id,
                    "detail": f"TreeKEM Direct Path KeyPackage committed at epoch {current_epoch}"
                })

            # Record telemetry snapshot
            if dev and not dev.is_quarantined:
                telemetry = {
                    "session_duration_sec": ev.get("session_duration_sec", 120.0),
                    "sync_frequency": ev.get("sync_frequency", 4.0),
                    "message_count_sent": ev.get("msg_count", 5),
                    "network_type": dev.network_type,
                    "network_ip": dev.ip_address,
                    "location_country": dev.country,
                    "active_timezone": dev.timezone,
                    "is_vpn": 1.0 if dev.network_type == "VPN" else 0.0,
                    "ip_changed": ev.get("ip_changed", 0.0),
                    "tz_changed": ev.get("tz_changed", 0.0)
                }
                dev.add_telemetry(telemetry)

        # 2. Behavioral Detection Inference
        for dev in self.devices.values():
            if not dev.is_quarantined and dev.telemetry_history:
                self.hmm.predict(dev)
                TrustScore.update(dev, dev.behavioral_risk, alpha=0.8)

        active_devs = [d for d in self.devices.values() if not d.is_quarantined and d.telemetry_history]
        if len(active_devs) >= 2:
            hists = [d.telemetry_history for d in active_devs]
            _, scores = self.graph_lstm.evaluate_devices(hists)
            for i, dev in enumerate(active_devs):
                if i < len(scores):
                    dev.graph_risk = scores[i]

        for dev in active_devs:
            self.fusion.predict(dev)

        # 3. Dual-Trigger QTK Decision
        quarantine_actions = []
        for dev in self.devices.values():
            if dev.is_quarantined:
                continue

            triggered, reason, detail = self.dual_trigger.dual_trigger_decision(dev, current_epoch)
            if triggered:
                dev.quarantine(current_epoch, reason.value)
                other_devs = [d for d in self.devices.values() if d != dev and not d.is_quarantined]
                shares = QuarantineManager.quarantine_device(dev, other_devs)
                action_info = {
                    "device_id": dev.device_id,
                    "reason": reason.value,
                    "detail": detail,
                    "shamir_shares_generated": len(shares.get("shares", {})),
                    "threshold": shares.get("threshold", 0)
                }
                quarantine_actions.append(action_info)
                self.mls_event_log.append({
                    "epoch": current_epoch,
                    "event_type": "QTK_QUARANTINE_INVOKED",
                    **action_info
                })

        return {
            "epoch": current_epoch,
            "quarantine_actions": quarantine_actions,
            "active_devices": [d.device_id for d in self.devices.values() if not d.is_quarantined]
        }

def run_proof_of_concept(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs an end-to-end MLS event lifecycle demonstration.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "raw"
        )
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================")
    print("Running Real MLS Protocol Lifecycle Proof-of-Concept")
    print("==================================================")

    poc = RealMLSProofOfConcept(group_id="mls_group_demo_icoco")
    poc.create_group(creator_id="alice_phone")
    poc.add_member(device_id="alice_laptop", name="Alice Laptop")
    poc.add_member(device_id="alice_standby_tablet", name="Alice Standby Tablet")
    poc.add_member(device_id="rogue_client", name="Rogue Terminal", is_rogue=True)

    timeline_results = []

    # Epoch 1 to 5: Normal operation with regular KeyUpdate Commits
    for ep in range(1, 6):
        events = [
            {"device_id": "alice_phone", "event_type": MLSEventType.KEY_UPDATE_COMMIT, "session_duration_sec": 120.0, "sync_frequency": 4.0, "msg_count": 8},
            {"device_id": "alice_laptop", "event_type": MLSEventType.KEY_UPDATE_COMMIT, "session_duration_sec": 150.0, "sync_frequency": 4.5, "msg_count": 10},
            {"device_id": "rogue_client", "event_type": MLSEventType.KEY_UPDATE_COMMIT, "session_duration_sec": 450.0, "sync_frequency": 16.0, "msg_count": 35}
            # Standby tablet is silent (no KeyUpdate Commit)
        ]
        res = poc.process_mls_epoch(events)
        timeline_results.append(res)
        print(f"Epoch {res['epoch']:2d}: Active Members = {len(res['active_devices'])} | Quarantines = {len(res['quarantine_actions'])}")

    # Epoch 6: Standby tablet exceeds delta_inact = 5 -> Quarantined by INACTIVITY
    # Rogue client exhibits high risk -> Quarantined by BEHAVIORAL
    events = [
        {"device_id": "alice_phone", "event_type": MLSEventType.KEY_UPDATE_COMMIT, "session_duration_sec": 110.0, "sync_frequency": 4.0, "msg_count": 6},
        {"device_id": "alice_laptop", "event_type": MLSEventType.KEY_UPDATE_COMMIT, "session_duration_sec": 130.0, "sync_frequency": 4.2, "msg_count": 7},
        {"device_id": "rogue_client", "event_type": MLSEventType.KEY_UPDATE_COMMIT, "session_duration_sec": 600.0, "sync_frequency": 22.0, "msg_count": 50, "ip_changed": 1.0}
    ]
    res = poc.process_mls_epoch(events)
    timeline_results.append(res)
    print(f"Epoch {res['epoch']:2d}: Active Members = {len(res['active_devices'])} | Quarantines = {len(res['quarantine_actions'])}")
    for q in res["quarantine_actions"]:
        print(f"  -> Quarantined: {q['device_id']} | Reason: {q['reason']} | Details: {q['detail']}")

    summary = {
        "group_id": poc.group_id,
        "total_epochs": poc.epoch_tracker.current_epoch,
        "devices": {d_id: str(d) for d_id, d in poc.devices.items()},
        "timeline_results": timeline_results,
        "mls_event_log": poc.mls_event_log
    }

    json_path = os.path.join(output_dir, "mls_poc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("==================================================")
    print(f"MLS Proof-of-Concept executed and saved to {json_path}")
    return summary

if __name__ == "__main__":
    run_proof_of_concept()
