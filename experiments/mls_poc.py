"""
experiments/mls_poc.py — Real MLS Protocol Lifecycle Proof-of-Concept

Demonstrates Dual-Trigger QTK behavioral detection integrated into a genuine
RFC 9420 MLS group session:

  - KeyPackage: real Ed25519-signed X25519 credentials
  - TreeKEM Commits: real HPKE path encryption + HKDF epoch secret derivation
  - Epoch tracking: actual MLS epoch number reported by MLSGroup.epoch
  - QTK Hook: DualTrigger.dual_trigger_decision() fires after every Commit
  - Application messages: AES-128-GCM encrypted with per-epoch, per-sender secrets

This replaces the former fake Python-dict simulation with cryptographically
correct MLS operations using the `cryptography` library (no Rust required).

References:
  RFC 9420 — The Messaging Layer Security (MLS) Protocol
  [Our Paper] §4 — QTK integration into MLS TreeKEM epochs
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real MLS implementation
from mls.key_package import KeyPackage
from mls.group import MLSGroup

# QTK behavioral detection pipeline
from simulator.device import Device, DeviceType, TrustState
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger, TriggerReason
from qtk.quarantine_state import QuarantineManager
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion


class RealMLSProofOfConcept:
    """
    End-to-end MLS + Dual-Trigger QTK integration.

    Each call to process_mls_epoch():
      1. Advances the MLS group via a real TreeKEM Commit
         (generates fresh path secrets, derives epoch secrets via HKDF)
      2. Injects behavioral telemetry into QTK device models
      3. Runs HMM, Graph-LSTM, TrustScore, RiskFusion inference
      4. Calls DualTrigger.dual_trigger_decision() to determine quarantine
      5. If triggered, calls QuarantineManager for Shamir secret sharing

    The MLS epoch number (group.epoch) is used as the QTK key_update_epoch,
    directly coupling protocol epoch progression to the inactivity trigger.
    """

    def __init__(self, group_id: str = "secure_mls_group_01"):
        self.group_id = group_id

        # QTK components
        self.epoch_tracker = EpochTracker()
        self.dual_trigger  = DualTrigger(delta_inact=5, theta_R=0.65)
        self.hmm           = HMMDetector()
        self.graph_lstm    = GraphLSTM(beta=0.8)
        self.fusion        = RiskFusion()
        self.devices: Dict[str, Device] = {}

        # MLS group (initialized in create_group)
        self.mls_group: Optional[MLSGroup] = None

        # Event log (combines MLS Commit records + QTK quarantine decisions)
        self.event_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Group Lifecycle
    # ------------------------------------------------------------------

    def create_group(self, creator_id: str) -> Dict[str, Any]:
        """
        Initialize a real MLS group with the creator as the first member.

        Generates a fresh KeyPackage (X25519 + Ed25519) for the creator,
        initializes the MLSGroup at epoch 0.
        """
        creator_kp = KeyPackage.generate(creator_id)
        assert creator_kp.verify(), "Creator KeyPackage self-signature invalid"

        self.mls_group = MLSGroup(group_id=self.group_id, creator_key_package=creator_kp)

        # QTK: register creator device
        phone = Device(
            device_id=creator_id,
            owner_id="alice",
            name=creator_id,
            device_type=DeviceType.PRIMARY,
            ip_address="172.16.23.10"
        )
        self.devices[creator_id] = phone
        self.epoch_tracker.reset(0)

        event = {
            "mls_epoch": self.mls_group.epoch,
            "event_type": "MLS_CREATE_GROUP",
            "group_id": self.group_id,
            "creator": creator_id,
            "tree_hash": self.mls_group.tree_hash.hex(),
            "creator_kp_ref": creator_kp.ref_hex,
        }
        self.event_log.append(event)
        return event

    def add_member(
        self,
        device_id: str,
        name: str,
        is_rogue: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a real KeyPackage for `device_id` and commit an Add to the group.

        - Verifies KeyPackage self-signature before admission
        - Commits a TreeKEM Add → real epoch advancement + HKDF secret derivation
        - Registers a QTK Device with realistic or rogue initial telemetry
        """
        assert self.mls_group is not None, "Group not created yet"

        # Generate real KeyPackage
        kp = KeyPackage.generate(device_id)
        assert kp.verify(), f"KeyPackage for {device_id} failed self-verification"

        # MLS: Commit Add → real epoch advance
        commit_record = self.mls_group.add(kp)

        # QTK: register device
        ip      = "185.220.101.5" if is_rogue else "172.16.23.20"
        network = "VPN"  if is_rogue else "WiFi"
        dev = Device(
            device_id=device_id,
            owner_id="alice",
            name=name,
            device_type=DeviceType.LINKED,
            ip_address=ip,
            network_type=network,
            initial_epoch=self.mls_group.epoch
        )
        self.devices[device_id] = dev
        self.epoch_tracker.sync_device_key(dev)

        event = {
            "mls_epoch": self.mls_group.epoch,
            "event_type": "MLS_ADD_MEMBER",
            "device_id": device_id,
            "is_rogue": is_rogue,
            "kp_ref": kp.ref_hex,
            "tree_hash": self.mls_group.tree_hash.hex(),
            "commit_secret": commit_record["commit_secret"],
            "epoch_secret":  commit_record["epoch_secret"],
        }
        self.event_log.append(event)
        return event

    # ------------------------------------------------------------------
    # Epoch Processing (MLS Commit + QTK Detection)
    # ------------------------------------------------------------------

    def process_mls_epoch(
        self,
        epoch_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Advance the MLS protocol epoch and run QTK behavioral detection.

        For each active device:
          1. Commit an MLS Update (fresh KeyPackage) if the device sent a
             KEY_UPDATE_COMMIT event — real HPKE path encryption, real epoch advance
          2. Inject behavioral telemetry into QTK device model
          3. Optionally send an encrypted ApplicationMessage to verify E2EE

        Then run the full QTK pipeline:
          HMM → Graph-LSTM → TrustScore → RiskFusion → DualTrigger

        Returns a summary including MLS epoch info and QTK quarantine decisions.
        """
        assert self.mls_group is not None, "Group not created"

        self.epoch_tracker.increment_epoch()
        qtk_epoch = self.epoch_tracker.current_epoch

        # ----------------------------------------------------------------
        # Phase 1: MLS Updates — one Commit per device that sent KEY_UPDATE
        # ----------------------------------------------------------------
        mls_commits: List[Dict[str, Any]] = []

        for ev in epoch_events:
            dev_id  = ev.get("device_id")
            ev_type = ev.get("event_type", "")
            dev     = self.devices.get(dev_id)

            if not dev or dev.is_quarantined:
                continue

            if ev_type == "MLS_KEY_UPDATE_COMMIT":
                # Real MLS Update Commit: generates new X25519 keypair for this
                # member's leaf, ratchets path secret up to root, advances epoch
                if dev_id in self.mls_group.active_members:
                    try:
                        commit_record = self.mls_group.update(dev_id)
                        # Sync QTK key epoch from real MLS epoch
                        self.epoch_tracker.sync_device_key(dev)
                        mls_commits.append({
                            "device_id":     dev_id,
                            "mls_epoch":     commit_record["epoch"],
                            "tree_hash":     commit_record["tree_hash"],
                            "commit_secret": commit_record["commit_secret"],
                        })
                    except Exception as exc:
                        mls_commits.append({"device_id": dev_id, "error": str(exc)})

            # Inject behavioral telemetry into QTK device
            telemetry = {
                "session_duration_sec": ev.get("session_duration_sec", 120.0),
                "sync_frequency":       ev.get("sync_frequency", 4.0),
                "message_count_sent":   ev.get("msg_count", 5),
                "network_type":         dev.network_type,
                "network_ip":           dev.ip_address,
                "location_country":     dev.country,
                "active_timezone":      dev.timezone,
                "is_vpn":               1.0 if dev.network_type == "VPN" else 0.0,
                "ip_changed":           ev.get("ip_changed", 0.0),
                "tz_changed":           ev.get("tz_changed", 0.0),
            }
            dev.add_telemetry(telemetry)

            # Optional: encrypt/decrypt an ApplicationMessage to verify E2EE
            if ev.get("send_app_message") and dev_id in self.mls_group.active_members:
                try:
                    payload = ev.get("app_message_plaintext", b"Hello MLS").encode() \
                        if isinstance(ev.get("app_message_plaintext", b""), str) \
                        else ev.get("app_message_plaintext", b"Hello MLS")
                    aad, ct = self.mls_group.send_application_message(dev_id, payload)
                    recovered = self.mls_group.receive_application_message(dev_id, aad, ct)
                    assert recovered == payload, "E2EE round-trip mismatch"
                    ev["_e2ee_verified"] = True
                except Exception as exc:
                    ev["_e2ee_error"] = str(exc)

        # ----------------------------------------------------------------
        # Phase 2: QTK Behavioral Inference
        # ----------------------------------------------------------------
        for dev in self.devices.values():
            if not dev.is_quarantined and dev.telemetry_history:
                self.hmm.predict(dev)
                TrustScore.update(dev, dev.behavioral_risk, alpha=0.8)

        active_devs = [
            d for d in self.devices.values()
            if not d.is_quarantined and d.telemetry_history
        ]
        if len(active_devs) >= 2:
            hists = [d.telemetry_history for d in active_devs]
            _, scores = self.graph_lstm.evaluate_devices(hists)
            for i, dev in enumerate(active_devs):
                if i < len(scores):
                    dev.graph_risk = scores[i]

        for dev in active_devs:
            self.fusion.predict(dev)

        # ----------------------------------------------------------------
        # Phase 3: Dual-Trigger QTK Decision
        # ----------------------------------------------------------------
        quarantine_actions = []
        for dev in self.devices.values():
            if dev.is_quarantined:
                continue

            triggered, reason, detail = self.dual_trigger.dual_trigger_decision(
                dev, qtk_epoch
            )
            if triggered:
                dev.quarantine(qtk_epoch, reason.value)
                other_devs = [
                    d for d in self.devices.values()
                    if d != dev and not d.is_quarantined
                ]
                shares = QuarantineManager.quarantine_device(dev, other_devs)

                action = {
                    "device_id":              dev.device_id,
                    "reason":                 reason.value,
                    "detail":                 detail,
                    "shamir_shares_generated": len(shares.get("shares", {})),
                    "threshold":              shares.get("threshold", 0),
                    "mls_epoch":              self.mls_group.epoch,
                    "qtk_epoch":              qtk_epoch,
                }
                quarantine_actions.append(action)

                # MLS Remove: quarantined member is removed from the group
                if dev.device_id in self.mls_group.active_members:
                    try:
                        remove_commit = self.mls_group.remove(dev.device_id)
                        action["mls_remove_commit"] = {
                            "new_epoch":  remove_commit["epoch"],
                            "tree_hash":  remove_commit["tree_hash"],
                        }
                    except Exception as exc:
                        action["mls_remove_error"] = str(exc)

                self.event_log.append({
                    "mls_epoch": self.mls_group.epoch,
                    "qtk_epoch": qtk_epoch,
                    "event_type": "QTK_QUARANTINE_INVOKED",
                    **action
                })

        return {
            "qtk_epoch":          qtk_epoch,
            "mls_epoch":          self.mls_group.epoch,
            "tree_hash":          self.mls_group.tree_hash.hex(),
            "mls_commits":        mls_commits,
            "quarantine_actions": quarantine_actions,
            "active_mls_members": self.mls_group.active_members,
            "active_devices":     [
                d.device_id for d in self.devices.values()
                if not d.is_quarantined
            ],
        }


# ---------------------------------------------------------------------------
# Proof-of-Concept Runner
# ---------------------------------------------------------------------------

def run_proof_of_concept(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs an end-to-end MLS + QTK proof-of-concept:

    Epoch 0:    Create group (Alice Phone)
    Add:        Alice Laptop  → real KeyPackage, real MLS Add Commit
    Add:        Alice Tablet  → starts as silent (no KEY_UPDATE_COMMIT)
    Add:        Rogue Client  → real KeyPackage, high anomaly telemetry

    Epochs 1-5: Normal operation — all legitimate members commit key updates
                Rogue exhibits: long sessions, high sync rate, IP changes

    Epoch 6:    Alice Tablet exceeds delta_inact=5 → INACTIVITY quarantine
                Rogue Client triggers behavioral threshold → BEHAVIORAL quarantine
                Both are MLS-Removed from the group after quarantine
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "raw"
        )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Real MLS Protocol Lifecycle + QTK Proof-of-Concept")
    print("Using: RFC 9420 TreeKEM (X25519 HPKE + Ed25519 Credentials)")
    print("=" * 60)

    poc = RealMLSProofOfConcept(group_id="mls_group_demo_icoco_2026")

    # ── Epoch 0: Create group ──────────────────────────────────────────
    create_ev = poc.create_group(creator_id="alice_phone")
    print(f"\n[Epoch 0] Group created")
    print(f"  Creator:    alice_phone")
    print(f"  Tree Hash:  {create_ev['tree_hash'][:16]}...")

    # ── Add members (each generates real KeyPackage + MLS Commit) ─────
    for dev_id, name, rogue in [
        ("alice_laptop",         "Alice Laptop",          False),
        ("alice_standby_tablet", "Alice Standby Tablet",  False),
        ("rogue_client",         "Rogue Terminal",        True),
    ]:
        ev = poc.add_member(device_id=dev_id, name=name, is_rogue=rogue)
        tag = " [ROGUE]" if rogue else ""
        print(f"  Added{tag}: {dev_id} → MLS epoch {ev['mls_epoch']} "
              f"| tree_hash={ev['tree_hash'][:16]}...")

    print(f"\n  Initial MLS members: {poc.mls_group.active_members}")
    print(f"  MLS epoch after adds: {poc.mls_group.epoch}")

    # ── Epochs 1-5: Normal operation ──────────────────────────────────
    print("\n── Epochs 1–5: Normal operation ──")
    timeline_results = []

    for ep in range(1, 6):
        events = [
            {   # Alice Phone: regular key update + app message
                "device_id": "alice_phone",
                "event_type": "MLS_KEY_UPDATE_COMMIT",
                "session_duration_sec": 120.0,
                "sync_frequency": 4.0,
                "msg_count": 8,
                "send_app_message": True,
                "app_message_plaintext": f"Hello from alice_phone at epoch {ep}",
            },
            {   # Alice Laptop: regular key update
                "device_id": "alice_laptop",
                "event_type": "MLS_KEY_UPDATE_COMMIT",
                "session_duration_sec": 150.0,
                "sync_frequency": 4.5,
                "msg_count": 10,
            },
            # Alice Standby Tablet: intentionally SILENT (no KEY_UPDATE_COMMIT)
            # Rogue Client: high anomaly but does commit (to track via HMM)
            {
                "device_id": "rogue_client",
                "event_type": "MLS_KEY_UPDATE_COMMIT",
                "session_duration_sec": 450.0,   # unusually long
                "sync_frequency": 16.0,          # very high
                "msg_count": 35,                 # abnormal burst
            },
        ]

        res = poc.process_mls_epoch(events)
        timeline_results.append(res)

        print(f"  [QTK ep {res['qtk_epoch']} / MLS ep {res['mls_epoch']}] "
              f"Active MLS: {res['active_mls_members']} "
              f"| Quarantines: {len(res['quarantine_actions'])}")

    # ── Epoch 6: Trigger quarantine ────────────────────────────────────
    print("\n── Epoch 6: Inactivity + Behavioral triggers ──")
    events_ep6 = [
        {
            "device_id": "alice_phone",
            "event_type": "MLS_KEY_UPDATE_COMMIT",
            "session_duration_sec": 110.0,
            "sync_frequency": 4.0,
            "msg_count": 6,
        },
        {
            "device_id": "alice_laptop",
            "event_type": "MLS_KEY_UPDATE_COMMIT",
            "session_duration_sec": 130.0,
            "sync_frequency": 4.2,
            "msg_count": 7,
        },
        # Rogue: extreme anomaly burst + IP change
        {
            "device_id": "rogue_client",
            "event_type": "MLS_KEY_UPDATE_COMMIT",
            "session_duration_sec": 600.0,
            "sync_frequency": 22.0,
            "msg_count": 50,
            "ip_changed": 1.0,
        },
        # Standby tablet: still silent (epoch_gap = 6 >= delta_inact=5)
    ]

    res6 = poc.process_mls_epoch(events_ep6)
    timeline_results.append(res6)

    print(f"  [QTK ep {res6['qtk_epoch']} / MLS ep {res6['mls_epoch']}] "
          f"Active MLS: {res6['active_mls_members']} "
          f"| Quarantines: {len(res6['quarantine_actions'])}")

    for q in res6["quarantine_actions"]:
        print(f"  → Quarantined: {q['device_id']} | Reason: {q['reason']}")
        if "mls_remove_commit" in q:
            rc = q["mls_remove_commit"]
            print(f"    MLS Remove Commit: epoch={rc['new_epoch']} "
                  f"tree_hash={rc['tree_hash'][:16]}...")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n── Final State ──")
    print(f"  MLS epoch:   {poc.mls_group.epoch}")
    print(f"  Tree hash:   {poc.mls_group.tree_hash.hex()[:24]}...")
    print(f"  MLS members: {poc.mls_group.active_members}")
    print(f"  QTK epoch:   {poc.epoch_tracker.current_epoch}")
    print(f"  Quarantined: "
          f"{[d.device_id for d in poc.devices.values() if d.is_quarantined]}")

    # ── Persist results ──────────────────────────────────────────────────
    summary = {
        "group_id":         poc.group_id,
        "final_mls_epoch":  poc.mls_group.epoch,
        "final_tree_hash":  poc.mls_group.tree_hash.hex(),
        "final_members":    poc.mls_group.active_members,
        "qtk_epoch":        poc.epoch_tracker.current_epoch,
        "devices": {
            d_id: {
                "quarantined": d.is_quarantined,
                "trust_score": d.trust_score,
                "behavioral_risk": d.behavioral_risk,
            }
            for d_id, d in poc.devices.items()
        },
        "timeline_results": timeline_results,
        "mls_event_log":    poc.mls_group.event_log,
        "qtk_event_log":    poc.event_log,
    }

    json_path = os.path.join(output_dir, "mls_poc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, default=str)

    print(f"\n  Results saved to: {json_path}")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    run_proof_of_concept()
