"""
experiments/mls_poc_openmls.py — Real OpenMLS + Dual-Trigger QTK PoC

Demonstrates that the existing QTK behavioral quarantine policy can operate
alongside a REAL OpenMLS 0.9.0 MLS group without modifying any MLS
cryptographic primitives.

ARCHITECTURE:
  Python behavioral pipeline ← JSON → Rust/OpenMLS subprocess
  [HMM][RiskFusion][DualTrigger]   [MlsGroup::new/add/remove]

BENCHMARK ISOLATION:
  This PoC writes ONLY to:
    results/raw/mls_poc.json
    results/tables/mls_poc_summary.tex
    results/tables/mls_poc_summary.csv
  The frozen benchmark (Tables 1–6, theta_R, seeds) is NOT touched.

SECURITY BOUNDARY:
  Python receives ONLY non-secret telemetry (epoch, message_count,
  event_type, ciphertext_size, simulated behavioral metadata).
  All MLS cryptographic state is owned by the Rust process.

TELEMETRY LIMITATIONS (§16):
  The following fields are NOT natively available from OpenMLS:
    network_type, network_ip, location_country, active_timezone
  These are set to placeholder values and documented as unavailable.
  The following fields are SIMULATED (injected by PoC scenario):
    session_duration_sec, sync_frequency, is_vpn, ip_changed, tz_changed
  These represent behavioral metadata that would come from device telemetry
  in a real deployment, not from MLS protocol internals.
"""

import os
import sys
import json
import time
import hashlib
import datetime
from typing import Any, Dict, List, Optional

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real OpenMLS bridge
from mls.openmls_bridge import (
    ProcessBridge,
    TelemetryAdapter,
    QuarantineInterface,
)

# Existing behavioral pipeline (unchanged)
from simulator.device import Device, DeviceType
from qtk.epoch_tracker import EpochTracker
from qtk.dual_trigger import DualTrigger, TriggerReason
from qtk.quarantine_state import QuarantineManager
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

# ── Configuration ─────────────────────────────────────────────────────────────
MEMBERS = ["Alice", "Bob", "Charlie", "Dave"]
ROGUE   = "Dave"
GROUP_ID = "qtk_openmls_poc_2026"
THETA_R  = 0.65   # Behavioral threshold (same as trained model)
DELTA_INACT = 5   # Inactivity threshold (same as benchmark)

NORMAL_EPOCHS   = 8   # Epochs of normal operation
ROGUE_START     = 4   # Epoch at which Dave begins rogue behavior

# ── Dave's scenario (active rogue — inactivity must NOT fire) ─────────────────
# Dave sends messages every epoch (so inactivity_age < delta_inact)
# but with anomalous behavioral metadata (high sync rate, VPN, IP changes)
DAVE_NORMAL_META = {
    "is_vpn": 0.0,
    "ip_changed": 0.0,
    "tz_changed": 0.0,
    "sync_frequency": 4.2,
    "session_duration_sec": 130.0,
    "time_since_last_activity_secs": 120.0,
}
DAVE_ROGUE_META = {
    "is_vpn": 1.0,
    "ip_changed": 1.0,
    "tz_changed": 0.5,
    "sync_frequency": 22.0,
    "session_duration_sec": 580.0,
    "time_since_last_activity_secs": 45.0,
}

# ── Legitimate member metadata (normal) ───────────────────────────────────────
LEGIT_META = {
    "is_vpn": 0.0,
    "ip_changed": 0.0,
    "tz_changed": 0.0,
    "sync_frequency": 4.0,
    "session_duration_sec": 120.0,
    "time_since_last_activity_secs": 120.0,
}


def run_poc(binary_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the full OpenMLS + QTK proof-of-concept.

    Returns a dict with all result fields for JSON serialization.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir  = os.path.join(project_root, "results", "raw")
    tables_dir   = os.path.join(project_root, "results", "tables")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)

    print("=" * 60)
    print("OpenMLS Proof-of-Concept Integration")
    print("Dual-Trigger QTK + Real OpenMLS 0.9.0")
    print("=" * 60)

    # ── QTK pipeline setup ────────────────────────────────────────────────────
    epoch_tracker = EpochTracker()
    dual_trigger  = DualTrigger(delta_inact=DELTA_INACT, theta_R=THETA_R)
    hmm           = HMMDetector()
    graph_lstm    = GraphLSTM(beta=0.8)
    fusion        = RiskFusion()

    qtk_devices: Dict[str, Device] = {}
    for identity in MEMBERS:
        d = Device(
            device_id=identity,
            owner_id="alice",
            name=identity,
            device_type=DeviceType.PRIMARY if identity == "Alice" else DeviceType.LINKED,
            ip_address="172.16.10.1",
        )
        qtk_devices[identity] = d

    # ── Result tracking ────────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "group_id_hash": hashlib.sha256(GROUP_ID.encode()).hexdigest()[:16],
        "initial_members": list(MEMBERS),
        "rogue_member": ROGUE,
        "theta_R": THETA_R,
        "delta_inact": DELTA_INACT,
        "attack_start_epoch": ROGUE_START,
        "risk_trigger_epoch": None,
        "trigger_reason": None,
        "risk_score": None,
        "removal_proposal_epoch": None,
        "commit_epoch": None,
        "post_commit_epoch": None,
        "remaining_members": None,
        "rogue_removed": False,
        "inactivity_triggered": False,
        "behavioral_triggered": False,
        "epoch_progression": [],
        "mls_events": [],
        "telemetry_boundary": {
            "exposed_to_python": list(TelemetryAdapter.__doc__ and [
                "epoch", "member_id", "event_type", "message_size_bytes",
                "message_count", "commit_count", "time_since_last_activity_secs",
                "simulated_is_vpn", "simulated_ip_changed", "simulated_tz_changed",
                "simulated_sync_frequency", "simulated_session_duration_sec"
            ] or []),
            "never_exposed_to_python": [
                "epoch_secret", "init_secret", "joiner_secret",
                "path_secret", "sender_data_secret", "encryption_secret",
                "exporter_secret", "private_signing_key", "hpke_key",
                "decrypted_application_content",
            ],
            "unavailable_from_mls": [
                "network_type", "network_ip", "location_country", "active_timezone"
            ],
            "simulated_behavioral_metadata": [
                "session_duration_sec", "sync_frequency",
                "is_vpn", "ip_changed", "tz_changed"
            ],
        },
        "openssl_version": None,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # ── Start Rust subprocess ─────────────────────────────────────────────────
    with ProcessBridge(binary_path) as bridge:
        quarantine_iface = QuarantineInterface(bridge)

        # ── Create real MLS group ─────────────────────────────────────────────
        print(f"\n[Setup] Creating real MLS group: {GROUP_ID}")
        resp = bridge.send_command("create_group", {
            "creator_id": "Alice",
            "group_id": GROUP_ID,
        })
        evt = TelemetryAdapter.extract(resp["event"])
        result["mls_events"].append(evt)
        epoch_after_create = resp["current_epoch"]
        print(f"  Group created. MLS epoch: {epoch_after_create}")
        print(f"  Members: {resp['current_members']}")

        # ── Add Bob, Charlie, Dave ─────────────────────────────────────────────
        for identity in ["Bob", "Charlie", "Dave"]:
            resp = bridge.send_command("add_member", {"identity": identity})
            evt = TelemetryAdapter.extract(resp["event"])
            result["mls_events"].append(evt)
            print(f"  Added {identity}: MLS epoch={resp['current_epoch']} "
                  f"members={resp['current_members']}")
            epoch_tracker.sync_device_key(qtk_devices[identity])

        print(f"\n  All members added. MLS epoch: {bridge.send_command('get_epoch')['current_epoch']}")
        print(f"  Group: {bridge.send_command('get_members')['current_members']}")

        # ── Main epoch loop ────────────────────────────────────────────────────
        print(f"\n── Epochs 1–{NORMAL_EPOCHS}: Simulation ──")
        quarantined = set()

        for ep in range(1, NORMAL_EPOCHS + 1):
            epoch_tracker.increment_epoch()
            qtk_epoch = epoch_tracker.current_epoch
            print(f"\n[QTK ep {qtk_epoch}]", end="")

            epoch_events = []

            # Each member sends an application message
            for identity in MEMBERS:
                if identity in quarantined:
                    continue

                is_rogue_epoch = (identity == ROGUE and ep >= ROGUE_START)
                meta = DAVE_ROGUE_META if is_rogue_epoch else (
                    DAVE_NORMAL_META if identity == ROGUE else LEGIT_META
                )

                resp = bridge.send_command("send_message", {
                    "sender_id": identity,
                    "metadata": meta,
                })
                safe_evt = TelemetryAdapter.extract(resp["event"])
                qtk_tel  = TelemetryAdapter.to_qtk_telemetry(safe_evt)
                epoch_events.append((identity, safe_evt, qtk_tel))

                # Inject telemetry into QTK device
                dev = qtk_devices[identity]
                dev.add_telemetry({
                    "session_duration_sec": qtk_tel["session_duration_sec"],
                    "sync_frequency":       qtk_tel["sync_frequency"],
                    "message_count_sent":   qtk_tel["message_count_sent"],
                    "network_type":         qtk_tel["network_type"],
                    "network_ip":           qtk_tel["network_ip"],
                    "location_country":     qtk_tel["location_country"],
                    "active_timezone":      qtk_tel["active_timezone"],
                    "is_vpn":               qtk_tel["is_vpn"],
                    "ip_changed":           qtk_tel["ip_changed"],
                    "tz_changed":           qtk_tel["tz_changed"],
                })
                epoch_tracker.sync_device_key(dev)
                result["mls_events"].append(safe_evt)

            mls_epoch = resp["current_epoch"]
            print(f" / MLS ep {mls_epoch}", end="")

            # ── QTK behavioral inference ───────────────────────────────────────
            active_devs = [
                d for d in qtk_devices.values()
                if not d.is_quarantined and d.telemetry_history
            ]

            for dev in active_devs:
                hmm.predict(dev)
                TrustScore.update(dev, dev.behavioral_risk, alpha=0.8)

            if len(active_devs) >= 2:
                hists = [d.telemetry_history for d in active_devs]
                _, scores = graph_lstm.evaluate_devices(hists)
                for i, dev in enumerate(active_devs):
                    if i < len(scores):
                        dev.graph_risk = scores[i]

            for dev in active_devs:
                fusion.predict(dev)

            # ── QTK Dual-Trigger decision ──────────────────────────────────────
            epoch_record = {
                "qtk_epoch":  qtk_epoch,
                "mls_epoch":  mls_epoch,
                "decisions":  [],
            }

            for dev in list(qtk_devices.values()):
                if dev.is_quarantined or dev.device_id in quarantined:
                    continue

                triggered, reason, detail = dual_trigger.dual_trigger_decision(
                    dev, qtk_epoch
                )

                dev_record = {
                    "member_id":       dev.device_id,
                    "behavioral_risk": round(float(dev.behavioral_risk), 4),
                    "trust_score":     round(float(dev.trust_score), 4),
                    "triggered":       triggered,
                    "reason":          reason.value if triggered else None,
                }
                epoch_record["decisions"].append(dev_record)

                if triggered:
                    print(f"\n  !! TRIGGER: {dev.device_id} | {reason.value} "
                          f"| risk={dev.behavioral_risk:.3f}")

                    if reason == TriggerReason.INACTIVITY:
                        result["inactivity_triggered"] = True

                    if reason == TriggerReason.BEHAVIORAL:
                        result["behavioral_triggered"]    = True
                        result["risk_trigger_epoch"]      = qtk_epoch
                        result["trigger_reason"]          = reason.value
                        result["risk_score"]              = round(float(dev.behavioral_risk), 4)
                        result["removal_proposal_epoch"]  = qtk_epoch

                    # Mark QTK quarantine
                    dev.quarantine(qtk_epoch, reason.value)
                    quarantined.add(dev.device_id)

                    # ── Quarantine action via authorized interface ──────────────
                    # Python policy requests removal.
                    # Rust performs the actual MLS Remove Commit.
                    print(f"  -> Requesting MLS removal of {dev.device_id} via QuarantineInterface")
                    removal_result = quarantine_iface.request_removal(dev.device_id)

                    result["commit_epoch"]      = removal_result["post_commit_epoch"]
                    result["post_commit_epoch"] = removal_result["post_commit_epoch"]
                    result["remaining_members"] = removal_result["remaining_members"]
                    result["rogue_removed"]     = (ROGUE not in removal_result["remaining_members"])

                    print(f"  -> MLS Remove Commit done. New epoch: {result['post_commit_epoch']}")
                    print(f"  -> Remaining members: {result['remaining_members']}")

                    # Log removal event
                    result["mls_events"].append(removal_result["removal_event"])

            # Status
            risks = {
                d.device_id: round(float(d.behavioral_risk), 3)
                for d in qtk_devices.values()
                if not d.is_quarantined
            }
            print(f" | risks={risks}", end="")
            result["epoch_progression"].append(epoch_record)

        # ── Final state ────────────────────────────────────────────────────────
        final_members_resp = bridge.send_command("get_members")
        final_members = final_members_resp.get("current_members", [])
        final_epoch   = final_members_resp.get("current_epoch", 0)

        if result["remaining_members"] is None:
            result["remaining_members"] = final_members
            result["post_commit_epoch"] = final_epoch

        result["final_mls_epoch"] = final_epoch
        result["final_mls_members"] = final_members

    # ── Write outputs ──────────────────────────────────────────────────────────
    print(f"\n\n── Final State ──")
    print(f"  MLS epoch:        {result.get('final_mls_epoch')}")
    print(f"  MLS members:      {result.get('final_mls_members')}")
    print(f"  QTK epoch:        {epoch_tracker.current_epoch}")
    print(f"  Behavioral risk:  {result.get('risk_score')}")
    print(f"  Trigger:          {result.get('trigger_reason')}")
    print(f"  Rogue removed:    {result.get('rogue_removed')}")
    print(f"  Inactivity fired: {result.get('inactivity_triggered')}")

    _write_json(results_dir, result)
    _write_csv(tables_dir, result)
    _write_latex(tables_dir, result)

    print("\n=" * 60)
    return result


# ── Output writers ─────────────────────────────────────────────────────────────

def _write_json(results_dir: str, result: Dict[str, Any]):
    path = os.path.join(results_dir, "mls_poc.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, default=str)
    print(f"\n  JSON: {path}")


def _write_csv(tables_dir: str, result: Dict[str, Any]):
    path = os.path.join(tables_dir, "mls_poc_summary.csv")
    rows = [
        ["field", "value"],
        ["group_id_hash",           result.get("group_id_hash", "")],
        ["initial_members",         ", ".join(result.get("initial_members", []))],
        ["rogue_member",            result.get("rogue_member", "")],
        ["theta_R",                 result.get("theta_R", "")],
        ["delta_inact",             result.get("delta_inact", "")],
        ["attack_start_epoch",      result.get("attack_start_epoch", "")],
        ["risk_trigger_epoch",      result.get("risk_trigger_epoch", "")],
        ["trigger_reason",          result.get("trigger_reason", "")],
        ["risk_score",              result.get("risk_score", "")],
        ["removal_proposal_epoch",  result.get("removal_proposal_epoch", "")],
        ["commit_epoch",            result.get("commit_epoch", "")],
        ["post_commit_epoch",       result.get("post_commit_epoch", "")],
        ["remaining_members",       ", ".join(result.get("remaining_members") or [])],
        ["rogue_removed",           result.get("rogue_removed", "")],
        ["inactivity_triggered",    result.get("inactivity_triggered", "")],
        ["behavioral_triggered",    result.get("behavioral_triggered", "")],
        ["final_mls_epoch",         result.get("final_mls_epoch", "")],
    ]
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"  CSV: {path}")


def _write_latex(tables_dir: str, result: Dict[str, Any]):
    path = os.path.join(tables_dir, "mls_poc_summary.tex")
    remaining = ", ".join(result.get("remaining_members") or [])
    initial   = ", ".join(result.get("initial_members") or [])
    rogue_str = "YES" if result.get("rogue_removed") else "NO"
    inact_str = "YES" if result.get("inactivity_triggered") else "NO"
    behav_str = "YES" if result.get("behavioral_triggered") else "NO"

    tex = rf"""\begin{{table}}[h]
\centering
\caption{{OpenMLS Proof-of-Concept: Real MLS + Dual-Trigger QTK Integration}}
\label{{tab:mls_poc}}
\begin{{tabular}}{{ll}}
\hline
\textbf{{Field}} & \textbf{{Value}} \\ \hline
Initial members & {initial} \\
Rogue member & {result.get('rogue_member', '')} \\
Attack start (epoch) & {result.get('attack_start_epoch', '')} \\
$\theta_R$ & {result.get('theta_R', '')} \\
$\delta_{{inact}}$ & {result.get('delta_inact', '')} \\
Risk trigger epoch & {result.get('risk_trigger_epoch', 'N/A')} \\
Trigger reason & {result.get('trigger_reason', 'N/A')} \\
Risk score & {result.get('risk_score', 'N/A')} \\
Removal commit epoch & {result.get('commit_epoch', 'N/A')} \\
Post-commit MLS epoch & {result.get('post_commit_epoch', 'N/A')} \\
Remaining members & {remaining} \\
Rogue removed & {rogue_str} \\
Inactivity triggered & {inact_str} \\
Behavioral triggered & {behav_str} \\ \hline
\end{{tabular}}
\end{{table}}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  LaTeX: {path}")


if __name__ == "__main__":
    binary = sys.argv[1] if len(sys.argv) > 1 else None
    run_poc(binary)
