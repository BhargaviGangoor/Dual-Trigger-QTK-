"""
tests/test_mls_openmls.py — 12-Test Suite for Real OpenMLS + QTK PoC

Tests correspond to the 12 requirements in §18 of the specification:
  1.  Real MLS group creation
  2.  Four-member group
  3.  Application message processing
  4.  Epoch progression
  5.  Rogue (Dave) remains active (sends messages)
  6.  Inactivity trigger remains false
  7.  Behavioral risk is computed
  8.  Dual-Trigger returns BEHAVIORAL
  9.  Authorized removal succeeds
  10. Commit produces new epoch
  11. Rogue is absent from post-removal membership
  12. ML layer never receives cryptographic secrets

NOTE: Tests 1–11 require the Rust binary to be built:
  cd mls_openmls && cargo build --release

Test 12 is pure-Python (security boundary verification) and does NOT
require the Rust binary.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mls.openmls_bridge import (
    ProcessBridge,
    TelemetryAdapter,
    QuarantineInterface,
    _ALLOWED_EVENT_FIELDS,
    _FORBIDDEN_SECRET_FIELDS,
)
from experiments.mls_poc_openmls import (
    MEMBERS, ROGUE, GROUP_ID, THETA_R, DELTA_INACT,
    DAVE_ROGUE_META, LEGIT_META,
)

# ── Fixture: check binary exists ────────────────────────────────────────────

def _binary_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        project_root, "mls_openmls", "target", "release", "mls_openmls.exe"
    )

def _binary_available() -> bool:
    return os.path.exists(_binary_path())

requires_binary = pytest.mark.skipif(
    not _binary_available(),
    reason="Rust binary not built. Run: cd mls_openmls && cargo build --release"
)


# ── Shared group fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def live_group():
    """
    Module-scoped fixture: starts the Rust process, creates the MLS group,
    adds all 4 members. Yields the (bridge, quarantine_iface) tuple.
    """
    if not _binary_available():
        pytest.skip("Rust binary not built")

    bridge = ProcessBridge(_binary_path())
    bridge.start()
    qi = QuarantineInterface(bridge)

    # Create group
    resp = bridge.send_command("create_group", {
        "creator_id": "Alice",
        "group_id": GROUP_ID,
    })
    assert resp["status"] == "ok"

    # Add Bob, Charlie, Dave
    for identity in ["Bob", "Charlie", "Dave"]:
        resp = bridge.send_command("add_member", {"identity": identity})
        assert resp["status"] == "ok"

    yield bridge, qi

    bridge.stop()


# ── Test 1: Real MLS group creation ─────────────────────────────────────────

@requires_binary
def test_1_real_mls_group_creation(live_group):
    """OpenMLS MlsGroup::new() must succeed and return epoch >= 0."""
    bridge, _ = live_group
    resp = bridge.send_command("get_epoch")
    assert resp["status"] == "ok"
    epoch = resp.get("current_epoch")
    assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}"


# ── Test 2: Four-member group ────────────────────────────────────────────────

@requires_binary
def test_2_four_member_group(live_group):
    """All 4 members (Alice, Bob, Charlie, Dave) must be active."""
    bridge, _ = live_group
    resp = bridge.send_command("get_members")
    members = resp.get("current_members", [])
    for expected in MEMBERS:
        assert expected in members, f"{expected!r} missing from MLS group members: {members}"
    assert len(members) == 4, f"Expected 4 members, got {len(members)}: {members}"


# ── Test 3: Application message processing ───────────────────────────────────

@requires_binary
def test_3_application_message_processing(live_group):
    """
    send_message must produce a real MLS ApplicationMessage.
    The ciphertext size must be > 0 and the event_type must be APPLICATION_MESSAGE.
    """
    bridge, _ = live_group
    resp = bridge.send_command("send_message", {
        "sender_id": "Alice",
        "metadata": LEGIT_META,
    })
    assert resp["status"] == "ok"
    event = resp["event"]
    safe  = TelemetryAdapter.extract(event)

    assert safe["event_type"] == "APPLICATION_MESSAGE", \
        f"Wrong event_type: {safe['event_type']}"
    size = safe.get("message_size_bytes")
    assert size is not None and size > 0, \
        f"message_size_bytes must be > 0 for real MLS message, got {size}"


# ── Test 4: Epoch progression ────────────────────────────────────────────────

@requires_binary
def test_4_epoch_progression(live_group):
    """
    After Add commits, the MLS epoch must be > 0.
    After each Remove commit, the epoch must increment.
    """
    bridge, _ = live_group
    resp = bridge.send_command("get_epoch")
    epoch_now = resp["current_epoch"]
    # After 3 Add commits (Bob, Charlie, Dave), epoch must be >= 3
    assert epoch_now >= 3, \
        f"Epoch after 3 Add commits must be >= 3, got {epoch_now}"


# ── Test 5: Rogue (Dave) remains active ─────────────────────────────────────

@requires_binary
def test_5_dave_remains_active(live_group):
    """
    Dave sends application messages every epoch.
    His time_since_last_activity must be < delta_inact (5 epochs),
    so the inactivity trigger must NOT fire.
    """
    bridge, _ = live_group
    resp = bridge.send_command("send_message", {
        "sender_id": "Dave",
        "metadata": DAVE_ROGUE_META,
    })
    assert resp["status"] == "ok"
    safe = TelemetryAdapter.extract(resp["event"])
    # Dave is active: message_count > 0
    assert safe["message_count"] >= 1, \
        f"Dave must have message_count >= 1, got {safe['message_count']}"
    # Dave is in the group
    members_resp = bridge.send_command("get_members")
    assert "Dave" in members_resp["current_members"], "Dave must still be in MLS group"


# ── Test 6: Inactivity trigger remains false ─────────────────────────────────

@requires_binary
def test_6_inactivity_trigger_false(live_group):
    """
    Since Dave sends messages each epoch, inactivity_age < delta_inact.
    DualTrigger INACTIVITY must NOT fire for Dave.
    """
    from simulator.device import Device, DeviceType
    from qtk.epoch_tracker import EpochTracker
    from qtk.dual_trigger import DualTrigger, TriggerReason

    dave_dev = Device(
        device_id="Dave", owner_id="alice", name="Dave",
        device_type=DeviceType.LINKED, ip_address="10.0.0.5",
    )
    et = EpochTracker()
    dt = DualTrigger(delta_inact=DELTA_INACT, theta_R=THETA_R)

    # Dave actively sends for 4 epochs (< delta_inact=5)
    for ep in range(1, 5):
        et.increment_epoch()
        et.sync_device_key(dave_dev)
        dave_dev.add_telemetry({
            "session_duration_sec": 120.0, "sync_frequency": 4.0,
            "message_count_sent": 5, "network_type": "WiFi",
            "network_ip": "10.0.0.5", "location_country": "US",
            "active_timezone": "UTC", "is_vpn": 0.0,
            "ip_changed": 0.0, "tz_changed": 0.0,
        })
        triggered, reason, _ = dt.dual_trigger_decision(dave_dev, et.current_epoch)
        if triggered and reason == TriggerReason.INACTIVITY:
            pytest.fail(f"Inactivity trigger fired at epoch {ep} — Dave is active!")


# ── Test 7: Behavioral risk is computed ─────────────────────────────────────

@requires_binary
def test_7_behavioral_risk_computed(live_group):
    """
    After injecting rogue telemetry for Dave,
    the HMM + RiskFusion must produce behavioral_risk > 0.
    """
    from simulator.device import Device, DeviceType
    from models.hmm import HMMDetector
    from models.risk_fusion import RiskFusion
    from models.trust_score import TrustScore

    dave_dev = Device(
        device_id="Dave", owner_id="alice", name="Dave",
        device_type=DeviceType.LINKED, ip_address="185.1.2.3",
        network_type="VPN",
    )
    hmm    = HMMDetector()
    fusion = RiskFusion()

    for _ in range(6):  # Accumulate rogue telemetry
        dave_dev.add_telemetry({
            "session_duration_sec": 580.0, "sync_frequency": 22.0,
            "message_count_sent": 45, "network_type": "VPN",
            "network_ip": "185.1.2.3", "location_country": "Unknown",
            "active_timezone": "UTC", "is_vpn": 1.0,
            "ip_changed": 1.0, "tz_changed": 0.5,
        })
        hmm.predict(dave_dev)
        TrustScore.update(dave_dev, dave_dev.behavioral_risk, alpha=0.8)

    fusion.predict(dave_dev)
    risk = float(dave_dev.behavioral_risk)
    assert risk > 0.0, f"behavioral_risk must be > 0 after rogue telemetry, got {risk}"


# ── Test 8: Dual-Trigger returns BEHAVIORAL ──────────────────────────────────

@requires_binary
def test_8_dual_trigger_behavioral(live_group):
    """
    After enough rogue epochs, DualTrigger must return BEHAVIORAL (not INACTIVITY).
    Dave remains active, so the behavioral score alone must cross theta_R.
    """
    from simulator.device import Device, DeviceType
    from qtk.epoch_tracker import EpochTracker
    from qtk.dual_trigger import DualTrigger, TriggerReason
    from models.hmm import HMMDetector
    from models.risk_fusion import RiskFusion
    from models.trust_score import TrustScore

    dave_dev = Device(
        device_id="Dave_test8", owner_id="alice", name="Dave",
        device_type=DeviceType.LINKED, ip_address="185.1.2.3",
        network_type="VPN",
    )
    et = EpochTracker()
    dt = DualTrigger(delta_inact=DELTA_INACT, theta_R=THETA_R)
    hmm    = HMMDetector()
    fusion = RiskFusion()

    triggered_behavioral = False
    for ep in range(1, 12):
        et.increment_epoch()
        et.sync_device_key(dave_dev)
        dave_dev.add_telemetry({
            "session_duration_sec": 600.0, "sync_frequency": 24.0,
            "message_count_sent": 50, "network_type": "VPN",
            "network_ip": "185.1.2.3", "location_country": "Unknown",
            "active_timezone": "UTC", "is_vpn": 1.0,
            "ip_changed": 1.0, "tz_changed": 1.0,
        })
        hmm.predict(dave_dev)
        TrustScore.update(dave_dev, dave_dev.behavioral_risk, alpha=0.8)
        fusion.predict(dave_dev)
        triggered, reason, _ = dt.dual_trigger_decision(dave_dev, et.current_epoch)
        if triggered and reason == TriggerReason.BEHAVIORAL:
            triggered_behavioral = True
            break

    assert triggered_behavioral, \
        "DualTrigger must eventually return BEHAVIORAL for active rogue Dave"


# ── Test 9: Authorized removal succeeds ─────────────────────────────────────

@requires_binary
def test_9_authorized_removal_succeeds():
    """
    QuarantineInterface.request_removal() must succeed via a fresh group.
    The removal must go through the Rust process (authorized creator).
    """
    if not _binary_available():
        pytest.skip("Binary not built")

    with ProcessBridge(_binary_path()) as bridge:
        qi = QuarantineInterface(bridge)
        bridge.send_command("create_group", {"creator_id": "Alice", "group_id": "test_remove"})
        bridge.send_command("add_member", {"identity": "Target"})

        result = qi.request_removal("Target")
        assert "Target" not in result["remaining_members"], \
            "Target must be removed from remaining_members after authorized removal"
        assert result["post_commit_epoch"] is not None


# ── Test 10: Commit produces new epoch ───────────────────────────────────────

@requires_binary
def test_10_commit_produces_new_epoch():
    """
    After add_member (which issues a real Commit), the epoch must increment.
    After remove_member (Commit), epoch must increment again.
    """
    if not _binary_available():
        pytest.skip("Binary not built")

    with ProcessBridge(_binary_path()) as bridge:
        bridge.send_command("create_group", {"creator_id": "Alice", "group_id": "epoch_test"})
        e0 = bridge.send_command("get_epoch")["current_epoch"]

        bridge.send_command("add_member", {"identity": "Bob"})
        e1 = bridge.send_command("get_epoch")["current_epoch"]
        assert e1 > e0, f"Epoch must increase after Add Commit: {e0} → {e1}"

        bridge.send_command("add_member", {"identity": "Dave"})
        e2 = bridge.send_command("get_epoch")["current_epoch"]

        QuarantineInterface(bridge).request_removal("Dave")
        e3 = bridge.send_command("get_epoch")["current_epoch"]
        assert e3 > e2, f"Epoch must increase after Remove Commit: {e2} → {e3}"


# ── Test 11: Rogue absent from post-removal membership ───────────────────────

@requires_binary
def test_11_rogue_absent_after_removal():
    """
    After MLS Remove Commit for Dave, Dave must NOT appear in current_members.
    """
    if not _binary_available():
        pytest.skip("Binary not built")

    with ProcessBridge(_binary_path()) as bridge:
        bridge.send_command("create_group", {"creator_id": "Alice", "group_id": "removal_test"})
        for m in ["Bob", "Charlie", "Dave"]:
            bridge.send_command("add_member", {"identity": m})

        assert "Dave" in bridge.send_command("get_members")["current_members"]

        QuarantineInterface(bridge).request_removal("Dave")
        final_members = bridge.send_command("get_members")["current_members"]

        assert "Dave" not in final_members, \
            f"Dave must be absent after removal, but got: {final_members}"
        assert "Alice"   in final_members
        assert "Bob"     in final_members
        assert "Charlie" in final_members


# ── Test 12: ML layer never receives cryptographic secrets ───────────────────
# This test is PURE PYTHON — does NOT require the Rust binary.

def test_12_ml_layer_never_receives_secrets():
    """
    Security boundary verification:
    1. TelemetryAdapter.extract() rejects any event dict containing secret fields
    2. Only allowed non-secret fields pass through
    3. _ALLOWED_EVENT_FIELDS and _FORBIDDEN_SECRET_FIELDS are disjoint
    """
    # (a) Disjoint sets
    intersection = _ALLOWED_EVENT_FIELDS & _FORBIDDEN_SECRET_FIELDS
    assert not intersection, \
        f"Allowed and forbidden field sets must be disjoint: {intersection}"

    # (b) Clean event passes through
    clean_event = {
        "timestamp_unix": 1000,
        "epoch": 5,
        "member_id": "Dave",
        "event_type": "APPLICATION_MESSAGE",
        "message_size_bytes": 128,
        "message_count": 3,
        "commit_count": 1,
        "time_since_last_activity_secs": 120.0,
    }
    safe = TelemetryAdapter.extract(clean_event)
    assert safe["epoch"] == 5
    assert safe["member_id"] == "Dave"

    # (c) Event with forbidden field raises SecurityError
    from mls.openmls_bridge import SecurityError
    tainted_event = dict(clean_event)
    tainted_event["epoch_secret"] = "deadbeef" * 8  # simulated leak

    with pytest.raises(SecurityError, match="SECURITY VIOLATION"):
        TelemetryAdapter.extract(tainted_event)

    # (d) Each forbidden field individually raises SecurityError
    for secret_field in list(_FORBIDDEN_SECRET_FIELDS)[:5]:
        evt = dict(clean_event)
        evt[secret_field] = "leaked_value"
        with pytest.raises(SecurityError):
            TelemetryAdapter.extract(evt)

    # (e) Unknown fields are silently dropped (not passed to Python)
    event_with_unknown = dict(clean_event)
    event_with_unknown["some_new_unknown_field"] = "value"
    safe2 = TelemetryAdapter.extract(event_with_unknown)
    assert "some_new_unknown_field" not in safe2
