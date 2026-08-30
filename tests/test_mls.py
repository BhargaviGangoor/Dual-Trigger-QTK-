"""
tests/test_mls.py — Unit tests for the real RFC 9420 MLS implementation

Tests:
  1. KeyPackage self-signature is valid
  2. Adding a member advances the MLS epoch
  3. Removing a member blanks the leaf and advances epoch
  4. Path secret derivation produces deterministically-different tree hashes
  5. ApplicationMessage encrypt/decrypt round-trip (E2EE)
  6. QTK quarantine fires after behavioral trigger, followed by MLS Remove
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from mls.key_package import KeyPackage
from mls.group import MLSGroup


# ---------------------------------------------------------------------------
# 1. KeyPackage self-signature
# ---------------------------------------------------------------------------

def test_key_package_self_signs():
    """KeyPackage.generate() must produce a valid self-signature."""
    kp = KeyPackage.generate("alice_phone")
    assert kp.verify(), "KeyPackage self-signature verification failed"
    assert len(kp.hpke_pub_bytes) == 32, "HPKE public key must be 32 bytes (X25519)"
    assert len(kp.sig_pub_bytes)  == 32, "Signature public key must be 32 bytes (Ed25519)"
    assert len(kp.ref)            == 4,  "KeyPackageRef must be 4 bytes"
    assert kp.identity == "alice_phone"


def test_key_package_tampered_identity_fails():
    """Tampering with identity after signing must cause verify() to fail."""
    kp = KeyPackage.generate("alice_phone")
    # Manually tamper with identity field
    object.__setattr__(kp, "identity", "evil_device")
    assert not kp.verify(), "Tampered KeyPackage should fail verification"


# ---------------------------------------------------------------------------
# 2. Add member advances epoch
# ---------------------------------------------------------------------------

def test_add_member_advances_epoch():
    """Each Add Commit must increment the MLS epoch by exactly 1."""
    creator_kp = KeyPackage.generate("alice_phone")
    group = MLSGroup(group_id="test_group_add", creator_key_package=creator_kp)
    assert group.epoch == 0, "Initial epoch should be 0"

    kp2 = KeyPackage.generate("alice_laptop")
    group.add(kp2)
    assert group.epoch == 1, "After Add Commit, epoch must be 1"

    kp3 = KeyPackage.generate("alice_tablet")
    group.add(kp3)
    assert group.epoch == 2, "After second Add Commit, epoch must be 2"

    assert "alice_laptop" in group.active_members
    assert "alice_tablet" in group.active_members
    assert group.num_members == 3


# ---------------------------------------------------------------------------
# 3. Remove member blanks leaf and advances epoch
# ---------------------------------------------------------------------------

def test_remove_member_advances_epoch_and_removes():
    """Remove Commit must blank the leaf, remove from active_members, advance epoch."""
    creator_kp = KeyPackage.generate("alice_phone")
    group = MLSGroup(group_id="test_group_remove", creator_key_package=creator_kp)

    kp2 = KeyPackage.generate("alice_laptop")
    group.add(kp2)
    assert group.epoch == 1

    epoch_before = group.epoch
    group.remove("alice_laptop")
    assert group.epoch == epoch_before + 1, "Remove Commit must advance epoch"
    assert "alice_laptop" not in group.active_members, "Removed member must not be in active_members"
    assert group.num_members == 1


# ---------------------------------------------------------------------------
# 4. Tree hash changes on every Commit
# ---------------------------------------------------------------------------

def test_path_secret_changes_tree_hash():
    """Every Commit (add/update/remove) must produce a different tree_hash."""
    creator_kp = KeyPackage.generate("creator")
    group = MLSGroup(group_id="test_group_hash", creator_key_package=creator_kp)

    hash_0 = group.tree_hash

    kp2 = KeyPackage.generate("member_b")
    group.add(kp2)
    hash_1 = group.tree_hash
    assert hash_1 != hash_0, "tree_hash must change after Add Commit"

    group.update("creator")
    hash_2 = group.tree_hash
    assert hash_2 != hash_1, "tree_hash must change after Update Commit"

    group.remove("member_b")
    hash_3 = group.tree_hash
    assert hash_3 != hash_2, "tree_hash must change after Remove Commit"

    # All hashes must be 32 bytes (SHA-256)
    for h in [hash_0, hash_1, hash_2, hash_3]:
        assert len(h) == 32, f"tree_hash must be 32 bytes, got {len(h)}"


# ---------------------------------------------------------------------------
# 5. ApplicationMessage encrypt / decrypt round-trip
# ---------------------------------------------------------------------------

def test_application_message_roundtrip():
    """AES-128-GCM E2EE round-trip: encrypt then decrypt must recover plaintext."""
    creator_kp = KeyPackage.generate("alice_phone")
    group = MLSGroup(group_id="test_group_e2ee", creator_key_package=creator_kp)

    kp2 = KeyPackage.generate("alice_laptop")
    group.add(kp2)

    plaintext = b"Secret MLS application message payload"
    aad, ciphertext = group.send_application_message("alice_phone", plaintext)
    assert ciphertext != plaintext, "Ciphertext must differ from plaintext"

    recovered = group.receive_application_message("alice_phone", aad, ciphertext)
    assert recovered == plaintext, "Decrypted plaintext must match original"


def test_application_message_wrong_epoch_fails():
    """Decryption with a stale (wrong-epoch) ciphertext must fail."""
    creator_kp = KeyPackage.generate("alice_phone")
    group = MLSGroup(group_id="test_group_stale", creator_key_package=creator_kp)
    kp2 = KeyPackage.generate("alice_laptop")
    group.add(kp2)

    # Encrypt at epoch 1
    aad_old, ct_old = group.send_application_message("alice_phone", b"old epoch message")

    # Advance epoch (Update Commit)
    group.update("alice_phone")

    # Trying to decrypt old ciphertext in new epoch must fail
    with pytest.raises((ValueError, Exception)):
        group.receive_application_message("alice_phone", aad_old, ct_old)


# ---------------------------------------------------------------------------
# 6. QTK quarantine triggers MLS Remove Commit
# ---------------------------------------------------------------------------

def test_qtk_quarantine_triggers_mls_remove():
    """
    End-to-end: a rogue device added to the MLS group should be quarantined
    by the QTK behavioral trigger, which then causes a real MLS Remove Commit
    (advancing the epoch and removing the member from the ratchet tree).
    """
    from experiments.mls_poc import RealMLSProofOfConcept

    poc = RealMLSProofOfConcept(group_id="test_qtk_mls_integration")
    poc.create_group(creator_id="alice_phone")
    poc.add_member("alice_laptop",         "Alice Laptop",        is_rogue=False)
    poc.add_member("alice_standby_tablet", "Alice Tablet",        is_rogue=False)
    poc.add_member("rogue_client",         "Rogue Terminal",      is_rogue=True)

    assert poc.mls_group.epoch > 0, "Epoch must have advanced after adds"
    initial_members = set(poc.mls_group.active_members)
    assert "rogue_client" in initial_members

    # Run 6 epochs — rogue accumulates high anomaly; tablet goes silent
    all_quarantines = []
    for ep in range(1, 7):
        events = [
            {"device_id": "alice_phone",  "event_type": "MLS_KEY_UPDATE_COMMIT",
             "session_duration_sec": 120.0, "sync_frequency": 4.0, "msg_count": 8},
            {"device_id": "alice_laptop", "event_type": "MLS_KEY_UPDATE_COMMIT",
             "session_duration_sec": 140.0, "sync_frequency": 4.2, "msg_count": 9},
            # Standby tablet: no KEY_UPDATE_COMMIT (silent → inactivity trigger)
            {"device_id": "rogue_client", "event_type": "MLS_KEY_UPDATE_COMMIT",
             "session_duration_sec": 500.0, "sync_frequency": 18.0, "msg_count": 40,
             "ip_changed": 1.0 if ep >= 4 else 0.0},
        ]
        res = poc.process_mls_epoch(events)
        all_quarantines.extend(res["quarantine_actions"])

    # At least one device should have been quarantined
    assert len(all_quarantines) >= 1, \
        f"Expected at least 1 quarantine, got {all_quarantines}"

    # Any quarantined device must have been MLS-removed
    for q in all_quarantines:
        dev_id = q["device_id"]
        assert dev_id not in poc.mls_group.active_members, \
            f"Quarantined device {dev_id!r} should be removed from MLS group"
        assert poc.devices[dev_id].is_quarantined, \
            f"Device {dev_id!r} must be marked as quarantined in QTK"

    # Legitimate devices must remain active
    assert "alice_phone"  in poc.mls_group.active_members
    assert "alice_laptop" in poc.mls_group.active_members
