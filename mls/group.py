"""
mls/group.py — RFC 9420 MLSGroup

Implements the core MLS group lifecycle:
  - create / add / remove / update (Commit-based)
  - Per-epoch secret derivation via HKDF chain [RFC 9420 §8]
  - Application message encryption with per-sender application_secret
  - Epoch state tracking (group_id, epoch, tree_hash, confirmed_transcript_hash)

QTK Integration:
  After every Commit (add/remove/update), the caller should invoke
  DualTrigger.dual_trigger_decision() using the epoch number as the key epoch.

Wire format / TLS encoding and the Delivery Service are out of scope for this PoC.

References:
  RFC 9420 §7  — Ratchet Tree
  RFC 9420 §8  — Key Schedule
  RFC 9420 §12 — Message Framing (simplified)
  RFC 9420 §13 — Proposals
  RFC 9420 §14 — Commit
"""

import os
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .crypto import (
    hkdf_extract,
    hkdf_expand_label,
    derive_secret,
    aead_seal,
    aead_open,
    hash_bytes,
    sign_with_label,
    verify_with_label,
    HASH_LEN,
    KEM_KEY_LEN,
    KEM_NONCE_LEN,
)
from .key_package import KeyPackage
from .ratchet_tree import RatchetTree


# ---------------------------------------------------------------------------
# Epoch Secrets [RFC 9420 §8.1]
# ---------------------------------------------------------------------------

@dataclass
class EpochSecrets:
    """
    All secrets derived for a single MLS epoch from the commit_secret.

    Derivation chain (RFC 9420 §8.1):
      joiner_secret  = HKDF-Extract(init_secret_prev, commit_secret)
      epoch_secret   = DeriveSecret(joiner_secret, "epoch")
      sender_data_secret = DeriveSecret(epoch_secret, "sender data")
      encryption_secret  = DeriveSecret(epoch_secret, "encryption")
      exporter_secret    = DeriveSecret(epoch_secret, "exporter")
      authentication_secret = DeriveSecret(epoch_secret, "authentication")
      init_secret        = DeriveSecret(epoch_secret, "init")
    """
    epoch:                  int
    commit_secret:          bytes
    joiner_secret:          bytes
    epoch_secret:           bytes
    sender_data_secret:     bytes
    encryption_secret:      bytes
    exporter_secret:        bytes
    authentication_secret:  bytes
    init_secret:            bytes   # carried forward to next epoch


def derive_epoch_secrets(
    init_secret_prev: bytes,
    commit_secret: bytes,
    epoch: int,
) -> EpochSecrets:
    """
    Full RFC 9420 §8.1 epoch key schedule derivation.
    """
    joiner_secret = hkdf_extract(init_secret_prev, commit_secret)
    epoch_secret  = derive_secret(joiner_secret, "epoch")

    sender_data_secret    = derive_secret(epoch_secret, "sender data")
    encryption_secret     = derive_secret(epoch_secret, "encryption")
    exporter_secret       = derive_secret(epoch_secret, "exporter")
    authentication_secret = derive_secret(epoch_secret, "authentication")
    init_secret           = derive_secret(epoch_secret, "init")

    return EpochSecrets(
        epoch=epoch,
        commit_secret=commit_secret,
        joiner_secret=joiner_secret,
        epoch_secret=epoch_secret,
        sender_data_secret=sender_data_secret,
        encryption_secret=encryption_secret,
        exporter_secret=exporter_secret,
        authentication_secret=authentication_secret,
        init_secret=init_secret,
    )


def derive_application_secret(epoch_secrets: EpochSecrets, sender_leaf: int) -> Tuple[bytes, bytes]:
    """
    Derive per-sender application key and nonce [RFC 9420 §15.1]:
      application_secret = DeriveSecret(encryption_secret, "application")
      key   = ExpandWithLabel(application_secret, "key",   sender, Nk)
      nonce = ExpandWithLabel(application_secret, "nonce", sender, Nn)
    """
    app_secret = derive_secret(epoch_secrets.encryption_secret, "application")
    sender_ctx = sender_leaf.to_bytes(4, "big")
    key   = hkdf_expand_label(app_secret, "key",   sender_ctx, KEM_KEY_LEN)
    nonce = hkdf_expand_label(app_secret, "nonce", sender_ctx, KEM_NONCE_LEN)
    return key, nonce


# ---------------------------------------------------------------------------
# Group Context [RFC 9420 §7.1]
# ---------------------------------------------------------------------------

@dataclass
class GroupContext:
    """
    GroupContext authenticated in every MLS message/Commit.
    """
    group_id:                    bytes
    epoch:                       int
    tree_hash:                   bytes
    confirmed_transcript_hash:   bytes
    extensions:                  bytes = b""

    def serialize(self) -> bytes:
        """Minimal serialization for signing / hashing."""
        return (
            len(self.group_id).to_bytes(2, "big") + self.group_id
            + self.epoch.to_bytes(8, "big")
            + self.tree_hash
            + self.confirmed_transcript_hash
        )


# ---------------------------------------------------------------------------
# MLSGroup
# ---------------------------------------------------------------------------

class MLSGroup:
    """
    RFC 9420 MLS Group.

    Manages the full TreeKEM lifecycle for a single MLS group:
      - Creator initializes at epoch 0
      - Each Add/Remove/Update generates a Commit advancing the epoch
      - Epoch secrets are re-derived after each Commit
      - Application messages are encrypted with per-epoch, per-sender secrets

    QTK Hook:
      Call group.epoch (property) after each add()/remove()/update() to get
      the current MLS epoch — this is the key_update_epoch for DualTrigger.
    """

    def __init__(self, group_id: str, creator_key_package: KeyPackage):
        """
        Initialize a new MLS group created by `creator_key_package`.

        Args:
            group_id            : Unique string identifier for this group
            creator_key_package : The creator's KeyPackage (becomes leaf 0)
        """
        self._group_id_str = group_id
        self._group_id     = hashlib.sha256(group_id.encode()).digest()[:16]

        # Ratchet tree
        self._tree = RatchetTree()
        creator_leaf = self._tree.add_leaf(creator_key_package)

        # Member registry: identity → leaf_index
        self._members: Dict[str, int] = {creator_key_package.identity: creator_leaf}
        self._leaf_to_identity: Dict[int, str] = {creator_leaf: creator_key_package.identity}

        # Epoch 0 — genesis commit secret = SHA-256(group_id || "genesis")
        genesis_commit = hashlib.sha256(self._group_id + b"genesis").digest()
        self._init_secret = bytes(HASH_LEN)   # initial init_secret = all zeros
        self._epoch_secrets = derive_epoch_secrets(self._init_secret, genesis_commit, 0)

        # Transcript hash (covers all proposals + commits)
        self._confirmed_transcript_hash = hashlib.sha256(
            self._group_id + b"epoch_0"
        ).digest()

        # Proposal queue (cleared after each Commit)
        self._pending_proposals: List[Dict[str, Any]] = []

        # History: epoch → EpochSecrets (for audit/debugging)
        self._epoch_history: List[EpochSecrets] = [self._epoch_secrets]

        # Event log
        self.event_log: List[Dict[str, Any]] = [{
            "epoch": 0,
            "event": "CREATE_GROUP",
            "group_id": group_id,
            "creator": creator_key_package.identity,
            "tree_hash": self.tree_hash.hex(),
        }]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def epoch(self) -> int:
        """Current MLS epoch number."""
        return self._epoch_secrets.epoch

    @property
    def tree_hash(self) -> bytes:
        """Current ratchet tree hash (changes after every Commit)."""
        return self._tree.tree_hash()

    @property
    def group_context(self) -> GroupContext:
        """Current GroupContext for this epoch."""
        return GroupContext(
            group_id=self._group_id,
            epoch=self.epoch,
            tree_hash=self.tree_hash,
            confirmed_transcript_hash=self._confirmed_transcript_hash,
        )

    @property
    def active_members(self) -> List[str]:
        """Identities of all current non-blank tree members."""
        return [identity for identity, _ in
                [(kp.identity, li) for li, kp in self._tree.active_members()]]

    @property
    def num_members(self) -> int:
        return len(self._members)

    # ------------------------------------------------------------------
    # Add Proposal + Commit [RFC 9420 §13.1, §14]
    # ------------------------------------------------------------------

    def add(self, key_package: KeyPackage) -> Dict[str, Any]:
        """
        Process an Add Proposal and immediately Commit it.

        1. Verify the KeyPackage self-signature
        2. Add the KeyPackage to the ratchet tree
        3. Generate UpdatePath (path secrets + HPKE encryptions)
        4. Advance epoch, re-derive all epoch secrets
        5. Update transcript hash

        Returns a dict with commit metadata for logging.
        """
        if not key_package.verify():
            raise ValueError(f"KeyPackage verification failed for {key_package.identity}")

        if key_package.identity in self._members:
            raise ValueError(f"Member {key_package.identity!r} already in group")

        # Proposal: Add
        self._pending_proposals.append({
            "type": "Add",
            "key_package_ref": key_package.ref_hex,
            "identity": key_package.identity,
        })

        # Apply to tree
        leaf_idx = self._tree.add_leaf(key_package)
        self._members[key_package.identity] = leaf_idx
        self._leaf_to_identity[leaf_idx] = key_package.identity

        # Commit: generate new path secrets from a random committer leaf
        return self._commit(operation=f"Add({key_package.identity})")

    # ------------------------------------------------------------------
    # Remove Proposal + Commit [RFC 9420 §13.3]
    # ------------------------------------------------------------------

    def remove(self, identity: str) -> Dict[str, Any]:
        """
        Process a Remove Proposal and immediately Commit it.

        Blanks the removed member's leaf and their direct path,
        then advances the epoch.
        """
        if identity not in self._members:
            raise KeyError(f"Member {identity!r} not found in group")

        leaf_idx = self._members.pop(identity)
        del self._leaf_to_identity[leaf_idx]

        self._pending_proposals.append({
            "type": "Remove",
            "removed_leaf": leaf_idx,
            "identity": identity,
        })

        self._tree.remove_leaf(leaf_idx)
        return self._commit(operation=f"Remove({identity})")

    # ------------------------------------------------------------------
    # Update (Key Rotation) [RFC 9420 §13.2]
    # ------------------------------------------------------------------

    def update(self, identity: str) -> Dict[str, Any]:
        """
        Process an Update Proposal for `identity` and immediately Commit.

        Generates a fresh X25519 keypair for this member's leaf,
        then ratchets the path secret up to root.
        """
        if identity not in self._members:
            raise KeyError(f"Member {identity!r} not found")

        new_kp = KeyPackage.generate(identity)
        leaf_idx = self._members[identity]

        # Replace leaf key package with fresh keypair
        node_idx = RatchetTree._leaf_index_to_node(leaf_idx)
        self._tree._nodes[node_idx].hpke_pub_bytes = new_kp.hpke_pub_bytes
        self._tree._nodes[node_idx].hpke_priv      = new_kp._hpke_priv
        self._tree._nodes[node_idx].key_package    = new_kp

        self._pending_proposals.append({
            "type": "Update",
            "identity": identity,
            "new_kp_ref": new_kp.ref_hex,
        })

        return self._commit(operation=f"Update({identity})", committer_leaf=leaf_idx)

    # ------------------------------------------------------------------
    # Internal Commit [RFC 9420 §14]
    # ------------------------------------------------------------------

    def _commit(self, operation: str, committer_leaf: Optional[int] = None) -> Dict[str, Any]:
        """
        Advance the MLS epoch by processing pending proposals as a Commit.

        1. Pick a committer (first active member, or explicit)
        2. Generate UpdatePath → commit_secret
        3. Re-derive all epoch secrets from new commit_secret
        4. Update transcript hash
        5. Clear proposal queue
        6. Log the event
        """
        active = self._tree.active_members()
        if not active:
            raise RuntimeError("Cannot Commit with no active members")

        if committer_leaf is None:
            committer_leaf, _ = active[0]

        # Generate path secrets + HPKE-encrypted path nodes
        commit_secret, encrypted_path = self._tree.generate_path_secrets(committer_leaf)

        # Advance epoch
        new_epoch = self.epoch + 1
        self._epoch_secrets = derive_epoch_secrets(
            self._epoch_secrets.init_secret,
            commit_secret,
            new_epoch
        )
        self._epoch_history.append(self._epoch_secrets)

        # Update transcript hash: hash(prev_transcript || commit_secret || operation)
        self._confirmed_transcript_hash = hashlib.sha256(
            self._confirmed_transcript_hash
            + commit_secret
            + operation.encode()
        ).digest()

        # Build commit record
        commit_record = {
            "epoch":          new_epoch,
            "event":          "COMMIT",
            "operation":      operation,
            "proposals":      list(self._pending_proposals),
            "tree_hash":      self.tree_hash.hex(),
            "commit_secret":  commit_secret.hex(),
            "epoch_secret":   self._epoch_secrets.epoch_secret.hex(),
            "init_secret":    self._epoch_secrets.init_secret.hex(),
            "encrypted_path_nodes": len(encrypted_path),
            "confirmed_transcript_hash": self._confirmed_transcript_hash.hex(),
        }
        self.event_log.append(commit_record)
        self._pending_proposals = []
        return commit_record

    # ------------------------------------------------------------------
    # Application Messages [RFC 9420 §15]
    # ------------------------------------------------------------------

    def send_application_message(self, sender_identity: str, plaintext: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt `plaintext` as an MLS ApplicationMessage from `sender_identity`.

        Returns (aad, ciphertext) where:
          aad        = group_context serialized (epoch + tree_hash)
          ciphertext = AES-128-GCM encrypt of plaintext
        """
        if sender_identity not in self._members:
            raise KeyError(f"Sender {sender_identity!r} not in group")

        sender_leaf = self._members[sender_identity]
        key, nonce  = derive_application_secret(self._epoch_secrets, sender_leaf)
        aad         = self.group_context.serialize()
        ciphertext  = aead_seal(key, nonce, plaintext, aad)
        return aad, ciphertext

    def receive_application_message(
        self,
        sender_identity: str,
        aad: bytes,
        ciphertext: bytes,
    ) -> bytes:
        """
        Decrypt an MLS ApplicationMessage from `sender_identity`.

        Raises ValueError if decryption fails (wrong epoch / wrong sender).
        """
        if sender_identity not in self._members:
            raise KeyError(f"Sender {sender_identity!r} not in group")

        sender_leaf = self._members[sender_identity]
        key, nonce  = derive_application_secret(self._epoch_secrets, sender_leaf)
        try:
            return aead_open(key, nonce, ciphertext, aad)
        except Exception as e:
            raise ValueError(f"ApplicationMessage decryption failed: {e}") from e

    # ------------------------------------------------------------------
    # Epoch Info (for QTK integration)
    # ------------------------------------------------------------------

    def epoch_info(self) -> Dict[str, Any]:
        """
        Return a compact summary of the current epoch state for QTK logging.
        """
        return {
            "group_id":   self._group_id_str,
            "epoch":      self.epoch,
            "tree_hash":  self.tree_hash.hex(),
            "members":    self.active_members,
            "num_members": self.num_members,
            "exporter_secret": self._epoch_secrets.exporter_secret.hex(),
        }

    def __repr__(self) -> str:
        return (
            f"MLSGroup(id={self._group_id_str!r}, epoch={self.epoch}, "
            f"members={self.active_members}, "
            f"tree_hash={self.tree_hash.hex()[:8]}...)"
        )
