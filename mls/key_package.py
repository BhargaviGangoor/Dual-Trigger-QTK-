"""
mls/key_package.py — RFC 9420 KeyPackage

A KeyPackage bundles a member's:
  - HPKE public key (X25519) — used for path encryption in TreeKEM
  - Credential (identity string) + Ed25519 signing public key
  - Self-signature over the above payload

This corresponds to RFC 9420 §10.
"""

import os
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .crypto import (
    generate_hpke_keypair,
    generate_signature_keypair,
    sign_with_label,
    verify_with_label,
    serialize_hpke_private,
    serialize_sig_private,
    hash_bytes,
)


@dataclass
class KeyPackage:
    """
    RFC 9420 §10 KeyPackage.

    Attributes:
        identity        : Human-readable member identity (e.g. "alice_phone")
        hpke_pub_bytes  : 32-byte X25519 public key for TreeKEM path encryption
        sig_pub_bytes   : 32-byte Ed25519 verifying key for credential
        signature       : Ed25519 signature over (identity + hpke_pub_bytes)
        ref             : 4-byte KeyPackageRef = SHA-256(payload)[:4]

    Private keys are stored separately and NOT serialized with the package:
        _hpke_priv      : X25519PrivateKey (for HPKE open operations)
        _sig_priv       : Ed25519PrivateKey (for signing commits)
    """
    identity:       str
    hpke_pub_bytes: bytes
    sig_pub_bytes:  bytes
    signature:      bytes
    ref:            bytes  # 4-byte short reference (KeyPackageRef)

    # Private keys — NOT part of the public KeyPackage wire format
    _hpke_priv: Optional[X25519PrivateKey] = field(default=None, repr=False)
    _sig_priv:  Optional[Ed25519PrivateKey] = field(default=None, repr=False)

    @classmethod
    def generate(cls, identity: str) -> "KeyPackage":
        """
        Generate a fresh KeyPackage for `identity`:
          1. Generate X25519 HPKE keypair
          2. Generate Ed25519 signature keypair
          3. Sign the payload with SignWithLabel("KeyPackageTBS", payload)
          4. Compute ref = SHA-256(payload)[:4]
        """
        hpke_priv, hpke_pub = generate_hpke_keypair()
        sig_priv, sig_pub   = generate_signature_keypair()

        # TBS (to-be-signed) payload = identity_len || identity || hpke_pub
        id_bytes = identity.encode("utf-8")
        payload  = (
            len(id_bytes).to_bytes(2, "big") + id_bytes
            + hpke_pub
        )

        signature = sign_with_label(sig_priv, "KeyPackageTBS", payload)
        ref = hashlib.sha256(payload).digest()[:4]

        return cls(
            identity=identity,
            hpke_pub_bytes=hpke_pub,
            sig_pub_bytes=sig_pub,
            signature=signature,
            ref=ref,
            _hpke_priv=hpke_priv,
            _sig_priv=sig_priv,
        )

    def verify(self) -> bool:
        """
        Verify the self-signature on this KeyPackage.
        Returns True if valid.
        """
        id_bytes = self.identity.encode("utf-8")
        payload  = (
            len(id_bytes).to_bytes(2, "big") + id_bytes
            + self.hpke_pub_bytes
        )
        return verify_with_label(self.sig_pub_bytes, "KeyPackageTBS", payload, self.signature)

    @property
    def ref_hex(self) -> str:
        """Human-readable 8-char hex KeyPackageRef."""
        return self.ref.hex()

    def __repr__(self) -> str:
        return (
            f"KeyPackage(identity={self.identity!r}, "
            f"ref={self.ref_hex}, "
            f"hpke_pub={self.hpke_pub_bytes.hex()[:16]}...)"
        )
