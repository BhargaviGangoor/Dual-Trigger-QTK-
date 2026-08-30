"""
mls/crypto.py — RFC 9420 MLS Cryptographic Primitives

Implements:
  - DHKEM(X25519, HKDF-SHA256)  [RFC 9180 §4]
  - HKDF-labeled Extract/Expand  [RFC 9420 §8.1]
  - AES-128-GCM AEAD            [RFC 9180]
  - Ed25519 signatures           [RFC 9420 §5.3]
  - MLS-specific labeled secret derivation [RFC 9420 §8]

All operations use the `cryptography` (>=40.0) library — no Rust / FFI required.
"""

import os
import struct
import hashlib
import hmac as hmac_mod
from typing import Tuple

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.hmac import HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

# ---------------------------------------------------------------------------
# Constants (MLS ciphersuite MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519)
# ---------------------------------------------------------------------------
MLS_VERSION     = b"\x00\x01"          # ProtocolVersion mls10
CIPHERSUITE_ID  = b"\x00\x01"          # CipherSuite value in wire format
MLS_PREFIX      = b"MLS 1.0 "          # RFC 9420 §8.1 label prefix
HASH_LEN        = 32                    # SHA-256 output bytes
KEM_NONCE_LEN   = 12                    # AES-128-GCM nonce
KEM_KEY_LEN     = 16                    # AES-128-GCM key bytes
SUITE_ID_KEM    = b"KEM\x00\x20"       # DHKEM(X25519) suite ID [RFC 9180 §7.1]
SUITE_ID_HPKE   = b"HPKE\x00\x20\x00\x01\x00\x01"  # full HPKE suite id

# ---------------------------------------------------------------------------
# Key Generation
# ---------------------------------------------------------------------------

def generate_hpke_keypair() -> Tuple[X25519PrivateKey, bytes]:
    """Generate an X25519 ECDH keypair. Returns (private_key, public_bytes)."""
    priv = X25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return priv, pub_bytes


def generate_signature_keypair() -> Tuple[Ed25519PrivateKey, bytes]:
    """Generate an Ed25519 signature keypair. Returns (private_key, public_bytes)."""
    priv = Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return priv, pub_bytes


def serialize_hpke_private(priv: X25519PrivateKey) -> bytes:
    """Serialize an X25519 private key to 32 raw bytes."""
    return priv.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption()
    )


def deserialize_hpke_private(raw: bytes) -> X25519PrivateKey:
    """Deserialize a 32-byte raw X25519 private key."""
    return X25519PrivateKey.from_private_bytes(raw)


def serialize_sig_private(priv: Ed25519PrivateKey) -> bytes:
    return priv.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption()
    )


def deserialize_sig_private(raw: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(raw)


# ---------------------------------------------------------------------------
# MLS-Labeled HKDF  [RFC 9420 §8.1]
# ---------------------------------------------------------------------------

def _i2osp(n: int, length: int) -> bytes:
    """Integer to Octet String (big-endian)."""
    return n.to_bytes(length, "big")


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-SHA256 Extract step [RFC 5869]."""
    if not salt:
        salt = bytes(HASH_LEN)
    h = HMAC(salt, SHA256(), backend=default_backend())
    h.update(ikm)
    return h.finalize()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """HKDF-SHA256 Expand step [RFC 5869]."""
    return HKDFExpand(
        algorithm=SHA256(),
        length=length,
        info=info,
        backend=default_backend()
    ).derive(prk)


def hkdf_expand_label(secret: bytes, label: str, context: bytes, length: int) -> bytes:
    """
    MLS-labeled HKDF-Expand [RFC 9420 §8.1]:
      HkdfLabel = length (uint16) || "MLS 1.0 " || label || context
    """
    label_bytes = label.encode("ascii")
    hkdf_label = (
        _i2osp(length, 2)
        + _i2osp(len(MLS_PREFIX + label_bytes), 1)
        + MLS_PREFIX
        + label_bytes
        + _i2osp(len(context), 4)
        + context
    )
    return hkdf_expand(secret, hkdf_label, length)


def derive_secret(secret: bytes, label: str) -> bytes:
    """MLS DeriveSecret(secret, label) = ExpandWithLabel(secret, label, "", Nh) [RFC 9420 §8]."""
    return hkdf_expand_label(secret, label, b"", HASH_LEN)


# ---------------------------------------------------------------------------
# DHKEM(X25519, HKDF-SHA256) — RFC 9180 §4, §7.1
# ---------------------------------------------------------------------------

def _extract_and_expand_kem(dh: bytes, kem_context: bytes) -> bytes:
    """KEM shared secret derivation: ExtractAndExpand [RFC 9180 §4.1]."""
    prk = hkdf_extract(SUITE_ID_KEM, dh)
    return hkdf_expand(prk, b"shared_secret" + kem_context, HASH_LEN)


def hpke_seal(recipient_pub_bytes: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes]:
    """
    HPKE one-shot encrypt [RFC 9180 §6.1] using DHKEM(X25519) + AES-128-GCM.

    Returns (enc, ciphertext) where enc = ephemeral X25519 public key (32 bytes).
    """
    # 1. Ephemeral keypair
    eph_priv = X25519PrivateKey.generate()
    eph_pub_bytes = eph_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )

    # 2. DH with recipient
    recipient_pub = X25519PublicKey.from_public_bytes(recipient_pub_bytes)
    dh_bytes = eph_priv.exchange(recipient_pub)

    # 3. KEM context = enc || recipient_pub
    kem_context = eph_pub_bytes + recipient_pub_bytes

    # 4. Shared secret
    shared_secret = _extract_and_expand_kem(dh_bytes, kem_context)

    # 5. Derive AEAD key + nonce from shared secret
    key  = hkdf_expand_label(shared_secret, "key",   b"", KEM_KEY_LEN)
    nonce = hkdf_expand_label(shared_secret, "nonce", b"", KEM_NONCE_LEN)

    # 6. AES-128-GCM encrypt
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

    return eph_pub_bytes, ciphertext


def hpke_open(
    recipient_priv: X25519PrivateKey,
    enc: bytes,
    ciphertext: bytes,
    aad: bytes = b""
) -> bytes:
    """
    HPKE one-shot decrypt [RFC 9180 §6.1].

    enc = ephemeral public key bytes (32 bytes).
    """
    # 1. DH with ephemeral public key
    eph_pub = X25519PublicKey.from_public_bytes(enc)
    dh_bytes = recipient_priv.exchange(eph_pub)

    # 2. Reconstruct recipient public key bytes
    recipient_pub_bytes = recipient_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    kem_context = enc + recipient_pub_bytes

    # 3. Shared secret
    shared_secret = _extract_and_expand_kem(dh_bytes, kem_context)

    # 4. Derive AEAD key + nonce
    key  = hkdf_expand_label(shared_secret, "key",   b"", KEM_KEY_LEN)
    nonce = hkdf_expand_label(shared_secret, "nonce", b"", KEM_NONCE_LEN)

    # 5. AES-128-GCM decrypt
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


# ---------------------------------------------------------------------------
# Ed25519 Signatures [RFC 9420 §5.3]
# ---------------------------------------------------------------------------

def sign_with_label(priv_key: Ed25519PrivateKey, label: str, content: bytes) -> bytes:
    """
    SignWithLabel(priv, label, content) [RFC 9420 §5.3]:
      Signature over: "MLS 1.0 " || label || content
    """
    to_sign = MLS_PREFIX + label.encode("ascii") + content
    return priv_key.sign(to_sign)


def verify_with_label(pub_key_bytes: bytes, label: str, content: bytes, signature: bytes) -> bool:
    """
    VerifyWithLabel(pub, label, content, sig) [RFC 9420 §5.3].
    Returns True if valid, False if InvalidSignature.
    """
    to_verify = MLS_PREFIX + label.encode("ascii") + content
    pub = Ed25519PublicKey.from_public_bytes(pub_key_bytes)
    try:
        pub.verify(signature, to_verify)
        return True
    except InvalidSignature:
        return False


# ---------------------------------------------------------------------------
# AEAD helpers for Application Messages [RFC 9420 §15]
# ---------------------------------------------------------------------------

def aead_seal(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    """AES-128-GCM seal using a pre-derived key and nonce."""
    aesgcm = AESGCM(key)
    return aesgcm.encrypt(nonce, plaintext, aad)


def aead_open(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
    """AES-128-GCM open using a pre-derived key and nonce."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


# ---------------------------------------------------------------------------
# Tree Hash helper
# ---------------------------------------------------------------------------

def hash_bytes(*parts: bytes) -> bytes:
    """SHA-256 hash of concatenated byte parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(_i2osp(len(p), 4))
        h.update(p)
    return h.digest()
