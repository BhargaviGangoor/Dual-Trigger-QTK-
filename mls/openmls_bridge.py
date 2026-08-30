"""
mls/openmls_bridge.py — Python ↔ Rust OpenMLS Bridge

Manages the Rust subprocess that runs the real OpenMLS group.
Communicates via JSON over stdin/stdout (newline-delimited).

SECURITY BOUNDARY:
  Python receives ONLY non-secret telemetry fields from MlsEvent.
  All MLS cryptographic state (epoch secrets, path secrets, private keys)
  is owned exclusively by the Rust process and never sent to Python.

AUTHORIZATION MODEL:
  The Python behavioral layer may REQUEST member removal via
  QuarantineInterface.request_removal(). The actual MLS Remove Commit
  is performed by the Rust process using the creator's signing key.
  The ML model has no direct access to MLS key material.
"""

import json
import subprocess
import sys
import os
import time
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Confirmed non-secret fields that Python may receive
_ALLOWED_EVENT_FIELDS = {
    "timestamp_unix",
    "epoch",
    "member_id",
    "event_type",
    "message_size_bytes",
    "message_count",
    "commit_count",
    "time_since_last_activity_secs",
    "peer_id",
    "simulated_is_vpn",
    "simulated_ip_changed",
    "simulated_tz_changed",
    "simulated_sync_frequency",
    "simulated_session_duration_sec",
}

# Fields that must NEVER appear in Python-side event dicts
_FORBIDDEN_SECRET_FIELDS = {
    "epoch_secret",
    "init_secret",
    "joiner_secret",
    "path_secret",
    "sender_data_secret",
    "encryption_secret",
    "exporter_secret",
    "private_key",
    "signing_key",
    "hpke_key",
    "decrypted_content",
    "plaintext",
}


class TelemetryAdapter:
    """
    Strips events to confirmed non-secret fields only.

    This is the security enforcement layer on the Python side.
    Even if the Rust side accidentally sent a secret field,
    this adapter would silently drop it.
    """

    @staticmethod
    def extract(event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return only the allowed non-secret fields from an event dict."""
        safe = {k: v for k, v in event_dict.items() if k in _ALLOWED_EVENT_FIELDS}
        # Verify no secret fields slipped through
        leaked = _FORBIDDEN_SECRET_FIELDS & set(event_dict.keys())
        if leaked:
            raise SecurityError(
                f"SECURITY VIOLATION: Forbidden fields found in MLS event: {leaked}"
            )
        return safe

    @staticmethod
    def to_qtk_telemetry(event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a safe MLS event to the feature format expected by the
        existing QTK behavioral pipeline (HMM / RiskFusion / DualTrigger).

        Feature availability:
          REAL (from OpenMLS):
            - epoch                       → key_update_epoch proxy
            - message_count               → message_count_sent
            - message_size_bytes          → available
            - event_type                  → available
          SIMULATED (injected by PoC scenario, documented per §16):
            - simulated_is_vpn            → is_vpn
            - simulated_ip_changed        → ip_changed
            - simulated_tz_changed        → tz_changed
            - simulated_sync_frequency    → sync_frequency
            - simulated_session_duration  → session_duration_sec
          UNAVAILABLE from real MLS (documented per §16):
            - network_type                → set to "Unknown"
            - network_ip                  → set to "0.0.0.0"
            - location_country            → set to "Unknown"
            - active_timezone             → set to "UTC"
        """
        return {
            # Real MLS fields
            "epoch":                  event.get("epoch", 0),
            "message_count_sent":     event.get("message_count", 0),
            "message_size_bytes":     event.get("message_size_bytes"),
            "event_type":             event.get("event_type", "UNKNOWN"),
            "commit_count":           event.get("commit_count", 0),
            "time_since_last_activity_secs": event.get("time_since_last_activity_secs", 0.0),
            # Simulated behavioral metadata (documented limitation)
            "session_duration_sec":   event.get("simulated_session_duration_sec", 120.0),
            "sync_frequency":         event.get("simulated_sync_frequency", 4.0),
            "is_vpn":                 event.get("simulated_is_vpn", 0.0),
            "ip_changed":             event.get("simulated_ip_changed", 0.0),
            "tz_changed":             event.get("simulated_tz_changed", 0.0),
            # Unavailable from MLS — documented
            "network_type":           "Unknown",
            "network_ip":             "0.0.0.0",
            "location_country":       "Unknown",
            "active_timezone":        "UTC",
        }


class SecurityError(Exception):
    pass


class ProcessBridge:
    """
    Manages the Rust mls_openmls subprocess.

    Sends JSON commands via stdin, receives JSON responses via stdout.
    The Rust process owns all MLS cryptographic state.
    """

    def __init__(self, binary_path: Optional[str] = None):
        if binary_path is None:
            # Default: look for the release binary next to this file
            project_root = Path(__file__).parent.parent
            if platform.system() == "Windows":
                binary_path = str(
                    project_root / "mls_openmls" / "target" / "release" / "mls_openmls.exe"
                )
            else:
                binary_path = str(
                    project_root / "mls_openmls" / "target" / "release" / "mls_openmls"
                )

        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"OpenMLS binary not found at: {binary_path}\n"
                f"Build it with: cd mls_openmls && cargo build --release"
            )

        self.binary_path = binary_path
        self._proc: Optional[subprocess.Popen] = None

    def start(self):
        """Start the Rust OpenMLS subprocess."""
        self._proc = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

    def stop(self):
        """Gracefully terminate the Rust process."""
        if self._proc and self._proc.poll() is None:
            try:
                self._send_raw({"cmd": "quit", "args": {}})
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def _send_raw(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON command and return the parsed response."""
        if not self._proc or self._proc.poll() is not None:
            raise RuntimeError("OpenMLS subprocess is not running")

        line = json.dumps(command) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

        response_line = self._proc.stdout.readline()
        if not response_line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(f"OpenMLS process closed unexpectedly. stderr: {stderr}")

        return json.loads(response_line.strip())

    def send_command(self, cmd: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a command and return the full response envelope.
        Raises RuntimeError if status == 'error'.
        """
        response = self._send_raw({"cmd": cmd, "args": args or {}})
        if response.get("status") == "error":
            raise RuntimeError(f"OpenMLS error: {response.get('error', 'unknown')}")
        return response

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class QuarantineInterface:
    """
    Policy-driven quarantine interface.

    AUTHORIZATION MODEL (§11):
      1. Python behavioral detector computes risk score R(member, t)
      2. DualTrigger policy returns BEHAVIORAL trigger
      3. Python calls QuarantineInterface.request_removal(member_id)
      4. This method sends a "remove_member" command to the Rust process
      5. Rust executes the real MLS Remove Commit with the creator's signing key
      6. OpenMLS advances the epoch — Dave is no longer an MLS group member

    The ML model NEVER has access to MLS key material.
    The Remove Commit is performed by the authorized creator's signing key
    held exclusively in the Rust process.
    """

    def __init__(self, bridge: ProcessBridge):
        self._bridge = bridge

    def request_removal(self, member_id: str) -> Dict[str, Any]:
        """
        Request MLS removal of `member_id` through the authorized interface.

        Returns non-secret removal event info:
          - event_type: MEMBER_REMOVED
          - epoch: new epoch after removal commit
          - remaining members (identity strings only)
        """
        response = self._bridge.send_command("remove_member", {"target_id": member_id})
        # Extract only non-secret fields
        event = response.get("event", {})
        safe_event = TelemetryAdapter.extract(event)
        return {
            "removal_event":       safe_event,
            "post_commit_epoch":   response.get("current_epoch"),
            "remaining_members":   response.get("current_members", []),
        }
