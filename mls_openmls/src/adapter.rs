// mls_openmls/src/adapter.rs
//
// Telemetry Boundary Adapter
// ==========================
// This module defines EXACTLY what information the Python behavioral layer
// may receive from the OpenMLS group.
//
// SECURITY BOUNDARY:
//   ✅ EXPOSED to Python: epoch, member_id, event_type, message_size_bytes,
//      commit_count, message_count, time_since_last_activity_secs
//   ❌ NEVER EXPOSED: epoch_secrets, path_secrets, private keys, HPKE secrets,
//      init_secret, joiner_secret, decrypted application content
//
// OpenMLS is the sole owner of all cryptographic state.
// This adapter only emits metadata that is observable by any group member
// without any access to private key material.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Non-secret MLS event — the ONLY information that crosses the
/// OpenMLS ↔ Python behavioral-pipeline boundary.
///
/// All fields are safe for the behavioral layer to observe.
/// No cryptographic secrets are present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlsEvent {
    /// Wall-clock timestamp (Unix seconds) — not a secret
    pub timestamp_unix: u64,

    /// Current MLS epoch number — public group state, not a secret
    pub epoch: u64,

    /// Identity string of the acting member — from BasicCredential
    pub member_id: String,

    /// Type of event — no crypto material
    pub event_type: MlsEventType,

    /// Size of ciphertext in bytes (NOT the plaintext or key) — safe metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_size_bytes: Option<usize>,

    /// Number of application messages sent by this member this epoch
    pub message_count: u64,

    /// Number of commits sent by this member this epoch
    pub commit_count: u64,

    /// Seconds since this member last sent any event (0 if first event)
    pub time_since_last_activity_secs: f64,

    /// Optional peer member ID (for Add/Remove events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer_id: Option<String>,

    /// Simulated contextual metadata (NOT from OpenMLS internals).
    /// These fields are externally injected by the PoC scenario
    /// (e.g., simulated VPN usage, IP change flags).
    /// Documented as "simulated behavioral metadata" per §16 of requirements.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulated_is_vpn: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulated_ip_changed: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulated_tz_changed: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulated_sync_frequency: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulated_session_duration_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MlsEventType {
    /// Real MLS: MlsGroup::new() was called
    GroupCreated,
    /// Real MLS: add_members() + merge_pending_commit()
    MemberAdded,
    /// Real MLS: create_message() (ciphertext size recorded, content discarded)
    ApplicationMessage,
    /// Real MLS: remove_members() + merge_pending_commit()
    MemberRemoved,
    /// Real MLS: self_update() + merge_pending_commit() for key rotation
    KeyUpdate,
}

impl MlsEvent {
    pub fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// JSON envelope sent from Rust → Python for every event.
/// Status "ok" or "error". On "ok", `event` is populated.
#[derive(Debug, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<MlsEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Current group members (safe: identity strings only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_members: Option<Vec<String>>,
    /// Current MLS epoch (safe: public group state)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_epoch: Option<u64>,
}

impl EventEnvelope {
    pub fn ok(event: MlsEvent, members: Vec<String>, epoch: u64) -> Self {
        Self {
            status: "ok".into(),
            event: Some(event),
            current_members: Some(members),
            current_epoch: Some(epoch),
            error: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            status: "error".into(),
            event: None,
            current_members: None,
            current_epoch: None,
            error: Some(msg.into()),
        }
    }
}

/// JSON command sent from Python → Rust.
#[derive(Debug, Deserialize)]
pub struct Command {
    pub cmd: String,
    #[serde(default)]
    pub args: serde_json::Value,
}
