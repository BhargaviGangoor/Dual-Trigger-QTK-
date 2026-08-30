// mls_openmls/src/group.rs
//
// Real OpenMLS Group Lifecycle
// ============================
// All MLS cryptographic operations are delegated to the openmls crate.
// This module does NOT implement any custom cryptography.
//
// OpenMLS is responsible for:
//   - TreeKEM ratchet tree operations
//   - HPKE path encryption
//   - HKDF epoch secret derivation
//   - Ed25519 credential signatures
//   - Commit processing and epoch advancement
//
// This wrapper:
//   - Manages member KeyPackages and credentials
//   - Delegates all group state to MlsGroup
//   - Extracts ONLY non-secret telemetry (epoch, member identity, event type)

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use openmls::prelude::*;
use openmls_basic_credential::SignatureKeyPair;
use openmls_memory_storage::MemoryStorage;
use openmls_rust_crypto::OpenMlsRustCrypto;

use crate::adapter::{MlsEvent, MlsEventType};

/// Ciphersuite used for this PoC:
/// MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519
const CIPHERSUITE: Ciphersuite =
    Ciphersuite::MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519;

/// Per-member credentials and key material.
/// SECURITY: These are held by the Rust process only.
/// Private keys NEVER leave the Rust process.
struct MemberState {
    credential_with_key: CredentialWithKey,
    signer: SignatureKeyPair,
    key_package: KeyPackage,
    storage: MemoryStorage,
    provider: OpenMlsRustCrypto,
    /// Running counters for telemetry (non-secret)
    message_count: u64,
    commit_count: u64,
    last_activity_unix: u64,
}

/// The real OpenMLS group wrapper.
pub struct OpenMlsGroup {
    /// The actual OpenMLS group — owns all cryptographic state
    group: MlsGroup,
    /// Creator's provider and storage (the "authorized remover")
    creator_provider: OpenMlsRustCrypto,
    creator_storage: MemoryStorage,
    creator_signer: SignatureKeyPair,
    /// All member states, keyed by identity string
    members: HashMap<String, MemberState>,
    /// Group ID (safe string label)
    pub group_id_label: String,
}

impl OpenMlsGroup {
    /// Create a new MLS group with the given creator identity.
    /// Returns the OpenMlsGroup and the first MlsEvent (non-secret).
    pub fn new(
        creator_id: &str,
        group_id_label: &str,
    ) -> Result<(Self, MlsEvent), String> {
        // 1. Real OpenMLS provider (crypto backend + storage)
        let provider = OpenMlsRustCrypto::default();
        let storage = MemoryStorage::default();

        // 2. Real Ed25519 credential for creator
        let credential = BasicCredential::new(creator_id.as_bytes().to_vec());
        let signer = SignatureKeyPair::new(
            CIPHERSUITE.signature_algorithm(),
        )
        .map_err(|e| format!("SignatureKeyPair::new failed: {e}"))?;
        signer.store(&storage).map_err(|e| format!("signer.store: {e}"))?;

        let credential_with_key = CredentialWithKey {
            credential: credential.into(),
            signature_key: signer.public().into(),
        };

        // 3. Real MLS group creation via OpenMLS
        let group_config = MlsGroupCreateConfig::builder()
            .ciphersuite(CIPHERSUITE)
            .build();

        let group = MlsGroup::new(
            &provider,
            &signer,
            &group_config,
            credential_with_key.clone(),
        )
        .map_err(|e| format!("MlsGroup::new failed: {e}"))?;

        let epoch = group.epoch().as_u64();
        let now = now_unix();

        let mut members = HashMap::new();

        // Generate initial KeyPackage for creator (for future add operations)
        let kp = KeyPackage::builder()
            .build(
                CryptoConfig {
                    ciphersuite: CIPHERSUITE,
                    version: ProtocolVersion::default(),
                },
                &provider,
                &signer,
                credential_with_key.clone(),
            )
            .map_err(|e| format!("KeyPackage::builder failed: {e}"))?;

        members.insert(
            creator_id.to_string(),
            MemberState {
                credential_with_key,
                signer: signer.clone(),
                key_package: kp,
                storage: MemoryStorage::default(),
                provider: OpenMlsRustCrypto::default(),
                message_count: 0,
                commit_count: 0,
                last_activity_unix: now,
            },
        );

        let event = MlsEvent {
            timestamp_unix: now,
            epoch,
            member_id: creator_id.to_string(),
            event_type: MlsEventType::GroupCreated,
            message_size_bytes: None,
            message_count: 0,
            commit_count: 0,
            time_since_last_activity_secs: 0.0,
            peer_id: None,
            simulated_is_vpn: None,
            simulated_ip_changed: None,
            simulated_tz_changed: None,
            simulated_sync_frequency: None,
            simulated_session_duration_sec: None,
        };

        Ok((
            Self {
                group,
                creator_provider: provider,
                creator_storage: storage,
                creator_signer: signer,
                members,
                group_id_label: group_id_label.to_string(),
            },
            event,
        ))
    }

    /// Add a new member to the MLS group.
    /// Uses real OpenMLS: add_members() + merge_pending_commit()
    /// Returns a non-secret MlsEvent.
    pub fn add_member(&mut self, new_identity: &str) -> Result<MlsEvent, String> {
        let now = now_unix();

        // 1. Generate real KeyPackage for new member
        let new_provider = OpenMlsRustCrypto::default();
        let new_storage = MemoryStorage::default();

        let new_credential = BasicCredential::new(new_identity.as_bytes().to_vec());
        let new_signer = SignatureKeyPair::new(CIPHERSUITE.signature_algorithm())
            .map_err(|e| format!("signer new: {e}"))?;
        new_signer
            .store(&new_storage)
            .map_err(|e| format!("signer store: {e}"))?;

        let new_cred_with_key = CredentialWithKey {
            credential: new_credential.into(),
            signature_key: new_signer.public().into(),
        };

        let new_kp = KeyPackage::builder()
            .build(
                CryptoConfig {
                    ciphersuite: CIPHERSUITE,
                    version: ProtocolVersion::default(),
                },
                &new_provider,
                &new_signer,
                new_cred_with_key.clone(),
            )
            .map_err(|e| format!("kp build: {e}"))?;

        // 2. Real OpenMLS: Add the KeyPackage to the group (creator commits)
        let (commit, welcome_out, _group_info) = self
            .group
            .add_members(
                &self.creator_provider,
                &self.creator_signer,
                &[new_kp.clone()],
            )
            .map_err(|e| format!("add_members: {e}"))?;

        // 3. Merge the commit — real epoch advancement
        self.group
            .merge_pending_commit(&self.creator_provider)
            .map_err(|e| format!("merge_pending_commit: {e}"))?;

        let epoch = self.group.epoch().as_u64();

        // 4. The new member joins using the Welcome message
        let mls_group_config = MlsGroupJoinConfig::default();
        let mut new_member_group = StagedWelcome::new_from_welcome(
            &new_provider,
            &mls_group_config,
            welcome_out.into_welcome().map_err(|e| format!("into_welcome: {e}"))?,
            None,
        )
        .map_err(|e| format!("StagedWelcome: {e}"))?
        .into_group(&new_provider)
        .map_err(|e| format!("into_group: {e}"))?;

        // Store new member state (private keys stay in Rust process)
        self.members.insert(
            new_identity.to_string(),
            MemberState {
                credential_with_key: new_cred_with_key,
                signer: new_signer,
                key_package: new_kp,
                storage: new_storage,
                provider: new_provider,
                message_count: 0,
                commit_count: 1,
                last_activity_unix: now,
            },
        );

        // Update creator commit count
        if let Some(creator_state) = self.members.values_mut().next() {
            creator_state.commit_count += 1;
            creator_state.last_activity_unix = now;
        }

        Ok(MlsEvent {
            timestamp_unix: now,
            epoch,
            member_id: "creator".to_string(),
            event_type: MlsEventType::MemberAdded,
            message_size_bytes: None,
            message_count: 0,
            commit_count: 1,
            time_since_last_activity_secs: 0.0,
            peer_id: Some(new_identity.to_string()),
            simulated_is_vpn: None,
            simulated_ip_changed: None,
            simulated_tz_changed: None,
            simulated_sync_frequency: None,
            simulated_session_duration_sec: None,
        })
    }

    /// Send a real MLS ApplicationMessage from `sender_id`.
    /// Content is random bytes — content is NOT exposed to Python.
    /// Only the ciphertext SIZE is reported (safe metadata).
    pub fn send_message(
        &mut self,
        sender_id: &str,
        simulated_metadata: SimulatedMetadata,
    ) -> Result<MlsEvent, String> {
        let now = now_unix();

        // Use creator as the actual MLS sender for the PoC
        // (all members share the same group state in this single-process PoC)
        let plaintext = format!(
            "PoC application message from {} at epoch {}",
            sender_id,
            self.group.epoch().as_u64()
        );
        let plaintext_bytes = plaintext.as_bytes();

        // Real OpenMLS: encrypt application message
        let mls_message = self
            .group
            .create_message(
                &self.creator_provider,
                &self.creator_signer,
                plaintext_bytes,
            )
            .map_err(|e| format!("create_message: {e}"))?;

        // Extract ONLY the ciphertext size — content is discarded
        let ct_bytes = mls_message
            .tls_serialize_detached()
            .map_err(|e| format!("tls_serialize: {e}"))?;
        let ciphertext_size = ct_bytes.len();

        let epoch = self.group.epoch().as_u64();

        // Update sender telemetry counters (non-secret)
        let (msg_count, commit_count, last_active) =
            if let Some(state) = self.members.get_mut(sender_id) {
                state.message_count += 1;
                state.last_activity_unix = now;
                (state.message_count, state.commit_count, state.last_activity_unix)
            } else {
                (1, 0, now)
            };

        let time_since = simulated_metadata
            .time_since_last_activity_secs
            .unwrap_or(0.0);

        Ok(MlsEvent {
            timestamp_unix: now,
            epoch,
            member_id: sender_id.to_string(),
            event_type: MlsEventType::ApplicationMessage,
            message_size_bytes: Some(ciphertext_size),
            message_count: msg_count,
            commit_count,
            time_since_last_activity_secs: time_since,
            peer_id: None,
            simulated_is_vpn: simulated_metadata.is_vpn,
            simulated_ip_changed: simulated_metadata.ip_changed,
            simulated_tz_changed: simulated_metadata.tz_changed,
            simulated_sync_frequency: simulated_metadata.sync_frequency,
            simulated_session_duration_sec: simulated_metadata.session_duration_sec,
        })
    }

    /// Remove a member from the MLS group.
    /// Uses real OpenMLS: remove_members() + merge_pending_commit()
    /// This is the quarantine action — initiated by Python policy decision,
    /// but executed through the authorized MLS adapter (creator's keys).
    ///
    /// AUTHORIZATION: Only the creator (authorized group member) can call this.
    /// The ML model requests removal via the JSON interface; the Rust layer
    /// performs the actual MLS operation with the creator's signing key.
    pub fn remove_member(
        &mut self,
        target_id: &str,
    ) -> Result<(MlsEvent, u64), String> {
        let now = now_unix();

        // Find target's leaf index in the current group
        let target_leaf = self
            .group
            .members()
            .find(|m| {
                m.credential
                    .serialized_content()
                    == target_id.as_bytes()
            })
            .map(|m| m.index)
            .ok_or_else(|| format!("Member {target_id:?} not found in group"))?;

        let epoch_before = self.group.epoch().as_u64();

        // Real OpenMLS: Remove proposal + Commit
        let (commit, _welcome, _group_info) = self
            .group
            .remove_members(
                &self.creator_provider,
                &self.creator_signer,
                &[target_leaf],
            )
            .map_err(|e| format!("remove_members: {e}"))?;

        // Real epoch advancement — OpenMLS performs all cryptographic state transitions
        self.group
            .merge_pending_commit(&self.creator_provider)
            .map_err(|e| format!("merge_pending_commit after remove: {e}"))?;

        let epoch_after = self.group.epoch().as_u64();

        // Remove from local registry
        self.members.remove(target_id);

        let event = MlsEvent {
            timestamp_unix: now,
            epoch: epoch_after,
            member_id: "creator".to_string(),
            event_type: MlsEventType::MemberRemoved,
            message_size_bytes: None,
            message_count: 0,
            commit_count: 1,
            time_since_last_activity_secs: 0.0,
            peer_id: Some(target_id.to_string()),
            simulated_is_vpn: None,
            simulated_ip_changed: None,
            simulated_tz_changed: None,
            simulated_sync_frequency: None,
            simulated_session_duration_sec: None,
        };

        Ok((event, epoch_after))
    }

    /// Return current active member identity strings (non-secret).
    pub fn current_members(&self) -> Vec<String> {
        self.group
            .members()
            .filter_map(|m| {
                String::from_utf8(m.credential.serialized_content().to_vec()).ok()
            })
            .collect()
    }

    /// Return current MLS epoch number (non-secret public group state).
    pub fn current_epoch(&self) -> u64 {
        self.group.epoch().as_u64()
    }
}

/// Simulated contextual metadata injected by the PoC scenario.
/// These fields are NOT from OpenMLS internals — they represent
/// external behavioral context (device environment) that would be
/// obtained from device telemetry in a real deployment.
/// Documented per §16: "feature unavailable in real MLS — simulated".
#[derive(Debug, Default, serde::Deserialize)]
pub struct SimulatedMetadata {
    pub is_vpn: Option<f64>,
    pub ip_changed: Option<f64>,
    pub tz_changed: Option<f64>,
    pub sync_frequency: Option<f64>,
    pub session_duration_sec: Option<f64>,
    pub time_since_last_activity_secs: Option<f64>,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
