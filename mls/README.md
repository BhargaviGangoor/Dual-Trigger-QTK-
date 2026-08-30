# OpenMLS PoC Integration — `mls_openmls/`

Real OpenMLS 0.9.0 adapter for the Dual-Trigger QTK behavioral detection
proof-of-concept. Demonstrates that the existing behavioral quarantine
policy can operate alongside a **real MLS implementation** without
modifying any MLS cryptographic primitives.

> **Paper-safe terminology:** This is an *OpenMLS proof-of-concept integration*
> and *real-MLS feasibility demonstration*. It is NOT a formal MLS security
> proof, compositional security proof, or proof of forward secrecy.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Rust (MSVC) | ≥ 1.91.0 (tested: 1.98.0) |
| OpenMLS | 0.9.0 (fetched by Cargo) |
| Python | 3.14.4 |
| `cryptography` lib | ≥ 40.0 (for pure-Python reference `mls/`) |

---

## Reproduction Instructions

### 1. Build the Rust binary (one-time, ~3–5 min first build)

```powershell
cd mls_openmls
cargo build --release
```

The binary will be at:
```
mls_openmls/target/release/mls_openmls.exe   (Windows)
mls_openmls/target/release/mls_openmls       (Linux/macOS)
```

### 2. Run the PoC

```powershell
# From project root
py -3 experiments/mls_poc_openmls.py
```

### 3. Run tests

```powershell
py -3 -m pytest tests/test_mls_openmls.py -v
```

Test 12 (`test_12_ml_layer_never_receives_secrets`) runs without the
binary. Tests 1–11 require the Rust binary (auto-skipped if not built).

### 4. Inspect results

```powershell
type results\raw\mls_poc.json
type results\tables\mls_poc_summary.csv
```

---

## Architecture

```
Python (experiments/mls_poc_openmls.py)
  │
  │  JSON stdin/stdout
  │
Rust (mls_openmls/src/main.rs)
  │
  └── OpenMLS 0.9.0 (openmls crate)
        MlsGroup::new()
        group.add_members()
        group.create_message()
        group.remove_members()
        group.merge_pending_commit()
```

---

## Module Structure

| File | Purpose |
|---|---|
| `Cargo.toml` | Rust crate manifest (openmls = "0.9", serde_json) |
| `src/adapter.rs` | `MlsEvent` struct — non-secret telemetry boundary |
| `src/group.rs` | `OpenMlsGroup` — real OpenMLS group lifecycle wrapper |
| `src/main.rs` | JSON stdin/stdout command loop |

---

## Security Boundary

### ✅ Exposed to Python (non-secret telemetry)

| Field | Source |
|---|---|
| `epoch` | `MlsGroup::epoch()` — public group state |
| `member_id` | `BasicCredential` identity string |
| `event_type` | `APPLICATION_MESSAGE`, `MEMBER_ADDED`, `MEMBER_REMOVED` |
| `message_size_bytes` | `len(tls_serialize())` — ciphertext size only |
| `message_count` | Running counter per member |
| `commit_count` | Running counter per member |
| `time_since_last_activity_secs` | Wall-clock delta |

### ❌ Never exposed to Python

| Field | Reason |
|---|---|
| `epoch_secret` | Forward secrecy material |
| `init_secret` | Epoch key schedule |
| `joiner_secret` | New member key material |
| `path_secret` | TreeKEM ratchet secret |
| `sender_data_secret` | Message encryption key |
| `encryption_secret` | Application key material |
| Private signing keys | Ed25519 credential keys |
| Decrypted application content | Confidential payload |

### ⚠️ Simulated fields (documented per §16)

The following fields are NOT available from OpenMLS protocol internals.
They are injected as scenario parameters and documented as
*simulated behavioral metadata*:

| Field | Limitation |
|---|---|
| `session_duration_sec` | Not an MLS field — simulated |
| `sync_frequency` | Not an MLS field — simulated |
| `is_vpn` | Not an MLS field — simulated |
| `ip_changed` | Not an MLS field — simulated |
| `tz_changed` | Not an MLS field — simulated |
| `network_type` | Not an MLS field — set to "Unknown" |
| `network_ip` | Not an MLS field — set to "0.0.0.0" |
| `location_country` | Not an MLS field — set to "Unknown" |
| `active_timezone` | Not an MLS field — set to "UTC" |

In a real deployment, these fields would come from device telemetry
(OS network APIs, MDM enrollment data), not from MLS protocol state.

---

## Authorization Model

```
Behavioral detector (Python)
        ↓ risk ≥ theta_R
DualTrigger → BEHAVIORAL
        ↓ policy decision
QuarantineInterface.request_removal("Dave")
        ↓ JSON command (remove_member)
Rust process (authorized creator's signing key)
        ↓ MlsGroup::remove_members() + merge_pending_commit()
OpenMLS performs cryptographic state transition
        ↓ new epoch
Dave is no longer an MLS group member
```

The ML model has **no direct access** to MLS key material.

---

## Known Limitations

1. **Single-process PoC**: All members share one Rust process and one
   `MlsGroup` instance. A production deployment would have separate
   processes per member communicating via a Delivery Service.

2. **Simulated behavioral metadata**: `session_duration_sec`,
   `sync_frequency`, `is_vpn`, `ip_changed`, `tz_changed` are injected
   by the scenario, not extracted from MLS events.

3. **No Delivery Service**: This PoC uses in-process Welcome/Commit
   distribution. Real MLS deployments require an authenticated
   Delivery Service.

4. **No formal security proof**: This PoC demonstrates implementation
   feasibility. It does NOT provide:
   - Formal MLS security proof
   - Compositional security proof
   - Proof of forward secrecy for the combined construction
   - Proof that the behavioral detector cannot cause denial of service

5. **OpenMLS version**: Uses `openmls = "0.9"` (requires Rust ≥ 1.91.0).
   The API may change in future OpenMLS releases.

---

## Version Information

| Component | Version |
|---|---|
| OpenMLS | 0.9.0 |
| Rust (required) | ≥ 1.91.0 |
| openmls_rust_crypto | 0.9 |
| openmls_basic_credential | 0.9 |
| openmls_memory_storage | 0.9 |

---

## Relationship to Frozen Benchmark

This PoC is **isolated** from the simulation benchmark:

| What | Status |
|---|---|
| Tables 1–6 | ✅ Unchanged |
| Benchmark seeds | ✅ Unchanged |
| theta_R (benchmark) | ✅ Unchanged |
| Model weights | ✅ Unchanged |
| `results/raw/mls_poc.json` | ← PoC output only |
| `results/tables/mls_poc_summary.*` | ← PoC table only |
