// mls_openmls/src/main.rs
//
// JSON stdin/stdout Command Loop
// ================================
// This binary is spawned as a subprocess by the Python behavioral pipeline.
// It reads JSON commands from stdin and writes JSON responses to stdout.
//
// PROTOCOL:
//   Python → Rust: {"cmd": "<command>", "args": {...}}  (newline-delimited JSON)
//   Rust → Python: {"status": "ok"|"error", "event": {...}, "current_members": [...], "current_epoch": N}
//
// COMMANDS:
//   create_group    {"creator_id": str, "group_id": str}
//   add_member      {"identity": str}
//   send_message    {"sender_id": str, "metadata": {...}}
//   remove_member   {"target_id": str}
//   get_epoch       {}
//   get_members     {}
//   quit            {}
//
// SECURITY: This process owns all MLS cryptographic state.
//   Python receives ONLY non-secret telemetry (MlsEvent fields).
//   Private keys, epoch secrets, path secrets never leave this process.

mod adapter;
mod group;

use std::io::{self, BufRead, Write};

use adapter::{Command, EventEnvelope, MlsEvent, MlsEventType};
use group::{OpenMlsGroup, SimulatedMetadata};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut mls_group: Option<OpenMlsGroup> = None;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(e) => {
                emit_error(&mut out, &format!("stdin read error: {e}"));
                break;
            }
        };

        let cmd: Command = match serde_json::from_str(&line) {
            Ok(c) => c,
            Err(e) => {
                emit_error(&mut out, &format!("JSON parse error: {e}"));
                continue;
            }
        };

        match cmd.cmd.as_str() {
            // ── create_group ──────────────────────────────────────────────
            "create_group" => {
                let creator_id = cmd.args["creator_id"]
                    .as_str()
                    .unwrap_or("Alice");
                let group_id = cmd.args["group_id"]
                    .as_str()
                    .unwrap_or("qtk_poc_group");

                match OpenMlsGroup::new(creator_id, group_id) {
                    Ok((g, event)) => {
                        let members = g.current_members();
                        let epoch = g.current_epoch();
                        mls_group = Some(g);
                        emit(&mut out, EventEnvelope::ok(event, members, epoch));
                    }
                    Err(e) => emit_error(&mut out, &e),
                }
            }

            // ── add_member ────────────────────────────────────────────────
            "add_member" => {
                let identity = cmd.args["identity"].as_str().unwrap_or("unknown");
                match mls_group.as_mut() {
                    None => emit_error(&mut out, "Group not created yet"),
                    Some(g) => match g.add_member(identity) {
                        Ok(event) => {
                            let members = g.current_members();
                            let epoch = g.current_epoch();
                            emit(&mut out, EventEnvelope::ok(event, members, epoch));
                        }
                        Err(e) => emit_error(&mut out, &e),
                    },
                }
            }

            // ── send_message ──────────────────────────────────────────────
            "send_message" => {
                let sender_id = cmd.args["sender_id"].as_str().unwrap_or("unknown");
                let meta: SimulatedMetadata = cmd
                    .args
                    .get("metadata")
                    .and_then(|m| serde_json::from_value(m.clone()).ok())
                    .unwrap_or_default();

                match mls_group.as_mut() {
                    None => emit_error(&mut out, "Group not created yet"),
                    Some(g) => match g.send_message(sender_id, meta) {
                        Ok(event) => {
                            let members = g.current_members();
                            let epoch = g.current_epoch();
                            emit(&mut out, EventEnvelope::ok(event, members, epoch));
                        }
                        Err(e) => emit_error(&mut out, &e),
                    },
                }
            }

            // ── remove_member ─────────────────────────────────────────────
            // This is the quarantine action.
            // Authorization: only called when Python policy requests removal.
            // The actual MLS Remove Commit is performed by the creator's signing key
            // (held by this Rust process — never exposed to Python).
            "remove_member" => {
                let target = cmd.args["target_id"].as_str().unwrap_or("unknown");
                match mls_group.as_mut() {
                    None => emit_error(&mut out, "Group not created yet"),
                    Some(g) => match g.remove_member(target) {
                        Ok((event, new_epoch)) => {
                            let members = g.current_members();
                            emit(&mut out, EventEnvelope::ok(event, members, new_epoch));
                        }
                        Err(e) => emit_error(&mut out, &e),
                    },
                }
            }

            // ── get_epoch ─────────────────────────────────────────────────
            "get_epoch" => {
                match mls_group.as_ref() {
                    None => emit_error(&mut out, "Group not created"),
                    Some(g) => {
                        let epoch = g.current_epoch();
                        let members = g.current_members();
                        let event = MlsEvent {
                            timestamp_unix: MlsEvent::now_unix(),
                            epoch,
                            member_id: "system".into(),
                            event_type: MlsEventType::KeyUpdate,
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
                        emit(&mut out, EventEnvelope::ok(event, members, epoch));
                    }
                }
            }

            // ── get_members ───────────────────────────────────────────────
            "get_members" => {
                match mls_group.as_ref() {
                    None => emit_error(&mut out, "Group not created"),
                    Some(g) => {
                        let members = g.current_members();
                        let epoch = g.current_epoch();
                        let event = MlsEvent {
                            timestamp_unix: MlsEvent::now_unix(),
                            epoch,
                            member_id: "system".into(),
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
                        emit(&mut out, EventEnvelope::ok(event, members, epoch));
                    }
                }
            }

            // ── quit ──────────────────────────────────────────────────────
            "quit" => {
                let _ = writeln!(out, r#"{{"status":"quit"}}"#);
                break;
            }

            unknown => {
                emit_error(&mut out, &format!("Unknown command: {unknown:?}"));
            }
        }
    }
}

fn emit(out: &mut impl Write, envelope: EventEnvelope) {
    if let Ok(json) = serde_json::to_string(&envelope) {
        let _ = writeln!(out, "{json}");
        let _ = out.flush();
    }
}

fn emit_error(out: &mut impl Write, msg: &str) {
    emit(out, EventEnvelope::error(msg));
}
