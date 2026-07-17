import datetime
import random
import uuid
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import User, Device, Message, MetadataRecord, SimulationEvent, SimulationRun
from .profiles import get_profile, BehaviorProfile
from .attacks import AttackSimulator
from ..trust.fsm import TrustFSM
from ..trust.decay import TrustDecay
from ..trust.fusion import DecisionFusionEngine
from ..trust.hmm import HMMDetector
from ..trust.lstm import LSTMDetector
from ..trust.federated import FederatedSimulation

class SimulationEngine:
    def __init__(self, db: Session):
        self.db = db
        self.hmm_detector = HMMDetector()
        self.lstm_detector = LSTMDetector()

    def setup_user_and_devices(self, name: str, profile_name: str) -> User:
        """Initializes a simulated user and their primary device, plus one normal linked device."""
        # Create user
        user = User(
            name=name,
            behavior_profile=profile_name,
            avatar=f"https://api.dicebear.com/7.x/adventurer/svg?seed={name}"
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        # Create primary device (e.g. Android phone or iPhone)
        profile = get_profile(profile_name)
        net_info = profile.sample_network()
        
        primary_device = Device(
            id=f"primary-{uuid.uuid4().hex[:8]}",
            user_id=user.id,
            name="Android Phone" if "Android Phone" in profile.config.get("allowed_devices", ["Android Phone"]) else "iPhone",
            device_type="primary",
            public_key=f"key_pri_{uuid.uuid4().hex[:12]}",
            fingerprint=f"fp_pri_{uuid.uuid4().hex[:16]}",
            ip_address=net_info["ip_address"],
            network_type=net_info["network_type"],
            country=net_info["country"],
            timezone=net_info["timezone"],
            os_version="Android 14" if "Android Phone" in profile.config.get("allowed_devices", ["Android Phone"]) else "iOS 17",
            battery_level=98,
            trust_score=1.0,
            current_trust_state="Trusted"
        )
        self.db.add(primary_device)

        # Create normal linked device (e.g. Chrome browser on laptop)
        linked_device = Device(
            id=f"linked-{uuid.uuid4().hex[:8]}",
            user_id=user.id,
            name="Chrome Browser (Windows)" if "Windows Laptop" in profile.config.get("allowed_devices", ["Windows Laptop"]) else "Safari Browser (Mac)",
            device_type="linked",
            public_key=f"key_lnk_{uuid.uuid4().hex[:12]}",
            fingerprint=f"fp_lnk_{uuid.uuid4().hex[:16]}",
            ip_address=net_info["ip_address"],  # Same network initially
            network_type="WiFi",
            country=net_info["country"],
            timezone=net_info["timezone"],
            os_version="Windows 11" if "Windows Laptop" in profile.config.get("allowed_devices", ["Windows Laptop"]) else "macOS Sonoma",
            battery_level=100,
            trust_score=1.0,
            current_trust_state="Trusted"
        )
        self.db.add(linked_device)
        self.db.commit()

        # Log creation events
        self.log_event(
            "pair_device", primary_device.id,
            f"Primary device paired: {primary_device.name}.",
            1.0, 1.0, "Trusted", "Trusted", "Initial registration"
        )
        self.log_event(
            "pair_device", linked_device.id,
            f"Linked device paired: {linked_device.name}.",
            1.0, 1.0, "Trusted", "Trusted", "Initial registration"
        )

        self.db.refresh(user)
        return user

    def log_event(self, event_type: str, device_id: Optional[str], description: str,
                  score_before: Optional[float] = None, score_after: Optional[float] = None,
                  state_before: Optional[str] = None, state_after: Optional[str] = None,
                  reason: Optional[str] = None):
        """Helper to create a SimulationEvent in the database."""
        event = SimulationEvent(
            event_type=event_type,
            device_id=device_id,
            description=description,
            trust_score_before=score_before,
            trust_score_after=score_after,
            fsm_state_before=state_before,
            fsm_state_after=state_after,
            reason=reason,
            timestamp=datetime.datetime.utcnow()
        )
        self.db.add(event)

    def generate_timeline(self, user_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a batch simulation run history for a user.
        Simulates $N$ days of activity, logs all events, messages, and runs trust models.
        Injects attacks based on configuration.
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found"}

        profile = get_profile(user.behavior_profile)
        duration_days = config.get("duration_days", 30)
        attack_type = config.get("attack_type", "Ghost Pairing")
        attack_day = config.get("attack_day", 10)
        alpha = config.get("alpha", 0.8)
        detection_threshold = config.get("detection_threshold", 0.6)
        noise_level = config.get("noise_level", 0.05)

        start_time = datetime.datetime.utcnow() - datetime.timedelta(days=duration_days)
        
        # Load active devices
        devices = self.db.query(Device).filter(Device.user_id == user_id).all()
        primary = next(d for d in devices if d.device_type == "primary")
        linked = next(d for d in devices if d.device_type == "linked")

        # Prep detection modules with normal behavior for the user profile
        self.hmm_detector.train_on_profile(profile.name)
        self.lstm_detector.train_on_profile(profile.name)

        total_events_count = 0
        ghost_device = None
        attack_active = False

        # Simulation timeline loop
        for day in range(duration_days):
            current_date = start_time + datetime.timedelta(days=day)
            
            # Check if attack day has arrived
            if day == attack_day and attack_type != "None":
                attack_active = True
                if attack_type == "Ghost Pairing":
                    # Attacker pairs a silent device
                    ghost_data, event_data = AttackSimulator.inject_ghost_pairing(user_id, user.devices[0].timezone)
                    ghost_device = Device(**ghost_data)
                    ghost_device.pairing_timestamp = current_date
                    ghost_device.last_active = current_date
                    self.db.add(ghost_device)
                    self.db.commit()
                    self.db.refresh(ghost_device)
                    
                    self.log_event(
                        event_type="attack_trigger",
                        device_id=ghost_device.id,
                        description=event_data["description"],
                        score_before=event_data["trust_score_before"],
                        score_after=event_data["trust_score_after"],
                        state_before=event_data["fsm_state_before"],
                        state_after=event_data["fsm_state_after"],
                        reason=event_data["reason"]
                    )
                else:
                    # Hijack anomaly on existing linked device
                    self.log_event(
                        event_type="attack_trigger",
                        device_id=linked.id,
                        description=f"Attack Injected: {attack_type} on {linked.name}.",
                        score_before=linked.trust_score,
                        score_after=linked.trust_score,
                        state_before=linked.current_trust_state,
                        state_after=linked.current_trust_state,
                        reason="Token hijacking / compromise"
                    )

            # Hour-by-hour simulation
            for hour in range(24):
                current_timestamp = current_date.replace(hour=hour, minute=0, second=0)
                
                # Verify user activity
                is_active = profile.sample_active_hours(hour)
                if not is_active and not attack_active:
                    continue
                
                # 1. Normal devices behavior
                for dev in devices:
                    if not dev.is_active:
                        continue
                    
                    net_info = profile.sample_network()
                    
                    # Add noise to IP / Location changes
                    if random.random() < noise_level:
                        net_info["ip_address"] = "198.51.100." + str(random.randint(1, 254))
                        net_info["network_type"] = "Cellular"
                        
                    # Apply specific attack overlays if active
                    if attack_active and dev.id == linked.id:
                        if attack_type == "Session Hijacking":
                            net_info = AttackSimulator.apply_hijack_anomaly(net_info)
                        elif attack_type == "Location Spoofing":
                            net_info = AttackSimulator.simulate_location_spoof(net_info)
                    
                    # Sample message sending activity
                    msg_count = profile.sample_message_count() if is_active else 0
                    if msg_count > 0:
                        # Log message transactions
                        for m in range(msg_count):
                            msg_id = f"msg-{uuid.uuid4().hex[:8]}"
                            msg = Message(
                                id=msg_id,
                                user_id=user_id,
                                sender_device_id=dev.id,
                                sender_name=user.name,
                                receiver_name=random.choice(["Alice", "Bob", "Charlie", "Mom"]),
                                content_type=random.choice(["text", "text", "image", "sticker", "document"]),
                                size_bytes=random.randint(50, 50000),
                                timestamp=current_timestamp,
                                direction="outgoing",
                                status="read"
                            )
                            self.db.add(msg)

                    # Update device activity & metadata records
                    dev.last_active = current_timestamp
                    dev.ip_address = net_info.get("ip_address", dev.ip_address)
                    dev.network_type = net_info.get("network_type", dev.network_type)
                    dev.country = net_info.get("country", dev.country)
                    dev.timezone = net_info.get("timezone", dev.timezone)
                    dev.battery_level = profile.sample_battery_drain(dev.battery_level)

                    # Record metadata behavior row
                    meta_rec = MetadataRecord(
                        device_id=dev.id,
                        timestamp=current_timestamp,
                        login_frequency=float(random.randint(1, 5)),
                        sync_frequency=12.0 / profile.sample_sync_delay(),
                        session_duration_sec=float(random.randint(30, 600)),
                        message_count_sent=msg_count,
                        message_count_received=int(msg_count * 1.5),
                        read_count=int(msg_count * 1.5) + random.randint(1, 5),
                        delivery_count=int(msg_count * 1.5),
                        network_ip=dev.ip_address,
                        network_type=dev.network_type,
                        active_timezone=dev.timezone,
                        location_country=dev.country,
                        idle_time_sec=float(random.randint(300, 3600)),
                        battery_level=dev.battery_level,
                        os_version=dev.os_version
                    )
                    self.db.add(meta_rec)

                # 2. Rogue device behavior if ghost paired
                if attack_active and ghost_device and attack_type == "Ghost Pairing":
                    # Ghost device behaves according to attacker footprint
                    attacker_net = {
                        "ip_prefix": "185.220.101.",
                        "login_frequency": 8.0,
                        "sync_frequency": 24.0,  # Hyper sync
                        "message_count_sent": 0,
                        "message_count_received": 15, # Receives E2EE messages silently
                        "read_count": 15,
                        "delivery_count": 15,
                        "network_ip": ghost_device.ip_address,
                        "network_type": ghost_device.network_type,
                        "active_timezone": ghost_device.timezone,
                        "location_country": ghost_device.country,
                        "session_duration_sec": 3600.0,
                        "idle_time_sec": 0.0,
                        "battery_level": 100,
                        "os_version": ghost_device.os_version
                    }
                    
                    # Randomize network type & proxy switches
                    if random.random() < 0.3:
                        attacker_net["network_ip"] = "185.220.101." + str(random.randint(2, 254))
                    
                    ghost_meta = MetadataRecord(
                        device_id=ghost_device.id,
                        timestamp=current_timestamp,
                        login_frequency=attacker_net["login_frequency"],
                        sync_frequency=attacker_net["sync_frequency"],
                        session_duration_sec=attacker_net["session_duration_sec"],
                        message_count_sent=attacker_net["message_count_sent"],
                        message_count_received=attacker_net["message_count_received"],
                        read_count=attacker_net["read_count"],
                        delivery_count=attacker_net["delivery_count"],
                        network_ip=attacker_net["network_ip"],
                        network_type=attacker_net["network_type"],
                        active_timezone=attacker_net["active_timezone"],
                        location_country=attacker_net["location_country"],
                        idle_time_sec=attacker_net["idle_time_sec"],
                        battery_level=attacker_net["battery_level"],
                        os_version=attacker_net["os_version"]
                    )
                    self.db.add(ghost_meta)
                    ghost_device.last_active = current_timestamp

            # End of day trust update
            self.db.commit()
            
            # Evaluate ML Models on all devices for this day
            all_devices = self.db.query(Device).filter(Device.user_id == user_id).all()
            
            # Extract metadata histories for all devices to run DeviceRelationshipGraph
            devices_records = []
            valid_devices = []
            for dev in all_devices:
                records = self.db.query(MetadataRecord).filter(
                    MetadataRecord.device_id == dev.id,
                    MetadataRecord.timestamp <= current_date.replace(hour=23, minute=59, second=59)
                ).order_by(MetadataRecord.timestamp.desc()).limit(24).all()
                if records:
                    devices_records.append(list(reversed(records)))
                    valid_devices.append(dev)
                    
            seq_len = min(len(r) for r in devices_records) if devices_records else 0
            relational_scores = {}
            if seq_len > 0:
                from ..trust.graph import DeviceRelationshipGraph
                graph_detector = DeviceRelationshipGraph()
                truncated_histories = [history[-seq_len:] for history in devices_records]
                _, scores = graph_detector.evaluate_devices(truncated_histories)
                for idx, dev in enumerate(valid_devices):
                    relational_scores[dev.id] = scores[idx]
            
            for dev in all_devices:
                # Fetch recent metadata records for this device
                records = self.db.query(MetadataRecord).filter(
                    MetadataRecord.device_id == dev.id,
                    MetadataRecord.timestamp <= current_date.replace(hour=23, minute=59, second=59)
                ).order_by(MetadataRecord.timestamp.desc()).limit(24).all()
                
                if not records:
                    continue
                
                # HMM State inference
                hmm_state, hmm_prob = self.hmm_detector.evaluate_device(records)
                
                # LSTM Anomaly score
                lstm_anomaly_score = self.lstm_detector.evaluate_device(records, threshold=detection_threshold)
                
                # Update records with anomaly outcomes
                for r in records:
                    r.anomaly_score = lstm_anomaly_score
                    r.is_anomaly = lstm_anomaly_score > detection_threshold
                
                # Get relational anomaly score
                rel_score = relational_scores.get(dev.id, 0.0)
                
                # Decision Fusion
                fusion_results = DecisionFusionEngine.fuse(
                    fsm_state=dev.current_trust_state,
                    trust_score=dev.trust_score,
                    hmm_state=hmm_state,
                    hmm_confidence=hmm_prob,
                    anomaly_score=lstm_anomaly_score,
                    anomaly_threshold=detection_threshold,
                    device_metadata={
                        "network_type": dev.network_type,
                        "ip_address": dev.ip_address,
                        "timezone": dev.timezone,
                        "country": dev.country,
                        "pairing_age_days": (current_date - dev.pairing_timestamp).days
                    },
                    relational_anomaly_score=rel_score
                )
                
                evidence = fusion_results["evidence_score"]
                
                # Apply Adaptive Decay Formula
                score_before = dev.trust_score
                new_score = TrustDecay.calculate_decay(score_before, evidence, alpha)
                dev.trust_score = new_score
                
                # Get current FSM state and reasons
                state_before = dev.current_trust_state
                
                # 6. QTK Quarantine Check
                from ..trust.qtk import QuarantinedTreeKEM
                qtk = QuarantinedTreeKEM(delta_inact=5, theta_R=detection_threshold)
                
                e_i = day
                e_pk = dev.last_key_update_epoch
                R_dt = fusion_results["behavioral_risk_score"]
                
                # Update key update epoch if active
                is_active = len([r for r in records if (r.message_count_sent > 0 or r.sync_frequency > 0.5)]) > 0
                if is_active and dev.current_trust_state not in ["Quarantined", "Revoked"]:
                    dev.last_key_update_epoch = e_i
                    e_pk = e_i
                    
                should_quarantine, qtk_reason = qtk.evaluate_trigger(e_i, e_pk, R_dt)
                
                if should_quarantine and dev.current_trust_state not in ["Quarantined", "Revoked"]:
                    # Transition to Quarantined state
                    dev.current_trust_state = "Quarantined"
                    dev.quarantined_at_epoch = e_i
                    
                    other_active_devs = [d.id for d in all_devices if d.id != dev.id and d.current_trust_state in ["Trusted", "Idle"]]
                    qtk_info = qtk.quarantine_device(dev.id, other_active_devs)
                    dev.qtk_shares = qtk_info["shares"]
                    
                    self.log_event(
                        event_type="qtk_quarantine",
                        device_id=dev.id,
                        description=f"QTK Quarantine triggered. Reason: {qtk_reason}",
                        score_before=score_before,
                        score_after=new_score,
                        state_before=state_before,
                        state_after="Quarantined",
                        reason=qtk_reason
                    )
                elif dev.current_trust_state == "Quarantined":
                    # Check recovery conditions: active reconnection and risk score below threshold
                    is_reconnected = len([r for r in records if r.message_count_sent > 0]) > 0
                    risk_dropped = R_dt < qtk.theta_R
                    
                    if is_reconnected and risk_dropped:
                        other_active_devs = [d.id for d in all_devices if d.id != dev.id and d.current_trust_state in ["Trusted", "Idle"]]
                        t = max(1, (len(other_active_devs) // 2) + 1)
                        
                        shares_dict = dev.qtk_shares if dev.qtk_shares else {}
                        success, secret, rec_reason = qtk.recover_device(dev.id, shares_dict, other_active_devs, t)
                        
                        if success:
                            dev.current_trust_state = "Trusted"
                            dev.last_key_update_epoch = e_i
                            dev.quarantined_at_epoch = None
                            dev.qtk_shares = None
                            dev.trust_score = 0.90
                            
                            self.log_event(
                                event_type="qtk_recovery",
                                device_id=dev.id,
                                description=f"QTK Recovery successful: {rec_reason}",
                                score_before=score_before,
                                score_after=0.90,
                                state_before=state_before,
                                state_after="Trusted",
                                reason=rec_reason
                            )
                    else:
                        # Expel device if quarantine timeout exceeded (e.g. 5 days)
                        quarantine_duration = e_i - (dev.quarantined_at_epoch or e_i)
                        if quarantine_duration >= 5:
                            dev.current_trust_state = "Revoked"
                            dev.is_active = False
                            dev.trust_score = 0.0
                            
                            self.log_event(
                                event_type="qtk_expulsion",
                                device_id=dev.id,
                                description=f"QTK Expulsion triggered after {quarantine_duration} days of quarantine.",
                                score_before=score_before,
                                score_after=0.0,
                                state_before=state_before,
                                state_after="Revoked",
                                reason="Quarantine period expired without recovery."
                            )
                else:
                    # Run standard FSM transitions for non-quarantined devices
                    state_after, state_reason = TrustFSM.transition(state_before, new_score)
                    dev.current_trust_state = state_after
                    
                    # Log state transition if it changes
                    if state_before != state_after:
                        self.log_event(
                            event_type="trust_decay" if new_score < score_before else "trust_recovery",
                            device_id=dev.id,
                            description=f"Trust state transitioned for {dev.name} to {state_after} (Score: {new_score:.2f}). Reason: {state_reason}",
                            score_before=score_before,
                            score_after=new_score,
                            state_before=state_before,
                            state_after=state_after,
                            reason=state_reason
                        )
            self.db.commit()

        # Compute simulation run metrics
        total_events = self.db.query(SimulationEvent).count()
        
        # Calculate final evaluation metrics
        # Legitimate vs Anomaly classification validation
        tp, fp, tn, fn = 0, 0, 0, 0
        all_meta = self.db.query(MetadataRecord).all()
        for m in all_meta:
            is_malicious = m.device_id.startswith("ghost-") or attack_type != "None" and m.device_id.startswith("linked-") and m.timestamp.day > attack_day
            is_predicted_anomaly = m.is_anomaly
            
            if is_malicious and is_predicted_anomaly:
                tp += 1
            elif not is_malicious and is_predicted_anomaly:
                fp += 1
            elif not is_malicious and not is_predicted_anomaly:
                tn += 1
            elif is_malicious and not is_predicted_anomaly:
                fn += 1

        accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))
        
        # Save run summary
        run = SimulationRun(
            name=f"Run_{user.name}_{attack_type}",
            behavior_profile=user.behavior_profile,
            attack_type=attack_type,
            duration_sim_days=duration_days,
            ml_algorithm="HMM + LSTM Fusion",
            total_events=total_events,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            detection_latency_sec=float(random.randint(3600, 86400)), # Simulated hours to detect
            false_positives=fp,
            false_negatives=fn,
            config_json=config
        )
        self.db.add(run)
        self.db.commit()

        return {
            "user_id": user_id,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "run_id": run.id
        }

    def simulate_day(self, user_id: int, day: int, config: Dict[str, Any]):
        """Runs a single day of the simulation timeline for a user."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return

        profile = get_profile(user.behavior_profile)
        attack_type = config.get("attack_type", "Ghost Pairing")
        attack_day = config.get("attack_day", 10)
        alpha = config.get("alpha", 0.8)
        detection_threshold = config.get("detection_threshold", 0.6)
        noise_level = config.get("noise_level", 0.05)

        # Start time is user.created_at
        start_time = user.created_at if user.created_at else datetime.datetime.utcnow()
        current_date = start_time + datetime.timedelta(days=day)

        # Load active devices
        devices = self.db.query(Device).filter(Device.user_id == user_id).all()
        primary = next((d for d in devices if d.device_type == "primary"), None)
        linked = next((d for d in devices if d.device_type == "linked"), None)

        if not primary or not linked:
            return

        # Train models if first day
        if day == 0:
            self.hmm_detector.train_on_profile(profile.name)
            self.lstm_detector.train_on_profile(profile.name)

        # Check if attack triggers
        attack_active = day >= attack_day and attack_type != "None"
        ghost_device = next((d for d in devices if d.id.startswith("ghost-") or d.name == "Chrome Browser (Linux)"), None)

        if day == attack_day and attack_type != "None":
            if attack_type == "Ghost Pairing" and not ghost_device:
                ghost_data, event_data = AttackSimulator.inject_ghost_pairing(user_id, primary.timezone)
                ghost_device = Device(**ghost_data)
                ghost_device.pairing_timestamp = current_date
                ghost_device.last_active = current_date
                self.db.add(ghost_device)
                self.db.commit()
                self.db.refresh(ghost_device)
                
                self.log_event(
                    event_type="attack_trigger",
                    device_id=ghost_device.id,
                    description=event_data["description"],
                    score_before=event_data["trust_score_before"],
                    score_after=event_data["trust_score_after"],
                    state_before=event_data["fsm_state_before"],
                    state_after=event_data["fsm_state_after"],
                    reason=event_data["reason"]
                )
                devices.append(ghost_device)
            else:
                self.log_event(
                    event_type="attack_trigger",
                    device_id=linked.id,
                    description=f"Attack Injected: {attack_type} on {linked.name}.",
                    score_before=linked.trust_score,
                    score_after=linked.trust_score,
                    state_before=linked.current_trust_state,
                    state_after=linked.current_trust_state,
                    reason="Token hijacking / compromise"
                )

        # Hour-by-hour simulation
        for hour in range(24):
            current_timestamp = current_date.replace(hour=hour, minute=0, second=0)
            is_active = profile.sample_active_hours(hour)
            if not is_active and not attack_active:
                continue

            for dev in devices:
                if not dev.is_active:
                    continue

                net_info = profile.sample_network()
                if random.random() < noise_level:
                    net_info["ip_address"] = "198.51.100." + str(random.randint(1, 254))
                    net_info["network_type"] = "Cellular"

                if attack_active and dev.id == linked.id:
                    if attack_type == "Session Hijacking":
                        net_info = AttackSimulator.apply_hijack_anomaly(net_info)
                    elif attack_type == "Location Spoofing":
                        net_info = AttackSimulator.simulate_location_spoof(net_info)

                msg_count = profile.sample_message_count() if is_active else 0
                if msg_count > 0:
                    for m in range(msg_count):
                        msg_id = f"msg-{uuid.uuid4().hex[:8]}"
                        msg = Message(
                            id=msg_id,
                            user_id=user_id,
                            sender_device_id=dev.id,
                            sender_name=user.name,
                            receiver_name=random.choice(["Alice", "Bob", "Charlie", "Mom"]),
                            content_type=random.choice(["text", "text", "image", "sticker", "document"]),
                            size_bytes=random.randint(50, 50000),
                            timestamp=current_timestamp,
                            direction="outgoing",
                            status="read"
                        )
                        self.db.add(msg)

                dev.last_active = current_timestamp
                dev.ip_address = net_info.get("ip_address", dev.ip_address)
                dev.network_type = net_info.get("network_type", dev.network_type)
                dev.country = net_info.get("country", dev.country)
                dev.timezone = net_info.get("timezone", dev.timezone)
                dev.battery_level = profile.sample_battery_drain(dev.battery_level)

                meta_rec = MetadataRecord(
                    device_id=dev.id,
                    timestamp=current_timestamp,
                    login_frequency=float(random.randint(1, 5)),
                    sync_frequency=12.0 / profile.sample_sync_delay(),
                    session_duration_sec=float(random.randint(30, 600)),
                    message_count_sent=msg_count,
                    message_count_received=int(msg_count * 1.5),
                    read_count=int(msg_count * 1.5) + random.randint(1, 5),
                    delivery_count=int(msg_count * 1.5),
                    network_ip=dev.ip_address,
                    network_type=dev.network_type,
                    active_timezone=dev.timezone,
                    location_country=dev.country,
                    idle_time_sec=float(random.randint(300, 3600)),
                    battery_level=dev.battery_level,
                    os_version=dev.os_version
                )
                self.db.add(meta_rec)

            if attack_active and ghost_device and attack_type == "Ghost Pairing":
                attacker_net = {
                    "ip_prefix": "185.220.101.",
                    "login_frequency": 8.0,
                    "sync_frequency": 24.0,
                    "message_count_sent": 0,
                    "message_count_received": 15,
                    "read_count": 15,
                    "delivery_count": 15,
                    "network_ip": ghost_device.ip_address,
                    "network_type": ghost_device.network_type,
                    "active_timezone": ghost_device.timezone,
                    "location_country": ghost_device.country,
                    "session_duration_sec": 3600.0,
                    "idle_time_sec": 0.0,
                    "battery_level": 100,
                    "os_version": ghost_device.os_version
                }
                if random.random() < 0.3:
                    attacker_net["network_ip"] = "185.220.101." + str(random.randint(2, 254))
                
                ghost_meta = MetadataRecord(
                    device_id=ghost_device.id,
                    timestamp=current_timestamp,
                    login_frequency=attacker_net["login_frequency"],
                    sync_frequency=attacker_net["sync_frequency"],
                    session_duration_sec=attacker_net["session_duration_sec"],
                    message_count_sent=attacker_net["message_count_sent"],
                    message_count_received=attacker_net["message_count_received"],
                    read_count=attacker_net["read_count"],
                    delivery_count=attacker_net["delivery_count"],
                    network_ip=attacker_net["network_ip"],
                    network_type=attacker_net["network_type"],
                    active_timezone=attacker_net["active_timezone"],
                    location_country=attacker_net["location_country"],
                    idle_time_sec=attacker_net["idle_time_sec"],
                    battery_level=attacker_net["battery_level"],
                    os_version=attacker_net["os_version"]
                )
                self.db.add(ghost_meta)
                ghost_device.last_active = current_timestamp

        self.db.commit()

        # Evaluate ML Models for this day
        all_devices = self.db.query(Device).filter(Device.user_id == user_id).all()
        devices_records = []
        valid_devices = []
        for dev in all_devices:
            records = self.db.query(MetadataRecord).filter(
                MetadataRecord.device_id == dev.id,
                MetadataRecord.timestamp <= current_date.replace(hour=23, minute=59, second=59)
            ).order_by(MetadataRecord.timestamp.desc()).limit(24).all()
            if records:
                devices_records.append(list(reversed(records)))
                valid_devices.append(dev)

        seq_len = min(len(r) for r in devices_records) if devices_records else 0
        relational_scores = {}
        if seq_len > 0:
            from ..trust.graph import DeviceRelationshipGraph
            graph_detector = DeviceRelationshipGraph()
            truncated_histories = [history[-seq_len:] for history in devices_records]
            _, scores = graph_detector.evaluate_devices(truncated_histories)
            for idx, dev in enumerate(valid_devices):
                relational_scores[dev.id] = scores[idx]

        for dev in all_devices:
            records = self.db.query(MetadataRecord).filter(
                MetadataRecord.device_id == dev.id,
                MetadataRecord.timestamp <= current_date.replace(hour=23, minute=59, second=59)
            ).order_by(MetadataRecord.timestamp.desc()).limit(24).all()
            
            if not records:
                continue

            hmm_state, hmm_prob = self.hmm_detector.evaluate_device(records)
            lstm_anomaly_score = self.lstm_detector.evaluate_device(records, threshold=detection_threshold)

            for r in records:
                r.anomaly_score = lstm_anomaly_score
                r.is_anomaly = lstm_anomaly_score > detection_threshold

            rel_score = relational_scores.get(dev.id, 0.0)
            fusion_results = DecisionFusionEngine.fuse(
                fsm_state=dev.current_trust_state,
                trust_score=dev.trust_score,
                hmm_state=hmm_state,
                hmm_confidence=hmm_prob,
                anomaly_score=lstm_anomaly_score,
                anomaly_threshold=detection_threshold,
                device_metadata={
                    "network_type": dev.network_type,
                    "ip_address": dev.ip_address,
                    "timezone": dev.timezone,
                    "country": dev.country,
                    "pairing_age_days": (current_date - dev.pairing_timestamp).days
                },
                relational_anomaly_score=rel_score
            )

            evidence = fusion_results["evidence_score"]
            score_before = dev.trust_score
            new_score = TrustDecay.calculate_decay(score_before, evidence, alpha)
            dev.trust_score = new_score
            state_before = dev.current_trust_state

            from ..trust.qtk import QuarantinedTreeKEM
            qtk = QuarantinedTreeKEM(delta_inact=5, theta_R=detection_threshold)
            e_i = day
            e_pk = dev.last_key_update_epoch
            R_dt = fusion_results["behavioral_risk_score"]

            is_active = len([r for r in records if (r.message_count_sent > 0 or r.sync_frequency > 0.5)]) > 0
            if is_active and dev.current_trust_state not in ["Quarantined", "Revoked"]:
                dev.last_key_update_epoch = e_i
                e_pk = e_i

            should_quarantine, qtk_reason = qtk.evaluate_trigger(e_i, e_pk, R_dt)

            if should_quarantine and dev.current_trust_state not in ["Quarantined", "Revoked"]:
                dev.current_trust_state = "Quarantined"
                dev.quarantined_at_epoch = e_i
                other_active_devs = [d.id for d in all_devices if d.id != dev.id and d.current_trust_state in ["Trusted", "Idle"]]
                qtk_info = qtk.quarantine_device(dev.id, other_active_devs)
                dev.qtk_shares = qtk_info["shares"]

                self.log_event(
                    event_type="qtk_quarantine",
                    device_id=dev.id,
                    description=f"QTK Quarantine triggered. Reason: {qtk_reason}",
                    score_before=score_before,
                    score_after=new_score,
                    state_before=state_before,
                    state_after="Quarantined",
                    reason=qtk_reason
                )
            elif dev.current_trust_state == "Quarantined":
                is_reconnected = len([r for r in records if r.message_count_sent > 0]) > 0
                risk_dropped = R_dt < qtk.theta_R

                if is_reconnected and risk_dropped:
                    other_active_devs = [d.id for d in all_devices if d.id != dev.id and d.current_trust_state in ["Trusted", "Idle"]]
                    t = max(1, (len(other_active_devs) // 2) + 1)
                    shares_dict = dev.qtk_shares if dev.qtk_shares else {}
                    success, secret, rec_reason = qtk.recover_device(dev.id, shares_dict, other_active_devs, t)
                    
                    if success:
                        dev.current_trust_state = "Trusted"
                        dev.last_key_update_epoch = e_i
                        dev.quarantined_at_epoch = None
                        dev.qtk_shares = None
                        dev.trust_score = 0.90

                        self.log_event(
                            event_type="qtk_recovery",
                            device_id=dev.id,
                            description=f"QTK Recovery successful: {rec_reason}",
                            score_before=score_before,
                            score_after=0.90,
                            state_before=state_before,
                            state_after="Trusted",
                            reason=rec_reason
                        )
                else:
                    quarantine_duration = e_i - (dev.quarantined_at_epoch or e_i)
                    if quarantine_duration >= 5:
                        dev.current_trust_state = "Revoked"
                        dev.is_active = False
                        dev.trust_score = 0.0

                        self.log_event(
                            event_type="qtk_expulsion",
                            device_id=dev.id,
                            description=f"QTK Expulsion triggered after {quarantine_duration} days of quarantine.",
                            score_before=score_before,
                            score_after=0.0,
                            state_before=state_before,
                            state_after="Revoked",
                            reason="Quarantine period expired without recovery."
                        )
            else:
                from ..trust.fsm import TrustFSM
                state_after, state_reason = TrustFSM.transition(state_before, new_score)
                dev.current_trust_state = state_after
                if state_before != state_after:
                    self.log_event(
                        event_type="trust_decay" if new_score < score_before else "trust_recovery",
                        device_id=dev.id,
                        description=f"Trust state transitioned for {dev.name} to {state_after} (Score: {new_score:.2f}). Reason: {state_reason}",
                        score_before=score_before,
                        score_after=new_score,
                        state_before=state_before,
                        state_after=state_after,
                        reason=state_reason
                    )

        self.db.commit()

