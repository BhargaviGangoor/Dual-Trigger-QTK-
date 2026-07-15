import sys
import numpy as np
from simulator.device import Device
from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice
from qtk.epoch_tracker import EpochTracker
from qtk.quarantine_state import ShamirSecretSharing, QuarantineManager
from qtk.dual_trigger import DualTrigger
from models.hmm import HMMDetector
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion

def test_device_creation():
    phone = LegitimateDevice("phone", "u1", "Phone", "primary", "Android 14", "Student")
    assert phone.device_id == "phone"
    assert phone.trust_score == 1.0
    assert phone.current_trust_state == "Trusted"
    print("  test_device_creation PASSED")

def test_shamir_secret_sharing():
    secret = 123456
    t = 3
    m = 5
    shares = ShamirSecretSharing.split_secret(secret, t, m)
    assert len(shares) == m
    
    reconstructed = ShamirSecretSharing.reconstruct_secret(shares[:t])
    assert reconstructed == secret
    print("  test_shamir_secret_sharing PASSED")

def test_hmm_evaluation():
    hmm = HMMDetector()
    hmm.train_on_profile("Student")
    
    # Generate mock histories (normal activity)
    normal_history = [
        {"session_duration_sec": 120.0, "sync_frequency": 4.0, "message_count_sent": 5, "network_ip": "192.168.1.10", "active_timezone": "EST"},
        {"session_duration_sec": 130.0, "sync_frequency": 4.2, "message_count_sent": 6, "network_ip": "192.168.1.10", "active_timezone": "EST"},
        {"session_duration_sec": 110.0, "sync_frequency": 3.8, "message_count_sent": 4, "network_ip": "192.168.1.10", "active_timezone": "EST"}
    ]
    
    state, conf = hmm.evaluate_device(normal_history)
    # normal state index should be 0 or transient 3
    assert state in [0, 3]
    assert 0.0 <= conf <= 1.0
    print("  test_hmm_evaluation PASSED")

def test_graph_lstm():
    graph_lstm = GraphLSTM(beta=0.8)
    
    # 2 devices, 3 timesteps
    history = [
        [
            {"session_duration_sec": 300, "sync_frequency": 5.0, "message_count_sent": 10, "network_ip": "192.168.1.10", "network_type": "WiFi", "location_country": "US", "active_timezone": "EST"},
            {"session_duration_sec": 400, "sync_frequency": 5.2, "message_count_sent": 12, "network_ip": "192.168.1.10", "network_type": "WiFi", "location_country": "US", "active_timezone": "EST"},
            {"session_duration_sec": 500, "sync_frequency": 5.5, "message_count_sent": 15, "network_ip": "192.168.1.10", "network_type": "WiFi", "location_country": "US", "active_timezone": "EST"}
        ],
        [
            {"session_duration_sec": 200, "sync_frequency": 4.8, "message_count_sent": 8, "network_ip": "192.168.1.12", "network_type": "WiFi", "location_country": "US", "active_timezone": "EST"},
            {"session_duration_sec": 250, "sync_frequency": 5.0, "message_count_sent": 9, "network_ip": "192.168.1.12", "network_type": "WiFi", "location_country": "US", "active_timezone": "EST"},
            {"session_duration_sec": 300, "sync_frequency": 5.1, "message_count_sent": 11, "network_ip": "192.168.1.12", "network_type": "WiFi", "location_country": "US", "active_timezone": "EST"}
        ]
    ]
    
    adj, scores = graph_lstm.evaluate_devices(history)
    assert adj.shape == (2, 2)
    assert len(scores) == 2
    assert 0.0 <= scores[0] <= 1.0
    print("  test_graph_lstm PASSED")

def test_risk_fusion_and_trigger():
    fusion = RiskFusion()
    trigger = DualTrigger(delta_inact=5, theta_R=0.65)
    
    # Compliant case
    R_dt = fusion.fuse(hmm_state=0, hmm_confidence=0.9, relational_anomaly_score=0.1, trust_score=0.95)
    assert R_dt < 0.65
    
    phone = LegitimateDevice("phone", "u1", "Phone", "primary", "Android 14", "Student")
    phone.update_key(10)
    
    triggered, reason = trigger.evaluate(current_epoch=11, device=phone, R_dt=R_dt)
    assert not triggered
    
    # Anomaly case
    R_dt_anom = fusion.fuse(hmm_state=2, hmm_confidence=0.95, relational_anomaly_score=0.8, trust_score=0.4)
    assert R_dt_anom >= 0.65
    
    triggered_anom, reason_anom = trigger.evaluate(current_epoch=11, device=phone, R_dt=R_dt_anom)
    assert triggered_anom
    assert "Behavioral anomaly" in reason_anom
    print("  test_risk_fusion_and_trigger PASSED")

def run():
    print("==================================================")
    print("Running Verification Tests for Rebuilt Codebase...")
    print("==================================================")
    try:
        test_device_creation()
        test_shamir_secret_sharing()
        test_hmm_evaluation()
        test_graph_lstm()
        test_risk_fusion_and_trigger()
        print("\nAll verification tests PASSED successfully!")
        print("==================================================")
        sys.exit(0)
    except AssertionError as e:
        print("\nAssertion failed during validation!")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
