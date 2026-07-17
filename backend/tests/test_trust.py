import sys
import os

# Adjust path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.trust.fsm import TrustFSM
from app.trust.decay import TrustDecay
from app.trust.fusion import DecisionFusionEngine
from app.trust.hmm import HMMDetector
from app.trust.lstm import LSTMDetector

def test_fsm_transitions():
    """Verifies that FSM changes states correctly based on trust score thresholds."""
    # Test Trusted State
    state, reason = TrustFSM.transition("Trusted", 0.95)
    assert state == "Trusted"
    
    state, reason = TrustFSM.transition("Trusted", 0.72)
    assert state == "Suspicious"
    
    state, reason = TrustFSM.transition("Trusted", 0.45)
    assert state == "Verification Required"
    
    # Test Revocation Limit
    state, reason = TrustFSM.transition("Suspicious", 0.15)
    assert state == "Revoked"

def test_adaptive_trust_decay():
    """Validates the math of adaptive decay calculation."""
    # Trust_t = alpha * Trust_{t-1} + (1 - alpha) * Evidence
    prev_score = 1.0
    evidence = 0.2  # severe anomaly
    alpha = 0.8
    
    new_score = TrustDecay.calculate_decay(prev_score, evidence, alpha)
    # Expected: 0.8 * 1.0 + 0.2 * 0.2 = 0.8 + 0.04 = 0.84
    assert new_score == 0.84

def test_hmm_initialization():
    """Confirms HMM initializes transition and emission parameters cleanly."""
    detector = HMMDetector()
    matrices = detector.get_matrices()
    assert len(matrices["transition_matrix"]) == 4
    assert len(matrices["emission_means"]) == 4
    assert detector.is_trained

def test_decision_fusion():
    """Checks the fusion of multiple ML signals and the generated explainer report."""
    metadata = {
        "network_type": "VPN",
        "ip_address": "185.220.101.4",
        "timezone": "Europe/Moscow",
        "country": "Russia",
        "pairing_age_days": 0
    }
    
    results = DecisionFusionEngine.fuse(
        fsm_state="Suspicious",
        trust_score=0.62,
        hmm_state=2,  # Ghost Device
        hmm_confidence=0.92,
        anomaly_score=0.85,
        anomaly_threshold=0.65,
        device_metadata=metadata
    )
    
    assert results["hmm_classification"] == "Ghost Device (Silent Sync Monitoring)"
    assert results["evidence_score"] < 0.5  # Should be heavily penalized
    assert len(results["reasons"]) > 0
    assert "vpn_connection" in results["feature_importance"]
    assert "behavioral_risk_score" in results

def test_shamir_secret_sharing():
    """Verifies that Shamir's Secret Sharing splits and reconstructs successfully."""
    from app.trust.qtk import ShamirSecretSharing
    secret = 123456
    t = 3
    m = 5
    
    shares = ShamirSecretSharing.split_secret(secret, t, m)
    assert len(shares) == m
    
    # Reconstruct with t shares (should succeed)
    reconstructed_secret = ShamirSecretSharing.reconstruct_secret(shares[:t])
    assert reconstructed_secret == secret
    
    # Reconstruct with t - 1 shares (should fail/give wrong key)
    reconstructed_wrong = ShamirSecretSharing.reconstruct_secret(shares[:t-1])
    assert reconstructed_wrong != secret

def test_device_relationship_graph():
    """Validates the GCN and Graph-LSTM operations on device relationship graphs."""
    from app.trust.graph import DeviceRelationshipGraph
    graph_detector = DeviceRelationshipGraph()
    
    # Simulate metadata history for 2 devices over 3 timesteps
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
    
    adj, scores = graph_detector.evaluate_devices(history)
    assert adj.shape == (2, 2)
    assert len(scores) == 2
    assert 0.0 <= scores[0] <= 1.0

def test_qtk_trigger():
    """Confirms the behavior-aware QTK trigger functions correctly."""
    from app.trust.qtk import QuarantinedTreeKEM
    qtk = QuarantinedTreeKEM(delta_inact=5, theta_R=0.65)
    
    # Compliant case
    triggered, reason = qtk.evaluate_trigger(current_epoch=10, last_active_epoch=9, R_dt=0.2)
    assert not triggered
    
    # Inactivity trigger case
    triggered, reason = qtk.evaluate_trigger(current_epoch=10, last_active_epoch=4, R_dt=0.3)
    assert triggered
    assert "Inactivity timer" in reason
    
    # Behavioral trigger case
    triggered, reason = qtk.evaluate_trigger(current_epoch=10, last_active_epoch=9, R_dt=0.75)
    assert triggered
    assert "Behavioral anomaly" in reason

