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
