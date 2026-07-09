from typing import Dict, Any

class DecisionFusionEngine:
    @staticmethod
    def fuse(fsm_state: str, trust_score: float, hmm_state: int, hmm_confidence: float,
             anomaly_score: float, anomaly_threshold: float, device_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combines multiple trust signals and generates:
        1. An evidence score (0.0 = completely suspicious/malicious, 1.0 = normal/trusted)
        2. A confidence score (0.0 to 1.0)
        3. An explainability report (reasoning breakdown)
        """
        # HMM State Map: 0: Normal, 1: Hijacked Session, 2: Ghost Device, 3: Network Anomaly
        hmm_state_names = {
            0: "Legitimate User Activity",
            1: "Compromised Session (Hijacking)",
            2: "Ghost Device (Silent Sync Monitoring)",
            3: "Unstable Connection Anomaly"
        }

        # Calculate base evidence score (1.0 = good, 0.0 = malicious)
        evidence_score = 1.0
        reasons = []
        feature_importance = {}

        # 1. HMM influence
        if hmm_state == 2:  # Ghost Device
            evidence_score -= 0.60
            reasons.append("HMM classified behavior as Ghost Device pairing pattern.")
            feature_importance["hmm_ghost_pattern"] = -0.60
        elif hmm_state == 1:  # Hijack
            evidence_score -= 0.50
            reasons.append("HMM classified behavior as Compromised Session hijacking.")
            feature_importance["hmm_hijack_pattern"] = -0.50
        elif hmm_state == 3:  # Network Anomaly
            evidence_score -= 0.15
            reasons.append("HMM detected transient network sync anomalies.")
            feature_importance["hmm_network_instability"] = -0.15
        else:
            feature_importance["hmm_normal_pattern"] = +0.10

        # 2. LSTM Anomaly score influence
        if anomaly_score > anomaly_threshold:
            excess = anomaly_score - anomaly_threshold
            penalty = min(0.40, excess * 0.8)
            evidence_score -= penalty
            reasons.append(f"LSTM flagged sequential behavior deviation (Score: {anomaly_score:.2f} > Threshold: {anomaly_threshold:.2f}).")
            feature_importance["lstm_sequence_deviation"] = -penalty
        else:
            evidence_score += 0.05
            feature_importance["lstm_sequence_normal"] = +0.05

        # 3. Static metadata anomalies
        if device_metadata.get("network_type") == "VPN":
            evidence_score -= 0.10
            reasons.append("Device connected via a virtual private network (VPN).")
            feature_importance["vpn_connection"] = -0.10
            
        # Location anomaly / pairing age
        pairing_age = device_metadata.get("pairing_age_days", 0)
        if pairing_age < 1:  # Brand new device
            evidence_score -= 0.05
            reasons.append("Freshly paired device (less than 24 hours old).")
            feature_importance["new_device_pairing"] = -0.05

        # Adjust range
        evidence_score = max(0.0, min(1.0, evidence_score))

        # Compute fusion confidence (how much the trust scores agree)
        # Confidence decays if HMM and LSTM disagree severely
        if (hmm_state in [1, 2]) and (anomaly_score < anomaly_threshold):
            confidence = 0.50
            reasons.append("Warning: Model disagreement. HMM indicates abuse, but LSTM shows normal sequencing.")
        elif (hmm_state == 0) and (anomaly_score > anomaly_threshold):
            confidence = 0.60
            reasons.append("Warning: Model disagreement. LSTM indicates sequence anomaly, HMM indicates standard user pattern.")
        else:
            confidence = 0.85 + (0.15 * hmm_confidence)

        # Write the explainability summary paragraph
        if evidence_score > 0.8:
            verdict = "Device is exhibiting highly legitimate and profile-consistent behavior. No threat patterns detected."
        elif evidence_score > 0.5:
            verdict = "Minor behavioral deviations detected. Device is under review and flag triggers trust decay."
        elif evidence_score > 0.2:
            verdict = f"Critical Alert: Severe anomalies. High likelihood of {hmm_state_names.get(hmm_state, 'compromise')}. Verification strongly advised."
        else:
            verdict = "Severe Cryptographic and Behavioral Threat! Immediate revocation recommended due to high confidence ghost or hijacked pairing signatures."

        return {
            "evidence_score": evidence_score,
            "confidence_score": round(confidence, 4),
            "verdict": verdict,
            "reasons": reasons,
            "feature_importance": feature_importance,
            "hmm_classification": hmm_state_names.get(hmm_state, "Unknown"),
            "lstm_anomaly_score": anomaly_score
        }
