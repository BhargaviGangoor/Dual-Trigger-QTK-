from typing import Tuple

class TrustFSM:
    @staticmethod
    def transition(current_state: str, trust_score: float) -> Tuple[str, str]:
        """
        Determines the next FSM state based on the current state and trust score.
        Returns:
            next_state (str): The new trust state.
            reason (str): Explanation for the state change.
        """
        if current_state == "Revoked":
            return "Revoked", "Once revoked, a device cannot be re-trusted without manual repair/re-pairing."
            
        if trust_score < 0.2:
            return "Revoked", f"Trust score ({trust_score:.2f}) dropped below critical threshold (0.20)."
            
        if current_state == "Trusted":
            if trust_score < 0.5:
                return "Verification Required", f"Severe trust drop ({trust_score:.2f}). Cryptographic re-verification required."
            elif trust_score < 0.8:
                return "Suspicious", f"Trust score ({trust_score:.2f}) dropped below threshold (0.80)."
            else:
                return "Trusted", "Device trust score is healthy."
                
        elif current_state == "Idle":
            if trust_score < 0.5:
                return "Verification Required", f"Device awoke with low trust ({trust_score:.2f})."
            elif trust_score < 0.8:
                return "Suspicious", f"Device awoke with moderate anomalies ({trust_score:.2f})."
            elif trust_score >= 0.85:
                return "Trusted", f"Device active and healthy (Trust: {trust_score:.2f})."
            else:
                return "Idle", "Device remains idle."
                
        elif current_state == "Suspicious":
            if trust_score < 0.5:
                return "Verification Required", f"Trust decayed further in Suspicious state ({trust_score:.2f})."
            elif trust_score >= 0.85:
                return "Trusted", f"Trust score successfully recovered ({trust_score:.2f}) due to consistent normal behavior."
            else:
                return "Suspicious", "Device remains suspicious."
                
        elif current_state == "Verification Required":
            if trust_score >= 0.9:
                return "Trusted", f"Cryptographic verification succeeded. Trust restored ({trust_score:.2f})."
            elif trust_score < 0.3:
                return "Revoked", f"Verification window expired / failed (Trust: {trust_score:.2f})."
            else:
                return "Verification Required", "Awaiting verification check."
                
        elif current_state == "Quarantined":
            if trust_score >= 0.8:
                return "Trusted", f"Quarantine lifted. Group secret shares reconstructed successfully ({trust_score:.2f})."
            elif trust_score < 0.25:
                return "Revoked", f"Device expelled from group key agreement due to critical trust drop ({trust_score:.2f})."
            else:
                return "Quarantined", "Device remains quarantined in key containment."
                
        return current_state, "No state change."
