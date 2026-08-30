from typing import Optional

class TrustScore:
    """
    Manages long-term device trust accumulation and decay:
    T_{t+1}(d) = alpha * T_t(d) + (1 - alpha) * B_t(d)
    
    where B_t(d) = 1 - P_c(d, t) is the immediate behavioral evidence trust score,
    and (1 - T_t(d)) is the accumulated behavioral distrust.
    """
    @staticmethod
    def calculate_decay(
        prev_trust: float,
        evidence_trust: float,
        alpha: float = 0.8,
        alpha_rec: Optional[float] = 0.50
    ) -> float:
        """
        Computes the updated trust score T_{t+1}(d) with asymmetric recovery:
        - When evidence indicates suspicious behavior (evidence_trust < 0.70), trust decays with factor alpha.
        - When evidence indicates clean benign behavior (evidence_trust >= 0.70), trust recovers faster with alpha_rec.
        This allows legitimate devices to recover promptly from transient network switches,
        while maintaining rapid penalization for persistent anomalies.
        """
        prev_trust = max(0.0, min(1.0, float(prev_trust)))
        evidence_trust = max(0.0, min(1.0, float(evidence_trust)))
        
        effective_alpha = alpha
        if alpha_rec is not None and evidence_trust >= 0.70:
            effective_alpha = alpha_rec

        effective_alpha = max(0.0, min(1.0, float(effective_alpha)))
        updated = effective_alpha * prev_trust + (1.0 - effective_alpha) * evidence_trust
        return round(float(updated), 4)

    @staticmethod
    def compute_distrust(trust_score: float) -> float:
        """Computes behavioral distrust: 1 - T_t(d)."""
        return round(1.0 - max(0.0, min(1.0, float(trust_score))), 4)

    @staticmethod
    def update(device, p_c: float, alpha: float = 0.8, alpha_rec: float = 0.50) -> float:
        """
        Calculates updated trust from HMM anomaly probability P_c(d,t)
        and updates the device's trust_score attribute.
        """
        evidence_trust = max(0.0, min(1.0, 1.0 - float(p_c)))
        prev_trust = getattr(device, "trust_score", 1.0)
        new_trust = TrustScore.calculate_decay(prev_trust, evidence_trust, alpha=alpha, alpha_rec=alpha_rec)
        device.trust_score = new_trust
        return new_trust
