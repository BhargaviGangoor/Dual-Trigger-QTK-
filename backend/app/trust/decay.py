class TrustDecay:
    @staticmethod
    def calculate_decay(prev_score: float, evidence_score: float, alpha: float = 0.8) -> float:
        """
        Computes the adaptive trust score.
        Formula: Trust_t = alpha * Trust_{t-1} + (1 - alpha) * Evidence_t
        Args:
            prev_score (float): Previous trust score (0.0 to 1.0)
            evidence_score (float): Behavior evidence score (0.0 = malicious, 1.0 = normal)
            alpha (float): Decay factor (0.0 to 1.0)
        """
        # Ensure scores are within bounds
        prev_score = max(0.0, min(1.0, prev_score))
        evidence_score = max(0.0, min(1.0, evidence_score))
        alpha = max(0.0, min(1.0, alpha))
        
        new_score = alpha * prev_score + (1.0 - alpha) * evidence_score
        return round(new_score, 4)

    @staticmethod
    def get_dynamic_alpha(device_type: str, network_type: str, base_alpha: float = 0.8) -> float:
        """
        Dynamically adjusts the decay factor alpha based on environmental factors.
        Higher alpha means slower decay (higher trust stability).
        """
        alpha = base_alpha
        
        # Primary devices are structurally less suspicious of sudden changes than linked web browser instances
        if device_type == "primary":
            alpha += 0.05
        else:
            alpha -= 0.05

        # VPN networks increase decay responsiveness (decay trust faster)
        if network_type == "VPN":
            alpha -= 0.10
        elif network_type == "Ethernet" or network_type == "WiFi":
            alpha += 0.02
            
        return max(0.1, min(0.99, alpha))
