class TrustScore:
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
        prev_score = max(0.0, min(1.0, prev_score))
        evidence_score = max(0.0, min(1.0, evidence_score))
        alpha = max(0.0, min(1.0, alpha))
        
        new_score = alpha * prev_score + (1.0 - alpha) * evidence_score
        return round(new_score, 4)

    @staticmethod
    def get_dynamic_alpha(device_type: str, network_type: str, base_alpha: float = 0.8) -> float:
        """
        Dynamically adjusts the decay factor alpha based on device and network type.
        Higher alpha means slower decay (more stable).
        """
        alpha = base_alpha
        
        if device_type == "primary":
            alpha += 0.05
        else:
            alpha -= 0.05

        if network_type == "VPN":
            alpha -= 0.10
        elif network_type in ["Ethernet", "WiFi"]:
            alpha += 0.02
            
        return max(0.1, min(0.99, alpha))

    @staticmethod
    def update(device, evidence_score: float, alpha: float = 0.8):
        """
        Calculates trust decay for a device and updates its trust_score and current_trust_state.
        Loads thresholds dynamically from backend/configs/thresholds.yaml.
        """
        device_type = getattr(device, "device_type", "linked")
        network_type = getattr(device, "network_type", "WiFi")
        
        dev_alpha = TrustScore.get_dynamic_alpha(device_type, network_type, alpha)
        prev_score = getattr(device, "trust_score", 1.0)
        new_score = TrustScore.calculate_decay(prev_score, evidence_score, dev_alpha)
        
        if hasattr(device, "update_trust"):
            device.update_trust(new_score)
        else:
            device.trust_score = new_score
            
        # Load thresholds dynamically
        import os
        import yaml
        
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "thresholds.yaml"
        )
        critical_revocation = 0.2
        suspicious_th = 0.8
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                trust_cfg = cfg.get("trust", {})
                critical_revocation = trust_cfg.get("critical_revocation_threshold", 0.2)
                suspicious_th = trust_cfg.get("suspicious_threshold", 0.8)
            except Exception:
                pass

        if new_score < critical_revocation:
            state = "Revoked"
        elif new_score < suspicious_th:
            state = "Suspicious"
        else:
            state = "Trusted"
            
        if hasattr(device, "update_trust_state"):
            device.update_trust_state(state)
        else:
            device.current_trust_state = state
        return new_score
