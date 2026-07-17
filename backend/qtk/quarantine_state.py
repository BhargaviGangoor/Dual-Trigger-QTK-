import random
from typing import List, Tuple, Dict, Any
from simulator.device import Device

# Mersenne Prime for Shamir's Secret Sharing arithmetic
SHAMIR_PRIME = 2147483647

def mod_inverse(a: int, m: int) -> int:
    """Computes the modular multiplicative inverse using the Extended Euclidean Algorithm."""
    a = a % m
    t, newt = 0, 1
    r, newr = m, a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        raise ValueError("a is not invertible")
    if t < 0:
        t = t + m
    return t

class ShamirSecretSharing:
    """
    Implements Shamir's (t, m) Threshold Secret Sharing scheme over a finite field
    using prime SHAMIR_PRIME = 2147483647 as defined in the research paper to protect
    a quarantined device's key material.
    """
    @staticmethod
    def split_secret(secret: int, t: int, m: int, prime: int = SHAMIR_PRIME) -> List[Tuple[int, int]]:
        """
        Splits a secret integer into m shares. Any t shares can reconstruct the secret
        via Lagrange interpolation.
        """
        if t > m:
            raise ValueError("Threshold t cannot be greater than total shares m")
        if secret >= prime:
            raise ValueError("Secret must be less than the prime field size")
            
        # Generate random coefficients for the polynomial P(x) = secret + a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1}
        coefficients = [secret] + [random.randint(1, prime - 1) for _ in range(t - 1)]
        
        shares = []
        for x in range(1, m + 1):
            y = 0
            for power, coeff in enumerate(coefficients):
                y = (y + coeff * pow(x, power, prime)) % prime
            shares.append((x, y))
            
        return shares

    @staticmethod
    def reconstruct_secret(shares: List[Tuple[int, int]], prime: int = SHAMIR_PRIME) -> int:
        """
        Reconstructs the secret from shares using Lagrange interpolation at x = 0.
        """
        if len(shares) == 0:
            raise ValueError("No shares provided for reconstruction")
            
        secret = 0
        for i, (x_i, y_i) in enumerate(shares):
            numerator = 1
            denominator = 1
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-x_j)) % prime
                    denominator = (denominator * (x_i - x_j)) % prime
            
            denom_inverse = mod_inverse(denominator, prime)
            lagrange_coeff = (numerator * denom_inverse) % prime
            
            secret = (secret + y_i * lagrange_coeff) % prime
            
        return secret

class QuarantineManager:
    """
    Manages the quarantine containment actions: splitting/reconstructing key material
    using Shamir's Secret Sharing.
    
    When a device enters quarantine, its key seed is split among the remaining active devices.
    Majority threshold t = ceil(m/2) + 1 is used, as detailed in Section IV.
    """
    @staticmethod
    def quarantine_device(target_device: Device, other_devices: List[Device]) -> Dict[str, Any]:
        """
        Quarantines a device by splitting its key material among other devices.
        Threshold t = ceil(m/2) + 1 (majority threshold).
        """
        m = len(other_devices)
        if m == 0:
            return {"secret": 0, "shares": {}}
            
        t = max(1, (m // 2) + 1)
        
        # Generate a simulated secret key seed
        secret_key_seed = random.randint(1000, 999999)
        shares = ShamirSecretSharing.split_secret(secret_key_seed, t, m)
        
        device_shares = {}
        for idx, dev in enumerate(other_devices):
            device_shares[dev.device_id] = shares[idx]
            
        target_device.is_quarantined = True
        target_device.current_trust_state = "Quarantined"
        target_device.qtk_shares = device_shares
        
        return {
            "secret": secret_key_seed,
            "threshold": t,
            "total_shares": m,
            "shares": device_shares
        }

    @staticmethod
    def recover_device(target_device: Device, active_reporters: List[str]) -> Tuple[bool, int, str]:
        """
        Reconstructs the quarantined device's key seed if enough active reporting devices supply shares.
        """
        if not target_device.is_quarantined or not target_device.qtk_shares:
            return False, 0, "Device is not quarantined or has no shares stored."

        shares_dict = target_device.qtk_shares
        m = len(shares_dict)
        threshold = max(1, (m // 2) + 1)
        
        available_shares = []
        for reporter_id in active_reporters:
            if reporter_id in shares_dict:
                available_shares.append(shares_dict[reporter_id])
                
        if len(available_shares) < threshold:
            return False, 0, f"Insufficient shares: got {len(available_shares)}, need {threshold} shares"
            
        try:
            reconstructed = ShamirSecretSharing.reconstruct_secret(available_shares[:threshold])
            target_device.is_quarantined = False
            target_device.current_trust_state = "Trusted"
            target_device.qtk_shares = {}
            return True, reconstructed, f"Reconstruction successful using {threshold} shares from reporters."
        except Exception as e:
            return False, 0, f"Reconstruction failed: {str(e)}"
