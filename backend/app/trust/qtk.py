import random
from typing import List, Tuple, Dict, Any

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
    @staticmethod
    def split_secret(secret: int, t: int, m: int, prime: int = SHAMIR_PRIME) -> List[Tuple[int, int]]:
        """
        Splits a secret integer into m shares. Any t shares can reconstruct the secret.
        """
        if t > m:
            raise ValueError("Threshold t cannot be greater than total shares m")
        if secret >= prime:
            raise ValueError("Secret must be less than the prime field size")
            
        # Generate random coefficients for the polynomial P(x) = secret + a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1}
        # using standard random generator
        coefficients = [secret] + [random.randint(1, prime - 1) for _ in range(t - 1)]
        
        shares = []
        for x in range(1, m + 1):
            y = 0
            # Evaluate polynomial at x
            for power, coeff in enumerate(coefficients):
                y = (y + coeff * pow(x, power, prime)) % prime
            shares.append((x, y))
            
        return shares

    @staticmethod
    def reconstruct_secret(shares: List[Tuple[int, int]], prime: int = SHAMIR_PRIME) -> int:
        """
        Reconstructs the secret from t shares using Lagrange interpolation at x = 0.
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
            
            # Multiply by modular inverse of denominator
            denom_inverse = mod_inverse(denominator, prime)
            lagrange_coeff = (numerator * denom_inverse) % prime
            
            secret = (secret + y_i * lagrange_coeff) % prime
            
        return secret

class QuarantinedTreeKEM:
    """
    Simulates the Quarantined-TreeKEM (QTK) protocol lifecycle.
    Manages epochs, inactive timer check, behavioral risk scoring trigger,
    secret sharing generation, and group recovery.
    """
    def __init__(self, delta_inact: int = 5, theta_R: float = 0.65):
        self.delta_inact = delta_inact
        self.theta_R = theta_R

    def evaluate_trigger(self, current_epoch: int, last_active_epoch: int, R_dt: float) -> Tuple[bool, str]:
        """
        Evaluates the modified QTK trigger:
        Quarantine(d) <=> (e_i - e_pk(d) >= delta_inact) OR (R(d,t) >= theta_R)
        """
        epoch_diff = current_epoch - last_active_epoch
        inactivity_trigger = epoch_diff >= self.delta_inact
        behavior_trigger = R_dt >= self.theta_R
        
        if inactivity_trigger and behavior_trigger:
            return True, f"Both triggers fired (Epoch Gap: {epoch_diff} >= {self.delta_inact}, Risk: {R_dt:.2f} >= {self.theta_R})"
        elif inactivity_trigger:
            return True, f"Inactivity timer expired (Epoch Gap: {epoch_diff} >= {self.delta_inact})"
        elif behavior_trigger:
            return True, f"Behavioral anomaly detected (Risk: {R_dt:.2f} >= {self.theta_R})"
            
        return False, "Device is trust-compliant."

    def quarantine_device(self, device_id: str, other_devices: List[str]) -> Dict[str, Any]:
        """
        Locks the device's key material on its behalf using a (t, m) secret sharing scheme.
        t = ceil(m/2) + 1 (majority threshold) or similar configuration.
        """
        m = len(other_devices)
        if m == 0:
            # Cannot quarantine if there are no other devices to hold shares
            return {"secret": 0, "shares": {}}
            
        t = max(1, (m // 2) + 1)
        
        # Generate a simulated secret key (e.g. key seed integer)
        secret_key_seed = random.randint(1000, 999999)
        shares = ShamirSecretSharing.split_secret(secret_key_seed, t, m)
        
        # Map shares to the other devices
        device_shares = {}
        for idx, other_dev in enumerate(other_devices):
            device_shares[other_dev] = shares[idx]
            
        return {
            "secret": secret_key_seed,
            "threshold": t,
            "total_shares": m,
            "shares": device_shares
        }

    def recover_device(self, device_id: str, device_shares: Dict[str, Tuple[int, int]], 
                       active_reporters: List[str], threshold: int) -> Tuple[bool, int, str]:
        """
        Reconstructs the quarantined device's key material using shares provided by active reporting devices.
        Returns:
            success (bool)
            reconstructed_secret (int)
            reason (str)
        """
        available_shares = []
        for reporter in active_reporters:
            if reporter in device_shares:
                available_shares.append(device_shares[reporter])
                
        if len(available_shares) < threshold:
            return False, 0, f"Insufficient shares: got {len(available_shares)}, need {threshold} for threshold reconstruction"
            
        try:
            # Use exactly threshold shares
            reconstructed = ShamirSecretSharing.reconstruct_secret(available_shares[:threshold])
            return True, reconstructed, f"Reconstruction successful using {threshold} shares from: {', '.join(active_reporters[:threshold])}"
        except Exception as e:
            return False, 0, f"Reconstruction failed: {str(e)}"
