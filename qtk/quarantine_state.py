import random
from typing import List, Tuple, Dict, Any, Optional

# Mersenne Prime for Shamir Secret Sharing arithmetic (2^31 - 1)
SHAMIR_PRIME = 2147483647

def mod_inverse(a: int, m: int) -> int:
    """Computes modular inverse using the Extended Euclidean Algorithm."""
    a = a % m
    t, newt = 0, 1
    r, newr = m, a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        raise ValueError("a is not invertible modulo m")
    if t < 0:
        t = t + m
    return t

class ShamirSecretSharing:
    """
    Implements Shamir's (t, m) Threshold Secret Sharing scheme over finite field Z_p.
    Used by QTK to split and protect quarantined device key material across peer clients.
    """
    @staticmethod
    def split_secret(secret: int, t: int, m: int, prime: int = SHAMIR_PRIME) -> List[Tuple[int, int]]:
        """
        Splits a secret integer into m shares such that any t shares can reconstruct it.
        Polynomial: P(x) = secret + a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1} mod p
        """
        if t > m:
            raise ValueError("Threshold t cannot exceed total shares m")
        if t < 1 or m < 1:
            raise ValueError("t and m must be positive integers")
        if secret >= prime or secret < 0:
            raise ValueError(f"Secret must be in range [0, {prime - 1}]")

        # Generate random polynomial coefficients
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
        Reconstructs the secret from at least t shares using Lagrange interpolation at x = 0.
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

            denom_inv = mod_inverse(denominator, prime)
            lagrange_l_i = (numerator * denom_inv) % prime
            secret = (secret + y_i * lagrange_l_i) % prime

        return (secret + prime) % prime

class QuarantineManager:
    """
    Manages cryptographic quarantine containment actions:
    Splits key material among active group members upon quarantine,
    and reconstructs the key upon verified recovery.
    """
    @staticmethod
    def quarantine_device(target_device, other_devices: List[Any]) -> Dict[str, Any]:
        """
        Quarantines target device by splitting its key seed among other active devices.
        Threshold t = ceil(m/2) + 1 (majority threshold).
        """
        m = len(other_devices)
        if m == 0:
            target_device.quarantine(getattr(target_device, "quarantined_epoch", 0), "No peers available")
            return {"secret": 0, "threshold": 0, "shares": {}}

        t = max(1, (m // 2) + 1)
        secret_key_seed = random.randint(1000, 999999)
        shares = ShamirSecretSharing.split_secret(secret_key_seed, t, m)

        device_shares = {}
        for idx, dev in enumerate(other_devices):
            device_shares[dev.device_id] = shares[idx]

        target_device.secret_shares = device_shares
        if not target_device.is_quarantined:
            target_device.quarantine(getattr(target_device, "quarantined_epoch", 0) or 1, "Cryptographic quarantine invoked")

        return {
            "secret": secret_key_seed,
            "threshold": t,
            "total_shares": m,
            "shares": device_shares
        }

    @staticmethod
    def recover_device(target_device, active_reporters: List[str]) -> Tuple[bool, int, str]:
        """
        Reconstructs quarantined device key seed if sufficient active reporters provide shares.
        """
        if not target_device.is_quarantined or not target_device.secret_shares:
            return False, 0, "Device is not quarantined or has no stored shares."

        shares_dict = target_device.secret_shares
        m = len(shares_dict)
        threshold = max(1, (m // 2) + 1)

        available_shares = []
        for reporter_id in active_reporters:
            if reporter_id in shares_dict:
                available_shares.append(shares_dict[reporter_id])

        if len(available_shares) < threshold:
            return False, 0, f"Insufficient shares: got {len(available_shares)}, need {threshold}"

        try:
            reconstructed = ShamirSecretSharing.reconstruct_secret(available_shares[:threshold])
            target_device.recover()
            target_device.secret_shares = None
            return True, reconstructed, f"Recovery successful with {threshold} shares."
        except Exception as e:
            return False, 0, f"Reconstruction failed: {str(e)}"
