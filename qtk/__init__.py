"""
Quarantined-TreeKEM (QTK) Core Protocol Module
Contains epoch tracking, inactivity checking, dual-trigger quarantine evaluation,
and Shamir secret sharing (t, m) key protection.
"""

from .epoch_tracker import EpochTracker
from .inactivity_trigger import InactivityTrigger
from .dual_trigger import DualTrigger, TriggerReason
from .quarantine_state import ShamirSecretSharing, QuarantineManager

__all__ = [
    "EpochTracker",
    "InactivityTrigger",
    "DualTrigger",
    "TriggerReason",
    "ShamirSecretSharing",
    "QuarantineManager",
]
