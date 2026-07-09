# Ghost Pairing Paper — Final Upgrade Plan (QTK-Integrated Version)
## PART 1 — LAYMAN VERSION

**The original problem:** Once a device is linked to a messaging account, it's trusted forever. A lost, sold, forgotten, or secretly-added device can silently keep reading your encrypted messages — "ghost pairing."

**Original solution:** Watch device behavior (logins, sync timing, idle time) with an HMM (statistical guesser) and LSTM (pattern-memory neural net), and move devices down a trust ladder (Trusted → Idle → Suspicious → Verification Required → Revoked).
