# Ghost Pairing Paper — Final Upgrade Plan (QTK-Integrated Version)
## PART 1 — LAYMAN VERSION

**The original problem:** Once a device is linked to a messaging account, it's trusted forever. A lost, sold, forgotten, or secretly-added device can silently keep reading your encrypted messages — "ghost pairing."

**Original solution:** Watch device behavior (logins, sync timing, idle time) with an HMM (statistical guesser) and LSTM (pattern-memory neural net), and move devices down a trust ladder (Trusted → Idle → Suspicious → Verification Required → Revoked).

**The upgrades  added, in order:**

1. **Watch the "friend group," not just the individual** — a Graph-based LSTM that looks at how your devices behave *relative to each other* (sync timing correlation, shared network patterns), not just each device alone. A ghost device sticks out because it doesn't "fit" with your other devices, even if its solo behavior looks fine.

2. **Combine signals with a learned fusion layer** — instead of a crude "flag if EITHER signal looks bad" rule, a small learned model decides how much to trust each signal.
