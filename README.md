# Ghost Pairing Paper — Final Upgrade Plan (QTK-Integrated Version)
## PART 1 — LAYMAN VERSION

**The original problem:** Once a device is linked to a messaging account, it's trusted forever. A lost, sold, forgotten, or secretly-added device can silently keep reading your encrypted messages — "ghost pairing."

**Original solution:** Watch device behavior (logins, sync timing, idle time) with an HMM (statistical guesser) and LSTM (pattern-memory neural net), and move devices down a trust ladder (Trusted → Idle → Suspicious → Verification Required → Revoked).

**The upgrades  added, in order:**

1. **Watch the "friend group," not just the individual** — a Graph-based LSTM that looks at how your devices behave *relative to each other* (sync timing correlation, shared network patterns), not just each device alone. A ghost device sticks out because it doesn't "fit" with your other devices, even if its solo behavior looks fine.

2. **Combine signals with a learned fusion layer** — instead of a crude "flag if EITHER signal looks bad" rule, a small learned model decides how much to trust each signal.

3. **Make suspicion cost something real (this is the big one)** — instead of just flagging a device, you found that a real, published cryptographic protocol called **Quarantined-TreeKEM (QTK)**, from a 2024 top security conference (CCS), already does something like this inside real group-messaging encryption (MLS, the modern replacement for ad hoc protocols like Signal's). QTK quarantines "ghost users" (their actual term!) who've been *inactive* for too long, locking their encryption keys behind a system where other members have to cooperate to unlock them. But QTK's trigger is dumb — it only checks "how long since this device was last active," a plain timer. It **cannot** catch a ghost device that's sneaky and stays quietly active while behaving suspiciously.

4. **Your real contribution:** swap QTK's dumb timer for your smart trust score (from #1 and #2 above). Now the system quarantines a device not just because it's been silent too long, but because it's *behaviorally or relationally* suspicious — catching the sneaky ghost devices QTK's timer completely misses.

5. **Prove it, and stress-test it** — run real experiments comparing your smart trigger against QTK's plain timer, test whether a clever attacker could fake "normal" behavior to sneak past you, and check whether your own mechanism could be abused to falsely lock out a real device (a legitimate laptop that's just been idle).

**Why this is a good final-year paper now:** You're no longer proposing something vague. You're making a precise, provable claim against a specific, real, respected paper (QTK, CCS 2024): "your timer misses this threat class, ours catches it — here's the proof." That's exactly the shape of a strong, fundable, publishable contribution.

---

## PART 2 — TECHNICAL VERSION: THE QTK-GROUNDED ARCHITECTURE

### A. What QTK actually does (know this cold — it's your direct comparison point)

- QTK = a TreeKEM-based Continuous Group Key Agreement (CGKA) protocol, fully compatible with the MLS standard (RFC 9420), associated with a **(t,m)-perfect secret sharing scheme**.
