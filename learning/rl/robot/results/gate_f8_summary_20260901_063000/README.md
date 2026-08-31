# Gate F8 force-safety pilot

- Completion: **PASS** (80 official sequences)
- Safety gate: **PASS**
- Selected candidate: `static_cap` (curved-only force action <= +0.50)
- Curved safety: control 11/16 -> candidate 16/16
- Force overloads: control 5 -> candidate 0
- Curved all4 delta: 0.0619 percentage points
- Training/PPO/promotion/release changes: none
- Production-ready: no; larger multi-seed validation requires separate approval.

A deterministic duplicate stage2 execution was excluded from official counts; all eight duplicate sequence CSV hashes matched their official counterparts exactly.
