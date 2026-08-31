# Gate E7 flat-surface release champion

This immutable-style package records the approved `seed20260831`, iteration 211
policy as the flat-surface release champion.

## Decision

- Final release/regression criteria: **PASS**
- Safety: 64/64; sensor fault steps: 0
- Existing quality pass: 48/64, equal to the previous champion
- Four-target passing area: 94.02828125%, +0.2440625 percentage points
- Mean control-step increase: +15.5145304948%, within the frozen +40% limit
- 100/95/90/80% passing tiles: 1149/1238/1286/1391
- Same-seed repeat: exact CSV match

## Package

- `model_gate_e7_seed1_it211.pt`: packaged checkpoint
- `manifest.json`: provenance, decision, dimensions, and protected hashes
- `evidence/`: copied final evaluation decision and CSV summaries
- `smoke_test.py`, `smoke_result.json`: reproducible CPU load/inference smoke
- `checksums.sha256`: package integrity manifest

The pre-existing `learning/rl/robot/champion/model_bc_robot_tesla12p7.pt` was
not overwritten. This directory is the authoritative flat release package;
consumers must opt in to its checkpoint path explicitly.

The packaging smoke did not run PhysX or training. Full same/cross PhysX release
evidence is preserved under `evidence/` and at the source path recorded in
`manifest.json`.
