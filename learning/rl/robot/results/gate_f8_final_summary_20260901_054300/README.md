# Gate F8 final — small force-safety constraint pilot

- Final decision: **PASS** (8/8 frozen checks)
- Selected candidate: **`static_cap`** — curved-only force action ≤ +0.50, flat untouched
- Frozen criteria: `learning/rl/robot/results/gate_f8_force_safety_plan_20260901_041500/acceptance_criteria.json`
- Parent policy: `gate_e7_seed1_it211` (`c733763b…52`), observation `base14`
- **Training performed: none. Champion/release unchanged. Nothing promoted.**
- Scope: 2 arms × 2 directions × 4 geometries × 4 factory profiles = 64 sequences,
  1,600 tile rows, 16 PhysX runs; seeds physics `20260901`, policy `20260831`,
  surface base `33000`

## Frozen final gate

| check | observed | rule | result |
|---|---|---|---|
| sequences | 64 | = 64 | PASS |
| candidate cylinder+freeform safety pass | 16 | > control 11 | PASS |
| candidate force overloads (both directions) | 0 | = 0 | PASS |
| flat safety pass | 8 | ≥ control 8 | PASS |
| flat quality pass | 6 | ≥ control 6 | PASS |
| mean curved all4 area drop vs control | 0.0038 pp | ≤ 1.0 pp | PASS |
| sensor fault steps | 0 | = 0 | PASS |
| flat action passthrough parity | identical | exact | PASS |

## Arm comparison (32 sequences each)

| metric | control | static_cap |
|---|---|---|
| force overloads | **5** | **0** |
| max raw force | **14.041 N** | 13.381 N |
| safety pass (all) | 27/32 | **32/32** |
| curved safety pass | 19/24 | **24/24** |
| curved quality pass | 14/24 | **18/24** |
| flat safety / quality | 8/8, 6/8 | 8/8, 6/8 (identical rows) |
| sensor fault steps | 0 | 0 |
| min force-action cap | 1.00 | 0.50 |

All five parent-policy safety failures are cylinder force overloads — three on
`same_xx`, two on `cross_xy` — and every one of them is removed by the static
cap.  No thermal or unstable-contact violation occurred in either arm.

## Quality cost

Paired per-sequence curved all4 area difference (control − static_cap):
mean **+0.0038 pp**, range −0.600 to +0.430 pp.  The shield clips the force
action on ~71 % of curved control steps, yet whole-surface quality is
statistically unchanged and curved *quality* pass count improves (14 → 18),
because control's overloaded cylinder sequences terminate early.

Flat is byte-identical to control across all 8 flat sequences (outcome, safety,
quality, control steps, contact steps, raw force max, final surface state,
final all4 area) — the passthrough contract holds, so there is no flat
non-regression risk.

## Interpretation and limits

- This validates a **deterministic action-space safety layer**, not a policy.
  The parent policy itself is unchanged; the shield is an inference-time clip.
- 1 env per profile per cell: 64 sequences total.  It is a pilot, sized to
  decide direction, not a release-grade statistic.
- Cylinder R=0.60 m is the only geometry where the parent overloads.  Sphere
  and freeform never overloaded even without a shield.
- `predictive_shield` was equally effective at stage 1 (0 overloads) but its
  extra dynamic reduction to −0.50 was never needed; `static_cap` was selected
  as the less intrusive arm.  See
  `gate_f8_stage2_plan_20260901_051200/stage2_selection.json`.

## Evidence

- stage 1 (`same_xx`, 3 arms): `gate_f8_stage1_summary_20260901_051000/`
- stage 2 (`cross_xy`, 2 arms): `gate_f8_stage2_{control,static_cap}_*_cross_seed33000_20260901_051200/`
- this directory: `combined_sequences.csv` (64), `combined_tile_area_fractions.csv` (1,600),
  `arm_geometry_summary.csv`, `flat_passthrough_parity.csv`, `acceptance_checks.csv`,
  `decision.json`, `provenance.json`, `SHA256SUMS`, `SOURCE_SHA256SUMS`
- code: `learning/rl/gate_f8_force_safety.py`, `gate_f8_force_safety_eval.py`,
  `gate_f8_force_safety_summarize.py`; CPU tests `learning/rl/tests/test_gate_f8_*.py`

## Next (requires separate approval)

The original F8 slot in the handoff document is flat+curve production training
and release.  That has **not** started.  A follow-up would decide whether the
validated static cap is carried into training as an action-space constraint or
kept as an inference-time safety layer.
