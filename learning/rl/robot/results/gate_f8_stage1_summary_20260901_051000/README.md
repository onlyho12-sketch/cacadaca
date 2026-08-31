# Gate F8 stage 1 — small force-safety constraint pilot (same_xx)

- Stage 1 decision: **PASS**
- Frozen criteria: `learning/rl/robot/results/gate_f8_force_safety_plan_20260901_041500/acceptance_criteria.json`
- Parent policy: `gate_e7_seed1_it211` (`c733763b…52`), observation `base14`, unchanged
- Training performed: **none**; promotion: **none**
- Arms: `control`, `static_cap`, `predictive_shield` — 16 sequences each
  (4 geometries × 4 factory profiles × 1 env), direction `same_xx`
- Seeds: physics `20260901`, policy `20260831`, surface base `33000`

## Result per arm and geometry

| arm | geometry | safety pass | quality pass | force overload | mean final all4 % | max raw force N |
|---|---|---|---|---|---|---|
| control | flat | 4/4 | 3/4 | 0 | 95.295 | 9.463 |
| control | cylinder | **1/4** | 1/4 | **3** | 94.838 | **14.041** |
| control | sphere | 4/4 | 3/4 | 0 | 95.295 | 11.834 |
| control | freeform | 4/4 | 3/4 | 0 | 95.293 | 12.412 |
| static_cap | flat | 4/4 | 3/4 | 0 | 95.295 | 9.463 |
| static_cap | cylinder | 4/4 | 3/4 | 0 | 95.180 | 13.381 |
| static_cap | sphere | 4/4 | 3/4 | 0 | 95.180 | 10.537 |
| static_cap | freeform | 4/4 | 3/4 | 0 | 95.182 | 12.019 |
| predictive_shield | flat | 4/4 | 3/4 | 0 | 95.295 | 9.463 |
| predictive_shield | cylinder | 4/4 | 3/4 | 0 | 95.177 | 13.843 |
| predictive_shield | sphere | 4/4 | 3/4 | 0 | 95.180 | 10.538 |
| predictive_shield | freeform | 4/4 | 3/4 | 0 | 95.180 | 13.266 |

The only safety failures in the whole stage are the parent policy's three
cylinder force overloads.  Both shields removed them without any thermal or
unstable-contact violation, and sensor fault steps are 0 in all 48 sequences.

## Frozen stage-2 entry rule

| candidate | curved(cyl+freeform) overloads vs control | sensor faults | flat safety | flat passthrough parity | entered |
|---|---|---|---|---|---|
| static_cap | 0 < 3 | 0 | 4/4 (= control) | identical | yes |
| predictive_shield | 0 < 3 | 0 | 4/4 (= control) | identical | yes |

Flat parity is exact: every flat sequence of both shields matches the control
row byte-for-byte on outcome, safety, quality, control steps, contact steps,
raw force max, final surface state, and final all4 area.  This confirms the
flat passthrough contract — no flat non-regression risk was introduced.

## Selected candidate: `static_cap`

Both candidates tied on the frozen entry metric (0 overloads) and on curved
safety (12/12), so the documented tiebreak decided: paired curved all4 area
drop versus control is `-0.0392` pp for `static_cap` and `-0.0375` pp for
`predictive_shield` (both are small *improvements* over control), and
`static_cap` is the less intrusive shield.  Intervention level differs little
in practice — both clip on ~71 % of curved steps — but `predictive_shield`
additionally drives the force action down to `-0.50`, which was never required
because the static cap alone already removed every overload.

See `learning/rl/robot/results/gate_f8_stage2_plan_20260901_051200/stage2_selection.json`
for the selection record and the tiebreak disclosure.

## Files

- `combined_sequences.csv` — 48 sequences
- `combined_tile_area_fractions.csv` — 1,200 tile rows
- `arm_geometry_summary.csv` — per arm/geometry aggregate
- `flat_passthrough_parity.csv` — per-sequence flat parity evidence
- `acceptance_checks.csv` — 9 frozen checks, all passed
- `decision.json` — machine-readable stage 1 verdict

## Next

Stage 2 runs `control` and `static_cap` on `cross_xy` (8 PhysX runs) and then
the frozen final gate.  No training, no promotion, no protected-file change.
