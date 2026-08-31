# Gate C step-over / path-direction paired evaluation

## Scope

- Gate C only: analytic geometry/coverage screening followed by a paired PhysX comparison.
- No BC/PPO training was performed (`training_performed=false`).
- Existing checkpoints, CSV/PPT results, Gate A/B/B2 outputs, and protected files were not overwritten.
- `new_car_mild` distributions remain **PT-DESIGN**, not a claim of measured vehicle-population statistics.

## C1 analytic screening

Source: `gate_c1_analytic_coverage_20260831_001831/`

- Screened 24 combinations: step-over ratios 0.18/0.25/0.32/0.40, three edge modes,
  and `same_xx`/`cross_xy` directions.
- Selected the top two: step-over ratio 0.40, `balanced_extend5`, with `cross_xy` and
  `same_xx`.
- Both selected candidates use a 44 mm step-over, five lines per sweep and 2.1 m total path.
  This is 47.5% shorter than the 4.0 m reference-style analytic path.
- Analytic ranking is a Gaussian exposure/coverage screen, not a measured removal prediction.

## C2 paired PhysX design

- Profiles: `legacy_stress`, `new_car_mild`.
- Directions: `cross_xy`, `same_xx`.
- 16 surfaces per group, 64 sequences total, one pass per sequence.
- All four groups use exactly the same 16 profile seeds. Within each profile, the complete
  initial diagnostic rows are exactly equal between directions.
- Combined rows: sequences 64, passes 64, tiles 3,200. NaN/Inf: 0.

## Group results

| profile | direction | quality | safety | GU final mean | total Ra (um) | fine Ra (um) | removal CV | waviness Ra (um) | center/edge ratio | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_stress | cross_xy | 0/16 | 16/16 | 62.5581 | 0.143210 | 0.071916 | 0.360333 | 0.091767 | 1.50908 | 0.982175 |
| legacy_stress | same_xx | 0/16 | 16/16 | 62.6217 | 0.144991 | 0.072057 | 0.362944 | 0.092921 | 1.49873 | 0.981819 |
| new_car_mild | cross_xy | 16/16 | 16/16 | 86.8876 | 0.037028 | 0.023171 | 0.288063 | 0.023800 | 1.34096 | 0.896106 |
| new_car_mild | same_xx | 16/16 | 16/16 | 86.8702 | 0.036929 | 0.023175 | 0.290179 | 0.023914 | 1.33559 | 0.896081 |

All 32 legacy sequences improved scratch maximum and passed the Ra threshold, but none reached
GU 70; only 5/16 in each direction met the Rz threshold. Their `fail_max_passes` outcome is the
intentional one-pass Gate C limit, not a safety failure. All 32 new-car sequences passed quality
and safety.

## Paired direction difference (`cross_xy - same_xx`)

| profile | GU | total Ra (um) | removal std (um) | removal CV | waviness Ra (um) | center-edge delta (um) | center/edge ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_stress | -0.06368 | -0.001782 | -0.001010 | -0.002611 | -0.001155 | +0.004166 | +0.010350 |
| new_car_mild | +0.01742 | +0.000098 | -0.000332 | -0.002116 | -0.000114 | +0.000671 | +0.005364 |

Negative removal-CV/waviness values favor `cross_xy`; positive center/edge values favor
`same_xx`. The effects are small and mixed compared with the profile-to-profile difference.
In `new_car_mild`, cross direction lowered removal std and CV on 16/16 paired seeds, but raised
the center-edge delta on 16/16. Therefore Gate C does not establish a universally superior
direction. `cross_xy` is retained only as a tentative uniformity candidate and `same_xx` as the
paired control.

## Interpretation limits

- The 0.40 candidates were compared against the 0.18-style reference only analytically; no
  16-seed paired PhysX baseline at 0.18 was run. Do not claim a measured improvement over that
  baseline from this Gate.
- A single pass was used by design. Legacy quality failure does not authorize extra passes here.
- Profile effects dominate path-direction effects. This Gate is a screen, not a learned-policy
  result and not a vehicle-field validation.

## Files

- `group_summary.csv`: four group counts, means, and initial-to-final changes.
- `paired_direction_deltas.csv`: 32 seed-level paired direction deltas.
- `paired_direction_summary.csv`: paired delta means and better-counts by profile.
- `combined_sequences.csv`, `combined_passes.csv`, `combined_tiles.csv`: preserved combined rows.
- `pairing_validation.json`: seed and complete initial-state equality checks.
- `checksums.sha256`: integrity manifest for this summary and the Gate C source/result evidence.

