# Gate C corrected factory-profile paired evaluation

## Scope and evidence

- Gate C1 analytic screen was reused from
  `gate_c1_analytic_coverage_20260831_001831`; it selected step-over 0.40,
  `balanced_extend5`, and the `cross_xy`/`same_xx` pair. Both paths are 2.1 m.
- Preserved `legacy_stress` evidence comes from `gate_c_summary_20260831_004613`
  and was not rerun or modified: 2 directions x 16 paired seeds = 32 sequences.
- Corrected factory evidence comes from `gate_c_factory_crossxy_48_20260831_014100`
  and `gate_c_factory_samexx_48_20260831_014600`: 3 profiles x 2 directions x
  16 paired seeds = 96 new sequences.
- The combined comparison therefore covers 128 sequences. The factory runs contain
  96 initial rows, 96 sequence rows, 96 pass rows, and 4,800 tile rows.
- Every factory range is **PT-DESIGN**, not a measured factory population distribution.
- No BC/PPO training was performed. Existing checkpoints, Gate A/B/B2 results, and
  protected common environments were not overwritten.

The corrected factory profiles keep the established height map, `SurfaceState`,
Ra/Rz/GU, clearcoat, removal, temperature, PPO, and path equations. No mar optical
sidecar or new state field is used.

## One-pass PhysX results

| Profile | cross quality / safety | same quality / safety | cross GU | same GU | cross total Ra (um) | same total Ra (um) | cross scratch (um) | same scratch (um) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `factory_prepolish` | 15/16 / 16/16 | 15/16 / 16/16 | 74.6581 | 74.6804 | 0.09286 | 0.09235 | 0.20112 | 0.20125 |
| `factory_prepolish_deep_defect` | 15/16 / 16/16 | 15/16 / 16/16 | 73.3481 | 73.3101 | 0.10929 | 0.10872 | 0.62583 | 0.62640 |
| `factory_prepolish_deep_stress` | 14/16 / 16/16 | 14/16 / 16/16 | 72.8105 | 72.7181 | 0.11289 | 0.11170 | 1.04296 | 1.04587 |

All 96 factory sequences completed with fallback count 0. The same 16 profile seeds
were used in both directions, and every paired initial diagnostic row is exact.
All numeric CSV fields are finite.

The ordinary and deep-defect profiles fail only seed 1194 in both directions. The
deep-stress profile fails seeds 1194 and 2358 in both directions. These are
one-pass `fail_max_passes` quality outcomes, not safety failures. Residual scratch
means of about 0.626 um and 1.04 um show that the deep-tail cases are not complete
deep-defect removal; they remain rework/process-limit diagnostics.

## Direction diagnosis

Values below are paired means of `cross_xy - same_xx`.

| Profile | GU | removal CV | removal waviness Ra (um) | center/edge ratio | scratch max (um) | max force (N) |
|---|---:|---:|---:|---:|---:|---:|
| `legacy_stress` | -0.06368 | -0.002611 | -0.001155 | +0.010350 | -0.000677 | +0.1543 |
| `factory_prepolish` | -0.02232 | +0.001847 | +0.000337 | +0.005240 | -0.000125 | -0.0526 |
| `factory_prepolish_deep_defect` | +0.03796 | -0.000081 | +0.000748 | +0.014329 | -0.000576 | +0.1493 |
| `factory_prepolish_deep_stress` | +0.09240 | +0.003850 | +0.001767 | +0.018304 | -0.002911 | +0.1727 |

Direction effects are small and profile-dependent. `same_xx` is generally the more
conservative factory control for removal uniformity/waviness and center-edge balance,
while `cross_xy` gives slightly better deep-profile GU and residual scratch. Quality
counts are identical by direction, so Gate C does not establish a universal winner;
both remain candidates for the next approved stage.

The 0.18 reference has analytic coverage evidence only, not a matched 16-seed PhysX
baseline. The 2.1 m candidate path is 47.5% shorter than the 4.0 m analytic reference,
but this is not a measured quality superiority claim over step-over 0.18.

## Files and preservation notes

- `all_profile_group_summary.csv`: preserved legacy plus corrected factory group table.
- `all_profile_paired_direction_summary.csv`: four-profile direction comparison.
- `combined_initial_diagnostics.csv`, `combined_sequences.csv`, `combined_passes.csv`,
  `combined_tiles.csv`: corrected factory evidence only.
- `group_summary.csv`, `paired_direction_deltas.csv`,
  `paired_direction_summary.csv`: corrected factory summaries.
- `pairing_validation.json`, `summary_metadata.json`: pairing, provenance, training flags.
- `checksums.sha256`: final code/result/document freeze manifest.

`gate_c_factory_smoke48_20260831_014000` is the intentionally incomplete 20-step
48-environment startup smoke. The `...014514` and `...015203` summary directories are
preserved interim summaries. They are not the final evidence folder and were not
deleted or overwritten.

## Stop point

Gate C is complete. Gate D and BC/PPO training have not started and require a new
scope/ETA report and explicit user approval.
