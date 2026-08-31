# Corrected Gate B2 — factory pre-polish surface profiles

## Decision implemented

This result supersedes the earlier mar-based `new_car_mild` design for future work.
The earlier code and outputs remain preserved for audit, but are not imported here.

- `legacy_stress`: established Ra 0.08, 4--12 scratches, 0.05--2.0 um; unchanged.
- `factory_prepolish`: central-ROI base Ra 0.10--0.12 um, 0--2 independent
  short scratches, 0.10--0.60 um.
- `factory_prepolish_deep_defect`: same base and shallow scratches, plus one
  0.60--1.00 um scratch.
- `factory_prepolish_deep_stress`: same base and shallow scratches, plus one
  1.00--1.50 um stress-tail scratch.
- Scratch width and Gaussian groove formula remain the established 2 mm formula.
  Segment centers, directions and lengths are sampled independently; segments are
  not deliberately joined, although random intersections are possible.
- All new ranges are **PT-DESIGN**, not measured factory-population distributions.

No `mar_density`, `mar_severity`, optical sidecar, or new `SurfaceState` field was
created.  Existing Ra/Rz/GU, clearcoat, removal, temperature, PPO observation/action/
reward, and Gate B path calculations were not changed.

## CPU distribution and reproducibility validation

Final folder: `gate_b2_factory_profiles_20260831_013248/`

- 256 seeds per profile, 1,024 rows total; NaN/Inf 0.
- Legacy adapter: 256/256 complete `SurfaceState` array/scalar exact match.
- Factory declared-range violations: 0.
- Factory `SurfaceState` field count equals legacy and no mar field exists.
- All three factory profiles use exactly the same seed-specific base roughness,
  clearcoat and shallow-scratch population.  Only the declared deep tail differs.
- Factory base Ra target over the central evaluation ROI: mean 0.109948 um,
  min 0.100155, max 0.119985.
- Shallow count frequencies over 256 seeds: 0/1/2 = 82/76/98.

Initial CPU means:

| profile | total Ra (um) | Rz (um) | scratch max (um) | GU |
|---|---:|---:|---:|---:|
| legacy_stress | 0.128350 | 2.068903 | 1.769672 | 54.5124 |
| factory_prepolish | 0.110797 | 0.893118 | 0.270752 | 71.9331 |
| factory_prepolish_deep_defect | 0.113079 | 1.120295 | 0.789042 | 69.1667 |
| factory_prepolish_deep_stress | 0.114802 | 1.400952 | 1.235460 | 69.0718 |

## Grouped physical-contact check

Run folder: `gate_b2_factory_physx_20260831_012700/`

- Three profiles x the same four seeds (`1000,1097,1194,1291`) in one 12-env process.
- 360 mm physical plate, 320 mm continuous state map, central 200 mm ROI.
- Established Gate B path: step-over ratio 0.184, ten lines, 4.0 m, one pass.
- Feed anchor 12.7 mm/s; physical contact enabled.
- Completed 12/12 in 6,304 control steps.  Safety 12/12, fallback 0.
- No training was performed.

| profile | quality | safety | GU before->after | total Ra change (um) | fine Ra change (um) | scratch max before->after (um) | removal CV | waviness Ra (um) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| factory_prepolish | 3/4 | 4/4 | 71.0372 -> 74.1386 | -0.032484 | -0.037997 | 0.291351 -> 0.212644 | 0.293404 | 0.055345 |
| deep_defect | 3/4 | 4/4 | 68.0575 -> 72.6834 | +0.004976 | -0.041288 | 0.748739 -> 0.456132 | 0.441921 | 0.095701 |
| deep_stress | 3/4 | 4/4 | 67.9486 -> 72.3158 | +0.020254 | -0.041353 | 1.182682 -> 0.828693 | 0.484337 | 0.109570 |

Seed 1194 is the one quality failure in every profile (final GU 69.8991,
68.6678 and 68.3832 respectively); all safety checks still pass.  The ordinary
profile has one zero-scratch seed, so its scratch-improved count is 3/4 by design.

## Interpretation

The path reduces fine texture in all profiles, but the deeper isolated defect causes
more spatially uneven removal.  Total Ra decreases for the ordinary profile, while it
increases for both deep profiles even though fine Ra and the scratch maximum improve.
This is the same removal-waviness separation observed in Gate B.

The existing quality gate counts a deep scratch as acceptable when it merely improves;
it does not require the residual depth to be small.  Therefore the 3/4 quality count for
`deep_stress` must not be reported as complete deep-scratch removal.  Its mean residual
scratch is still 0.828693 um and it is a rework/stress diagnostic profile.

## Preserved intermediate outputs

- `gate_b2_factory_profiles_20260831_012332`: full-map rather than central-ROI target scaling.
- `gate_b2_factory_profiles_20260831_012425`: correct ROI target before explicit all-seed
  range-violation counter was added.
- `gate_b2_factory_smoke_20260831_012500`: intentional incomplete 20-step 12-env startup check.
- `gate_b2_factory_summary_20260831_013148`: summary before initial-to-final delta columns.

The final evidence is `gate_b2_factory_profiles_20260831_013248`,
`gate_b2_factory_physx_20260831_012700`, and this summary folder.

## Stop point

Corrected Gate B2 is complete.  Gate C factory runs and BC/PPO training have not started.

