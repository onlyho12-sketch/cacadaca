# Gate D observation expansion — non-training validation

## Scope

This Gate adds observation-only alternatives without changing the established
surface, quality, removal, thermal, action, reward, path, or PPO equations.
Protected common environments and previous results are unchanged.

- `base14`: established observation, byte-for-byte prefix and frozen baseline.
- `global20`: `base14` plus six cached inspection values.
- `spatial120`: `global20` plus four 5x5 maps (100 values).

The six global values are normalized inspection pass, signed GU/Ra/Rz target
gaps, minimum clearcoat safety margin, and center-minus-edge removal. Positive
quality gap means worse than the target and negative means margin beyond the
target. The four maps are tile scratch maximum, cumulative removal mean,
under-polished fraction, and over-polished fraction.

All new normalization scales are **PT-DESIGN**, not measured process
distributions. Initial inspection is cached during pass 1. If another pass is
requested, the immediately preceding pass-end inspection replaces the cache.
The full 200 mm ROI is therefore not reprocessed at every 20 Hz control step.

## Evidence paths

- Final CPU validation:
  `learning/rl/robot/results/gate_d_observation_cpu_signed_20260831_021220/`
- Final PhysX smoke:
  `learning/rl/robot/results/gate_d_observation_physx_signed_20260831_021225/`
- Preserved inputs: Gate C factory cross/same runs `...014100` and `...014600`.

Earlier `...021004`, `...021146`, and PhysX `...021016` are preserved intermediate
outputs. `...021004` covered initial inspection only; `...021146` added pass-end
inspection; `...021016` preceded the signed-gap correction. They are not final
Gate D evidence and were not deleted or overwritten.

The prior Gate C checksum manifest included the two living handoff documents.
Gate D necessarily appends to those documents, so only those two old manifest
entries are expected to change. Gate C code and data entries remain frozen and
are rechecked through `preserved_gate_c_code_data.sha256`. Gate D deliberately
does not put living handoff documents in its own immutable manifest.

## CPU validation result

- 192 vectors = 2 directions x 2 inspection stages x 3 profiles x 16 seeds.
- Each vector contains the 106-value extension used after the established 14 values.
- All numeric values are finite.
- The 48 initial cross/same pairs are exact, proving the direction comparison starts
  from identical observation extensions.
- Pass-end exact pair count is 0/48, as expected because the two paths generated
  small but real direction-dependent removal states.
- For `same_xx`, 75/318 profile-feature summaries vary across seeds initially and
  302/318 vary after the pass. Nonzero summaries increase from 75 to 305.
- Initial removal/pass/center-edge values are correctly zero. After the pass,
  removal, under/over, pass number, and center-edge channels become active.
- Signed Ra/Rz gaps remain informative even when a surface is already below the
  pass limit; the earlier excess-only prototype was rejected because those channels
  collapsed to zero.

Mean `same_xx` signed global values:

| profile | stage | GU gap | Ra gap | Rz gap | clearcoat margin | center-edge delta |
|---|---|---:|---:|---:|---:|---:|
| factory prepolish | initial | -0.2069 | -0.4382 | -0.5477 | 0.6313 | 0.0000 |
| factory prepolish | after pass | -0.4680 | -0.5383 | -0.5952 | 0.6232 | 0.0571 |
| deep defect | initial | +0.0180 | -0.4279 | -0.4361 | 0.6313 | 0.0000 |
| deep defect | after pass | -0.3310 | -0.4564 | -0.4571 | 0.6216 | 0.0910 |
| deep stress | initial | +0.0489 | -0.4191 | -0.2836 | 0.6313 | 0.0000 |
| deep stress | after pass | -0.2718 | -0.4415 | -0.3103 | 0.6216 | 0.0917 |

These are normalized synthetic diagnostic values, not measured GU or
profilometer distributions.

## PhysX smoke result

- Three environments: one for each corrected factory profile, seeds paired at 1000.
- Physical contact, `same_xx`, 2.1 m Gate C path, zero residual action.
- 20 control steps with snapshots at 0/1/5/10/20.
- 2,310 long-form observation samples and 154 feature summaries are finite.
- `base14 == global20[:14]` and `global20 == spatial120[:20]` exactly for every
  snapshot. No episode completed; this is intentionally a startup/integration
  smoke, not a quality or policy evaluation.
- `training_performed=false`; no checkpoint was loaded or created.

## Gate decision

`spatial120` is the recommended observation candidate for the next, separately
approved BC stage because it preserves the established 14-value prefix, contains
the global inspection control, and is the only variant that locates scratch and
removal imbalance. `global20` remains the lower-dimensional ablation control and
`base14` remains the frozen baseline.

This is an information/integration decision only. It does not claim that a
120-dimensional policy performs better; that requires separately approved BC
training and identical-seed policy evaluation.

## Stop point

Gate D observation design and non-training validation are complete. BC dataset
generation, BC training, PPO, Gate E, and curved-surface work have not started.
