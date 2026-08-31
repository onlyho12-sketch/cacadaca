# Gate F1 isolated curved-surface port

Gate F1 selectively ports three external ideas into new local files only:

1. flat/cylinder/sphere/freeform height and unit-normal formulas;
2. upward-wound deterministic trimesh generation and local-normal force projection;
3. CPU geometry tests plus an unexecuted entry point for the later F4 PhysX smoke.

## Result

- CPU geometry tests: **17/17 PASS**
- Static syntax checks: **5/5 PASS**
- Isaac/PhysX executed: **no**
- BC/PPO training executed: **no**
- Existing files modified by F1: **none**
- Protected hashes: unchanged from the Gate E7 release manifest

The default Gate F configuration uses the established factory planar ROI stack as a
base, but the curved implementation is isolated in a new subclass.  The vertical-pad
condition is explicit.  Enabling local-normal pad alignment raises an error until the
separately approved Gate F5 implementation exists, preventing an unvalidated alignment
mode from being reported as complete.

## External provenance

The documented external commit was `1d020643687e516dbf56453caa8db88eab554adb`.
At F1 start, remote `learning-bc-v2` pointed to
`cb9ee7e7beca0c40fb6669db659f90681bfcca1c`.  The four intervening commits modify
the protected v5/bridge path, not the selected curved geometry/env/test sources.  F1
records the current commit and per-source hashes in `provenance.json`; no external
checkpoint, result, public environment, or v5 change was copied wholesale.

## Scope boundary

`learning/rl/gate_f_curved_physx_smoke.py` is prepared but was not executed.  Flat
parity is Gate F2, geometry/coverage screening is Gate F3, and the first curved PhysX
contact execution is Gate F4.  No claim about curved contact stability, pad alignment,
zero-shot quality, or learned curved policy is made by this result.

