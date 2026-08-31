# Gate F5 normal-alignment remediation

## Decision

**PASS (4/4 surface pairs).**  Every final normal-aligned run passed all
frozen Gate F4 safety/quality checks and all Gate F5 comparison checks.  No
threshold was changed and no training was performed.

The remediation compensates the measured steady differential-IK pad-axis bias
on curved surfaces with a conservative world-tangent command bias of
`(-0.35, -0.03)` degrees.  Flat remains unmodified and preserves its exact
`[1, 0, 0, 0]` xyzw target and parity behavior.

## Alignment results

- flat: p95 `0.540914 deg`; all 29/29 checks passed
- cylinder: p95 `0.125115 deg`; ratio `0.013663`; all 27/27 passed
- sphere: p95 `0.123541 deg`; ratio `0.010260`; all 27/27 passed
- freeform: p95 `0.315062 deg`; ratio `0.144337`; all 27/27 passed

## Final normal runs

- `gate_f5r_normal_flat_cross_20260831_171300`
- `gate_f5r_normal_cylinder_r0p60_cross_20260831_171600`
- `gate_f5r_normal_sphere_r0p60_cross_20260831_171900`
- `gate_f5r_normal_freeform_seed0_cross_20260831_171000`

The prior `gate_f5_alignment_summary_20260831_170400` FAIL remains preserved
as pre-remediation evidence.  Further Gate F or post-F5 work requires separate
user approval.
