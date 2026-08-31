# Gate F5 normal-aligned pad comparison

## Decision

**FAIL (3/4 surface pairs passed).**  All four final normal-aligned PhysX runs
completed successfully without safety violations and passed every frozen Gate
F4 check.  Flat parity passed, and cylinder and sphere passed every Gate F5
comparison.

The only failed frozen check is freeform
`curved_alignment_error_ratio`: observed `0.3037822630`, required `<= 0.25`.
The freeform absolute steady alignment-error p95 was `0.6631100416 deg`, which
does pass its independent `<= 2 deg` limit.  No acceptance threshold was
changed after execution.

## Final normal runs

- `gate_f5_normal_flat_cross_final_20260831_165500`
- `gate_f5_normal_cylinder_r0p60_cross_final_20260831_165800`
- `gate_f5_normal_sphere_r0p60_cross_final_20260831_170100`
- `gate_f5_normal_freeform_seed0_cross_final_20260831_165200`

The implementation uses Isaac Lab's installed `xyzw` quaternion convention,
preserves the established flat link6 target `[1, 0, 0, 0]`, measures the
actual contact-disk local `+Y` outward axis, and couples face-center translation
to the commanded orientation.  Earlier probes and superseded runs remain in
place but are excluded from this aggregation.

No training was performed.  Further freeform alignment remediation or any
post-F5 work requires separate user approval.
