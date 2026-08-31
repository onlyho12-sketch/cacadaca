# Superseded quaternion-order interpretation

The physical run completed safely and matched the vertical flat baseline, but
the Gate F5 implementation and trace labels still interpreted Isaac Lab pose
quaternions as `wxyz`.  The installed Isaac Lab version uses `xyzw`; this run
is preserved as evidence and excluded from final F5 aggregation.  Final runs
use the corrected `xyzw` target composition and actual pad-body local `+Y`
outward-axis measurement.
