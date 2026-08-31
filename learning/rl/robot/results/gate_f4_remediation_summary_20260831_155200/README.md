# Gate F4 remediation result

Decision: **PASS**.  All four representative candidates passed all 20 criteria
that were frozen before the original F4 PhysX runs.  No threshold was relaxed,
no training was performed, and Gate F5 was not started.

## Remediation

The original completed candidates showed the measured pad gap consistently
about 3.15--3.18 mm below the internal clearance command under contact.  A
Gate-F-only 0.5 mm positive-Z target feed-forward was frozen before rerun.  It
does not change the surface geometry, force setpoint, controller gains, path,
physics seed, or acceptance criteria.

The earlier freeform exits were reproduced only in PTY-backed executions.  A
non-PTY run passed the previous exit region and completed all 3307 control
steps, so the evidence identifies an execution-session shutdown rather than a
freeform geometry or PhysX numerical failure.

## Results

| Candidate | Checks | Gap p95 | Raw force max | Force-error p95 | Contact rate |
|---|---:|---:|---:|---:|---:|
| flat / cross | 20/20 | 3.844 mm | 7.596 N | 0.065 N | 99.969% |
| cylinder R0.60 / cross | 20/20 | 3.701 mm | 9.558 N | 1.240 N | 99.875% |
| sphere R0.60 / cross | 20/20 | 3.618 mm | 13.539 N | 1.311 N | 99.969% |
| freeform seed 0 / cross | 20/20 | 3.857 mm | 10.587 N | 1.128 N | 99.969% |

All candidates completed with zero sensor faults, zero fallback steps, zero
no-contact removal errors, and no hard-force, thermal, or unstable-contact
latches.  Mean ROI removal was 0.210--0.212 um and coverage was 96.01--96.27%.

## Scope

This remains a PT-DESIGN synthetic curved PhysX diagnostic with one environment,
vertical pad control, zero action, and no training.  It is not real-cell
validation.  Gate F5 requires separate user approval.
