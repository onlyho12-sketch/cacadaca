# Gate F4 curved PhysX diagnostic

Decision: **FAIL** under criteria frozen before the PhysX runs.  No training
was performed, and Gate F5 was not started.

## Candidate results

| Candidate | Completion | Frozen checks | Main result |
|---|---:|---:|---|
| flat / cross | complete | 19/20 | gap tracking p95 4.344 mm > 4.000 mm |
| cylinder R0.60 / cross | complete | 19/20 | gap tracking p95 4.197 mm > 4.000 mm |
| sphere R0.60 / cross | complete | 19/20 | gap tracking p95 4.115 mm > 4.000 mm |
| freeform seed 0 / cross | incomplete | 2/20 | Isaac entered quick framework shutdown before pass completion |

The three completed candidates had zero sensor faults, zero fallback steps,
zero no-contact removal errors, and no hard-force, thermal, or unstable-contact
latches.  Their post-first-contact rates exceeded 99.9%, and their force
tracking checks passed.  The shared gap p95 miss is therefore retained as a
real frozen-criterion failure rather than relaxed after observation.

The freeform candidate was attempted three times with identical frozen
physics/path/seed settings.  Each attempt ended without a Python traceback or
completed output.  The instrumented third attempt reported step 1900 before
Isaac recorded a normal `quick framework shutdown`; its flushed trace through
step 1750 was finite, had zero sensor faults, and raw force stayed at or below
9.042 N.  It fails closed because the pass did not complete.

## Scope

This is a PT-DESIGN synthetic curved PhysX diagnostic with one environment,
vertical pad control, zero action, and no training.  It is not real-cell
validation.  F5 requires separate user approval and must not start from this
failed gate without diagnosis.
