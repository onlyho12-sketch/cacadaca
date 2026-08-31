# Gate F6 flat-release zero-shot curved evaluation

## Decision

- **F6 diagnostic completion: PASS** — 64/64 sequences and 1,600/1,600 tiles.
- **Zero-shot sufficiency: FAIL** — cylinder and freeform violate the frozen
  all-sequence safety and paired quality-count requirements.
- No training was performed and the official flat release was read-only.

The evaluated policy is the official Gate E7 flat release, seed 20260831,
iteration 211, SHA-256
`c733763bfebd7fcead01cf2574bbd1621c4660b9347143cccb35e0e83449d052`.
It received its original 14-dimensional observation only; curvature and local
normal were not added.

## Geometry results

| geometry | safety | quality | force overload | all-4 cell area | mean steps |
|---|---:|---:|---:|---:|---:|
| flat | 16/16 | 12/16 | 0 | 91.7931% | 4153.6 |
| cylinder R=0.60 m | 11/16 | 8/16 | 5 | 91.6038% | 3296.8 |
| sphere R=0.60 m | 16/16 | 12/16 | 0 | 91.7863% | 4150.0 |
| freeform seed 0 | 11/16 | 8/16 | 5 | 91.6838% | 3359.3 |

Sphere passes every frozen paired zero-shot check.  Cylinder and freeform each
have five force-overload terminations and four fewer quality passes than the
paired flat evaluation.  Sensor-fault steps are zero for all 64 sequences.

The negative mean-step deltas for cylinder and freeform are not speed gains:
early force-overload termination censors their run time.  Likewise, their
small all-4 area deltas must not be used to claim success because the initial
synthetic maps already have high local pass area and safety termination takes
precedence.

## Bottleneck

The F5 geometric contact and normal-alignment layer passed independently.  In
F6, the flat policy reacts to changed curved-contact observations without an
explicit curvature/local-normal input.  Cylinder force action is higher than
flat on average (`0.7468` vs `0.6735`), and freeform is higher again (`0.7611`).
The reproducible overloads therefore isolate a policy observation/action
generalization bottleneck, rather than a sensor fault or quaternion/mesh
failure.

The complete per-case failure list is in `geometry_failures.csv`.  F7 should
compare a small curvature/local-normal observation ablation and force-action
safety behavior, but requires separate user approval before any pilot or
training.
