# Gate F9 multi-seed paired validation

- Decision: **FAIL** (integration candidate: False)
- Official sample: 64 runs, 256 sequences, 6400 tile rows; duplicates excluded: 0
- Entry curved (cylinder+freeform) safety: control 52 -> candidate 53
- Curved (cyl+sph+free) safety: control 84/96 -> candidate 85/96
- Force overloads: control 12 -> candidate 11
- Mean paired curved all4 drop: 0.064375 pp (limit 1.0)
- Curved control times are censored by early hard terminations; see censoring_summary.csv before comparing control steps.
- Training/PPO/promotion/release changes: none. A PASS is an integration candidate only; integration/release is a separately approved stage.
