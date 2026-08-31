# Gate F2 flat parity and non-regression

Gate F2 verifies that the isolated Gate F geometry reduces exactly to the
established flat definitions without rerunning Gate E or starting Isaac/PhysX.

## Result

- Combined F1+F2 CPU tests: **23/23 PASS**
- Flat height over the full map and mesh margin: exactly `0 m`
- Flat normal: exactly `(0, 0, 1)`
- Local-normal force projection: exactly `abs(Fz)` for 2,048 deterministic vectors
- Flat target Z: exactly `work_top + clearance`
- Flat measured gap: exactly `pad_face_z - work_top`
- Flat 41x41 mesh: 1,681 vertices and 3,200 upward-wound triangles
- Cylinder/sphere at radius `1e8 m`: flat within the frozen numerical tolerances
- Factory profile quality-state arrays: unchanged while evaluating flat macro geometry
- Isaac/PhysX executed: **no**
- Training executed: **no**
- Existing/production files modified by F2: **none**

This is a definition-level parity result.  It does not claim runtime PhysX parity;
the first curved PhysX contact execution remains Gate F4.  Gate F3 may now screen
curvature, normal variation, pad footprint, coverage, edge margin, and tile geometry
using CPU calculations only.

