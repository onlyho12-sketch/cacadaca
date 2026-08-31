# Gate F3 curved geometry and analytic coverage screen

## Result

- Geometry candidates: 12
- same/cross path candidates: 24
- 5x5 tile rows: 600
- finite candidates: 24/24
- effective analytic coverage: minimum 100%
- physical mesh projected footprint support: 24/24
- quality-state-map projected footprint support: 18/24
- CPU regression tests after correction: **29/29 PASS**
- Isaac/PhysX executed: **no**
- BC/PPO training executed: **no**

All metrics are `PT-DESIGN` analytic geometry diagnostics.  Coverage is a
normalized local-tangent Gaussian footprint and is not physical removal.

## Normal consistency correction

The first preserved screen at `gate_f3_geometry_coverage_20260831_235207`
revealed that the imported cylinder/sphere normal signs did not match the
derivatives of their negative-sag height fields.  The R=0.60 m cylinder showed
an impossible 21.52 mm tangent-plane deviation across a 55 mm pad radius.

The isolated Gate F normal X/Y signs were corrected and verified against
numerical height gradients before this screen was generated.  The corrected
R=0.60 m cylinder/sphere maximum tangent-plane deviations are 2.523/2.526 mm.
The external source, shared environments, and protected files were not changed.

Because the F1 checksum manifest includes live source paths, its original
geometry/test source hashes now describe the preserved pre-correction state and
are superseded by this result.  The F1 result files themselves were not edited.

## F4 representatives

The frozen PT-DESIGN rule selects the median surface-tilt geometry per family,
then the lower-CV path direction.  The first F4 contact diagnostic candidates are:

| Candidate | Path | Path max tilt | Footprint normal change | Tangent deviation | Physical margin |
|---|---|---:|---:|---:|---:|
| flat | cross_xy | 0.000° | 0.000° | 0.000 mm | 21.981 mm |
| cylinder R=0.60 m | cross_xy | 9.887° | 5.256° | 2.523 mm | 21.152 mm |
| sphere R=0.60 m | cross_xy | 13.051° | 5.259° | 2.526 mm | 20.523 mm |
| freeform seed 0 | cross_xy | 2.533° | 3.705° | 1.901 mm | 21.965 mm |

R=0.30 m cylinder and R=0.30/0.45 m sphere paths remain in the CSV but are not
sent to the first F4 smoke: their projected pad footprints leave the 320 mm
quality map by up to 4.664 mm.  They still have at least 15.336 mm physical mesh
support, so they are retained as later stress candidates rather than labelled
invalid geometry.

