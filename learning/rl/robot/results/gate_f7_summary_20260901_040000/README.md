# Gate F7 small observation ablation

- Completion: **PASS**
- Observation-extension gate: **NO-GO**
- Best held-out arm: `base14`
- base14 test force MAE: `0.014785069`
- Best extension (`normal17`) test force MAE: `0.015087314`
- Candidate PhysX evaluation: not entered because the frozen offline entry criterion failed.
- Promotion/release modification: none.
- F8: not started; separate approval is required.

All three arms used the same 19,200 samples, split, parent initialization, seed,
optimizer settings, and epoch count.  The result supports keeping base14 for the
next safety-control experiment rather than scaling this observation extension.
