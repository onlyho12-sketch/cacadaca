# Gate F9 incident — Isaac asset server unreachable, runs 22-38 aborted

Recorded: 2026-09-01 10:10 KST. No frozen criterion, seed, cap value or threshold
was changed by this incident or its recovery.

## What happened

Network connectivity was intentionally turned off on the workstation at about
`08:56` while Gate F9 was running. Isaac Sim resolves the ground-plane USD asset
through `omni.client`, which requires the remote asset server:

```
gate_f_curved_polish_env._setup_scene
  -> isaaclab.sim.spawners.from_files.spawn_ground_plane
  -> sim.utils.prims.create_prim / add_usd_reference
  -> isaaclab.utils.assets.check_file_path
  -> omni.client.stat        <-- needs network
```

With no network this call fails, so every evaluation launched from `08:56:58`
onward aborted during scene creation, before writing any output.

`isaacsim/python.sh` returned exit status `0` for these aborted runs, so the
original `run_f9.sh` treated them as successful: it logged `done`, appended the
path to `f9_run_dirs.txt`, and continued. The failure was therefore silent.

## Impact

- Runs 1-21 (`seed40000` complete, `seed41000` first 5) are **valid and intact**.
- Runs 22-38 (17 runs) produced **empty directories, 0 files each**. No partial
  or corrupt CSV was written, so no invalid row can reach the official sample.
- Run 39 was hung in the same call and was stopped during recovery.
- Elapsed time lost: about 1 h 10 min.

Aborted runs were **not deleted**. Each was renamed with the suffix
`_aborted_network_20260901_101013`, matching how earlier gates preserved
incomplete/superseded directories as audit records.

## Why no invalid data can enter the official sample

Three independent guards held:

1. `gate_f8_force_safety_eval.py` writes `sequences.csv` and
   `tile_area_fractions.csv` before `metadata.json`, so the presence of
   `metadata.json` marks a completed run. The aborted runs have none.
2. `gate_f9_summarize.py` requires exactly 64 runs, each with matching
   checkpoint hash, `all_finite`, `completed == expected`, and full seed
   pairing. A short or empty sample fails aggregation rather than passing.
3. `run_f9.sh` refuses to reuse a directory that exists without
   `metadata.json`, so a resume cannot silently continue into a partial run.

## Fixes applied

- `run_f9.sh` now verifies that `metadata.json` and `sequences.csv` exist after
  every evaluation and exits with status 3 if they do not. The exit code of
  `python.sh` is no longer trusted on its own.
- `f9_run_dirs.txt` was regenerated from directories that actually contain
  `metadata.json` (21 entries). The polluted 38-entry version is preserved as
  `f9_run_dirs_polluted_20260901_101013.txt`.

## Operating requirement for the remainder of the run

Network access to the Isaac asset server must stay available for the whole run.
A `systemd-inhibit` lock (`sleep:handle-lid-switch`, block mode) allows the lid
to be closed without suspending, but the network must remain up.
