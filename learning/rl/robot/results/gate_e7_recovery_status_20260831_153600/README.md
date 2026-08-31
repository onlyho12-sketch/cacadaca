# Gate E7 PPO main training recovery status (2026-08-31 15:36 KST)

Gate E7 is **in progress**, not evaluated and not eligible for champion promotion.
All paths below are new outputs; no previous result was overwritten.

## Completed and preserved

- Seed `20260831`, surface base `21000`: 261 iterations, 150,336 samples,
  integration PASS.
- Output: `../gate_e7_train_seed20260831_150336_20260831_110000/`
- Final model SHA-256:
  `b40712e35b8e881330cfc3f529cffe1012d01f3c35441bf80709d6b42aebe7dd`
- Frozen champion SHA-256 remains:
  `5fa1a65a60a90ffca5116a8d75d20a34749ebd69d71bb6f27287e0957f49b14a`

## Interrupted seed and evidence

- Seed `20260832`, surface base `22000` completed iterations `0..196` and
  preserved 197 per-iteration models.
- Last complete model: `../gate_e7_train_seed20260832_150336_20260831_113000/model_196.pt`
- Completed samples: `197 * 12 * 48 = 113,472`.
- `model_196.pt` SHA-256:
  `afe11120f80fc57e499e4d127f9e92888b43816a0c8c3f65671c9b0aefa367ad`
- After iteration 196, one Isaac physics collection step made no progress for
  more than 15 minutes while still consuming CPU/GPU. SIGINT and SIGTERM were
  not handled; only the two exact training PIDs were terminated with SIGKILL.
- After termination, the NVIDIA driver reported 100% utilization with no compute
  process and Isaac returned `cuInit failed (CUresult 999)`. A reboot is required.

## Recovery implementation

- New isolated runner: `learning/rl/gate_e7_ppo_resume_train.py`
- It verifies checkpoint iteration 196, advances to the next iteration, and runs
  exactly iterations `197..260` (64 iterations, 36,864 additional samples).
- It refuses an existing output directory and does not change the parent model.
- Script SHA-256 at this handoff:
  `a68214dad3b7d3500a1473361cb80dbda8b4d027102a761c0369eb238b2909bf`
- Isaac checkpoints do not contain live environment state. The resumed segment
  therefore starts a new same-seed environment episode while continuing the
  actor/critic weights from iteration 196. Report this seed as a segmented run.

## Exact post-reboot command

Run only after `nvidia-smi` is no longer stuck at 100% and a CUDA preflight passes:

```bash
/home/rokey/isaacsim-6.0.1/python.sh learning/rl/gate_e7_ppo_resume_train.py \
  --checkpoint learning/rl/robot/results/gate_e7_train_seed20260832_150336_20260831_113000/model_196.pt \
  --out_dir learning/rl/robot/results/gate_e7_train_seed20260832_resume_after196_36864_20260831_153700 \
  --expected_saved_iteration 196 --remaining_iterations 64 \
  --cumulative_target_iterations 261 --envs_per_profile 4 \
  --num_steps_per_env 48 --gae_lambda 0.95 --seed 20260832 \
  --surface_seed_base 22000 --viz none
```

Then train independent seed `20260833`/surface base `23000` from the frozen
champion for 261 iterations. Only after all three seeds exist, screen the five
planned checkpoints per seed (`52/105/158/211/260`) on aligned unseen surfaces.
The interrupted seed uses `model_52/105/158` from the original directory and
the corresponding `211/260` models from the resume directory.

No Gate E7 checkpoint has been evaluated or promoted yet.

## Completion update (2026-08-31 18:28 KST)

The post-reboot recovery completed successfully. Seed `20260832` resumed exactly
at iterations `197..260`, integration PASS, and cumulative nominal samples are
150,336. Seed `20260833` also completed 261 iterations/150,336 samples with
integration PASS. All 15 checkpoint screens and the final champion+three-policy
same/cross comparison completed. Final summary:
`../gate_e7_main_summary_seed25000_20260831_181500/`.

No checkpoint was promoted automatically. The frozen champion remains unchanged.
