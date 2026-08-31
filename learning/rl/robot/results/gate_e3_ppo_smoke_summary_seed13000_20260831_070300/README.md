# Gate E3 base14 PPO integration smoke

- PPO integration gate: **PASS**
- Scope: 12 envs x 48 steps x 3 iterations = 1,728 nominal samples
- Critic warmup / actor update: 2 / 1 iterations
- Actor relative L2 drift: 0.00304908019
- Paired unseen one-pass executions: 8
- Safety champion/smoke: 8/8 / 8/8
- Quality champion/smoke: 6/8 / 6/8
- Mean smoke - champion: GU +0.006821, Ra -0.000541 um, scratch +0.000162 um
- This is a PT-DESIGN integration smoke, not a performance result or full PPO run.
- No checkpoint was promoted; the frozen champion remains unchanged.
