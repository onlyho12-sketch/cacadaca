# Gate E5 PPO credit-assignment diagnostic

- Frozen champion, factory profiles only: 12/12 one-pass executions completed.
- Safety: 12/12; quality: 12/12; sensor faults: 0.
- Episode steps min/mean/max: 2942 / 3269.00 / 3562.
- With 48-step PPO rollouts, first/all terminal events occur after 62 / 75 iterations.
- At most the last 48 steps directly share a terminal-bearing GAE rollout: mean path fraction 1.475%.
- Terminal reward min/mean/max: 654.469727 / 691.073919 / 715.381348.
- Gate E4 used only 480 steps per environment, below even this diagnostic's minimum episode length, so it contained zero terminal events by construction.
- gamma=0.9995 reduces long-horizon decay but does not remove the 48-step rollout boundary; terminal credit still reaches only the final rollout directly.
- Reward/action/physics formulas were not changed. No training or checkpoint promotion was performed.
