# Gate E5 exact PPO credit summary

- Frozen factory one-pass executions: 12/12; safety and quality 12/12; sensor faults 0.
- Episode steps min/mean/max: 2942 / 3269.00 / 3562.
- Gate E4 used 480 steps/env, so terminal events were mathematically impossible.
- Current 48-step rollouts first/all see terminal after 62 / 75 iterations.
- Exact rollout-local direct-credit steps min/mean/max: 6 / 21.00 / 40, only 0.647% of the path on average.
- With current lambda=0.95, terminal GAE weight at that rollout's start min/mean/max: 0.132663 / 0.418397 / 0.771848.
- A PT-DESIGN lambda=0.99 would make those values 0.662677 / 0.815864 / 0.948615, but may increase variance.
- Terminal reward min/mean/max: 654.469727 / 691.073919 / 715.381348; mean terminal/|dense| ratio 3.214.
- Hypothetical uninterrupted gamma=0.9995 weight at episode start is 0.196218 on average, but this is not realized across separate PPO storage updates.
- Conclusion: raising gamma alone cannot solve the credit boundary. Full PPO should not start yet.
- Next pilot recommendation: equal-sample comparison of current 48-step/lambda=0.95 and 128-step/lambda=0.99, with strict checkpoint safety screens.
- No reward formula, environment, policy, or checkpoint was changed.
