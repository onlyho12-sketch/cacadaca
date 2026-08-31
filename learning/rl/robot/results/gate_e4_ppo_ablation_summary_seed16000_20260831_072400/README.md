# Gate E4 base14 short PPO ablation

- Training: 12 envs, 5 critic warmup + 5 actor iterations, 5,760 nominal samples.
- Actor relative L2 drift after actor5: 0.0145206288.
- Screen: champion and actor1/actor3/actor5, 8 executions each; all safety 8/8.
- Screen quality: champion 3/8, actor1 3/8, actor3 3/8, actor5 4/8.
- Actor5 was selected for confirmation, not promotion.
- Confirmation: 32 paired conditions / 64 executions across same_xx and cross_xy.
- Safety champion/actor5: 32/32 / 32/32; sensor faults 0/0.
- Quality champion/actor5: 20/32 / 20/32.
- Mean actor5 - champion: GU -0.024530, Ra -0.001447 um, Rz -0.004501 um, scratch +0.001989 um, clearcoat -0.004081 um, control steps -34.375.
- Verdict: safety integration passes, but quality count is tied and GU/scratch regress.
- No checkpoint is promoted; the frozen champion remains unchanged.
- This is PT-DESIGN short ablation, not full PPO or a production result.
