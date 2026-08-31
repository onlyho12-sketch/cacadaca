# Gate E2 factory-only BC 파일럿 최종 결과

## 결론

- 335,702-sample factory-only BC 파일럿과 256-execution paired PhysX 평가는 완료했다.
- 새 BC 세 모델 모두 frozen champion으로 **승격하지 않는다**.
- `base14`는 가장 근접했지만 champion을 개선하지 못했다.
- `global20`은 base14보다 일반화가 나빴다.
- `spatial120`은 offline/PhysX 모두 가장 나빠 탈락한다.
- 기존 champion은 그대로 보존한다. 100만 sample 본 수집과 PPO는 시작하지 않는다.

## 데이터와 학습

- factory 성공 sequence 112/112, 총 335,702 samples, NaN/Inf 0.
- profile sequence 비중: factory/deep defect/deep stress = 56/34/22.
- path sequence 비중: same/cross = 74/38.
- 비중은 실측 생산분포가 아닌 PT-DESIGN이다.
- seed split sample 수: train 232,498 / validation 50,745 / test 52,459.
- legacy 학습 sample은 0이다.

| 모델 | test MSE | 판정 |
|---|---:|---|
| base14 | 0.000543475 | 승격 안 함, 가장 가까운 ablation |
| global20 | 0.001186834 | 승격 안 함 |
| spatial120 | 0.005269270 | 탈락 |

교사 label은 frozen champion의 base14 action이다. 따라서 global/spatial feature에 직접
의존하는 정답 신호가 없고, spatial120은 epoch 2 이후 validation 악화로 조기 종료했다.

## 최종 1-pass paired PhysX

조건은 네 profile × 두 경로 × 동일 미사용 8 seed × 네 정책 = 256 executions이다.
모든 정책은 `max_passes=1`, physical contact, seed base 10000이다.

| 정책 | 전체 성공 | factory 성공 | legacy 성공 | 안전 | fault |
|---|---:|---:|---:|---:|---:|
| frozen champion | 42/64 | 42/48 | 0/16 | 64/64 | 0 |
| BC base14 | 42/64 | 42/48 | 0/16 | 64/64 | 0 |
| BC global20 | 42/64 | 42/48 | 0/16 | 64/64 | 0 |
| BC spatial120 | 40/64 | 40/48 | 0/16 | 64/64 | 0 |

champion 대비 전체 paired 평균:

| 정책 | ΔGU | Δscratch (μm) | Δclearcoat (μm) |
|---|---:|---:|---:|
| BC base14 | -0.036669 | +0.002477 | -0.000518 |
| BC global20 | -0.173298 | +0.011872 | +0.006153 |
| BC spatial120 | -1.012848 | +0.036614 | +0.036819 |

base14/global20은 성공 수는 같지만 GU와 scratch가 소폭 악화되어 새 champion으로 바꿀
근거가 없다. spatial120은 deep stress 성공이 12/16으로 champion의 14/16보다 낮고,
전체 GU/scratch도 더 나쁘다.

## 보존 중간 결과

`gate_e2_eval_bc_base14_same_32_seed10000_20260831_045000/`과
`gate_e2_eval_frozen_champion_same_32_seed10000_20260831_051000/`은 2-pass 확장 stress다.
pass1-only BC를 미학습 legacy pass2에 외삽하므로 최종 비교에서는 제외했다. 삭제하지
않았으며 base14 stress에서 legacy 안전 7/8, force overload 1건을 기록했다.

## 다음 작업

대규모 동일-label BC를 반복하지 않는다. 먼저 다음 중 하나를 승인받아야 한다.

1. 공간 feature에 실제로 의존하고 안전한 expert/target을 새로 설계한 소규모 파일럿,
2. 기존 champion을 유지한 채 BC를 종료하고 PPO 전 안전·보상·평가 계획만 설계.

어느 경우든 기존 champion, 보호 파일, 기존 결과를 덮어쓰지 않는다.
