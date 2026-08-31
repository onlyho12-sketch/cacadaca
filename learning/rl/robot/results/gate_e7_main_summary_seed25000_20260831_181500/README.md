# Gate E7 flat PPO multi-seed main training

## 결론

- 독립 training seed 3개에 각각 150,336 samples, 총 451,008 nominal samples를 학습했다.
- seed 20260832는 iteration 196 뒤 Isaac 정체로 재부팅 후 197~260회를 이어간 segmented run이다.
- 15 checkpoint aligned screen과 대표 3개+champion의 unseen same/cross 32표면 평가를 완료했다.
- 네 정책 모두 안전 32/32, sensor fault 0이다.
- 순수 품질 최고는 `s20260833_it211`: 품질 26/32, all-4 92.4247%, 평균 6300.3 step.
- 품질-시간 균형 후보는 `s20260832_it260`: 품질 26/32, all-4 92.2953%, 평균 4882.8 step.
- 기존 champion은 품질 20/32, all-4 91.7875%, 평균 3655.9 step.
- 사전 승인 임계가 없으므로 champion을 자동 교체하지 않았다.

## 정책당 800타일의 내부 all-4 통과면적

| 정책 | 100% | ≥95% | ≥90% | ≥80% |
|---|---:|---:|---:|---:|
| champion | 534 | 592 | 598 | 649 |
| raw quality | 550 | 613 | 617 | 658 |
| quality-time | 540 | 603 | 615 | 655 |

타일 평균 하나로 전체 타일을 통과 처리하지 않았고, 각 타일 내부 cell의 all-4 통과면적을 기준으로 집계했다.

## 다음 승인 항목

1. 순수 품질 최고와 품질-시간 균형 중 배포 목표를 선택한다.
2. 허용 가능한 작업시간 증가율과 필요한 품질/타일 개선 임계를 정한다.
3. 선택 정책을 release/regression 검증한 뒤에만 champion 승격 여부를 결정한다.
