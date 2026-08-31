# Gate C1 — analytic geometry/coverage screen

Isaac/학습 없이 200×200 mm ROI, 320×320 mm 품질맵, Ø110 mm pad의 Gaussian footprint를
경로 중심선을 따라 적분한 PT-DESIGN 노출 진단이다. 실제 제거량이나 실측 coverage가 아니다.

## 비교 범위

- step-over ratio: 0.18, 0.25, 0.32, 0.40
- 방향: 동일방향 2회 `same_xx`, X/Y 각 1회 `cross_xy`
- 경계 모드:
  - `legacy_start`: 기존 spacing/2 시작
  - `balanced`: 같은 line 수를 ROI 중앙에 대칭 배치
  - `balanced_extend5`: 대칭 배치 + 품질맵 여유 안에서 선분 양끝 5 mm 연장
- 총 24조건, 모두 pad footprint가 품질맵 안에 있고 analytic any/effective coverage 100%

## 상위 결과

| 순위 | step-over | edge | 방향 | CV | 중앙/가장자리 | under 80% | 경로 |
|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 0.40 | balanced+5 mm | cross_xy | 0.1463 | 1.201 | 13.92% | 2.10 m |
| 2 | 0.40 | balanced+5 mm | same_xx | 0.1504 | 1.201 | 14.84% | 2.10 m |
| 3 | 0.32 | balanced+5 mm | cross_xy | 0.1654 | 1.244 | 15.96% | 2.52 m |
| 현행 근사 | 0.18 | legacy | same_xx | 0.2251 | 1.388 | 23.46% | 4.00 m |

상위 2개는 현행 근사 대비 경로 길이 47.5% 감소, CV 약 33~35% 감소다. 좁은 step-over는
같은 경계 손실을 더 많은 중앙 line에서 누적해 이 geometry에서는 자동으로 더 균일하지 않았다.

## PhysX 선정

상위 1~2개 제한과 교차경로 진단을 동시에 만족시키기 위해 같은 0.40/balanced+5 mm에서
방향만 다른 다음 두 조건을 선정한다.

1. `so0p40_balext5_crossxy`
2. `so0p40_balext5_samexx`

각 조건은 `legacy_stress`와 `new_car_mild` 각각 같은 seed 16표면으로 1-pass 평가한다.
BC/PPO는 학습하지 않는다. 0.40의 sparse line이 실제 scratch 제거와 제거 waviness에 미치는
영향은 analytic 결과로 확정하지 않고 PhysX paired 결과로 판정한다.

끝점 5 mm 연장은 exact path endpoint에서 품질맵 footprint 여유를 모두 사용하지만 밖으로
나가지는 않는다. 물리 Workpiece에는 추가 20 mm 여유가 남는다. 이는 현재 시뮬레이션 screen
조건이며 실제 로봇 오차 허용 경로로 바로 사용하면 안 된다.

## 산출물

- `analytic_candidates.csv`: 24조건 scalar와 uniformity rank
- `analytic_tiles.csv`: 조건별 5×5 normalized exposure
- `analytic_summary.json`: 설정과 전체 결과
- `selected_candidates.csv`: PhysX 상위 2개

단위검증은 24조건과 geometry 8항목을 모두 통과했다.
