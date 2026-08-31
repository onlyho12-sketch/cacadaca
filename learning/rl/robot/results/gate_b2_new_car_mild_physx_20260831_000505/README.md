# Gate B2 — legacy_stress 보존과 new_car_mild 분리

## 판정

Gate B2 완료조건을 모두 통과했다.

- `legacy_stress`: 기존 `make_flat_patch()`와 **256/256 seed에서 모든 배열·scalar exact match**
- `new_car_mild`: 256/256 seed 유한값, 선언 범위 통과
- base micro 높이, 개별 scratch 깊이 지도, mar 광학 sidecar, cumulative removal/clearcoat를 분리
- mar density/severity는 `micro_height_um`에 합치지 않으며 기존 Ra 계산값을 바꾸지 않음
- `new_car_mild` 4환경×1 pass PhysX: **품질 4/4, 안전 4/4, fallback 0**
- 기존 Gate A/B와 과거 결과는 재실행·덮어쓰기하지 않음

`new_car_mild`의 scratch 깊이·개수·혼합비와 mar 분포/광학 전달식은 실측 결합분포가 없는
**PT-DESIGN**이다. 실제 신차 모집단을 측정한 결과로 표현하면 안 된다.

## 구현 분리

- `learning/polytwin/surface_profiles.py`
  - 기존 생성기 무수정 `legacy_stress` adapter
  - 별도 `new_car_mild` generator
  - base micro, individual scratch, mar sidecar 분리
- `learning/polytwin/mar_optics.py`
  - sub-grid mar density/severity와 removal에 따른 광학 감쇠
  - height/Ra 비결합
- `learning/rl/env/profiled_planar_roi_diagnostics.py`
  - profile GU/mar와 기존 Ra/removal waviness/coverage의 동시 출력
- 신규 profiled Robot cfg/env 및 전용 평가기

기존 `surface_state.py::make_flat_patch()`, 공용 `polish_env.py/cfg.py`, Gate B 환경과 v5 파일은
수정하지 않았다.

## 256-seed 초기 분포

최종 검증 폴더:
`learning/rl/robot/results/gate_b2_initial_profiles_20260831_001029/`

| 지표 | legacy_stress 평균 | new_car_mild 평균 |
|---|---:|---:|
| base micro-Ra (μm) | 0.07963 | 0.04912 |
| scratch 포함 전체 Ra (μm) | 0.12975 | 0.04931 |
| 전체 Rz (μm) | 2.10150 | 0.43597 |
| 개별 scratch 최대깊이 (μm) | 1.76967 | 0.12142 |
| 생성 scratch 수 | 기존 상태에 미보관, 설계 4–12 | 0.78125, 설계 0–3 |
| mar density | 0 | 0.06482 |
| mar severity | 0 | 0.09824 |
| q_mar | 1 | 0.95037 |
| 초기 GU proxy | 54.33 | 85.27 |
| clearcoat min (μm) | 41.58 | 41.55 |

`new_car_mild` scratch 수의 0/1/2/3 빈도는 118/89/36/13이다. 초기 tile 최저 GU 분포가
넓은 것은 개별 scratch가 있는 tile의 기존 `q_scratch`가 낮아지기 때문이며, mar를 굵은
groove로 만든 결과가 아니다.

검증 결과:

- `legacy_exact_all=true`
- `all_finite=true`
- `all_declared_bounds_ok=true`
- 단위검증: legacy 16-seed exact 포함 23개 PASS

## new_car_mild 4환경×1 pass PhysX

고정 조건은 Gate B와 같은 360 mm 물리평판, 320 mm 연속맵, 중앙 200 mm ROI, Ø110 mm pad,
step-over 20.24 mm, 10 raster line, 4.0 m 경로다. 기존 14차원 champion과 12.7 mm/s 기준을
그대로 사용했고 BC/PPO 학습은 하지 않았다.

| 지표 | 초기 평균 | pass 1 평균 | 변화 |
|---|---:|---:|---:|
| profile GU | 84.897 | 87.341 | +2.444 |
| 전체 Ra (μm) | 0.04808 | 0.05074 | +0.00265 |
| 전체 Rz (μm) | 0.39130 | 0.37134 | −0.01996 |
| fine Ra (μm) | 0.03833 | 0.01504 | **−0.02329** |
| fine Rz (μm) | 0.32765 | 0.18201 | **−0.14565** |
| mar remaining severity | 0.06851 | 0.02093 | **−0.04758** |
| q_mar | 0.96446 | 0.98901 | **+0.02456** |
| removal 평균 (μm) | 0 | 0.30885 | +0.30885 |
| removal waviness Ra (μm) | 0 | 0.04917 | 별도 생성 |
| removal waviness std (μm) | 0 | 0.06206 | 별도 생성 |
| 중앙−가장자리 제거량 (μm) | 0 | 0.09672 | 별도 생성 |
| coverage | 0 | 99.41% | +99.41%p |
| scratch max (μm) | 0.13207 | 0.06955 | −0.06251 |
| clearcoat min (μm) | 42.180 | 41.872 | −0.308 |

평균 force used 5.168 N, 최대 force 평균 6.903 N, feed 15.788 mm/s, peak temperature
26.62 °C, sensor fallback 0이다. 네 표면 모두 최종 품질·안전 판정을 통과했다. 단, n=4의
구조 점검 결과이므로 legacy 대비 우월성이나 새 champion 성능을 뜻하지 않는다.

### 평균 5×5 removal 지도 (μm)

```text
0.168 0.277 0.305 0.303 0.250
0.233 0.341 0.375 0.380 0.321
0.270 0.352 0.376 0.386 0.329
0.276 0.359 0.369 0.370 0.316
0.221 0.285 0.308 0.299 0.253
```

### 평균 5×5 coverage

```text
0.860 1.000 1.000 1.000 1.000
0.994 1.000 1.000 1.000 1.000
1.000 1.000 1.000 1.000 1.000
1.000 1.000 1.000 1.000 1.000
0.998 1.000 1.000 1.000 1.000
```

전체 env×tile 최저 coverage는 82%이며 중앙부 제거가 더 크다. 따라서 mild 표면에서도
Gate B와 같은 geometry/path 불균일 신호가 남고, Gate C에서 step-over 후보를 geometry와
coverage로 먼저 줄여야 한다.

## 산출물과 실패/중간 폴더

최종 초기분포:

- `gate_b2_initial_profiles_20260831_001029/initial_profile_distributions.csv`
- `gate_b2_initial_profiles_20260831_001029/initial_profile_summary.json`
- `gate_b2_initial_profiles_20260831_001029/parameter_provenance.json`

최종 PhysX:

- `initial_diagnostics.csv`: 4행
- `profile_sequences.csv`: 4행
- `profile_passes.csv`: 4행
- `profile_tiles.csv`: 초기 100 + pass 1 100 = 200행
- `metadata.json`
- `artifacts_sha256.txt`

보존하되 최종 근거로 사용하지 않는 폴더:

- `gate_b2_initial_profiles_20260831_000016/`: GU 평가 후 cached residual을 비교한 검증순서 오류
- `gate_b2_initial_profiles_20260831_000046/`, `...000158/`: 통과했으나 CSV sentinel/범위 자동판정 개선 전
- `gate_b2_profiled_smoke_20260831_000438/`: 의도적으로 20 step만 수행한 기동 점검

## 다음 Gate

Gate C에서는 BC/PPO를 학습하지 않는다. 먼저 analytic geometry/coverage로 step-over와
same/cross-direction 후보를 줄인 뒤 상위 1~2개만 두 profile 각각 같은 seed 16표면으로
paired PhysX 비교한다. Gate B2 결과 설명과 사용자 승인 전에는 시작하지 않는다.
