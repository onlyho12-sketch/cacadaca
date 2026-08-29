# 차량 검사 시스템 연동 — 150셀 BC 정책 결과 export

> 2026-08-28. 차량 검사 시스템 요구사항(6영역×5×5셀, 셀별 전후 품질 CSV) 대응 패키지.
> **정책은 PPO 가 아니라 BC(수제 dwell 정책 모방) 체크포인트다** — 모든 산출물에 BC 로 표기한다.

## 1. 구성 파일

| 파일 | 내용 |
|---|---|
| `export_vehicle_results.py` | 입력 CSV → BC 정책 + 논문 기반 물리모델 실행 → 결과 CSV + 요약 JSON |
| `make_example_input.py` | 6영역×25셀 = 150셀 입력 예제 생성기 |
| `vehicle_150_cells.csv` | 입력 CSV 예제 (검사 시스템이 공급할 스키마) |
| `rl_vehicle_150_results.csv` | 150셀 전부 실행한 최종 출력 CSV |
| `rl_vehicle_150_results_summary.json` | 셀별 실패 원인 + 영역/자세별 집계 |
| 체크포인트 | `../rl/champion/model_bc.pt` (BC 챔피언 — RESULTS.md 6장) |

## 2. 실행 명령

```bash
cd <repo>
# 입력 예제 재생성 (선택)
~/isaacsim/python.sh learning/vehicle_export/make_example_input.py

# 본 실행 — Isaac Sim 앱 불필요 (torch+rsl_rl+numpy). 150셀 ≈ 2~5분 (8워커)
~/isaacsim/python.sh learning/vehicle_export/export_vehicle_results.py \
    --checkpoint learning/rl/champion/model_bc.pt \
    --input  learning/vehicle_export/vehicle_150_cells.csv \
    --output learning/vehicle_export/rl_vehicle_150_results.csv \
    --workers 8
# 스모크: --limit 5
```

## 3. 입력 CSV 스키마 (검사 시스템 → 우리)

`region_id, region_name, cell_id, position_x_m/y/z, normal_x/y/z,
init_roughness_um, init_scratch_um, init_ra_um, init_rz_um,
init_clearcoat_um, init_gu_proxy, surface_seed(선택)`

- 셀 합성에 실제로 쓰는 값: **init_ra_um(정상부 거칠기), init_scratch_um(최대 스크래치 깊이),
  init_clearcoat_um(평균 두께), surface_seed(재현성)**.
- init_roughness/rz/gu 는 참고열 — twin 이 합성 표면에서 재계산한 값을 출력의 before 로 쓴다
  (전/후가 같은 정의로 비교되도록. 4장).

## 4. 출력 CSV — 전/후 보존

요구 스키마 그대로 + 추적열. 셀 하나 = 120×120 mm 대표 patch 하나의 에피소드.

- 전후 품질: `roughness_(before|after)`(=전역 Rq), `scratch_*_um`(기하학적 valley 최대깊이),
  `ra/rz_*_um`(전역, 스크래치 골 포함), `clearcoat_initial/removed/remaining(_min)_um`,
  `gu_proxy_(before|after)`.
- 공정: `force_n`(달성힘 평균 — 명령 아님), `rpm`, `feed_speed_mm_s`(적용 평균),
  `step_over_ratio`, `pass_count`(recipe), `policy_action_force/feed`(잔차 평균, [-1,1]).
- Isaac Sim 시각화: before 열로 초기 상태맵, after 열로 결과 상태맵을 채색하면 된다.
  (셀 내부 분포 지도가 필요하면 `learning/polytwin/export_cell_dataset.py` 참고 — 셀내 격자 단위.)

## 5. 판정 열 — literature-derived project target

이 프로젝트는 실제 폴리싱 실험 없이, 제공된 자동차 도장·연마 논문의 실험 결과를 근거로
디지털 트윈을 구현한다. 아래 4종 기준은 그 문헌을 바탕으로 프로젝트가 채택한
**문헌 기반 목표값(논문 기반 디지털 트윈 판정값)**이며, **프로젝트 자체 실측·보정값이 아니다.**
결과가 기준에 미달하면 목표값을 낮추지 않고 그대로 실패로 판정한다.

| 열 | 기준 | 성격 |
|---|---|---|
| `gu_target_pass` | 20° GU proxy after ≥ **70** | literature-derived project target |
| `ra_target_pass` | Ra after ≤ **0.20 μm** | literature-derived project target |
| `rz_target_pass` | Rz after ≤ **2.0 μm** | literature-derived project target |
| `clearcoat_safe` | 잔여 최소 ≥ **35 μm** | literature-derived project target (검사 시스템과 통일) |
| `scratch_improved` | after < before (before<0.05 면 자동 통과) | 파생 판정 |
| `overall_pass` | 위 전부 + 완주 + 힘한계(14 N) 미위반 | |
| `failure_reason` | 실패 항목을 `;` 로 연결 (수치 포함) | |

### Clearcoat 35 μm 기준과 GU 스케일 (중요)

GU proxy 의 clearcoat 품질항이 이 상수를 쓰므로 **GU 스케일이 전체적으로 ~0.7 내려갔다**
(행동 동일한 baseline 68.18→67.45). 30 기준 시절 수치와 직접 비교 금지. 35 기준 공식:
baseline GU 67.45/scratch 1.073 μm vs **BC 67.77/0.448 μm (Δscratch −58%)**.
이 스케일에서 "GU ≥ 70" 은 다수 셀이 미달하지만, **목표값은 낮추지 않고 그대로 실패로
판정한다** (프로젝트 방침). 실측 보정은 Gate 7 의 일이다.

## 6. 평면 학습의 한계 (요구 7)

- 이 정책은 **120×120 mm 평면 patch 에서만 학습**됐다. 전 셀의 `evaluation_mode` =
  `flat_trained_bc_policy_inference` — **곡면 재학습·검증 결과가 아니다.**
- 수직면(도어/펜더, tilt>45°)은 원 시뮬의 side 접촉 상수(soft limit 6 N)로 물리만 바꿔 추론한다.
  요약 JSON 의 top/side 구분은 **같은 평면 학습 정책의 자세별 결과**이지 곡면 일반화 증거가 아니다.
- 곡면 일반화 완료 판정은 Gate 4 (02 §6 표면 family, 평면/원통/구면/자유곡면 train/val/test 분리)
  수행 후에만 가능 — 현재 미수행 (RL_WORKLOG 7장 과제 3).

## 7. 정직성 경계

모든 품질 수치는 논문 근거 모델의 **SYNTHETIC** 출력이다. 실측 아님. 스크래치는 절차 생성한
합성 결함이며, 실전 결함맵은 μm급 검사 장비가 공급해야 한다 (RL_WORKLOG 5장).
