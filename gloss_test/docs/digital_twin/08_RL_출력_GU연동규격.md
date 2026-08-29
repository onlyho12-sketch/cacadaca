# 08. Isaac Lab 출력 → GU 평가기 연동 규격

## 1. 목적

강화학습 담당자의 Isaac Lab 결과를 PolyTwin GU 평가기에 연결하기 위한 셀 단위 CSV 계약이다.
현재 제공되는 예제는 연동 시험용 합성 데이터이며 PPO 학습결과가 아니다.

## 2. 계산 경계

```text
Force/RPM/Feed/Step-over/Pass
  → Isaac Lab 폴리싱 물리모델
  → 제거량·Clearcoat·Ra/Rz·Roughness·Scratch
  → Isaac Sim 광학모델
  → relative_gloss_before/after_not_gu
  → 이 저장소의 GU proxy 평가기
  → predicted 20° GU proxy와 70 GU 판정
```

Force/RPM/Feed를 GU 식에 직접 넣지 않는다. 물리모델 출력만 있고 광학 출력이 없으면 GU 계산을
중단한다. `relative_gloss_*_not_gu`는 정상 기준 정반사 응답을 1.0으로 정규화한 광학 출력이다.

## 3. 필수 열

| 분류 | 필수 열 |
|---|---|
| 출처 | `data_origin` |
| 식별 | `episode_id`, `cell_id`, `grid_row`, `grid_column` |
| 위치·법선 | `position_x_m`, `position_y_m`, `position_z_m`, `normal_x`, `normal_y`, `normal_z` |
| 공정 | `force_n`, `rpm`, `feed_mm_s`, `step_over_ratio`, `pass_count` |
| 표면 | `roughness_before/after`, `scratch_before/after`, `ra_before_um`, `ra_after_um`, `rz_before_um`, `rz_after_um` |
| 도막 | `clearcoat_before_um`, `clearcoat_after_um`, `clearcoat_removed_um` |
| 광학 | `relative_gloss_before_not_gu`, `relative_gloss_after_not_gu` |

`step_over_ratio`는 고정 패드 지름에 대한 경로 간격 비율이며 범위는 0~1이다. 좌표 단위는 m,
Feed는 mm/s, 도막·Ra·Rz는 μm이다. 법선은 월드 좌표계의 단위벡터다.

## 4. 자동 검증

- `(episode_id, cell_id)` 중복 금지
- 좌표·공정·표면·광학값은 유한한 수
- 법선 길이는 `1 ± 0.001`
- 공정·표면값은 음수 금지, `step_over_ratio`는 0~1
- 상대 광택은 0~1
- `clearcoat_after_um ≤ clearcoat_before_um`
- `before - after = removed` 오차 0.001 μm 이하
- 광학 열이 없으면 GU proxy를 임의 추정하지 않고 실패

## 5. 담당자 출력 예

형식 예시는 `examples/rl_polishing_output_example.csv`에 있다. 이 파일의
`data_origin=synthetic_interface_example_not_rl`은 실제 RL 결과가 아니라는 표시다. 실제 결과는
예를 들어 `isaac_lab_ppo_validation_seed_42`처럼 출처와 validation seed를 식별할 수 있게 기록한다.

## 6. 실행

```bash
cd /home/rokey/cacadaca/gloss_test
./run_rl_result_evaluation.sh /받은/경로/rl_polishing_output.csv \
  results/rl_received_run_001
```

인수 없이 실행하면 합성 예제로 연동만 검사한다.

```bash
./run_rl_result_evaluation.sh
```

출력:

```text
rl_cells_with_gu_proxy.csv
rl_gu_proxy_summary.json
```

`rl_cells_with_gu_proxy.csv`는 원본 열을 보존하면서 전·후 GU proxy, 개선량, 70 GU 통과 여부를
추가한다. JSON은 episode별 평균 GU proxy, 통과 셀 수·비율, 최소 잔여 Clearcoat를 기록한다.

## 7. 인수 조건

실제 RL 결과를 받았다고 판정하려면 `data_origin`이 합성 예제 표기가 아니어야 하고, 자동 검증을
통과해야 한다. 평가기가 실행된 사실만으로 폴리싱 물리모델이나 PPO 학습이 완료됐다고 판정하지
않는다.

## 8. 차량 6영역 150셀 입력과 상태 맵 준비

Local 20°로 검증한 BMW 6영역의 위치·법선과 합성 초기 상태를 결합해 RL 담당자 전달용
150셀 입력을 만든다.

```bash
./run_vehicle_rl_150_input_export.sh
```

기본 입력은 6영역 × 25셀이고 `data_origin=synthetic_initial_state_not_rl`로 기록된다. 각 셀의
초기 Clearcoat 두께는 그대로 보존하며, 사용한 seed와 출처를 CSV·JSON에 남긴다. 문헌 기반
합성 초기분포는 `uniform(40, 50) μm`이며 실제 차량 측정분포가 아니다.

RL 담당자가 전·후 품질과 행동값을 채운 150셀 결과를 반환하면 다음 순서로 검증하고 렌더
상태로 변환한다.

```bash
./run_vehicle_rl_150_adapter.sh validate \
  /받은/경로/rl_vehicle_150_results.csv
./run_vehicle_rl_150_state_prepare.sh
```

출력:

```text
rl_vehicle_150_cells_normalized.csv
rl_vehicle_150_validation.json
rl_vehicle_150_validation.txt
vehicle_rl_150_render_cells.csv
vehicle_rl_150_state_summary.json
vehicle_rl_150_state_report.txt
states/<region_id>_rl_state_maps.npz
```

기본 Clearcoat 안전한계 `35 μm`는 `PT-DESIGN`이며 CLI에서 변경할 수 있다. 안전한계 미달 또는
70 GU proxy 미달 셀도 삭제하지 않고 실패 마스크로 저장해 이후 차량 시각화에서 표시한다.
