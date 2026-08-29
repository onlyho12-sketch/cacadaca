# Isaac Sim 20° Gloss 디지털 트윈 실험 기록

이 폴더는 자동차 Clearcoat의 결함과 폴리싱 전·후 광택 변화를 Isaac Sim에서
검사하기 위한 독립 실험 환경이다. 평면에서 시작해 해석 곡면, 실제 BMW Z4 USD
Mesh, RL 출력 연동, 여러 곡면 일반화 순서로 확장했다.

이 문서는 2026-08-28 기준 구현·실행·결과·근거 수준·재현 명령과 남은 작업을 정리한다.

## 1. 현재 상태

- 20 cm 평면 Roughness sweep과 5×5 공간 검사
- 단일 결함, Roughness Mask, Scratch/Swirl Normal Map
- 폴리싱 5단계와 전체 분산 결함 전·후 비교
- 표면 법선 기준 Local 20° 측정
- 원통·구면·완만/강한 자유곡면 실제 RTX 비교
- BMW Z4 보닛·루프·양쪽 도어·양쪽 앞 펜더 120×120 mm 5×5 검사
- 논문 앵커 기반 20° GU proxy
- RL 출력 CSV 검증과 차량 상태 맵 규격
- 셀별 Clearcoat 초기·제거·잔량 보존
- 합성 RL 상태를 BMW 재질에 연결한 전·후 검사
- 6종 표면 150점 일반화 벤치마크

| 일반화 표면 | Local 20° | 합성 상태 | 실제 RTX 전·후 |
|---|---:|---:|---:|
| 평면 | 완료 | 완료 | 완료 |
| 볼록 원통 | 완료 | 완료 | 완료 |
| 오목 원통 | 완료 | 완료 | 완료 |
| 볼록 구면 | 완료 | 완료 | 완료 |
| 완만한 복합곡면 | 완료 | 완료 | 완료 |
| 강한 복합곡면 | 완료 | 완료 | 완료 |

## 2. 광택값 구분

| 필드 | 의미 | 실제 GU인가 |
|---|---|---:|
| `hdr_roi_mean_intensity` | Tone mapping 전 RTX `HdrColor` ROI 평균 | 아니오 |
| `relative_gloss_*_not_gu` | 정상 기준점 대비 상대 반사 | 아니오 |
| `predicted_20deg_gu_literature_proxy` | 상대광택을 논문 앵커에 대응한 지표 | 아니오 |
| 20° GU | Gloss Meter·표준판으로 측정·보정한 값 | 현재 없음 |

실제 Gloss Meter 보정은 하지 않으며 현재 계획에도 없다. CSV에서 `not_gu`,
`proxy`, `actual_gloss_meter_calibrated=false`를 의도적으로 유지한다.

## 3. 실행 환경과 측정 조건

- 작업공간: `/home/rokey/cacadaca`
- Isaac Sim: `/home/rokey/isaacsim-6.0.1`
- Isaac Python: `/home/rokey/isaacsim-6.0.1/python.sh`
- 차량: `/home/rokey/cacadaca/scan_obj/car.usd`
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- 수치 측정: RTX Path Tracing
- 측정 후 GUI: RTX Real-Time 2.0

`config/gloss_config.py`의 주요 기본값:

```text
입사/검출각       Local Normal 기준 20° / 20°
광원·카메라 거리  0.42 m
패널              0.20 m
해상도            512×512
Path Tracing      128 spp
HDR ROI           영상 중심 폭 8%
반복              5회
Clearcoat IOR     1.5
```

GUI의 분홍 구, 주황 측정점, 배경과 overview 조명은 측정 완료 후 추가되므로 CSV에
영향을 주지 않는다. 숨은 측정 Render Product도 GUI 전환 전에 해제해 렉을 줄였다.

## 4. Local 20° 원리

각 측정점의 위치 `p`, 단위 법선 `n`, 경로 접선으로 Local
tangent/bitangent/normal 좌표계를 만든다. 광원과 카메라를 법선 양쪽 20°에 놓고
측정점을 순회한다. 한 카메라로 곡면 전체를 동시에 20°로 보는 방식이 아니다.

각 셀에서 다음을 검증한다.

1. 광원–법선 각도 20°
2. 카메라–법선 각도 20°
3. 정반사 방향과 카메라 방향 일치
4. 10 mm footprint 내부 법선 변화
5. HDR가 양수·유한값인지

차량이 특정 방위의 광원/카메라를 가려 HDR가 0이면 각도는 유지하고 `+Y`,
`-X`, `-Y` 접선 방위를 재시도한다. 사용 방위는 CSV에 기록한다.

## 5. 재질과 결함

- Clearcoat: `UsdPreviewSurface`의 weight, roughness, IOR
- Roughness 결함: 1024×1024 Mask와 부드러운 경계
- Scratch/Swirl: 기하를 파내지 않는 절차적 Normal Map
- Clearcoat 두께: μm 상태값으로 보존하며 실제 두께 Mesh가 아님

Clearcoat 합성 기본 범위는 40–50 μm이고 시드를 기록한다. 안전선 35 μm는 논문
직접값이 아닌 `PT-DESIGN`이다. 안전선 이상이면 정상 광학층을 유지하고, 미만이면
integrity를 감소시킨다. 단지 40 μm와 48 μm라는 이유만으로 광택을 다르게 만들지 않는다.

## 6. 문헌 근거와 설계값

참고 자료:

- `coatings-11-01320.pdf`, DOI `10.3390/coatings11111320`
- `The_Effect_of_Detergents_on_the_Appearance_of_Auto.pdf`
- `자동차_표면_연마_방법의_비교_분석.pdf`
- `목표_품질_데이터.pdf`

`white_automotive_literature_composite_v1`:

- `Ra=0.0805 μm`: 2016년식 흰색 Toyota 시편의 노출 전 두 평균 0.078/0.083
- `20° Gloss=88.8 GU`: 제조사가 특정되지 않은 백색 아크릴 상용차 전체 평균
- RGB, IOR, shader roughness: Isaac Sim 설계값

따라서 “Toyota 차량 평균 88.8 GU”로 해석하지 않는다. 70 GU도 보편 법규가 아니라
프로젝트 proxy 목표다. 근거 태그는 `L-DERIVED`, `PT-DESIGN`,
`synthetic_*_not_rl`로 구분한다.

GU proxy v1:

```python
GU_proxy = 25 + clip(relative_gloss, 0, 1) * (78 - 25)
```

차량/곡면 프로필 일부는 상단 앵커 88.8을 사용한다. 실제 공식은 각
`gu_proxy_summary.json`에 저장된다.

## 7. 실험 순서와 결과

### 7.1 평면 Roughness sweep

Clearcoat roughness `0.02, 0.05, 0.10, 0.20, 0.40`을 동일 20° 조건에서 측정했다.

```bash
cd /home/rokey/cacadaca/gloss_test
./run_test.sh
./run_test.sh --normal 1,0,0 --tag normal_x
./run_test.sh --normal 0,1,0 --tag normal_y
```

### 7.2 평면 5×5와 단일 결함

```bash
./run_test.sh --spatial-grid 5 --scan-roughness 0.10 --tag spatial_5x5
./run_test.sh --spatial-grid 5 --scan-roughness 0.10 \
  --defect-cell 4,4 --defect-roughness 0.30 --defect-size-m 0.030 \
  --tag defect_5x5 --no-headless --keep-open
```

분홍 구는 정반사 법칙에 맞춰 배치한 실제 장면 물체다. 결함 위치에서는 반사된 구가
흐려진다. 구 자체는 결함이나 측정 결과 마커가 아니다.

### 7.3 폴리싱 5단계

```text
roughness       0.30 → 0.23 → 0.16 → 0.12 → 0.10
normal strength 2.80 → 1.80 → 0.90 → 0.35 → 0.00
```

```bash
./run_polishing_progression.sh
```

이 단계는 몇 N이 몇 μm를 제거하는 물리 실험이 아니라, 광학 상태 개선을 검사기가
검출하는지 확인한 단계다.

### 7.4 전체 분산 결함

시드 `20260827`, 중앙 (3,3) 한 셀만 정상으로 두고 나머지 24셀을 다르게 만들었다.

```bash
./run_distributed_roughness_test.sh
```

```text
결함부 GU proxy mean  36.556 → 71.719
70 GU pass             0/24 → 18/24
```

### 7.5 곡면 기하와 RTX

```bash
./run_curved_local_20_validation.sh   # 자세만 검증, RTX/GU 아님
./run_curved_defect_comparison.sh
./run_cylinder_sphere_comparison.sh
```

| 표면 | 상대광택 Before | After | 개선 |
|---|---:|---:|---:|
| 완만한 자유곡면 | 0.2166 | 0.9308 | 24/24 |
| 볼록 원통 | 0.2311 | 0.9259 | 24/24 |
| 볼록 구면 | 0.2201 | 0.9318 | 24/24 |

구면은 구 전체가 아니라 반지름 500 mm의 200×200 mm 패치다.

### 7.6 BMW Z4 실제 Mesh

```bash
./run_vehicle_mesh_rtx_scan.sh
./run_vehicle_defect_comparison.sh
```

BMW 내장 DomeLight가 전용 측정광을 오염시켜 측정 전에 intensity를 0으로 override했다.
주황 구는 측정 완료 후 추가하는 25개 위치 마커다.

```text
결함부 relative mean  0.2652 → 0.9669
개선 셀             24/24
GU proxy mean       41.917 → 86.624
70 GU pass           0/24 → 24/24
```

### 7.6.1 BMW Z4 다중 영역 Local 20°와 실제 RTX

기존 보닛 전용 하향 raycast를 임의 검사 평면의 `axis_u`, `axis_v`,
`ray_direction`을 받는 방식으로 확장했다. 이에 따라 수평에 가까운 보닛·루프뿐 아니라
법선이 ±X에 가까운 도어·펜더 옆면도 각 점의 법선 기준 Local 20°로 측정한다.

검사 영역은 보닛, 루프, -X/+X 측 도어, -X/+X 측 앞 펜더 상부의 총 6개다. 각 영역은
120×120 mm, 5×5이며, 펜더는 바퀴 아치의 급격한 경계를 피하고 10 mm 측정 footprint가
안정적으로 놓이는 도장면을 사용했다.

```bash
# Local 20°·Mesh hit·10 mm footprint 기하 검증만 수행
./run_vehicle_multi_region_local_20.sh

# 기하 검증, 6개 영역 Path Tracing, 최종 집계를 모두 재실행
./run_vehicle_multi_region_benchmark.sh
```

```text
기하 Local 20° 영역       6/6 통과
Mesh 측정점               150/150
10 mm footprint           150/150
실제 RTX 영역             6/6 통과
양수·유한 HdrColor        150/150
```

| 영역 | HDR ROI mean | HDR min–max | 방위 재시도 | 판정 |
|---|---:|---:|---:|---:|
| 보닛 | 0.753179 | 0.204386–0.965089 | 1 | 통과 |
| 루프 | 0.469020 | 0.104700–0.860074 | 1 | 통과 |
| -X 측 도어 | 0.433899 | 0.095345–0.909158 | 1 | 통과 |
| +X 측 도어 | 0.430867 | 0.090979–0.910539 | 1 | 통과 |
| -X 측 앞 펜더 | 0.340721 | 0.072431–0.942836 | 1 | 통과 |
| +X 측 앞 펜더 | 0.344510 | 0.072692–0.939195 | 1 | 통과 |

모든 영역의 첫 셀은 초기 접선 방위에서 차량 형상에 가려 HDR가 0이었고, 입사·검출
20°는 그대로 유지하면서 `axis_v` 방위로 바꿔 양수값을 얻었다. 위 HDR 차이는 GU가
아니며, 영역별 절대 광택 우열로 해석하지 않는다. 현재는 동일 Clearcoat
roughness `0.10`의 단일 상태 검사이고 각 영역의 결함 폴리싱 전·후 비교는 아니다.

### 7.6.2 BMW Z4 다중 영역 검사 이동 시각화

7.6.1에서 실제 Path Tracing 검사가 끝난 6개 영역 150개 측정점을 그대로 불러와,
광원이 각 지점으로 이동하는 모습을 RTX Real-Time 2.0 화면에서 반복 재생한다.

```bash
cd /home/rokey/cacadaca/gloss_test
./run_vehicle_multi_region_visualization.sh
```

화면 표시는 다음과 같다.

- 노란 발광 구: 현재 측정점에 대응하는 이동 광원 위치
- 주황 작은 점: 현재 검사 중인 차체 표면 위치
- 차체의 국소적인 밝은 반사: 현재 광원이 비추는 영역
- 우측 상태 창: 영역, 5×5 셀, 전체 진행 순서, 반복 횟수

광원은 셀별 표면 법선과 접선 좌표계로 계산한 Local 20° 위치로 이동하며, 검사 순서는
보닛 → 루프 → 양쪽 도어 → 양쪽 앞 펜더다. 검출기 위치는 계산에 포함되지만 화면에는
표시하지 않는다. 고정 Key/Fill/Dome Light는 모두 끄고, 노란 광원 위치에서 함께 움직이는
80×80 mm RectLight가 현재 검사점 주변을 직접 비춘다. 노란 구와 주황 마커는 재생용이다.
이 플레이어는 이미 저장된 측정점을 시각화할 뿐 Path Tracing HDR,
상대광택, GU proxy 등의 기존 수치 결과를 다시 계산하거나 변경하지 않는다.

기본 재생은 창을 닫을 때까지 반복한다. 속도는 다음처럼 바꿀 수 있다.

```bash
./run_vehicle_multi_region_visualization.sh \
  --dwell-seconds 0.55 \
  --transition-seconds 0.25 \
  --region-pause-seconds 0.7
```

첫 프레임과 설정 메타데이터는
`results/vehicle_multi_region_visualization/`에, 전체 터미널 출력은 그 아래 `logs/`에
저장된다.

### 7.6.3 BMW Z4 다중 랜덤 시드 반복성 검사

한 가지 결함 배치에서만 결과가 좋아지는지 확인하기 위해 시드 `20260828`부터
`20260832`까지 5개 결함 배치를 만들었다. 각 시드는 6개 차량 영역 × 25셀 = 150셀이며,
총 750셀의 Roughness, Scratch, Ra/Rz, Clearcoat 두께, 상대광택, GU proxy 전·후 상태를
생성한다. 이 상태는 RL 결과나 실제 연마 측정값이 아니라 시드가 기록된 합성 설계값이다.

```bash
cd /home/rokey/cacadaca/gloss_test

# 5개 시드, 750셀 합성 상태·판정·대표 RTX 계획 생성
./run_vehicle_seed_repeatability.sh

# 첫 시드의 6영역 대표점 실제 Path Tracing 전·후 측정
./run_vehicle_seed_representative_rtx.sh 20260828
```

영역마다 최대 결함, 중간 결함, 정상 셀, 중앙 기준점을 선택한다. 중복되는 셀이 있으면
제거하므로 보통 영역당 4점이며, 중앙점은 RTX 상대값의 기준으로 항상 포함된다. 전체
150셀은 합성 판정을 수행하고 실제 RTX는 대표점만 수행함으로써 반복 시간을 제한한다.

5개 시드의 합성 판정 결과:

| 시드 | GU proxy 평균 Before → After | 70 proxy 통과 | Clearcoat 통과 | 폴리싱 후 최소 두께 |
|---:|---:|---:|---:|---:|
| 20260828 | 54.10 → 75.55 | 150/150 | 150/150 | 37.243 μm |
| 20260829 | 52.54 → 75.39 | 150/150 | 150/150 | 37.275 μm |
| 20260830 | 54.03 → 75.54 | 150/150 | 150/150 | 38.116 μm |
| 20260831 | 53.62 → 75.50 | 150/150 | 150/150 | 37.291 μm |
| 20260832 | 53.48 → 75.49 | 150/150 | 150/150 | 37.593 μm |

시드 `20260828`의 실제 RTX 대표 검사 결과:

```text
실제 RTX 영역 통과        6/6
결함 대표점 HDR 개선     18/18
양수·유한 HDR            24/24
합성 상태→RTX 재질 일치  24/24
```

`HdrColor`는 GU가 아니며, 위 70 기준은 실측 GU가 아니라 문헌 기반 GU proxy다. 실제 RTX
결과도 5개 시드 전체가 아니라 첫 시드의 대표점 증거다. 합성 750셀 성공과 실제 대표점
성공은 별도 필드로 저장하며 서로 대신하지 않는다.

### 7.7 RL 출력 인터페이스

필수 범주: 셀/위치/법선, Force/RPM/Feed/step-over/pass, Roughness/Scratch,
Ra/Rz, Clearcoat 초기·제거·잔량, 명시된 광학 출력.

검증기는 필수 열, 숫자, 단위법선, [0,1] 범위, Clearcoat 물질수지를 확인한다.
Force/RPM/Feed만 보고 GU를 임의 생성하지 않는다.

```bash
./run_rl_result_evaluation.sh /경로/rl_polishing_output.csv results/rl_received_run_001
```

RL 담당자에게 전달할 차량 6영역 150셀 초기 입력 생성:

```bash
./run_vehicle_rl_150_input_export.sh
```

입력은 `synthetic_initial_state_not_rl`로 기록한다. 담당자가 반환한 150셀 결과는 다음과 같이
검증하고 렌더 상태로 변환한다.

```bash
./run_vehicle_rl_150_adapter.sh validate \
  /받은/경로/rl_vehicle_150_results.csv
./run_vehicle_rl_150_state_prepare.sh
```

계약은
[`docs/digital_twin/08_RL_출력_GU연동규격.md`](docs/digital_twin/08_RL_출력_GU연동규격.md)에 있다.

### 7.8 150셀 RL 상태를 BMW에 연결

```bash
./run_vehicle_rl_150_visualization.sh
```

이 시각화는 검증된 RL 결과 CSV가 준비된 후 사용한다. 합성 상태로 실행한 결과는 인터페이스
검증일 뿐 RL 정책 성능으로 판정하지 않는다.

### 7.9 6종 표면 일반화

```bash
./run_surface_generalization_benchmark.sh
```

시드 `20260828`로 6개 표면 150셀에 Roughness, Scratch, Clearcoat를 생성했다.

```text
Local 20°       6/6
10 mm footprint 150/150
합성 GU/safety  150/150
실제 RTX        6/6
```

실제 RTX까지 추가한 최종 결과는 다음과 같다.

```text
실제 RTX 수행/통과  6/6
평면 HDR 개선       25/25
오목 원통 HDR 개선  25/25
```

평면 실제 전·후 비교:

```text
결함부 relative mean  0.252152 → 0.903609
GU proxy mean          41.087 → 82.650
70 GU pass              0/24 → 24/24
```

오목 원통 실제 전·후 비교:

```text
결함부 relative mean  0.316618 → 0.936572
GU proxy mean          45.200 → 84.753
70 GU pass              0/24 → 24/24
```

강한 복합곡면 실제 결과:

```text
HDR mean            0.240111 → 0.708476
HDR 개선            25/25
결함 상대광택       0.309171 → 0.968259
GU proxy mean       44.725 → 86.682
70 GU pass          24/24
```

최종 보고서에서 기하 Local 20°는 6/6, 실제 RTX 수행 및 통과도 6/6이다. 여기서
RTX 값은 `HdrColor` 반사 측정이고, GU 값은 실측이 아닌 문헌 기반 proxy다.

## 8. GUI 명령

```bash
# 평면
./run_test.sh --no-headless --keep-open

# 강한 곡면
./run_curved_rtx_scan.sh --surface-profile freeform_strong \
  --distributed-roughness improved --roughness-seed 20260828 \
  --tag freeform_strong_view --no-headless --keep-open

# BMW
./run_vehicle_mesh_rtx_scan.sh --distributed-roughness improved \
  --tag vehicle_improved_view --no-headless --keep-open

# BMW 6영역 150점 이동 광원 시각화
./run_vehicle_multi_region_visualization.sh
```

25점 Path Tracing이 먼저 끝난 뒤 GUI로 전환된다.

## 9. 결과·로그 구조

```text
results/<tag>/
├── logs/                 전체 터미널 로그
├── images/               PNG preview
├── raw/                  HDR/preview NumPy
├── assets/               Roughness/Scratch/Clearcoat texture
├── *_results.csv         셀별 값
├── *_summary.json        판정·메타데이터
├── *_heatmap.png         공간 분포
└── *_scene.usda          USD 장면
```

주요 종합 폴더:

- `distributed_roughness_comparison`
- `curved_distributed_comparison`
- `cylinder_white_clearcoat_comparison`
- `sphere_white_clearcoat_comparison`
- `vehicle_hood_defect_comparison`
- `vehicle_rl_state_comparison`
- `generalization_freeform_strong_comparison`
- `generalization_plane_comparison`
- `generalization_concave_cylinder_comparison`
- `surface_generalization_benchmark`
- `vehicle_multi_region_local_20`
- `vehicle_multi_region_benchmark`
- `vehicle_multi_region_visualization`
- `vehicle_seed_repeatability`

## 10. 코드 구조와 정리 결과

- `run_gloss_sweep.py`: 평면 측정
- `run_curved_rtx_scan.py`: 해석 곡면 RTX
- `run_vehicle_mesh_rtx_scan.py`: 실제 USD Mesh RTX
- `gloss_geometry.py`: Local 20°
- `reflection_measurement.py`: HDR ROI
- `distributed_roughness.py`: 분산 결함
- `masked_defect_material.py`: Clearcoat texture 연결
- `mesh_surface_sampling.py`: 차량 raycast/법선
- `vehicle_region_profiles.py`: BMW 다중 영역 검사 평면·방향
- `validate_vehicle_multi_region_local_20.py`: 6영역 기하·footprint 검증
- `aggregate_vehicle_multi_region_results.py`: 6영역 RTX 결과 통합
- `play_vehicle_multi_region_inspection.py`: 150점 이동 광원 GUI 재생
- `vehicle_seed_repeatability.py`: 5시드 750셀 합성 반복성 상태·판정
- `measurement_cell_selection.py`: 대표 RTX 부분 셀 선택
- `aggregate_vehicle_seed_representative_rtx.py`: 대표 실제 RTX 전·후 집계
- `gu_proxy.py`: GU proxy와 근거 메타데이터
- `evaluate_rl_output.py`: RL CSV 검증
- `vehicle_state_textures.py`: 차량 재질 texture
- `surface_generalization_benchmark.py`: 6종 일반화
- `aggregate_curved_rtx_evidence.py`: 실제 RTX 증거 통합
- `export_vehicle_rl_input_150.py`: 6영역 150셀 RL 초기 입력·기하 검증

평면·곡면·차량 비교기는 CSV와 판정 기준이 달라 단순 중복이 아니다. 2026-08-28
Python/Shell 참조를 전수 조사했고 삭제 가능한 미사용 소스는 없었다.
`__pycache__` 3개와 `.pyc` 59개만 시스템 휴지통으로 이동했다.
자동 테스트 중 재생성된 캐시도 검증 종료 후 다시 휴지통으로 이동했으며, 현재 프로젝트
안의 `__pycache__`와 `.pyc`는 0개다.

## 11. RL 담당자 전달용 6영역 150셀 초기 입력

현재 차량 Local 20° 검사 기하와 합성 시드 초기 상태를 결합해 RL 추론 입력 CSV를 만든다.
기본 시드는 `20260828`이며, 이 파일에는 폴리싱 전 상태만 있다. Force/Feed 행동과 폴리싱 후
상태는 RL 담당자가 계산해서 별도의 결과 CSV로 돌려줘야 한다.

```bash
cd /home/rokey/cacadaca/gloss_test
./run_vehicle_rl_150_input_export.sh
```

다른 합성 시드나 출력 경로를 선택할 수도 있다.

```bash
./run_vehicle_rl_150_input_export.sh 20260829 \
  results/rl_vehicle_150_input/vehicle_150_cells_seed_20260829.csv
```

기본 출력:

```text
results/rl_vehicle_150_input/vehicle_150_cells.csv
results/rl_vehicle_150_input/vehicle_150_cells.summary.json
results/rl_vehicle_150_input/vehicle_150_cells.report.txt
```

CSV에는 6영역 × 25셀의 ID, 위치, 월드 법선, Roughness/Scratch/Ra/Rz, 초기
Clearcoat 두께, 상대광택, GU proxy가 들어간다. 위치·법선은 Local 20° 기준 기하와 자동
대조한다. `data_origin=synthetic_initial_state_not_rl`이므로 실제 RL 결과나 실측값으로
표현하면 안 된다.

RL 담당자는 이 입력을 읽고 기존 출력 계약에 맞춰 Force/RPM/Feed/Step-over/Pass와 전·후
품질을 채운 `rl_vehicle_150_results.csv`를 반환해야 한다. 받은 결과는 다음 명령으로 검증한다.

```bash
./run_vehicle_rl_150_adapter.sh validate \
  /받은/경로/rl_vehicle_150_results.csv
```

### 11.1 전달받은 학습 정책의 차량 150셀 연결

2026-08-29에 전달받은 학습 결과를 `learning/`에 보존하고 다음 두 체크포인트를 동일한 BMW
150셀에 적용했다.

- `learning/rl/champion/model_bc.pt`: BC 챔피언
- `learning/rl/champion/model_terminal_ppo_it400.pt`: 종말보상 PPO 평가 후보

전체 추론·변환·검증·렌더 상태 생성:

```bash
cd /home/rokey/cacadaca
./gloss_test/run_learning_rl_vehicle_150.sh
```

BC 폴리싱 전·후 이동 검사광 시각화:

```bash
VEHICLE_RL_150_CELLS=/home/rokey/cacadaca/gloss_test/results/learning_rl_vehicle_150/bc_champion/render_states/vehicle_rl_150_render_cells.csv \
  ./gloss_test/run_vehicle_rl_150_visualization.sh --phase cycle --loop
```

최종 동시 통과는 BC `14/150`, 종말보상 PPO `11/150`이었다. BC의 평균 GU proxy는
`62.946`, PPO는 `60.987`이므로 현재 챔피언은 BC로 유지한다. 두 정책 모두 차량 좌표·법선
오차 0, Clearcoat 물질수지 150/150을 통과했다. 화면용 Clearcoat는 패치 평균이 아니라
안전 판정용 최소 잔량을 사용한다.

상세 결과는
[`results/learning_rl_vehicle_150/FINAL_EVALUATION.md`](results/learning_rl_vehicle_150/FINAL_EVALUATION.md)에
기록했다.

## 12. 자동 테스트

```bash
cd /home/rokey/cacadaca
python3 -m unittest discover -s gloss_test/tests -p 'test_*.py'
```

2026-08-28 기준 자동 테스트는 재질 근거, Local 20°, footprint, Mesh
sampling, GU proxy, RL 계약/물질수지, 차량 texture, Clearcoat safety, 시드 재현성을
검사한다.

## 13. 제한과 남은 순서

- GU는 실측이 아닌 문헌 proxy다.
- Force/RPM/Feed → 제거량·Ra/Rz·GU 변화는 전달받은 문헌 기반 합성 디지털 트윈 모델이다.
- BC와 종말보상 PPO 체크포인트를 BMW 대표 6영역 150셀에 연결했지만 차량 전체 검사는 아니다.
- 정책은 120×120 mm 평면 patch 학습본이므로 곡면 일반화 학습 완료를 주장할 수 없다.
- 현재 최고 성능인 BC도 프로젝트 동시 기준 통과가 14/150이므로 품질 목표를 달성하지 못했다.
- 합성 반복성은 5개 시드지만 실제 RTX 반복성은 첫 시드 대표점만 확인했다.

남은 순서:

1. GU와 최종 품질이 같은 방향으로 증가하도록 PPO 보상함수 또는 정책 개선
2. 같은 고정 BMW 150셀에서 개선 정책 재평가
3. 미학습 곡면·직선 표면 일반화 평가
4. 이후 차량 전체 도장면 검사점 자동 생성으로 범위 확장
