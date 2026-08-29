# PolyTwin 강화학습 파라미터 논문 근거 검증 및 전달안

> **구버전/논문 추적 참고용:** 이 문서는 실제 실험 재보정을 전제로 작성됐다.
> 현재 구현기준은 [`digital_twin/README.md`](digital_twin/README.md)이며, 논문값과 PolyTwin
> 설계식을 이용한 Isaac Lab 내부 synthetic 디지털 트윈을 최종모델로 사용한다.

## 1. 문서 목적

이 문서는 PolyTwin의 자동차 Clearcoat 폴리싱 강화학습 환경에 사용할 초기 상태,
공정 파라미터, 품질 목표 및 안전 제한을 정리하고, 각 값이 다음 중 어디에 해당하는지
구분하기 위한 전달 문서다.

- **논문 직접값**: 특정 논문의 실험에서 실제 사용하거나 측정한 값
- **논문 기반 확장값**: 논문 직접값을 중심으로 강화학습 탐색을 위해 넓힌 범위
- **PolyTwin 설계값**: 논문에 공식 기준으로 제시된 값이 아니라 프로젝트 내부에서 정한
  학습 성공조건 또는 안전 제한

논문 직접값과 PolyTwin 설계값을 구분하는 이유는, 논문 대부분이 서로 다른 재료·공구·패드·
연마재·장비를 사용하며 한 번의 실험 조건이 전체 자동차 Clearcoat 공정의 안전범위나
최적범위를 의미하지 않기 때문이다. 논문 직접값은 기본 조건과 기준 정책으로 사용하고,
PolyTwin 설계값은 강화학습의 탐색·허용·안전 범위를 정의하는 데 사용한다.

---

## 2. 패드 크기 고정 전제

PolyTwin의 모든 학습, 검증 및 비교 실험에서는 **동일한 패드 직경과 동일한 접촉면 형상**을
사용한다. 따라서 PolyTwin 내부에서는 Contact Force를 동일 패드 조건끼리 직접 비교할 수 있다.

단, 논문과 PolyTwin 사이에는 패드 재질, 강성, 연마재 및 실제 접촉면적이 다를 수 있으므로
논문의 Force를 그대로 자동차 Clearcoat의 안전범위라고 간주하지 않는다.

다음 값은 환경 설정과 모든 실행 로그에 반드시 고정·기록한다.

```text
Pad Diameter        : 프로젝트 확정값 1개로 고정
Pad Material        : 프로젝트 확정 재질로 고정
Pad Contact Geometry: 동일 형상으로 고정
Abrasive/Compound   : 학습 단계별 사용 제품 또는 모델 명시
```

동일 패드를 사용하더라도 곡면과 Tool Angle에 따라 실제 접촉면적이 달라질 수 있으므로 다음
파생값도 관측 또는 로그에 포함하는 것이 바람직하다.

```python
contact_pressure = contact_force / actual_contact_area
```

---

## 3. 최종 검증표

| 항목 | PolyTwin 초기값 | 근거 구분 | 검증 결과 |
|---|---:|---|---|
| Initial Ra | 0.30~0.70 μm | 논문 기반 확장값 | 수치는 확인되지만 황동 시편이며 자동차 Clearcoat 직접범위가 아님 |
| Initial Rz | 1.2~4.2 μm | PolyTwin 임시 설계값 | `Ra×4~6` 경험식에서 파생, 최종적으로 Height Map에서 직접 계산 필요 |
| Contact Force | 5~20 N | 논문 기반 확장값 | 금형강·항공기 프라이머에서 확인, 자동차 Clearcoat 직접 안전범위는 아님 |
| Feed Speed | 1~8 mm/s | 논문 기반 확장값 | 논문의 50~500 mm/min을 단위 변환한 범위 |
| RPM | 3000~6000 rpm | 논문 기반 확장값 | 자동차 직접 실험값은 5000 rpm, 주변 범위는 RL 탐색용 |
| Step-over | 패드 직경의 30~50% | PolyTwin 설계값 | 경로 간격의 중요성은 논문 근거가 있으나 숫자 자체는 직접값이 아님 |
| Clearcoat Thickness | 40~50 μm | 논문 직접값 | 자동차 Clearcoat 대표 두께로 확인됨 |
| Scratch Depth | 0.05~2 μm | 논문 직접값 | 자동차 Clearcoat 스크래치 범위로 확인됨 |
| Clearcoat 목표 제거량 | 1~7 μm | PolyTwin 설계값 | 손상 깊이와 전체 두께를 참고한 프로젝트 목표 |
| Clearcoat 절대 상한 | 10 μm | PolyTwin 설계값 | 공식 안전기준이 아닌 보수적 시뮬레이션 제한 |
| Removal Map MAE | ≤2 μm | PolyTwin 설계값 | 관련 폴리싱 연구의 μm급 잔차를 참고한 초기 기준 |
| 최종 Ra 성공범위 | 0.070~0.100 μm | 논문 기반 확장값 | Toyota 시편 초기 Ra와 분산을 참고해 확장한 범위 |
| 최종 Ra 최고보상 | 0.075~0.095 μm | PolyTwin 설계값 | 약 0.08 μm 중심의 보상구간 |
| 20° Gloss | 추후 추가 | 검증 진행 중 | Isaac Sim 시험 및 실제 GU 보정 통과 후 추가 |

---

## 4. Initial Ra 검증

### 논문에서 확인된 내용

Denkena et al.은 황동 시편을 일관된 초기 상태로 만들기 위해 평면 시편을
`Ra=0.3~0.4 μm`로 준비했다. 검증 단계에서는 초기 Ra가 약 `0.7 μm`인 구간도 다뤘으며,
Feed Rate를 조절하고 반복 폴리싱하여 목표 Ra에 접근했다.

### PolyTwin 적용

```text
Initial Ra = Uniform(0.30, 0.70) μm
```

이 범위는 자동차 Clearcoat 실측 분포가 아니라, 다양한 초기 상태에서 정책을 학습시키기 위한
**severe synthetic condition**으로 표시한다. 실제 자동차 폴리싱 전 Ra 데이터가 확보되면
해당 분포로 교체하거나 범위를 축소한다.

### 출처

- Denkena, B., Dittrich, M. A., Nguyen, H. N., Bild, K. (2021),
  *Self-optimizing process planning of multi-step polishing processes*,
  Production Engineering 15, 563–571.
  [DOI 및 Springer 원문](https://doi.org/10.1007/s11740-021-01042-6)

---

## 5. Initial Rz 검증 및 수정 권장

`Rz=4×Ra` 또는 이와 유사한 관계는 제조 현장에서 사용된 경험식이지만 Ra와 Rz는 서로 다른
형상 정보를 나타내며, 공정과 표면 프로파일에 따라 비율이 달라진다. 관련 연구는 이 변환의
적용성이 제한적임을 지적한다.

따라서 다음 방식은 실측 데이터 확보 전 임시 초기화에만 사용한다.

```python
initial_rz = initial_ra * random.uniform(4.0, 6.0)
```

최종 구현에서는 Ra와 Rz를 독립적으로 임의 생성하지 않고 동일한 Surface Height Map에서
각각 계산해야 한다.

```text
Surface Height Map 생성
        ↓
동일 표면에서 Ra 계산
동일 표면에서 Rz 계산
```

### 출처

- Palásti-Kovács, B., Sipos, S., Czifra, Á.,
  *Interpretation of “Rz = 4×Ra” and other roughness parameters in the evaluation
  of machined surfaces*.
  [확장 연구 원문](https://epa.oszk.hu/02400/02461/00051/pdf/EPA02461_acta_polytechnica_hungarica_2014_05_01.pdf)

---

## 6. Contact Force 검증

### 논문 직접값

NAK80 금형강의 로봇 Adaptive Hydraulic Polishing 연구에서는 다음 최적조건을 보고했다.

```text
P180: 20 N, 5000 rpm, 5 mm/s → Ra 0.08 μm
P400: 10 N, 4000 rpm, 5 mm/s → Ra 0.044 μm
P800: 20 N, 5000 rpm, 5 mm/s
```

항공기 에폭시 프라이머 연마 연구에서는 Force에 따라 Ra가 단조롭게 개선되지 않았다.

```text
5 N  → Ra 1.653 μm
15 N → Ra 1.350 μm
20 N → Ra 1.606 μm
```

이는 과도한 Force가 깊은 연마 흔적과 열·점착을 증가시켜 표면 품질을 악화시킬 수 있음을
보여준다.

### PolyTwin 적용

```text
Contact Force = 5~20 N
```

이 값은 자동차 Clearcoat의 공식 안전범위가 아니라 금형강과 코팅 프라이머 연구를 참고한
초기 RL 탐색범위다. 패드 크기는 프로젝트에서 고정하며, 향후 동일 패드와 실제 Clearcoat
시편의 실험 결과로 범위를 재보정한다.

### 출처

- Shi, D. et al. (2025), *Parameter Optimization and Surface Roughness Prediction
  for the Robotic Adaptive Hydraulic Polishing of NAK80 Mold Steel*.
  [MDPI 원문](https://doi.org/10.3390/pr13040991)
- Shi, D. et al. (2025), *Process Parameter Optimization and Removal Depth Prediction
  for Robotic Adaptive Hydraulically Controlled Grinding of Aircraft Skin Primer*.
  [MDPI 원문](https://doi.org/10.3390/technologies13110498)

---

## 7. Feed Speed 검증

Denkena et al.은 Feed Rate를 `50~500 mm/min` 범위에서 10단계로 변화시켰다.

```text
50 mm/min  = 0.833 mm/s
500 mm/min = 8.333 mm/s
```

따라서 PolyTwin에서는 반올림한 탐색범위를 사용한다.

```text
Feed Speed = 1~8 mm/s
대표값 = 5 mm/s
```

느린 Feed는 접촉 체류시간을 늘리므로 일반적으로 더 큰 Roughness 감소와 제거량을 유발할 수
있다. 반복 횟수가 증가할수록 Roughness 감소 효과가 줄거나 경우에 따라 증가할 수도 있다는
점도 전이모델에 반영해야 한다.

### 출처

- [Denkena et al., 2021, Springer 원문](https://doi.org/10.1007/s11740-021-01042-6)
- [Shi et al., 2025, MDPI 원문](https://doi.org/10.3390/pr13040991)

---

## 8. RPM 검증

자동차 차체 repair polishing 자동화를 위한 Kakinuma et al.의 실험은 실제 차량용
urethane polymer coated steel plate, Ø32 mm wool buff 및 약 10 μm alumina abrasive를
사용하고 spindle speed를 `5000 min⁻¹`로 고정했다. Oba et al.의 숙련공 폴리싱 기술 재현
연구에서도 `5000 min⁻¹`를 사용했다.

따라서 다음과 같이 구분한다.

```text
논문 자동차 직접값: 5000 rpm
PolyTwin 기본값   : 5000 rpm
PolyTwin 탐색범위 : 3000~6000 rpm
```

`3000~6000 rpm` 전체가 자동차 논문의 직접범위는 아니다. 또한 DA 폴리셔의 OPM과 회전식
스핀들의 RPM은 동일한 물리량이 아니므로 PolyTwin 공구 모델이 어떤 방식을 나타내는지
명시해야 한다.

### 출처

- Kakinuma, Y. et al. (2013), *Development of 5-axis polishing machine capable of
  simultaneous trajectory, posture, and force control*.
  [ScienceDirect 원문](https://doi.org/10.1016/j.cirp.2013.03.135)
- Oba, Y. et al. (2016), *Replication of skilled polishing technique with
  serial–parallel mechanism polishing machine*.
  [ScienceDirect 원문](https://doi.org/10.1016/j.precisioneng.2016.03.006)

---

## 9. Step-over 검증 및 정의

관련 연구는 인접 경로의 pitch와 polishing ribbon overlap이 Material Removal 균일도에
중요하며, 부적절한 경로 간격이 과소·과다 연마와 edge effect를 발생시킬 수 있음을 보여준다.

PolyTwin에서는 Step-over를 다음 하나의 정의로 통일한다.

```python
step_over_spacing_ratio = path_center_spacing / pad_diameter
overlap_ratio = 1.0 - step_over_spacing_ratio
```

초기 탐색범위:

```text
Step-over spacing ratio = 0.30~0.50
실제 overlap ratio      = 0.70~0.50
대표 Step-over           = 0.40
```

`30~50%` 자체는 자동차 Clearcoat 국제표준이나 논문 직접값이 아니라, 경로 밀도를 탐색하기
위한 PolyTwin 설계값이다.

### 출처

- Zhang, L. et al. (2017), *Polishing path planning for physically uniform overlap
  of polishing ribbons on freeform surface*.
  [Springer 원문](https://doi.org/10.1007/s00170-017-0466-z)
- Tam, H. Y., Cheng, H. (2010), *An investigation of the effects of the tool path
  on the removal of material in polishing*.
  [ScienceDirect 원문](https://doi.org/10.1016/j.jmatprotec.2010.01.012)

---

## 10. Clearcoat 두께와 Scratch 깊이

자동차 Clearcoat 연구에서 대표 두께 `40~50 μm`, 일반적인 Scratch 깊이 `0.05~2 μm`가
보고됐다. 별도의 자동차 코팅 리뷰에서도 대표 Clearcoat 두께를 `40~50 μm`로 제시한다.

```text
Initial Clearcoat Thickness = Uniform(40, 50) μm
Representative Scratch Depth = 0.05~2 μm
```

이 두 범위는 자동차 Clearcoat에 대한 논문 직접값으로 사용할 수 있다.

### 출처

- Bertrand-Lambotte, P. et al. (2002), *Understanding of automotive clearcoats
  scratch resistance*.
  [ScienceDirect 원문](https://doi.org/10.1016/S0040-6090(02)00943-4)
- *Progress in waterborne polymer dispersions for coating applications:
  commercialized systems and new trends* (2024).
  [RSC 원문](https://doi.org/10.1039/D4SU00267A)

---

## 11. Clearcoat 제거량 제한

PolyTwin 초기 제한:

```text
목표 제거량 : 1~7 μm
주의 영역   : 7~10 μm
실패        : >10 μm
```

이는 논문에서 규정한 공식 안전상한이 아니다. Clearcoat 전체 두께 `40~50 μm`와 일반적인
Scratch 깊이 `0.05~2 μm`를 참고해, RL이 Ra만 낮추기 위해 도막을 과도하게 제거하지 못하도록
정한 프로젝트 내부 제한이다.

동일 패드 및 실제 Clearcoat 시편을 이용한 `Force/RPM/Feed/Dwell → Removal` 실험 데이터가
확보되면 이 범위를 재보정한다.

---

## 12. Removal Map MAE 검증

Ng et al.은 숙련 작업자의 Contact Force, Feed Rate, Tool Path 등을 수집하고 이론적 Material
Removal Map을 생성한 뒤 실제 3D 스캔 결과로 모델을 보정했다. 이 연구는 PolyTwin의
`공정 파라미터 → Removal Map → 실측 보정` 구조를 직접 뒷받침한다.

Chen et al.은 다른 폴리싱 공정에서 양·음 residual error의 표준편차를 약 `1.79 μm`,
`1.72 μm` 수준으로 보고했다. 다만 이 값은 자동차 Clearcoat의 MAE 기준이 아니다.

따라서 다음 기준은 PolyTwin 초기 검증용 설계값으로만 사용한다.

```text
MAE ≤ 1 μm     : 매우 우수
1 < MAE ≤ 2 μm : 성공
2 < MAE ≤ 3 μm : 부분 패널티
MAE > 3 μm     : 실패 또는 큰 패널티
```

### 출처

- Ng, C. W. et al. (2014), *A Method for Capturing the Tacit Knowledge in the
  Surface Finishing Skill by Demonstration for Programming a Robot*.
  [IEEE DOI](https://doi.org/10.1109/ICRA.2014.6907031)
- Chen, K. et al. (2024), *On the path planning method of polishing tool based on
  the abrasive belt flap wheel*.
  [Springer 원문](https://doi.org/10.1007/s00170-024-13821-3)

---

## 13. 최종 Ra 목표 검증

Toyota 2016 흰색 자동차 도장 시편의 외부 노출 전 Clearcoat Ra는 다음과 같이 측정됐다.

```text
Group 1: 0.078 ± 0.011 μm
Group 2: 0.083 ± 0.009 μm
```

이 값은 모든 신차에 적용되는 공식 규격이 아니라 해당 Toyota 시편의 측정 결과다. PolyTwin은
약 `0.08 μm`를 목표 중심으로 사용하고, 측정 분산과 시뮬레이션 오차를 고려해 성공범위를
확장한다.

```text
최고보상: 0.075 ≤ Final Ra ≤ 0.095 μm
성공    : 0.070 ≤ Final Ra ≤ 0.100 μm
부분보상: 0.100 < Final Ra ≤ 0.120 μm
품질미달: Final Ra > 0.120 μm
```

Ra가 `0.07 μm`보다 더 낮아졌다는 이유만으로 추가 보상하지 않는다. 과도한 Force, RPM,
Dwell 또는 Pass Count를 통해 Clearcoat를 불필요하게 제거하는 정책을 방지해야 한다.

### 출처

- Alsoufi, M. S. et al. (2017), *The Effect of Detergents on the Appearance of
  Automotive Clearcoat Systems Studied in an Outdoor Weathering Test*.
  [논문 원문](https://doi.org/10.4236/msa.2017.87036)

---

## 14. 권장 RL 구성

### Phase 0: 환경 검증

- 패드 크기·재질 고정
- Tool Path와 Tool Angle 고정
- Force, Feed, RPM, Step-over 변화가 Removal/Ra에 반영되는지 단위시험
- 안전 제한과 로그 검증

### Phase 1: 기본 정책 학습

```python
action = {
    "contact_force": [5.0, 20.0],
    "feed_speed": [1.0, 8.0],
    "rpm": [3000.0, 6000.0],
    "step_over_spacing_ratio": [0.30, 0.50],
}
```

```python
observation = {
    "surface_height_map": "map",
    "ra_map": "map",
    "rz_map": "map",
    "clearcoat_thickness_map": "map",
    "removal_map": "map",
    "tool_pose": "vector",
    "contact_force": "scalar",
    "feed_speed": "scalar",
    "rpm": "scalar",
    "dwell_map": "map",
    "pass_count_map": "map",
}
```

Dwell Time과 Pass Count는 독립 Action으로 중복 입력하지 않고 실제 체류시간과 동일 영역 통과
횟수에서 누적 계산한다.

### Phase 2: 경로와 자세 확장

- Tool Path를 고수준 이산 Action 또는 별도 planner로 추가
- Tool Angle/Posture를 연속 Action으로 추가
- 곡률에 따른 실제 접촉면적과 접촉압력 반영

### Phase 3: 20° Gloss 추가

현재 Isaac Sim 20° 광택값은 정상부 대비 상대값이며 실제 GU가 아니다. 다음 조건이 모두
완료된 뒤 Gloss를 Observation, Reward 및 Success 조건에 추가한다.

1. Isaac Sim 상대 광택 반복성·결함 검출·전후 개선 검증 통과
2. 실제 20° Gloss Meter 또는 표준판 데이터 확보
3. Isaac Sim 측정값과 실제 GU의 보정식 검증
4. 검증 데이터에서 허용 오차 확인

추후 성공조건 예시:

```text
평균 20° Gloss > 70 GU
AND Ra/Rz 목표범위
AND Clearcoat Removal 제한 이내
AND Removal Map 오차 제한 이내
```

`70 GU`는 관련 자동차 보수도장 연구에서 제안된 평가기준이며 범용 법정 규격으로 표현하지
않는다.

---

## 15. Reward 및 종료조건

보상은 최종 절대값만 반복해서 주기보다 이전 스텝 대비 품질비용 감소량을 기본으로 한다.

```python
reward = previous_quality_cost - current_quality_cost
reward -= time_penalty
reward -= overforce_penalty
reward -= excessive_removal_penalty

if all_quality_targets_passed:
    reward += success_bonus
```

품질비용에는 다음 항목을 포함한다.

- Ra 목표구간과의 거리
- Rz 목표구간과의 거리
- 위치별 표면 품질 불균일
- Removal Map 목표 오차
- Clearcoat 제거량
- 미작업 영역
- 안전 접촉력 초과
- 불필요한 작업시간과 반복경로
- 추후 보정된 20° GU 목표 미달 및 불균일

초기 성공조건:

```text
0.070 ≤ Final Ra ≤ 0.100 μm
AND Clearcoat Removal ≤ 10 μm
AND Removal Map MAE ≤ 2 μm
```

초기 실패조건:

```text
Clearcoat Removal > 10 μm
OR Contact Force 안전한계 초과
OR 최대 episode 시간/step 초과
```

이 성공·실패 기준 중 `10 μm`와 `MAE≤2 μm`는 PolyTwin 초기 설계값이며, 실제 자동차
Clearcoat 실험 후 재보정한다.

---

## 16. 강화학습 담당자 필수 전달사항

```text
1. 이 문서의 전체 범위를 자동차 Clearcoat 국제표준이라고 표현하지 않는다.

2. Initial Ra 0.30~0.70 μm는 황동 연구에서 가져온 severe synthetic condition이다.

3. Rz=Ra×4~6은 임시값이다. 최종적으로 동일 Height Map에서 Ra와 Rz를 각각 계산한다.

4. 패드 크기와 재질은 전체 학습·검증에서 고정하고 설정과 로그에 기록한다.

5. Force 5~20 N은 초기 탐색범위이며 실제 Clearcoat 제거 실험으로 재보정한다.

6. 자동차 논문 직접 RPM은 5000 rpm이다. 3000~6000 rpm은 RL 확장범위다.

7. Step-over는 중심 간격/패드 직경으로 정의한다. overlap과 혼용하지 않는다.

8. 제거 상한 10 μm와 Removal MAE≤2 μm는 PolyTwin 설계기준이다.

9. 학습 매 스텝마다 RTX Path Tracing을 실행하지 않는다.
   빠른 Roughness/Removal surrogate로 학습하고 RTX는 정책 최종평가에 사용한다.

10. 20° Gloss는 실제 GU 보정이 끝난 뒤 최종 평가항목으로 추가한다.
```

---

## 17. 핵심 참고문헌

1. Denkena et al. (2021), *Self-optimizing process planning of multi-step polishing
   processes*. [DOI](https://doi.org/10.1007/s11740-021-01042-6)
2. Shi et al. (2025), *Parameter Optimization and Surface Roughness Prediction for the
   Robotic Adaptive Hydraulic Polishing of NAK80 Mold Steel*.
   [DOI](https://doi.org/10.3390/pr13040991)
3. Shi et al. (2025), *Process Parameter Optimization and Removal Depth Prediction for
   Robotic Adaptive Hydraulically Controlled Grinding of Aircraft Skin Primer*.
   [DOI](https://doi.org/10.3390/technologies13110498)
4. Kakinuma et al. (2013), *Development of 5-axis polishing machine capable of simultaneous
   trajectory, posture, and force control*.
   [DOI](https://doi.org/10.1016/j.cirp.2013.03.135)
5. Oba et al. (2016), *Replication of skilled polishing technique with serial–parallel
   mechanism polishing machine*.
   [DOI](https://doi.org/10.1016/j.precisioneng.2016.03.006)
6. Zhang et al. (2017), *Polishing path planning for physically uniform overlap of polishing
   ribbons on freeform surface*. [DOI](https://doi.org/10.1007/s00170-017-0466-z)
7. Tam and Cheng (2010), *An investigation of the effects of the tool path on the removal
   of material in polishing*. [DOI](https://doi.org/10.1016/j.jmatprotec.2010.01.012)
8. Bertrand-Lambotte et al. (2002), *Understanding of automotive clearcoats scratch
   resistance*. [DOI](https://doi.org/10.1016/S0040-6090(02)00943-4)
9. Ng et al. (2014), *A Method for Capturing the Tacit Knowledge in the Surface Finishing
   Skill by Demonstration for Programming a Robot*.
   [DOI](https://doi.org/10.1109/ICRA.2014.6907031)
10. Chen et al. (2024), *On the path planning method of polishing tool based on the abrasive
    belt flap wheel*. [DOI](https://doi.org/10.1007/s00170-024-13821-3)
11. Alsoufi et al. (2017), *The Effect of Detergents on the Appearance of Automotive
    Clearcoat Systems Studied in an Outdoor Weathering Test*.
    [DOI](https://doi.org/10.4236/msa.2017.87036)
12. *Progress in waterborne polymer dispersions for coating applications: commercialized
    systems and new trends* (2024). [DOI](https://doi.org/10.1039/D4SU00267A)

---

## 18. 결론

PolyTwin 초기 강화학습은 동일한 패드를 사용하고 Force, Feed Speed, RPM, Step-over를 먼저
학습하는 구조가 적절하다. 논문 직접값은 기본조건과 모델 구축의 출발점으로 사용하되, 서로
다른 재료와 장비에서 얻은 값을 자동차 Clearcoat의 공식 안전범위로 해석하지 않는다.

초기 RL 설계값으로 학습 파이프라인을 검증한 뒤, 동일 패드와 실제 Clearcoat 시편에서 얻은
`Force/RPM/Feed/Dwell → Ra/Rz/Removal/GU` 데이터로 전이모델, 탐색범위, Reward 및
종료조건을 순차적으로 재보정한다.
