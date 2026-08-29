> ⛔️ **구버전 — 현재 구현기준이 아니다.**
> 2026-08-27 전달된 `learning/polytwin_docs/` 문서 세트가 이 문서를 대체한다.
> 해당 세트의 `README.md`가 이 파일을 명시적으로 "구버전 문서"로 지정했다.
> **논문 출처 추적이나 과거 의사결정 확인에만 사용한다.**
>
> 가장 큰 차이: 이 문서는 실측 보정 전까지 품질 모델(제거량·Ra/Rz·GU)을 금지했지만,
> 새 문서 세트는 **논문 근거값으로 지금 구현하라**고 지시한다
> (`07_구현_인수인계.md` 7장: "GU calibration 장비가 없다는 이유로 GU proxy 구현을 생략하지 않는다").

---

# PolyTwin 자동차 Clearcoat 폴리싱 강화학습 구현 전달서

- 작성일: 2026-08-27
- 문서 상태: 상세 구현·검증·인수인계 명세
- 대상 독자: PolyTwin 담당자, 후속 개발자 및 작업을 이어받는 AI
- 대상: Isaac Lab 폴리싱 강화학습 구현 담당자
- 목적: 현재 시뮬레이터에서 실제로 학습 가능한 부분과, 실측 보정 후에만 가능한 품질 최적화를 분리하여 구현 방향을 전달한다.
- 패드 전제: 학습·검증 전 과정에서 패드 직경, 재질, 접촉면 형상은 하나로 고정한다.

---

## 0. 담당자에게 먼저 전달할 결론

현재 Isaac Lab에서 우선 학습시킬 것은 **로봇이 곡면을 따라가며 안정적으로 접촉력과 이동속도를 유지하는 제어 정책**이다.

현재 시뮬레이터에는 다음 관계가 자동차 Clearcoat 기준으로 검증되어 있지 않다.

```text
힘 + RPM + 이동속도 + 체류시간 + 패스 수
                    ↓
실제 제거 깊이 + Ra + Rz + 20° Gloss
```

따라서 현 단계에서 `Ra`, `Rz`, 실제 제거량, 실제 `20° GU`를 매 스텝 RL 보상으로 사용했다고 주장하면 안 된다. 값이 생성되더라도 실물과 연결되지 않은 임의 수치이기 때문이다.

전체 작업은 다음처럼 나눈다.

| 구분 | 해결할 문제 | 현재 권장 방법 |
|---|---|---|
| 공정조건 최적화 | 어떤 힘·RPM·속도·경로 간격을 써야 품질이 좋아지는가 | 실험 데이터 기반 서러게이트 + BO |
| 로봇 접촉·경로 제어 | 정해진 조건을 곡면에서 안전하고 균일하게 수행하는가 | Isaac Lab 강화학습 |
| 안전 제한 | 과압, 과도한 체류, 도막 과다 제거, 과열 위험을 막는가 | RL 밖의 하드 제한 + 종료조건 |
| 최종 품질 검증 | Ra/Rz, 제거량, 잔존 도막, 20° GU가 목표를 통과하는가 | 실측 및 보정된 모델, RTX 최종평가 |

이전 문서나 리뷰에서 사용한 `L1`, `L2`, `L3`는 논문의 공식 용어가 아니라 위 역할을 설명하기 위한 임시 명칭이다. 이 문서에서는 혼동을 줄이기 위해 각각 `공정조건 최적화`, `로봇 접촉·경로 제어`, `안전 제한`이라고 부른다.

### 최종 학습 목표

이 프로젝트의 강화학습 목표는 특정 자동차 한 대의 좌표와 형상을 외우는 정책이 아니다.
다양한 국소 표면 형상에서 접촉·경로 제어를 학습한 뒤, 새 자동차나 새 물체의 표면을
스캔하면 동일한 정책을 재사용할 수 있는 **형상 일반화 폴리싱 정책**을 구축한다.

```text
다양한 평면·곡면 패치 생성
        ↓
국소 좌표계에서 접촉·경로 정책 학습
        ↓
학습에 사용하지 않은 곡률·방향으로 검증
        ↓
새 자동차/새 물체 스캔 및 경로 생성
        ↓
각 경로점의 국소 법선·곡률을 정책에 입력
        ↓
동일 정책으로 폴리싱 수행
```

정책은 `특정 자동차의 특정 좌표`가 아니라 `현재 패드 주변의 표면 형상과 제어 오차`를
관측해야 한다. 패드, compound 및 재료조건처럼 제거 특성에 직접 영향을 주는 항목은 환경
설정으로 명시하며, 형상 일반화와 재료 공정모델을 섞지 않는다.

---

## 1. 상대방에게 요청할 작업 범위

### 지금 바로 구현할 것

1. 고정된 목표 공정조건을 입력받는 Isaac Lab 환경
2. 곡면 법선과 경로를 따라가는 로봇 제어
3. 목표 접촉력 유지
4. 목표 TCP 이동속도 유지
5. 패드 자세 오차 최소화
6. 경로 커버리지와 중복 방문 기록
7. 과압·접촉손실·관통·로봇 충돌에 대한 하드 제한
8. 학습과 평가 로그 저장
9. 보지 못한 곡률·결함 위치에 대한 별도 평가
10. 평면·볼록면·오목면·자유곡면을 생성하는 표면 패치 랜덤화
11. 학습 형상과 평가 형상을 완전히 분리한 일반화 평가
12. 새 물체의 스캔 결과를 국소 관측으로 변환하는 입력 인터페이스

### 지금은 구현하지 않거나, 보상에서 분리할 것

1. 근거 없이 만든 `Force/RPM/Feed → Ra` 계산식
2. 근거 없이 만든 `Force/RPM/Feed → 20° GU` 계산식
3. 매 RL 스텝마다 RTX Path Tracing을 실행하는 방식
4. 5~20 N 전체를 현재 로봇의 직접 행동범위로 사용하는 방식
5. 모든 에피소드에 동일한 1~7 μm 제거 목표를 부여하는 방식
6. `Rz = Ra × 4~6`을 최종 품질의 정답으로 사용하는 방식

---

## 2. 현재 코드와 반드시 맞춰야 하는 사실

현재 `scripts/polishing_v5_modules/common.py` 기준값은 다음과 같다.

```text
패드 반경                   = 0.055 m
패드 직경                   = 0.110 m
패드 높이                   = 0.070 m
패드 회전속도               = 314.16 rad/s ≈ 3000 rpm
가상 패드 강성              = 350 N/m

상단 평탄면 목표 접촉력     = 8.0 N
상단 급경사 목표 접촉력     = 5.0 N
측면 평탄면 목표 접촉력     = 6.0 N
측면 급경사 목표 접촉력     = 3.5 N

상단 하드 제한              = 14.0 N
측면 하드 제한              = 6.0 N
경로 진행 허용 상한         = 14.0 N

상단 경로 중심 간격         = 0.09 m
측면 경로 중심 간격         = 0.05 m
```

따라서 검토리뷰에 적힌 `목표 5 N`, `강성 500 N/m`, `패드 150 mm`, `회전 5 rad/s`는 현재 v5 값이 아니라 과거 코드 기준이다. 그 값으로 현재 구조를 비판한 계산은 그대로 적용할 수 없다.

다만 핵심 결론은 남는다. 현재 안전 제한에서 20 N을 명령하면 다음 문제가 생긴다.

- 상단에서는 14 N 제한에 걸린다.
- 측면에서는 6 N 제한에 걸린다.
- 가상 스프링 강성 350 N/m에서 20 N은 약 57 mm 압축에 해당한다.
- 현재 접촉 제어를 바꾸지 않고 5~20 N을 그대로 행동공간으로 쓰면 정책 명령과 안전 계층이 충돌한다.

따라서 **현재 접촉제어 RL의 목표 힘은 기존 장비가 유지할 수 있는 영역에서만 변화**시킨다. 논문에서 정리한 5~20 N은 향후 실제 Clearcoat 실험을 위한 후보 범위이지, 현재 Isaac Lab 정책에 그대로 넣을 확정 범위가 아니다.

### Step-over 정의

이 프로젝트에서는 다음 정의만 사용한다.

```python
step_over_spacing_ratio = path_center_spacing / pad_diameter
overlap_ratio = 1.0 - step_over_spacing_ratio
```

현재 110 mm 패드 기준:

| 구역 | 중심 간격 | spacing ratio | overlap ratio |
|---|---:|---:|---:|
| 상단 | 90 mm | 81.8% | 18.2% |
| 측면 | 50 mm | 45.5% | 54.5% |

검토리뷰의 `150 mm 패드에서 90 mm 간격` 계산도 과거 값이다. 앞으로 로그·설정·논문에는 `spacing ratio`인지 `overlap ratio`인지 이름을 반드시 붙인다.

위 `90/50 mm`는 현재 v5의 `rail/full` 모드에 해당한다. `path_generator.py`의 기존 4대 고정
`multi` 모드는 별도로 상단 `30 mm`를 사용하므로 서로 혼용하지 않는다. 실행 모드와 경로
간격을 같은 config에 기록한다. 파일 수정시각만으로 기준 버전을 결정하지 않고, RL 환경이
사용한 git commit과 config를 실행 결과에 함께 저장한다.

---

## 3. 강화학습 문제 정의

### 3.1 한 에피소드의 의미

한 에피소드는 다음 중 하나의 제한된 패널 구역을 연마하는 과정으로 정의한다.

```text
초기화
  → 패널 형상·곡률·목표 경로·목표 힘·목표 속도 설정
  → 접근
  → 접촉
  → 경로 추종 및 폴리싱
  → 구역 완료 또는 안전 실패
```

처음부터 차량 전체를 한 에피소드로 사용하지 않는다. 작은 패널 구역에서 학습한 뒤 구역 수와 곡률 난도를 늘린다.

### 3.2 학습 정책이 담당하는 것

정책은 기준 제어기 전체를 대체하지 않고, 기준 제어기에 작은 잔차 보정을 주는 방식이 안전하다.

권장 구조:

```text
경로 생성기 + 법선 기반 기준 자세 + 기존 힘 제어기
                         ↓
                RL 잔차 보정값 추가
                         ↓
                      로봇 명령
```

이 구조에서 RL은 다음을 개선한다.

- 곡률 변화에서 발생하는 힘 오차
- 접촉 시작 시 오버슈트
- 측면 이동 시 힘 손실
- 경로 코너에서의 속도·자세 오차
- 구역별 커버리지 불균일

### 3.3 행동공간 권장안

구현 전에 담당자가 현재 사용 중인 `action/observation/reward/termination` 표를 먼저 공유한다.
이미 확정하거나 구현한 잔차 행동이 있다면 임의로 교체하지 않고, 기존 사양을 기준으로 부족한
항목만 추가한다. 특히 임피던스 계수 보정과 기하학적 위치·자세 보정은 서로 다른 설계이므로
이름만 비슷하다는 이유로 혼합하지 않는다.

새로 정의해야 하는 경우의 기하학적 잔차 행동 예시는 다음과 같다.

```python
action = {
    "delta_normal_offset": continuous(-1, 1),
    "delta_feed_scale": continuous(-1, 1),
    "delta_tool_tilt_x": continuous(-1, 1),
    "delta_tool_tilt_y": continuous(-1, 1),
}
```

실제 단위로 변환할 때는 반드시 제한한다. 보정폭은 임의의 고정값으로 확정하지 않고, RL을
끄고 기준 제어기만 실행한 로그에서 정상상태 오차와 과도응답의 `p95`를 구해 결정한다.

```text
법선 방향 위치 보정       : 기준 제어기 잔여 위치오차 p95 기반
Feed 속도 배율 보정        : 기준 속도오차 p95 및 안전속도 기반
패드 자세 보정             : 기준 법선 정렬오차 p95 기반
```

RMPFlow의 전체 추종지연과 RL이 보정할 잔여오차를 혼동하지 않는다. 기존 feed-forward 및 기준
제어기를 적용한 뒤에도 남는 오차를 RL 행동범위로 삼는다. 행동의 최대 변화율도 제한하여
패드 튐과 접촉력 스파이크를 방지한다.

정책 실행 주파수와 물리 시뮬레이션 주파수도 분리한다. `10/20/30 Hz` 등의 후보를 동일한
baseline에서 비교하고 힘 RMSE, 힘 스파이크, 경로오차 및 계산시간으로 정책 주파수와 control
decimation을 확정하여 config에 기록한다.

`Contact Force`, `RPM`, `Feed Speed`, `Step-over` 네 값을 에피소드 시작 때 한 번 선택하고 끝까지 고정한다면 이것은 순차 제어 RL이라기보다 공정조건 탐색 문제다. 이 네 값을 찾는 작업은 뒤의 BO/서러게이트 절에서 다룬다.

### 3.4 관측공간 권장안

최소 관측:

```python
observation = {
    "joint_position": "normalized vector",
    "joint_velocity": "normalized vector",
    "tcp_pose_error": "position + orientation error",
    "surface_normal_local": "3D vector",
    "surface_curvature_or_tilt": "scalar/vector",
    "principal_curvatures_local": "k1, k2",
    "contact_force": "scalar",
    "force_error": "target - measured",
    "force_derivative": "scalar",
    "tcp_speed": "scalar/vector",
    "feed_speed_error": "scalar",
    "path_tangent_local": "3D vector",
    "path_progress": "0~1",
    "distance_to_path": "scalar",
    "previous_action": "vector",
}
```

### 3.5 좌표계 불변성

새 자동차나 새 물체로 일반화하려면 표면의 월드 좌표를 정책의 주된 입력으로 사용하지 않는다.
관측은 패드 또는 TCP 기준의 국소 좌표계로 변환한다.

```text
월드 좌표의 표면점·법선·경로
             ↓ TCP/pad frame으로 변환
국소 표면 법선
국소 경로 접선
국소 경로 오차
주곡률 k1, k2
국소 경계 방향
```

물체의 이름, 특정 자동차의 vertex index, 특정 위치의 절대 XYZ에 정책이 의존하지 않게 한다.
절대 위치는 로봇 도달 가능성·관절한계·충돌 검사를 위한 planner와 안전 계층에서 사용한다.

가능하면 추가할 항목:

- 패드의 실제 접촉면적 또는 접촉면적 근사값
- 누적 체류시간 맵의 현재 셀 값
- 해당 셀 방문 횟수
- 주변 셀 커버리지
- 곡면 경계까지의 거리
- 로봇 특이점·관절한계까지의 거리

단위가 다른 관측값을 그대로 넣지 않는다. 각 값은 물리적으로 의미 있는 범위로 나누고 `[-1,1]` 또는 `[0,1]`로 정규화한다.

예시:

```python
force_error_norm = clip(force_error / force_error_scale, -1.0, 1.0)
speed_error_norm = clip(speed_error / speed_error_scale, -1.0, 1.0)
path_error_norm = clip(path_error / path_error_limit, -1.0, 1.0)
tilt_error_norm = clip(tilt_error_deg / tilt_error_limit_deg, -1.0, 1.0)
```

---

## 4. 현재 단계의 RL 보상함수

### 4.1 원칙

현재 RL 보상은 `품질 결과를 추측하는 보상`이 아니라 `정해진 공정을 정확하고 안전하게 수행하는 보상`이어야 한다.

권장 보상 항목:

1. 접촉력 추종
2. TCP 이동속도 추종
3. 경로 추종
4. 패드 법선 정렬
5. 균일한 커버리지
6. 행동 변화량 억제
7. 시간과 불필요한 중복 경로 억제

### 4.2 정규화된 예시

```python
e_force = clip(abs(force - target_force) / force_tolerance, 0.0, 1.0)
e_speed = clip(abs(speed - target_speed) / speed_tolerance, 0.0, 1.0)
e_path  = clip(path_distance / path_tolerance, 0.0, 1.0)
e_tilt  = clip(tilt_error_deg / tilt_tolerance_deg, 0.0, 1.0)
e_force_rate = clip(abs(force_derivative) / force_rate_limit, 0.0, 1.0)
e_action = clip(norm(action - previous_action) / action_delta_limit, 0.0, 1.0)

delta_coverage = max(0.0, current_coverage - previous_coverage)

r_force = 1.0 - e_force
r_speed = 1.0 - e_speed
r_path  = 1.0 - e_path
r_tilt  = 1.0 - e_tilt

reward = (
    0.35 * r_force
    + 0.20 * r_speed
    + 0.20 * r_path
    + 0.15 * r_tilt
    + 0.10 * delta_coverage
    - 0.05 * e_force_rate
    - 0.05 * e_action
    - 0.01 * step_cost
)
```

누적 coverage 값을 매 스텝 반복 보상하지 않고 새로 작업된 면적의 증가분만 보상한다. 가중치는
초기 제안값이며 합이 반드시 1이어야 하는 것은 아니다. 중요한 것은 원 단위의 Ra, μm, N,
m/s를 한 식에 바로 더하지 않고 먼저 `[0,1]`로 정규화하는 것이다.

목표 힘과 하드 제한이 상단·측면에서 다르므로 `force_tolerance`, `force_rate_limit` 및 종료조건도
구역별 config에서 가져온다. 상단에서 학습한 정규화식을 측면에 그대로 적용하더라도 실제 N
기준의 제한까지 동일하게 복사하지 않는다.

### 4.3 보상과 하드 제한을 구분

다음은 단순 음의 보상만 주고 계속 진행시키지 않는다.

- 접촉력 하드 제한 초과
- 패드 또는 공구의 차체 관통
- 비패드 링크와 차체의 충돌
- 관절한계 또는 위험한 특이점 진입
- NaN/Inf 상태
- 일정 시간 이상 접촉 상실
- 경로 이탈 후 복구 불가

이 경우 에피소드를 즉시 종료하고 실패 원인을 로그에 기록한다.

### 4.4 열 위험의 현재 처리

실제 온도 모델이 없다면 `F × RPM / Feed` 하나만으로 실제 온도라고 주장하지 않는다. 이 값은 냉각, 패드 면적, 재료, compound, 주변 온도를 반영하지 못한다.

현재 단계에서는 안전 프록시로 다음을 사용한다.

```text
동일 위치 연속 체류시간 상한
동일 셀 누적 체류시간 상한
동일 셀 통과 횟수 상한
저속·고접촉력·고RPM 조합 경고
```

온도 센서 또는 열전달 모델이 추가되면 실제 온도와 냉각시간을 관측·종료조건에 포함한다.

---

## 5. Isaac Lab 기준 환경 이식 검증

강화학습을 시작하기 전에 현재 Isaac Sim v5의 기준 동작이 Isaac Lab 환경에서도 재현되는지
검증한다. 학습 문제와 환경 이식 오류를 분리하기 위한 필수 단계다.

### 5.1 이식 대상

- 차량 또는 시험 표면과 로봇·패드 asset
- RMPFlow 또는 현재 사용하는 기준 motion controller
- 경로 접선과 표면 법선에 따른 기준 자세 계산
- 가상 스프링 접촉력 및 필터링
- 곡률·구역별 목표 접촉력
- 상단·측면별 안전 제한과 과압 후퇴
- 접촉손실·관통·비패드 링크 충돌 감지
- 경로 진행도·커버리지·체류시간 계산
- episode reset 및 randomized surface 재생성

### 5.2 학습 전 baseline 검사

RL action을 모두 0으로 고정하고 다음을 확인한다.

```text
Isaac Sim v5 기준 제어기
          vs
Isaac Lab 안의 기준 제어기(action=0)
```

동일 표면·경로·seed에서 다음 지표를 비교한다.

- TCP 궤적
- 접촉력 시계열
- Force RMSE/p95/peak
- 경로 오차
- 법선 정렬오차
- 접촉 유지율
- 완료시간
- 안전 이벤트

차이가 발생하면 RL 학습 전에 asset scale, 좌표축, physics dt, controller dt, solver 설정,
접촉모델 및 센서 필터를 먼저 수정한다.

### 5.3 초기화 검증

한 에피소드가 끝난 뒤 다음 상태가 이전 에피소드에서 누적되지 않아야 한다.

```text
로봇 joint position/velocity
RMPFlow/controller 내부 상태
패드 접촉력 필터
경로 index와 progress
coverage/dwell/pass-count map
이전 action
표면 패치와 결함 seed
종료 원인 플래그
```

동일 seed reset은 동일한 초기상태를 재현하고, 다른 seed는 지정된 분포에서 새로운 형상을
생성해야 한다.

---

## 6. 형상 일반화 학습환경

### 6.1 표면 패치 종류

학습환경은 자동차 전체 모델 하나를 반복하지 않고 다음 국소 표면 패치를 절차적으로 생성한다.

| 표면 | 학습 목적 |
|---|---|
| 평면 | 기준 접촉력·속도 제어 |
| 경사진 평면 | 월드 방향 변화에 대한 불변성 |
| 볼록 원통면 | 한 방향 곡률 변화 |
| 오목 원통면 | 패드 중심·가장자리 접촉 변화 |
| 볼록 구면 | 두 방향 곡률 변화 |
| 오목 구면 | 오목부 접근과 자세 안정화 |
| 완만한 자유곡면 | 위치에 따라 연속적으로 변하는 곡률 |
| 곡률 전이 패치 | 평면에서 곡면, 볼록에서 오목으로 바뀌는 구간 |

각 패치는 충분한 면적과 경계 여유를 가져야 하며, 패드가 경계 밖으로 나가는 경우를 안전
계층과 경로 planner가 검출한다.

### 6.2 에피소드 랜덤화 항목

```python
episode_context = {
    "surface_family": "plane/cylinder/sphere/freeform/transition",
    "principal_curvature_k1": "sampled",
    "principal_curvature_k2": "sampled",
    "surface_orientation": "sampled",
    "patch_position": "within robot workspace",
    "path_direction": "sampled in tangent plane",
    "path_start": "sampled",
    "path_pattern": "line/raster/contour",
    "target_force": "sampled from region-safe config",
    "target_feed_speed": "sampled from configured set",
    "friction": "domain randomized",
    "contact_stiffness": "domain randomized",
    "sensor_noise": "domain randomized",
    "sensor_delay": "domain randomized",
}
```

랜덤값을 한 번에 전 범위로 뿌리지 않고 커리큘럼 단계에 따라 분산을 넓힌다. 표면 형상,
제어조건, 센서조건의 seed를 따로 기록하여 실패조건을 재현할 수 있게 한다.

### 6.3 형상 생성과 실제 물체 입력을 동일 인터페이스로 통일

절차적으로 생성한 패치와 실제 스캔 물체는 정책 입장에서 같은 형식으로 보여야 한다.

```python
LocalSurfaceContext = {
    "contact_point_local": [x, y, z],
    "normal_local": [nx, ny, nz],
    "tangent_local": [tx, ty, tz],
    "principal_curvature": [k1, k2],
    "boundary_distance": scalar,
    "path_error_local": [ex, ey, ez],
}
```

훈련 시에는 analytic surface에서 이 값을 계산하고, 새 자동차·새 물체에서는 point cloud 또는
mesh의 근방점으로 법선과 곡률을 추정한다. 단위, 축 순서, 법선 방향 및 smoothing 설정을
동일하게 유지한다.

### 6.4 학습·검증·최종평가 분리

단순히 random seed만 바꾸는 것으로 끝내지 않고 형상 자체를 분리한다.

```text
Train
  지정된 평면·원통·구면·자유곡면 family와 곡률 조합

Validation
  Train 사이의 미사용 곡률·방향·경로 시작점

Test
  학습에 사용하지 않은 자유곡면 seed
  학습에 사용하지 않은 곡률 전이
  실제 자동차 또는 물체 mesh의 국소 패널
```

정책 checkpoint 선택에는 validation만 사용하고 test 결과를 보고 다시 학습하지 않는다.

### 6.5 새 자동차·새 물체 적용 파이프라인

```text
1. 물체 표면 스캔 또는 mesh 로드
2. 접근 가능한 작업영역 분할
3. 표면 법선·주곡률·경계 계산
4. 로봇 도달성과 충돌을 반영해 경로 생성
5. 각 경로점을 LocalSurfaceContext로 변환
6. BO/공정설정에서 목표 힘·속도·RPM·Step-over 입력
7. RL 정책이 잔차 접촉·속도·자세 보정
8. 안전 계층이 최종 명령 제한 및 필요 시 종료
9. 커버리지와 품질평가 결과 저장
```

경로 생성기와 로봇별 작업영역 분할은 정책 밖에서 담당한다. RL 정책은 도달 불가능한 위치를
억지로 해결하지 않으며, 현재 경로점 주변의 국소 접촉 제어에 집중한다.

### 6.6 다중 로봇 적용

레일/오버헤드 로봇과 측면 로봇은 설치방향, 중력 영향 및 힘 제한이 다르다. 공통 정책을
사용하려면 다음 context를 관측에 포함하거나, 동일한 정규화 규칙을 가진 robot별 adapter를
둔다.

```text
robot/mount type
gravity vector in TCP frame
region type: top/side
target force
soft/hard force limits
joint-limit margin
```

기준 정책 검증은 한 로봇에서 먼저 완료한 뒤 다른 설치방향으로 확장하며, 전체 차량 평가는
각 로봇의 결과를 같은 지표 정의로 합산한다.

### 6.7 공정조건을 입력받는 조건부 정책

하나의 정책이 서로 다른 폴리싱 조건을 수행하려면 목표 공정조건을 observation의 context로
입력한다. 정책이 공정조건 자체를 매 스텝 임의로 바꾸는 구조와 구분한다.

```python
process_context = {
    "target_contact_force": scalar,
    "target_feed_speed": scalar,
    "target_rpm": scalar,
    "step_over_spacing_ratio": scalar,
    "tool_path_type": categorical,
    "target_tool_angle": vector,
}
```

`Dwell Time`과 `Pass Count`는 동일 위치의 실제 체류시간과 통과 횟수에서 누적 계산한다.
패드 크기는 프로젝트 전 과정에서 고정하므로 정책 입력으로 변화시키지 않고 config와 로그에
기록한다.

구조는 다음과 같다.

```text
공정조건 최적화기 또는 작업계획기
  → 목표 Force/Feed/RPM/Step-over/Path/Posture 결정
                         ↓ process context
국소 표면 관측 + 기준 제어기 + RL 잔차 정책
  → 조건을 곡면에서 안정적으로 실행
```

학습 에피소드에서는 안전·검증된 config 안에서 process context를 변화시킨다. 각 항의 조합을
순서대로 전부 열거하지 않고, 균형 표본추출로 전체 조건공간을 덮고 구간별 표본 수를 로그에
남긴다. RPM이 실제 제거·열 모델과 연결되기 전에는 회전 목표 추종 context로만 취급하며,
Ra/Rz/GU 품질 개선을 학습했다고 해석하지 않는다.

---

## 7. 커리큘럼 학습 순서

### Stage A: 평면 접촉력 유지

- 평면 패널
- 한 방향 직선 경로
- 고정 목표 속도
- 고정 목표 힘
- RPM은 고정
- Step-over 없음
- 성공: 접촉력 RMSE, 오버슈트, 접촉 유지율 통과

### Stage B: 완만한 곡면

- 곡률 랜덤화
- 법선 방향 변화
- 기존 행동공간 유지
- 성공: 힘 오차와 자세 오차를 동시에 통과

### Stage C: 래스터 경로

- 여러 줄의 경로
- 코너 및 줄 전환 포함
- 커버리지와 중복방문 평가
- Step-over는 에피소드 설정값으로 고정

### Stage D: 초기 상태 랜덤화

- 패널 위치·방향
- 곡률
- 마찰계수와 접촉강성의 작은 변화
- 목표 힘·속도의 안전범위 내 변화
- 센서 노이즈와 지연
- 스크래치 위치 및 난이도 맵

스크래치 맵은 현재 단계에서 `어디를 더 오래/정확히 작업할지`를 결정하는 문맥으로 사용할 수 있지만, 실제 Ra/Rz 개선량으로 변환했다고 주장하지 않는다.

### Stage E: 보지 못한 조건 평가

학습에 사용하지 않은 다음 조건으로 별도 평가한다.

- 새로운 곡률
- 새로운 패널 방향
- 새로운 결함 위치
- 다른 경로 시작점
- 센서 노이즈 증가

훈련 seed와 평가 seed를 분리한다.

---

## 8. 성공조건과 평가 지표

구체적인 허용값은 현재 제어 담당자가 가진 사양을 우선 사용하되, 최소한 다음을 보고한다.

```text
Force RMSE
Force 최대 오버슈트
Force 변화율 RMSE/95 percentile/최대값
Force spike 횟수
목표 힘 허용대역 내 시간 비율
TCP speed RMSE
경로 오차 평균/95 percentile/최대값
패드 법선 각도 오차 평균/최대값
유효 접촉 유지율
경로 커버리지
중복 방문율
작업 완료시간
안전 종료 횟수와 원인
미학습 곡률·미학습 표면에서의 일반화 성능
```

평균만 기록하면 순간 과압을 숨길 수 있으므로 평균, 95 percentile, 최댓값을 함께 기록한다.

성공 여부는 하나의 총보상만으로 결정하지 말고 핵심 제약을 모두 통과시킨다.

```python
success = (
    force_rmse <= force_rmse_limit
    and force_peak <= force_hard_limit
    and force_rate_p95 <= force_rate_limit
    and contact_ratio >= contact_ratio_min
    and path_p95 <= path_error_limit
    and coverage >= coverage_min
    and not safety_terminated
)
```

---

## 9. 공정조건 최적화는 어떻게 할 것인가

### 9.1 왜 현재 RL과 분리하는가

다음 값을 에피소드 시작 전에 한 번 선택하고 패널 작업이 끝난 뒤 최종 품질을 한 번 평가한다면,
이는 매 순간 상태에 따라 행동하는 순차 제어 문제가 아니라 비용이 큰 공정조건 탐색 문제다.

```text
Contact Force
Feed Speed
RPM
Step-over
Tool Path
Tool Angle/Posture
```

반면 로봇은 폴리싱 중 매 제어 스텝마다 곡률, 힘 오차, 경로 오차를 보고 위치·속도·자세를
조정해야 한다. 이 부분은 PPO와 같은 강화학습 정책이 담당할 수 있다.

```text
에피소드 시작 전에 레시피 1회 선택 → BO가 담당
에피소드 중 매 스텝 잔차 제어       → PPO가 담당
하드 안전 제한과 강제 종료           → 규칙 기반 안전 계층이 담당
최종 Ra/Rz/Removal/GU 판정           → 실측·보정된 품질 평가기가 담당
```

### 9.2 BO와 PPO의 핵심 차이

| 항목 | BO: Bayesian Optimization | PPO: Proximal Policy Optimization |
|---|---|---|
| 알고리즘 종류 | 비싼 black-box 함수의 입력을 찾는 최적화 | 순차적 의사결정을 학습하는 강화학습 |
| PolyTwin 질문 | 어떤 공정 레시피가 품질을 만족하는가 | 그 레시피를 곡면에서 어떻게 안정적으로 실행하는가 |
| 선택 시점 | 실험·에피소드 시작 전 1회 | 제어 주기마다 반복 |
| 입력 | 공정조건과 초기 결함·재료 context | 현재 힘·속도·경로·법선·곡률·이전 행동 |
| 출력 | Force/RPM/Feed/Step-over 등의 레시피 | 위치·속도·자세·제어기 계수의 잔차 보정 |
| 평가 신호 | 한 실험의 최종 Ra/Rz/제거량/GU/시간 | 매 스텝 힘·속도·경로·자세 오차와 종료 결과 |
| 데이터 특성 | 적은 횟수의 비싼 평가 | 많은 시뮬레이션 rollout |
| 학습 결과 | 최적 후보 레시피와 불확실도 | 상태를 행동으로 변환하는 policy network |
| 실행 시 사용 | 작업 recipe 생성 시 | 로봇 운전 중 실시간 inference |
| 적합한 차원 | 보통 저차원 연속·혼합 파라미터 | 고차원 연속 관측과 순차 행동 |
| 안전 처리 | 제약조건과 feasible domain | action clipping + 별도 safety layer + termination |

둘 중 하나를 선택해 다른 하나를 버리는 구조가 아니다. BO는 공정 레시피를 만들고 PPO는
레시피를 실행한다.

### 9.3 PPO가 해결하는 문제의 정확한 정의

PPO가 받는 문제는 다음 MDP 형태로 정의한다.

```text
State/Observation s_t
  현재 로봇·패드·국소표면·경로·접촉 상태

Action a_t
  기준 제어기에 더하는 작은 잔차 보정

Transition
  Isaac Lab physics + 기준 제어기 + 안전 계층

Reward r_t
  힘·속도·경로·자세 추종, 신규 coverage, 부드러운 행동

Termination
  작업 완료 또는 과압·관통·충돌·복구불가
```

PPO는 actor와 critic을 학습한다.

- Actor `π(a|s)`: 현재 관측에서 잔차 행동의 확률분포를 출력한다.
- Critic `V(s)`: 현재 상태에서 앞으로 받을 누적보상을 추정한다.
- Rollout: 여러 에피소드에서 `(s, a, r, done)` 전이를 수집한다.
- Advantage: 실제 결과가 critic 예상보다 얼마나 좋았는지 계산한다.
- Clipped update: 새 정책이 한 번에 너무 크게 바뀌지 않도록 확률비율을 제한한다.

개념식:

```python
ratio = new_policy_prob(action | state) / old_policy_prob(action | state)
ppo_objective = min(
    ratio * advantage,
    clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantage,
)
```

PPO가 학습하는 것은 다음 함수다.

```python
residual_action_t = policy(
    local_surface_state_t,
    robot_state_t,
    tracking_error_t,
    process_context,
)
```

PPO가 직접 출력해서는 안 되는 것은 다음과 같다.

- 근거 없는 `final Ra` 또는 `20° GU`
- 실제 실험 없이 확정한 최적 RPM
- 안전 계층을 무시하는 최종 joint command
- 특정 자동차의 절대 좌표에만 유효한 waypoint index

이 문서에서 PPO는 대표적인 연속제어 알고리즘으로 설명한다. 담당자가 이미 CHEQ 또는 다른
잔차 RL 알고리즘을 구현하고 있다면 알고리즘을 자동으로 PPO로 교체하지 않는다. 환경의
관측·행동·보상·안전·평가 인터페이스는 유지하고, 알고리즘 변경은 동일 baseline의 실험으로
결정한다.

### 9.4 BO가 해결하는 문제의 정확한 정의

BO는 다음 black-box 목적함수를 최소화하거나 최대화한다.

```python
quality_score, constraints = evaluate_polishing_recipe(
    recipe,
    initial_surface_context,
    material_context,
)
```

레시피 예시:

```python
recipe = {
    "target_contact_force_n": 7.0,
    "feed_speed_mm_s": 4.0,
    "rpm": 4000,
    "step_over_spacing_ratio": 0.40,
    "tool_path_type": "raster",
    "tool_angle_deg": [0.0, 0.0],
}
```

평가 결과 예시:

```python
result = {
    "final_ra_um": 0.09,
    "final_rz_um": 0.48,
    "removal_mae_um": 0.7,
    "healthy_overremoval_um": 0.2,
    "minimum_remaining_clearcoat_um": 38.5,
    "max_temperature_c": 48.0,
    "gloss_20deg_gu": 72.0,
    "process_time_s": 83.0,
    "feasible": True,
}
```

위 결과 숫자는 데이터 구조를 보여주는 예시이며 PolyTwin 확정값이 아니다.

BO의 반복 절차는 다음과 같다.

1. 초기 실험계획으로 여러 레시피를 평가한다.
2. `recipe → quality/constraint` 관계를 근사하는 확률적 모델을 학습한다.
3. acquisition function으로 다음 후보를 선택한다.
4. 후보를 실제 시편 또는 검증된 공정모델에서 평가한다.
5. 결과를 데이터셋에 추가한다.
6. 종료조건까지 2~5를 반복한다.

대표 acquisition의 의미:

- Expected Improvement: 현재 최고점보다 개선될 기대량이 큰 후보 선택
- Upper Confidence Bound: 높은 예측값과 높은 불확실도의 균형
- Probability of Improvement: 기준을 넘을 확률이 큰 후보 선택
- Constrained acquisition: 안전·품질 제약을 만족할 확률까지 고려

PolyTwin에서는 단순 최고 Gloss가 아니라 여러 제약을 동시에 고려해야 하므로 constrained BO
또는 feasible 후보만 평가하는 안전한 탐색구조를 사용한다.

### 9.5 PolyTwin에서 BO를 채택한 이유

1. 실제 Clearcoat 실험은 시편·측정·패드 상태관리 때문에 평가비용이 크다.
2. 공정 레시피의 주요 변수 수가 비교적 적다.
3. 품질 결과가 패널 작업 종료 후에 주로 확인되는 종단 평가다.
4. 실제 시편에서 PPO가 요구하는 대량 rollout을 수행하기 어렵다.
5. BO는 적은 데이터에서 예측 불확실도를 이용해 다음 실험을 선택할 수 있다.
6. 도막 잔존두께·온도·힘 같은 제약조건을 후보 선택에 포함할 수 있다.
7. 어떤 레시피를 왜 다음 후보로 선택했는지 기록하기 쉽다.

BO가 항상 PPO보다 우수하다는 뜻은 아니다. 다음과 같이 문제의 시간구조가 다르기 때문에
채택한 것이다.

```text
조건을 한 번 선택하고 최종 점수만 받음 → BO
상태를 계속 관측하며 행동을 바꿈       → PPO/RL
```

표면의 위치별 결함을 보고 힘·속도를 실시간 변경하여 품질까지 최적화하려면 순차 품질모델이
준비된 뒤 PPO의 역할을 확대할 수 있다. 그 전에는 품질모델이 없는 RL에 가상의 Ra/GU 보상을
주는 방식은 사용하지 않는다.

### 9.6 두 종류의 서러게이트를 구분

`서러게이트`라는 단어가 두 의미로 사용될 수 있으므로 구현과 로그에서 이름을 구분한다.

#### A. BO 내부 확률 서러게이트

```text
목적: 다음 실험 후보 선택
입력: recipe + context
출력: 목적함수 평균 예측 + 불확실도
예: Gaussian Process, Random Forest 계열, TPE 계열
```

#### B. 빠른 공정 품질 서러게이트

```text
목적: 실제 연마 또는 고비용 물리모델 대신 빠르게 품질상태 전이 예측
입력: 이전 surface state + 위치별 pressure/speed/dwell/pass
출력: 다음 removal/height/Ra/Rz/GU/temperature state
예: 회귀모델, neural network, calibrated analytic model
```

BO가 내부적으로 사용하는 모델과 RL rollout에 사용하는 빠른 품질모델은 동일할 수도 있지만
자동으로 동일한 것은 아니다. 다음 이름으로 분리한다.

```text
bo_acquisition_model
process_quality_surrogate
```

### 9.7 BO 목적함수와 제약조건

BO의 품질 점수는 원 단위 값을 바로 더하지 않고 정규화한다.

```python
objective_cost = (
    w_ra * ra_target_distance_norm
    + w_rz * rz_target_distance_norm
    + w_map * removal_map_error_norm
    + w_healthy * healthy_overremoval_norm
    + w_gloss * gloss_shortfall_norm
    + w_time * process_time_norm
)
```

안전항은 낮은 가중치의 soft penalty로만 처리하지 않는다.

```python
feasible = (
    minimum_remaining_clearcoat >= remaining_clearcoat_limit
    and max_temperature <= temperature_limit
    and peak_force <= force_hard_limit
    and no_collision
)
```

`feasible=False`인 후보는 품질점수가 좋아도 최적 레시피로 선택하지 않는다.

Gloss가 아직 실제 GU로 보정되지 않았다면 `w_gloss=0`으로 두고 상대 광택은 보조 분석값으로만
저장한다. 해당 상태를 config의 `gloss_calibrated=false`로 명시한다.

### 9.8 BO에서 PPO로 넘기는 인터페이스

BO 결과는 자유형식 문장이 아니라 버전이 있는 구조화 파일로 전달한다.

```json
{
  "schema_version": "1.0",
  "recipe_id": "recipe_00042",
  "material_id": "clearcoat_dataset_v1",
  "pad_id": "fixed_pad_v1",
  "compound_id": "compound_v1",
  "target_contact_force_n": 7.0,
  "feed_speed_mm_s": 4.0,
  "rpm": 4000,
  "step_over_spacing_ratio": 0.40,
  "tool_path_type": "raster",
  "tool_angle_deg": [0.0, 0.0],
  "hard_limits": {
    "peak_force_n": "REQUIRED_CONFIG",
    "max_temperature_c": "REQUIRED_CONFIG",
    "minimum_remaining_clearcoat_um": "REQUIRED_CONFIG"
  },
  "quality_model_version": "REQUIRED_CONFIG",
  "gloss_calibrated": false
}
```

PPO 환경은 이 파일을 `process_context`로 읽고 각 값을 정규화하여 observation에 포함한다.
PPO는 recipe 원본을 수정하지 않고 잔차 명령만 낸다.

```python
baseline_command = controller(state, recipe)
residual_command = ppo_policy(observation, recipe_context)
safe_command = safety_filter(baseline_command + residual_command)
```

PPO가 지속적으로 큰 잔차를 내야만 recipe를 유지할 수 있다면 다음 중 하나다.

- BO recipe가 해당 형상에 부적절함
- 기준 제어기 성능이 부족함
- PPO action scale이 잘못됨
- 관측 정규화 또는 좌표계가 잘못됨

이 경우 policy가 안전 제한과 싸우도록 두지 않고 원인을 분리해 수정한다.

### 9.9 전체 학습·운영 데이터 흐름

```text
[실제 Clearcoat 실험]
Force/RPM/Feed/Step-over/Path/Posture
초기·최종 height, Ra, Rz, removal, temperature, GU
                  ↓
[process_quality_dataset]
                  ↓
[품질 서러게이트 학습 및 독립 검증]
                  ↓
[BO]
안전하고 품질점수가 높은 recipe 추천
                  ↓ recipe.json
[Isaac Lab 기준 제어기 + PPO 잔차 정책]
다양한 국소 곡면에서 recipe 실행
                  ↓
[RTX 선택 평가 + 실제 시편 최종 검증]
                  ↓
[새 데이터 추가 및 BO/서러게이트 재보정]
```

### 9.10 BO 의사코드

```python
dataset = load_calibrated_experiments()
search_space = load_recipe_search_space()

while not stopping_condition(dataset):
    bo_acquisition_model.fit(dataset)

    candidate = propose_candidate(
        model=bo_acquisition_model,
        search_space=search_space,
        constraints=hard_constraints,
    )

    result = run_real_or_validated_evaluation(candidate)
    validate_measurement_quality(result)
    dataset.append(candidate, result)
    save_dataset_atomically(dataset)

best_recipe = select_best_feasible_recipe(dataset)
export_recipe_json(best_recipe)
```

### 9.11 PPO 학습 의사코드

```python
env = PolyTwinLocalSurfaceEnv(
    surface_generator=procedural_surface_generator,
    baseline_controller=ported_v5_controller,
    safety_filter=hard_safety_layer,
    recipe_sampler=validated_recipe_context_sampler,
)

assert compare_action_zero_baseline(env)

for iteration in range(max_iterations):
    rollout = collect_rollout(
        policy=ppo_policy,
        env=env,
        randomized_surface=True,
        split="train",
    )

    ppo_policy.update(rollout)
    metrics = evaluate_fixed_seeds(ppo_policy, split="validation")

    save_latest_checkpoint()
    if metrics_pass_acceptance(metrics):
        save_best_checkpoint()

test_metrics = evaluate_once(best_checkpoint, split="test")
export_policy_and_metrics(best_checkpoint, test_metrics)
```

### 9.12 BO와 PPO 결과를 혼동하지 않는 로그 명칭

```text
BO 결과
  recipe_id
  acquisition_value
  predicted_objective_mean/std
  measured_objective
  constraint feasibility

PPO 결과
  policy_checkpoint_id
  rollout reward
  Force/Speed/Path/Tilt metrics
  coverage and termination
  recipe_id used by episode

최종 품질 결과
  specimen_id
  initial/final metrology
  calibrated GU
  recipe_id
  policy_checkpoint_id
```

하나의 최종 실험을 재현하려면 `specimen_id + recipe_id + policy_checkpoint_id + seed + config`
조합이 모두 기록되어야 한다.

### 9.13 공정 품질 서러게이트의 입출력

서러게이트는 실제 폴리싱 실험을 대신해 결과를 빠르게 예측하는 대리 모델이다.

```text
입력
  초기 height/scratch map
  힘, RPM, 속도, Step-over
  dwell map, pass count
  패드·compound 식별자
  곡률과 접촉면적

출력
  removal map
  final height map
  Ra/Rz
  잔존 clearcoat 두께
  추후 보정된 20° GU
```

모델 학습 시 train/validation/test 시편을 분리하고, 동일 시편의 여러 측정 위치가 서로 다른
split으로 섞이지 않게 specimen 단위로 분할한다. 출력별 MAE뿐 아니라 bias, p95 error,
불확실도 calibration 및 안전 제약 오판율을 보고한다.

### 9.14 필요한 캘리브레이션 데이터

동일한 패드와 compound를 사용하여 다음을 기록한다.

```text
패드 직경·재질·상태
compound 제품과 도포량
초기 도막 두께
초기 height map, Ra, Rz
스크래치 깊이·밀도·위치
접촉력
RPM
Feed Speed
Step-over spacing ratio
Dwell map
Pass count map
표면온도 또는 최대온도
최종 removal map
최종 Ra, Rz
20° Gloss GU
```

패드 크기를 통일하더라도 패드 마모와 compound 상태가 달라지면 결과가 달라질 수 있으므로 함께 로그에 남긴다.

---

## 10. 품질 모델이 준비된 뒤 추가할 보상

### 10.1 제거 모델

초기 모델은 Preston 형태를 참고할 수 있다.

```python
delta_h = k_p * pressure * relative_speed * dwell_time
```

그러나 `k_p`는 자동차 Clearcoat, 고정 패드, compound, 온도 조건으로 실측 보정해야 한다. 금형강이나 황동 논문의 계수를 그대로 사용하면 안 된다.

### 10.2 목표 제거맵

모든 에피소드에 동일한 `1~7 μm` 제거 목표를 주지 않는다. 초기 height/scratch map에서 위치별 목표를 만든다.

```text
결함이 거의 없는 위치 → 제거 목표가 매우 작음
얕은 스크래치 위치   → 필요한 만큼만 제거
깊은 스크래치 위치   → 안전상한 안에서 더 큰 제거 목표
이미 정상인 위치      → 추가 제거 억제
```

즉 `target_removal_map = f(initial_height_map, scratch_map, safety_margin)` 형태가 되어야 한다.

### 10.3 Ra와 Rz

Ra와 Rz는 하나에서 다른 하나를 단순 환산하지 않는다.

```text
동일한 최종 height map
      ├─ Ra 계산
      ├─ Rz 계산
      └─ 최대 잔존 스크래치 깊이 계산
```

Ra는 평균 거칠기라 드문 깊은 스크래치를 숨길 수 있다. 따라서 최종 성공조건에는 Rz 또는 최대 잔존 스크래치 깊이가 반드시 들어가야 한다.

### 10.4 Removal Map 오차

`MAE ≤ 2 μm` 하나만 사용하면 목표 제거량이 작은 구역에서는 아무 작업을 하지 않아도 통과할 수 있다. 절대·상대 기준을 함께 사용한다.

```python
mae_abs = mean(abs(removal_map - target_removal_map))

valid = target_removal_map > target_epsilon
mae_rel = mean(
    abs(removal_map[valid] - target_removal_map[valid])
    / maximum(target_removal_map[valid], relative_floor)
)
```

정상영역은 별도로 과다 제거율을 평가한다.

```python
healthy_overremoval = mean(
    maximum(0, removal_map[healthy_mask] - healthy_removal_allowance)
)
```

### 10.5 제거량 패널티 중복 방지

`Removal Map 오차`와 `Clearcoat 총 제거량`은 목적이 다르므로 함께 사용할 수 있다.

- Removal Map 오차: 필요한 위치를 정확히 깎았는가
- 총 제거량/정상부 제거: 필요 이상으로 도막을 소비했는가

다만 총 제거량 전체를 무조건 선형 패널티로 주면 `아무것도 하지 않는 정책`이 유리해진다. 목표를 달성하는 데 필요한 제거에는 패널티를 주지 않고, 초과 제거만 벌한다.

```python
excess_removal = mean(maximum(0, removal_map - target_removal_map - tolerance))
```

### 10.6 품질 보상 예시

모든 항을 `[0,1]`로 정규화한 후 사용한다.

```python
quality_cost = (
    w_ra * ra_target_distance_norm
    + w_rz * rz_target_distance_norm
    + w_map * removal_map_error_norm
    + w_healthy * healthy_overremoval_norm
    + w_clearcoat * clearcoat_risk_norm
    + w_heat * heat_risk_norm
    + w_time * process_time_norm
)

reward_quality = previous_quality_cost - current_quality_cost
```

`현재보다 실제 품질비용을 줄였는가`를 보상하면 같은 최종값을 매 스텝 중복해서 보상하는 문제를 줄일 수 있다.

### 10.7 품질 성공조건 예시

정확한 기준값은 실측 데이터로 재보정한다.

```text
Ra 목표구간 통과
AND Rz 또는 최대 잔존 스크래치 기준 통과
AND Removal Map 절대·상대오차 통과
AND 정상영역 과다제거 기준 통과
AND 잔존 Clearcoat 두께 안전기준 통과
AND 온도 안전기준 통과
AND 추후 20° GU 기준 통과
```

도막 제거 7~10 μm가 주의 구간이면서 `≤10 μm 성공`인 것은 논리적 모순은 아니다. 낮은 점수로 안전상한 이내 성공을 허용할 수 있기 때문이다. 단, 다음처럼 상태를 구분해 기록해야 한다.

```text
목표 품질 성공   : 필요한 제거만 수행하고 모든 품질기준 통과
허용 성공        : 절대 안전상한은 넘지 않았지만 과다 제거로 감점
안전 실패        : 잔존 도막/온도/힘의 하드 제한 위반
```

---

## 11. 20° Gloss를 RL에 넣는 시점

현재 `gloss_test`의 Isaac Sim 결과는 정상부 대비 **상대 광택**이며 실제 Gloss Unit(GU)이 아니다.

그러므로 지금은 다음 용도로만 사용한다.

- 결함 위치의 상대 반사 저하 시각화
- 5×5 위치별 상대 광택 비교
- 폴리싱 전후 개선 방향 확인
- RTX 측정 파이프라인 반복성 확인

다음 절차가 끝난 뒤에만 실제 `20° GU` 보상 또는 성공조건으로 추가한다.

1. 표준 광택판 또는 실제 자동차 시편을 20° Gloss Meter로 측정
2. 동일 시편·동일 위치를 Isaac Sim 장면으로 구성
3. 조명, 센서, 재질 파라미터를 고정
4. 시뮬레이션 상대값과 실제 GU의 보정식 구축
5. 학습에 쓰지 않은 시편으로 오차 검증
6. 허용오차를 통과한 경우만 GU 예측값 사용

RTX Path Tracing은 계산비용이 크므로 RL 매 스텝에 넣지 않는다.

```text
훈련 중  : 빠른 서러게이트 또는 상대 품질맵
정책 평가: 선택된 에피소드만 RTX Path Tracing
최종 검증: 실제 20° Gloss Meter
```

논문에서 확인한 평균 `20° Gloss > 70 GU`는 해당 자동차 보수도장 연구에서 제안한 평가기준이지 모든 차량에 적용되는 법정 규격이 아니다. 프로젝트 시편과 고객 요구조건으로 최종 기준을 확정한다.

---

## 12. 검토리뷰에 대한 수용·반박 정리

### 12.1 플랜트 모델이 없다는 지적

**수용한다.** 현재 자동차 Clearcoat용 `공정 파라미터 → Ra/Rz/Removal/GU` 모델이 없다. 따라서 품질 최적화 보상을 당장 계산할 수 없다.

다만 이것이 접촉·경로 제어 RL까지 불가능하다는 뜻은 아니다. 힘, 속도, 자세, 경로 오차는 현재 시뮬레이션에서 관측 가능하므로 제어 정책은 먼저 학습할 수 있다.

### 12.2 5~20 N이 현재 시뮬레이터와 충돌한다는 지적

**핵심은 수용하지만 근거 수치는 수정한다.** 리뷰가 인용한 값은 현재 v5와 다르다.

```text
리뷰 기준: 5 N, 500 N/m, 3~10 N, 150 mm, 5 rad/s
현재 v5 : 곡률별 3.5~8 N, 350 N/m, 상단 14 N/측면 6 N 제한,
          110 mm, 314.16 rad/s
```

현재 값에서도 20 N은 안전 제한과 충돌하므로 지금 행동공간으로 쓰지 않는다. 5~20 N은 실험 후보범위로만 남긴다.

### 12.3 제거량 이중 패널티 지적

**부분 수용한다.** Removal Map 오차와 총 제거량은 각각 공간 정확도와 도막 소비를 보므로 의미가 다르다. 따라서 둘을 같이 쓴다고 자동으로 이중 계산은 아니다.

하지만 모든 제거량에 선형 패널티를 주는 식은 잘못이다. 정상부 과다 제거 또는 목표 초과분만 패널티로 바꾼다.

### 12.4 고정 제거 목표가 부적절하다는 지적

**수용한다.** 목표 제거맵은 초기 height/scratch map에서 위치별로 생성한다.

단, 리뷰가 제안한 `바탕 Ra 0.08~0.15 μm` 전체 범위도 자동차 손상면의 직접 실측 근거가 확보된 확정값은 아니다. 이 역시 임시 분포로 표시하고 실측 후 교체해야 한다.

### 12.5 Rz가 성공조건에 없다는 지적

**수용한다.** Ra만으로는 깊은 잔존 스크래치를 놓칠 수 있다. Rz 또는 최대 잔존 스크래치 깊이를 성공조건에 추가한다.

다만 `Rz ≤ Ra의 4~6배`를 그대로 정답으로 쓰지 않는다. 동일 height map에서 Ra와 Rz를 각각 계산한다.

### 12.6 Removal MAE ≤ 2 μm 지적

**수용한다.** 고정 절대 MAE만 사용하지 않고 절대오차, 상대오차, 정상부 과다 제거를 함께 본다. 목표값이 0에 가까운 영역 때문에 상대오차만 단독으로 쓰는 것도 피한다.

### 12.7 열 모델 누락 지적

**수용한다.** 열은 중요한 실패모드다. 다만 `F × RPM / Feed`는 실제 온도가 아니라 위험 프록시일 뿐이다. 우선 체류시간·반복횟수 제한을 적용하고 실제 온도 데이터 또는 열 모델이 준비되면 교체한다.

### 12.8 이 문제는 RL보다 BO에 가깝다는 지적

**조건부로 맞다.** 네 공정 파라미터를 에피소드마다 한 번 고르면 BO/밴딧 문제다. 반대로 표면 상태와 곡률을 보며 매 스텝 접촉·속도·자세를 조정하면 RL 문제다.

따라서 공정조건은 BO/서러게이트, 로봇의 순차 제어는 RL로 나눈다.

### 12.9 Step-over 정의가 모호하다는 지적

**정의 명시는 수용하지만 리뷰의 현재 코드 계산은 반박한다.** 현재 패드는 150 mm가 아니라
110 mm다. 또한 현재 v5 `rail/full` 실행은 상단 90 mm, 측면 50 mm이며, 리뷰의 상단 30 mm는
기존 4대 고정 `multi` 모드 호출값이다. 이 프로젝트는 `중심 간격/패드 직경`인 spacing
ratio로 통일하고 실행 mode를 함께 기록한다. overlap을 쓸 때는 별도 이름으로 표시한다.

### 12.10 RPM이 현재 물리 효과가 없다는 지적

**핵심은 수용하지만 현재 수치는 반박한다.** v5 회전속도는 5 rad/s가 아니라 314.16 rad/s다. 그러나 회전 애니메이션이 실제 제거량과 Ra를 변화시키는 검증된 재료 모델로 연결되지 않았으므로, 현재 RPM은 품질 최적화 행동으로서 충분한 물리 효과가 없다.

### 12.11 보상 정규화가 필요하다는 지적

**전적으로 수용한다.** 모든 보상 항은 허용범위로 정규화하고 가중치를 명시한다. 안전 위반은 큰 음의 보상만 주기보다 즉시 종료한다.

### 12.12 7~10 μm 감점과 ≤10 μm 성공이 모순이라는 지적

**모순이라고 단정하는 것은 반박한다.** 안전상한 안에서 품질은 통과했지만 도막을 필요 이상 소비한 `허용 성공`은 가능하다. 다만 `목표 성공/허용 성공/안전 실패`를 구분해 보고해야 한다.

---

## 13. 실행 로그 필수 항목

각 실행은 최소 다음 파일을 남긴다.

```text
config.json
episode_metrics.csv
termination_summary.json
training_curve.csv 또는 TensorBoard 로그
best_policy_checkpoint
evaluation_summary.json
```

`config.json` 필수값:

```text
git commit hash
Isaac Sim / Isaac Lab 버전
random seed
surface geometry seed / path seed / sensor seed
train/validation/test split id
surface family / principal curvatures / orientation
패드 직경·높이·재질·접촉강성
compound 또는 재료모델 버전
목표 힘과 안전 제한
목표 속도
RPM
Step-over spacing ratio
관측·행동 정규화 범위
reward 가중치
episode 종료조건
physics dt / control decimation
학습 알고리즘과 hyperparameter
```

에피소드별 로그:

```text
episode id / seed
패널과 곡률 조건
총 reward와 항목별 reward
Force RMSE / p95 / peak
Force derivative RMSE / p95 / peak / spike count
Speed RMSE
Path error mean / p95 / max
Tilt error mean / max
Contact ratio
Coverage / revisit ratio
작업시간
종료 사유
```

기준선 측정과 최종평가에서는 상태 출력용 저주파 로그만 사용하지 않는다. 힘 peak와 변화율이
유실되지 않도록 physics/control 스텝 단위 원시 CSV를 별도로 저장하고, 요약 통계는 이 원시
로그에서 계산한다.

품질 모델 추가 후:

```text
initial/final height map 경로
initial/final Ra, Rz
target/predicted/actual removal map 경로
잔존 도막 두께
온도 관련 지표
상대 광택과 보정된 20° GU
```

---

## 14. 담당자 완료 체크리스트

### 접촉·경로 RL 완료 조건

- [ ] RL 행동이 기존 안전 계층과 충돌하지 않는다.
- [ ] RL action을 0으로 했을 때 Isaac Lab baseline이 기존 v5 동작을 재현한다.
- [ ] 기존 담당자의 확정 action/observation/reward 사양과 충돌 여부를 확인했다.
- [ ] 관측과 보상의 모든 항이 정규화되어 있다.
- [ ] 평면에서 먼저 수렴한 뒤 곡면 커리큘럼을 적용했다.
- [ ] 학습 seed와 평가 seed가 분리되어 있다.
- [ ] Force 평균뿐 아니라 RMSE, p95, peak를 기록한다.
- [ ] Force 변화율과 spike 횟수를 기록한다.
- [ ] 경로·자세·속도·커버리지 지표를 기록한다.
- [ ] 과압, 관통, 충돌은 즉시 종료된다.
- [ ] 현재 상대 광택을 실제 GU라고 부르지 않는다.
- [ ] RTX Path Tracing을 매 학습 스텝 실행하지 않는다.
- [ ] 학습 전후 정책을 동일 조건에서 비교한 평가표가 있다.
- [ ] Train/validation/test 형상과 seed가 분리되어 있다.
- [ ] 미학습 곡률·자유곡면·실제 물체 패널에서 일반화 평가를 수행했다.
- [ ] 누적 coverage가 아니라 스텝별 신규 coverage 증가분을 보상한다.

### 품질 최적화 착수 조건

- [ ] 동일 패드·compound의 실제 폴리싱 데이터가 있다.
- [ ] 입력 파라미터와 제거량/Ra/Rz 관계를 검증했다.
- [ ] 위치별 target removal map을 생성할 수 있다.
- [ ] 정상부 과다 제거를 별도로 계산한다.
- [ ] 열 또는 체류시간 안전모델이 있다.
- [ ] 서러게이트를 학습·검증 데이터로 분리 평가했다.
- [ ] BO가 제안한 조건을 독립 실험으로 재검증했다.

### 20° Gloss 추가 조건

- [ ] 실제 20° Gloss Meter 데이터가 있다.
- [ ] Isaac Sim 값과 실제 GU 보정식이 있다.
- [ ] 보정에 쓰지 않은 시편으로 오차를 검증했다.
- [ ] 프로젝트의 최종 GU 합격기준이 확정됐다.

---

## 15. 담당자에게 그대로 보낼 요청문

```text
현재 단계에서는 Isaac Lab으로 자동차 Clearcoat의 실제 Ra/Rz/GU 개선량을
직접 학습시키지 말고, 정해진 폴리싱 조건을 곡면에서 안전하게 수행하는
접촉·경로 제어 정책을 먼저 학습해 주세요.

목표는 특정 자동차 좌표를 외우는 정책이 아니라 평면·볼록면·오목면·자유곡면에서
학습한 뒤 새로운 자동차와 물체의 국소 표면에도 적용되는 형상 일반화 정책입니다.

1. 현재 Isaac Sim v5의 경로/법선/힘/안전 제어를 Isaac Lab에 옮기고,
   RL action=0에서 원본과 같은 baseline이 나오는지 먼저 검증하세요.
2. 기존 경로/법선/힘 제어기를 기준 정책으로 유지하고 RL은 작은 잔차 보정만 하세요.
   기존에 확정된 잔차 action 사양이 있다면 교체하지 말고 먼저 공유하세요.
3. 정책 관측은 특정 물체의 절대 XYZ가 아니라 TCP/pad 국소 좌표계로 구성하세요.
   국소 법선, 경로 접선, 주곡률 k1/k2, 힘·속도·자세·경로 오차를 사용하세요.
4. 평면, 경사 평면, 볼록/오목 원통면, 볼록/오목 구면, 자유곡면 및 곡률 전이
   패치를 절차적으로 생성하여 curriculum과 domain randomization을 적용하세요.
5. Train/validation/test는 seed뿐 아니라 곡률 조합과 자유곡면 자체를 분리하고,
   test에는 실제 자동차 또는 물체 mesh의 미학습 패널을 포함하세요.
6. 관측에는 힘 오차, 힘 변화율, TCP 속도, 경로 오차, 법선/자세 오차,
   곡률, path progress, previous action을 넣어 주세요.
7. 보상은 힘, 속도, 경로, 자세, 신규 coverage 증가분, 힘 변화율 및 행동 변화량으로
   구성하고 모든 항을 0~1로 정규화하세요. 누적 coverage를 반복 보상하지 마세요.
8. 과압, 차체 관통, 비패드 링크 충돌, 장시간 접촉손실은 음의 보상만 주지 말고
   즉시 에피소드를 종료하세요.
9. 현재 v5 목표 힘은 곡률/구역별 약 3.5~8N이며 상단 14N, 측면 6N 제한입니다.
   논문 기반 5~20N을 현재 행동공간으로 바로 사용하지 마세요.
10. 패드는 하나의 크기로 고정하고 직경·재질·강성·compound를 모든 로그에 기록하세요.
11. Step-over는 path center spacing / pad diameter로 통일하고 overlap과 혼용하지 마세요.
12. 행동 보정폭과 정책 주파수는 기준 제어기의 매 스텝 로그에서 잔여오차 p95를 구하고,
    여러 주파수 후보를 비교한 뒤 확정하세요.
13. 결과는 reward 하나가 아니라 Force RMSE/p95/peak, Force 변화율/spike,
    속도오차, 경로오차, 자세오차, 접촉률, 커버리지, 일반화 성능 및
    안전종료 원인으로 평가하세요.
14. RPM/힘/속도/Step-over의 최적 조합은 실제 Clearcoat 데이터가 생긴 뒤
    서러게이트+BO로 찾고, 그 결과를 RL 제어기의 목표조건으로 넘기세요.
15. 현재 Isaac Sim 광택은 상대값이고 실제 GU가 아니므로 RL 성공조건에 넣지 마세요.
    실제 20° Gloss Meter와 보정이 끝난 뒤 추가합니다.

구현 전에 현재 코드 상수와 action/observation/reward/termination 표를 먼저 공유하고,
구현 후에는 고정 seed 평가 결과와 로그 경로를 전달해 주세요.
```

---

## 16. 논문 근거와 값의 해석

| 항목 | 확인된 근거 | PolyTwin에서의 사용 방법 |
|---|---|---|
| 자동차 직접 RPM 5000 rpm | Kakinuma 2013, Oba 2016 | 기준 후보값. 3000~6000은 탐색 확장값이지 직접 표준이 아님 |
| Feed 50~500 mm/min | Denkena 2021 | 0.83~8.33 mm/s로 환산 가능하나 황동 연구임 |
| Force 5~20 N | NAK80·항공기 프라이머 연구 | 실제 Clearcoat 실험 후보범위. 현 시뮬 직접 행동범위 아님 |
| Clearcoat 40~50 μm | 자동차 Clearcoat 문헌 | 초기 도막 두께 분포 참고 |
| Scratch 0.05~2 μm | 자동차 Clearcoat 문헌 | 초기 결함 깊이 분포 참고 |
| Toyota 초기 Ra 약 0.08 μm | Alsoufi 2017 | 특정 시편 측정값. 전 차량 공식 규격이 아님 |
| 20° Gloss 평균 70 GU 초과 | Coatings 2021 연구 | 해당 연구의 평가기준. 범용 법정 규격이 아님 |

핵심 원칙은 `논문에 나온 입력범위`와 `그 입력이 우리 Clearcoat에서 만드는 결과`를 분리하는 것이다. 재료·패드·compound가 다르면 결과 모델은 그대로 이식할 수 없다.

### 주요 참고문헌

1. Denkena et al. (2021), *Self-optimizing process planning of multi-step polishing processes*. [DOI](https://doi.org/10.1007/s11740-021-01042-6)
2. Shi et al. (2025), *Parameter Optimization and Surface Roughness Prediction for the Robotic Adaptive Hydraulic Polishing of NAK80 Mold Steel*. [DOI](https://doi.org/10.3390/pr13040991)
3. Shi et al. (2025), *Process Parameter Optimization and Removal Depth Prediction for Robotic Adaptive Hydraulically Controlled Grinding of Aircraft Skin Primer*. [DOI](https://doi.org/10.3390/technologies13110498)
4. Kakinuma et al. (2013), *Development of 5-axis polishing machine capable of simultaneous trajectory, posture, and force control*. [DOI](https://doi.org/10.1016/j.cirp.2013.03.135)
5. Oba et al. (2016), *Replication of skilled polishing technique with serial-parallel mechanism polishing machine*. [DOI](https://doi.org/10.1016/j.precisioneng.2016.03.006)
6. Zhang et al. (2017), *Polishing path planning for physically uniform overlap of polishing ribbons on freeform surface*. [DOI](https://doi.org/10.1007/s00170-017-0466-z)
7. Tam and Cheng (2010), *An investigation of the effects of the tool path on the removal of material in polishing*. [DOI](https://doi.org/10.1016/j.jmatprotec.2010.01.012)
8. Bertrand-Lambotte et al. (2002), *Understanding of automotive clearcoats scratch resistance*. [DOI](https://doi.org/10.1016/S0040-6090(02)00943-4)
9. Ng et al. (2014), *A Method for Capturing the Tacit Knowledge in the Surface Finishing Skill by Demonstration for Programming a Robot*. [DOI](https://doi.org/10.1109/ICRA.2014.6907031)
10. Alsoufi et al. (2017), *The Effect of Detergents on the Appearance of Automotive Clearcoat Systems Studied in an Outdoor Weathering Test*. [DOI](https://doi.org/10.4236/msa.2017.87036)
11. Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
    [arXiv 원문](https://arxiv.org/abs/1707.06347)
12. Snoek, Larochelle, Adams (2012), *Practical Bayesian Optimization of Machine Learning
    Algorithms*. [NeurIPS 원문](https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html)

---

## 17. 역할 및 시스템 결정

현재 상대방은 특정 자동차의 좌표를 외우는 방식이 아니라 **다양한 평면·볼록면·오목면·
자유곡면의 국소 형상과 공정조건을 관측하여 접촉력·속도·경로·자세·커버리지를 제어하는
Isaac Lab 조건부 잔차 정책**을 학습한다.

학습된 정책은 새 자동차 또는 물체를 스캔한 뒤 생성된 경로, 법선, 곡률 및 목표 공정조건을
동일한 `LocalSurfaceContext + process_context` 인터페이스로 받아 실행한다. 물체 전체의 절대
좌표와 이름은 정책이 아니라 경로 planner와 안전 계층이 처리한다.

우리 쪽에서는 다음을 준비한다.

1. 초기 결함 및 height map 정의
2. 위치별 목표 제거맵 생성 규칙
3. 실제 Clearcoat 폴리싱 캘리브레이션 데이터 형식
4. Ra/Rz/잔존 도막/열/20° GU 최종 평가 인터페이스
5. 품질 서러게이트와 BO 목적함수

실측 품질 모델이 확보되면 BO가 공정조건을 제안하고, Isaac Lab의 RL 정책은 그 조건을 실제 곡면에서 안정적으로 실행한다. 마지막으로 RTX 및 실제 계측으로 최종 품질을 검증한다.

---

## 18. 다른 AI가 구현할 때 사용하는 규범

이 절부터는 구현 해석 차이를 줄이기 위한 명세다.

다음 용어를 사용한다.

- **MUST**: 구현·검증에 반드시 필요한 요구사항
- **MUST NOT**: 사용하면 결과 해석이 무효가 되는 금지사항
- **SHOULD**: 특별한 사유가 없으면 적용하는 권장사항
- **MAY**: 구현 선택사항
- **REQUIRED_CONFIG**: 현재 근거로 숫자를 확정하지 않고 실측·팀 결정으로 채울 설정값

다른 AI는 `REQUIRED_CONFIG`를 임의의 숫자로 바꿔 완료 처리하면 안 된다. 해당 값이 비어 있으면
환경 시작을 중단하거나 `calibration_mode`에서만 실행해야 한다.

### 18.1 단위 규칙

내부 물리 계산의 기본단위:

```text
길이와 위치       : m
시간              : s
힘                : N
각도              : rad
각속도            : rad/s
곡률              : 1/m
도막·거칠기 보고값: μm
Gloss             : GU, 보정 완료 후에만 사용
```

사람이 읽는 config에서 mm/s, rpm, degree를 허용할 수 있으나 환경 진입 시 한 번만 SI 단위로
변환한다. 변수명에 단위를 붙인다.

```python
feed_speed_m_s = feed_speed_mm_s / 1000.0
spin_rad_s = rpm * 2.0 * pi / 60.0
tool_angle_rad = radians(tool_angle_deg)
```

단위를 이름에 쓰지 않은 `speed`, `angle`, `removal` 같은 필드는 새 인터페이스에 만들지 않는다.

### 18.2 좌표계 규칙

```text
world frame : 물체 배치, 로봇 workspace, 충돌검사
base frame  : 로봇 kinematics와 관절 상태
TCP frame   : 정책의 국소 관측 기준
surface frame:
  z축 = 표면 외향 법선
  x축 = 경로 진행 접선
  y축 = z × x로 만든 접평면 보조축
```

법선 방향은 물체 바깥쪽으로 통일한다. 스캔에서 법선 부호가 뒤집힐 수 있으므로 로봇 접근방향
또는 mesh orientation으로 부호를 정리하고, 연속 경로에서 인접 법선의 dot product가 음수이면
flip한다.

---

## 19. Isaac Lab 환경의 한 스텝 처리 순서

환경의 `step(action)`은 다음 순서를 MUST 따른다.

```text
1. policy action 수신
2. action 범위 [-1,1] clipping
3. 물리단위 잔차로 변환
4. action rate limit 적용
5. 기준 제어기 명령 계산
6. 기준 명령 + 잔차 명령 결합
7. 안전 필터 적용
8. control decimation 동안 physics 진행
9. 센서·접촉력·로봇 상태 갱신
10. 국소 표면·경로 상태 계산
11. coverage/dwell/pass map 갱신
12. reward 항목별 계산
13. success/failure/timeout 판정
14. observation 정규화
15. 원시 로그와 episode accumulator 기록
16. observation, reward, terminated, truncated, info 반환
```

안전 필터는 policy 이후에 위치하여 최종 명령을 제한한다. 하지만 policy가 안전 필터에 계속
의존하는 것을 숨기지 않도록 다음도 기록한다.

```text
raw_policy_action
rate_limited_action
baseline_command
pre_safety_command
post_safety_command
safety_intervention_flag
safety_intervention_reason
```

### 19.1 Reset 순서

```text
1. episode seed 확정 및 기록
2. train/validation/test split 확인
3. surface family와 geometry seed 선택
4. 표면 patch 생성 또는 test mesh patch 로드
5. process recipe/context 선택
6. robot/mount/region 선택
7. 로봇 pose와 joint velocity 초기화
8. controller/filter/RL history 초기화
9. 경로 생성 및 path index=0
10. coverage/dwell/pass map을 0으로 초기화
11. 물리 안정화 스텝 수행
12. 초기 관측 계산
13. 초기 상태 snapshot과 config hash 저장
```

reset 직후 이미 접촉력이 비정상적으로 크거나 표면 안쪽에 패드가 있으면 해당 episode를 학습에
포함하지 않고 `invalid_reset`으로 기록한다.

### 19.2 접근·접촉·폴리싱 단계 구분

모든 보상을 전체 episode에서 똑같이 활성화하지 않는다.

| 단계 | 활성 보상 | 비활성 또는 제한 보상 |
|---|---|---|
| Approach | 경로 접근, 자세 정렬, 시간 | Force 추종, coverage |
| Contact acquisition | 완만한 힘 상승, 자세, spike 억제 | Feed 추종의 비중 축소 |
| Polishing | Force/Feed/Path/Tilt, Δcoverage | 접근거리 보상 |
| Line transition | 접촉 안전, 경로 재진입, 행동 안정 | coverage 중복 보상 |
| Completion | 성공 보너스 1회 | 지속형 성공 보상 |

단계 판정값 `phase_id`를 observation과 로그에 포함한다.

---

## 20. 관측 텐서 상세 명세

아래 표는 최소 논리필드다. 실제 tensor index와 shape는 담당자가 확정하여 별도
`observation_schema.json`으로 저장한다.

| 필드 | 좌표계 | 정규화 기준 | 목적 |
|---|---|---|---|
| joint_position | joint | 각 관절 lower/upper limit | 자세와 한계 인식 |
| joint_velocity | joint | 관절별 허용속도 | 동적상태 인식 |
| tcp_position_error | surface | path tolerance | 경로 추종 |
| tcp_orientation_error | surface | tilt tolerance | 패드 정렬 |
| surface_normal | TCP | 이미 unit vector | 국소 표면 방향 |
| path_tangent | TCP | unit vector | 진행 방향 |
| principal_curvature_k1/k2 | surface | config curvature scale | 형상 일반화 |
| contact_force | scalar | region hard limit | 현재 접촉상태 |
| force_error | scalar | region tolerance | 목표 힘 추종 |
| force_derivative | scalar | region rate limit | 튐·슬램 감지 |
| tcp_linear_velocity | TCP | target/max speed | 실제 이동상태 |
| feed_speed_error | scalar | speed tolerance | 속도 추종 |
| path_progress | scalar | 이미 0~1 | 작업 진행 |
| local_coverage | scalar/map | 이미 0~1 | 미작업 위치 인식 |
| local_dwell | scalar/map | dwell limit | 과다 체류 방지 |
| previous_action | policy | 이미 -1~1 | 행동 진동 억제 |
| target_force | scalar | region hard limit | 조건부 정책 context |
| target_feed | scalar | configured max | 조건부 정책 context |
| target_rpm | scalar | configured max | recipe context |
| step_over_ratio | scalar | 0~1 | recipe context |
| gravity_vector | TCP | unit vector | 설치방향 일반화 |
| joint_limit_margin | joint | 0~1 | 도달성·안전 |
| boundary_distance | surface | pad radius | 패치 경계 인식 |
| phase_id | categorical | one-hot | 단계별 행동 구분 |

### 20.1 관측 누락 처리

- 센서값이 일시 누락되면 0을 실제값처럼 넣지 않는다.
- 해당 값의 last valid value와 `valid_mask`를 함께 제공한다.
- 일정 시간 이상 invalid이면 접촉손실 또는 sensor failure 종료조건을 적용한다.
- NaN/Inf는 정규화 전에 검출하고 즉시 종료한다.

예시:

```python
observation["contact_force"] = normalized_last_valid_force
observation["contact_force_valid"] = 0.0
```

### 20.2 관측 정규화 통계

물리적 한계가 있는 값은 config 한계로 정규화한다. 데이터 기반 표준화를 사용하는 값은 train
split에서만 mean/std를 계산하고 validation/test 결과로 갱신하지 않는다.

정규화 통계는 checkpoint와 함께 저장한다.

```text
policy.pt
observation_normalizer.json
action_scaler.json
environment_config.json
```

---

## 21. 행동 명령 상세 명세

정책의 raw output은 모든 축에서 `[-1,1]`로 제한한다. 물리값 변환은 환경이 담당한다.

### 21.1 기하 잔차안

```python
delta_normal_m = raw[0] * normal_offset_scale_m
delta_feed_m_s = raw[1] * feed_residual_scale_m_s
delta_tilt_x_rad = raw[2] * tilt_x_scale_rad
delta_tilt_y_rad = raw[3] * tilt_y_scale_rad
```

### 21.2 임피던스 잔차안

기존 담당자가 임피던스 계수 잔차를 사용 중이면 다음처럼 명시한다.

```python
k_cmd = k_base * (1.0 + raw_delta_k * k_ratio_limit)
zeta_cmd = zeta_base * (1.0 + raw_delta_zeta * zeta_ratio_limit)
c_cmd = 2.0 * zeta_cmd * sqrt(k_cmd * effective_mass)
force_target_cmd = force_target_base * (1.0 + raw_delta_force * force_ratio_limit)
feed_cmd = feed_base * (1.0 + mapped_delta_feed)
```

기하 잔차안과 임피던스 잔차안을 동시에 모두 활성화하면 서로 같은 힘 오차를 다른 경로로
보정할 수 있다. 동시 사용 시 ablation으로 각 action의 기여도를 검증하고, 필요하지 않은 축은
제거한다.

### 21.3 보정폭 결정 절차

```text
1. RL action=0 baseline 실행
2. 표면 family와 구역별 오차분포 수집
3. 정상상태 bias, p95, peak 계산
4. baseline이 처리해야 할 구조적 오차 먼저 수정
5. 남은 잔차 p95를 기준으로 action scale 후보 생성
6. 후보별 step response와 안전개입률 비교
7. 최종 scale을 config에 기록
```

policy action이 연속 100% saturation되는 비율도 기록한다. saturation이 높으면 action 범위가
작거나 기준 제어기·observation에 문제가 있다는 신호다.

---

## 22. 보상·종료 계산 상세 명세

### 22.1 항목별 계산

```python
force_cost = huber(
    (measured_force_n - target_force_n) / force_tolerance_n
)

speed_cost = huber(
    (measured_feed_m_s - target_feed_m_s) / speed_tolerance_m_s
)

path_cost = huber(path_error_m / path_tolerance_m)
tilt_cost = huber(tilt_error_rad / tilt_tolerance_rad)
force_rate_cost = clip(abs(force_rate_n_s) / force_rate_limit_n_s, 0, 1)
action_rate_cost = mean(square(action_t - action_t_minus_1))

delta_coverage = max(0, coverage_t - coverage_t_minus_1)
revisit_cost = newly_added_dwell_on_already_complete_cells
```

Huber loss의 delta와 모든 tolerance는 config에 기록한다. 항목별 raw value, normalized cost,
weighted reward를 각각 로그에 남겨 한 항이 전체 보상을 지배하는지 확인한다.

### 22.2 종료 종류

Gymnasium/Isaac Lab 의미를 구분한다.

```text
terminated=True
  작업 성공 또는 물리·안전 실패로 MDP 종결

truncated=True
  최대 step/time 등 외부 제한으로 중단
```

권장 reason code:

```text
success_path_complete
failure_force_hard_limit
failure_force_spike
failure_tool_penetration
failure_nonpad_collision
failure_joint_limit
failure_contact_lost
failure_path_unrecoverable
failure_nan_inf
failure_sensor_invalid
truncated_max_steps
truncated_max_time
invalid_reset
```

reason code는 자유문장 대신 enum으로 저장하고, 부가설명은 별도 message 필드에 둔다.

### 22.3 Reward hacking 검사

다음 정책이 높은 보상을 받지 않는지 반드시 시험한다.

- 표면에 접촉하지 않고 경로만 빠르게 이동
- 같은 쉬운 셀을 반복 방문
- 아주 느리게 움직여 Force 오차만 최소화
- 안전 필터가 모든 명령을 고치도록 과도한 행동 출력
- 종료 직전까지 진행하지 않고 step reward만 수집
- 정상부만 닦고 결함부를 회피

각 편법 정책을 deterministic test로 만들어 reward 총합과 각 항을 회귀시험한다.

---

## 23. 절차적 표면 생성 상세 명세

표면 patch는 analytic function 또는 mesh로 생성하되, 정책 입력은 동일한 국소 인터페이스로
변환한다.

### 23.1 기본 표면식

평면:

```text
z(x,y) = ax + by + c
```

포물형 자유곡면:

```text
z(x,y) = 0.5·k1·x² + 0.5·k2·y² + cross·x·y
```

원통면과 구면은 지정 반경으로 mesh를 만들고 외향 법선을 해석적으로 계산한다. 자유곡면에는
저주파 basis를 더할 수 있으나 패드 크기보다 작은 비물리적 고주파 요철을 무작위로 만들지
않는다.

### 23.2 형상 유효성 검사

생성된 patch는 episode 시작 전에 다음을 통과해야 한다.

- mesh self-intersection 없음
- 법선 연속성 검사 통과
- 패드보다 충분한 경계 여유
- 로봇 workspace 안에 존재
- 기준 경로의 모든 점이 도달 가능
- 경로에서 급격한 normal flip 없음
- 설정된 곡률 단위와 부호 일관성

### 23.3 형상 split manifest

```json
{
  "split_version": "1.0",
  "train": ["surface ids"],
  "validation": ["surface ids"],
  "test": ["surface ids"],
  "generator_version": "generator git hash",
  "parameter_distribution": "path to config",
  "no_surface_id_overlap": true
}
```

실제 차량 mesh의 일부를 test에 넣을 때 train용 절차표면을 그 test mesh에 맞춰 역으로 조정한
뒤 같은 test 결과를 보고하면 leakage다. 정책 선택이 끝난 후 test를 한 번 평가한다.

---

## 24. 학습 설정과 모델 선택 규칙

하이퍼파라미터 숫자는 다른 장비·환경의 값을 복사해 확정하지 않는다. 다음 항목은 반드시
config에 노출한다.

```text
algorithm name/version
policy and value network architecture
activation function
observation normalization
reward scaling
learning rate and schedule
rollout horizon
number of parallel environments
batch/minibatch size
number of update epochs
gamma
GAE lambda
PPO clip epsilon
entropy coefficient
value loss coefficient
gradient norm limit
policy inference frequency
physics frequency
control decimation
total environment steps
checkpoint interval
evaluation interval
```

모델 선택은 train reward 최고점이 아니라 고정 validation seed의 제약 통과율과 핵심 지표로
수행한다.

```text
1차: 안전 실패가 없는 checkpoint
2차: Force/Path/Contact/Coverage 필수조건 통과
3차: 통과 모델 중 작업시간과 제어 안정성이 우수한 모델
```

서로 다른 알고리즘을 비교할 때 환경 step 수, 평가 seed, network parameter 규모와 wall-clock을
함께 보고한다.

---

## 25. 평가 프로토콜

### 25.1 비교 대상

```text
Baseline A: 기존 v5 기준 제어기
Baseline B: Isaac Lab 이식 제어기, RL action=0
Candidate : Isaac Lab 기준 제어기 + RL 잔차 정책
```

Baseline A와 B가 먼저 일치해야 Candidate 성능을 해석할 수 있다.

### 25.2 반복 평가

각 surface family, robot/region, recipe context별로 여러 고정 seed를 사용한다. 보고서에는 단일
최고 영상 대신 전체 반복의 평균, 표준편차, median, p95, worst case와 성공률을 기록한다.

### 25.3 일반화 결과표

| Split | Surface | Recipe ID | Success rate | Force RMSE | Force peak | Path p95 | Contact | Coverage | Time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Validation | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Test synthetic | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Test real mesh | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 25.4 Ablation

다음 구성요소를 하나씩 제거해 효과를 확인한다.

- 곡률 관측 제거
- previous action 제거
- force derivative 보상 제거
- Δcoverage 보상 제거
- domain randomization 제거
- 잔차 RL 전체 제거

정책 개선이 실제로 어느 구성요소에서 왔는지 확인하고 불필요한 복잡성을 제거한다.

---

## 26. 파일·데이터 산출물 규격

권장 구조:

```text
polytwin_rl/
  configs/
    environment.yaml
    robot_top.yaml
    robot_side.yaml
    reward.yaml
    ppo.yaml
    bo_search_space.yaml
    surface_distribution.yaml
  envs/
    local_surface_env.py
    observations.py
    actions.py
    rewards.py
    terminations.py
    safety.py
  surfaces/
    generator.py
    estimators.py
    split_manifest.json
  controllers/
    baseline_adapter.py
    residual_adapter.py
  optimization/
    bo_runner.py
    process_surrogate.py
    recipe_schema.json
  tests/
    test_reset.py
    test_units.py
    test_frames.py
    test_action_zero_baseline.py
    test_reward_hacking.py
    test_termination.py
  outputs/<run_id>/
    resolved_config.json
    observation_schema.json
    action_schema.json
    raw_steps.parquet
    episode_metrics.csv
    termination_summary.json
    checkpoints/
    evaluation_summary.json
    videos/
```

실제 담당자 저장소 구조가 다르면 이름을 강제로 맞출 필요는 없지만, 위 책임을 담당하는 파일과
산출물이 어디에 있는지 `HANDOFF.md`에 매핑해야 한다.

### 26.1 Run ID와 재현성

```text
run_id
git commit
dirty working tree 여부
resolved config hash
surface split version
recipe id
policy checkpoint id
random seeds
Isaac Sim/Isaac Lab/driver/GPU 정보
```

최종 보고서 숫자는 반드시 run_id로 원시 로그까지 추적할 수 있어야 한다.

---

## 27. 다른 AI 또는 담당자의 최종 회신 형식

작업을 넘겨받은 AI/담당자는 완료라고만 답하지 말고 다음 형식으로 회신한다.

```text
1. 현재 구현 상태
   - Isaac Lab environment 경로
   - 사용 알고리즘
   - 구현 완료/미완료 기능

2. 기준 사양
   - action/observation/reward/termination 표
   - 패드와 힘 제한
   - policy/physics frequency

3. 포팅 검증
   - v5 vs action=0 baseline 비교 결과
   - 불일치 항목과 원인

4. 학습 결과
   - run_id/checkpoint
   - train/validation curve
   - 고정 seed 평가표

5. 형상 일반화
   - 사용 surface family
   - split manifest
   - 미학습 synthetic/real mesh 결과

6. 안전성
   - termination reason별 횟수
   - safety intervention rate
   - worst force/path/contact 사례

7. 재현 명령
   - 환경 생성
   - 학습
   - checkpoint 평가
   - 영상 재생

8. 산출물 경로
   - config/log/checkpoint/video/report

9. 미확정 REQUIRED_CONFIG
   - 값
   - 미확정 이유
   - 확정에 필요한 실험
```

이 회신에 원시 로그, resolved config, checkpoint 또는 재현 명령이 빠져 있으면 완료로 판정하지
않는다.

---

## 28. 미확정 설정값 레지스트리

아래 값은 코드에 임시값이 존재하더라도 실물 공정 규격 또는 최종 RL 사양으로 자동 확정하지
않는다. 담당자는 각 항목의 `value`, `source`, `approved_by`, `date`, `config_path`를 채운다.

| 항목 | 현재 문서 상태 | 확정 방법 | 잘못 확정했을 때 영향 |
|---|---|---|---|
| canonical code commit | REQUIRED_CONFIG | 실행 기준 저장소·commit 팀 지정 | 서로 다른 상수로 결과 비교 |
| 고정 패드 직경·높이 | 현재 v5 110/70 mm | 실제 사용 패드와 시뮬 asset 대조 | Step-over·접촉면적 오류 |
| 패드 재질·강성 | REQUIRED_CONFIG | 패드 사양·압축 시험 | 힘-압입 관계 오류 |
| compound | REQUIRED_CONFIG | 실제 제품·도포량 고정 | 제거·열·Gloss 모델 무효 |
| 상단/측면 목표 힘 | 현재 v5 제어값 존재 | baseline·안전·실물 시험 | 접촉손실 또는 과압 |
| 상단/측면 force tolerance | REQUIRED_CONFIG | baseline error와 품질 허용치 | 보상·성공률 왜곡 |
| force derivative limit | REQUIRED_CONFIG | 매 스텝 force 로그 분석 | spike 미검출 또는 과민 종료 |
| target feed와 tolerance | REQUIRED_CONFIG | 경로 제어·공정실험 | 시간·제거량·열 오류 |
| action scale/rate limit | REQUIRED_CONFIG | action=0 잔여오차 p95·step test | saturation·진동 |
| policy frequency | REQUIRED_CONFIG | 주파수 후보 비교 | 불안정 또는 계산 낭비 |
| physics dt/solver | REQUIRED_CONFIG | v5 baseline 재현 시험 | 포팅 결과 불일치 |
| 곡률 train distribution | REQUIRED_CONFIG | 대상 물체 스캔 통계·패드 형상 | 일반화 실패 |
| validation/test surface manifest | REQUIRED_CONFIG | 고정 manifest 생성 | 데이터 누수 |
| reward tolerance/weights | REQUIRED_CONFIG | normalized metric·ablation | 특정 항 지배·게이밍 |
| episode max step/time | REQUIRED_CONFIG | 정상 완료시간 분포 | 조기 truncation 또는 낭비 |
| PPO hyperparameters | REQUIRED_CONFIG | validation 기반 tuning | 미수렴·불안정 |
| BO search space | REQUIRED_CONFIG | 장비·안전·실물 실험 근거 | 위험 후보 생성 |
| BO stopping condition | REQUIRED_CONFIG | 실험예산·개선량·불확실도 | 너무 이른/늦은 종료 |
| Preston/process coefficient | 미보정 | 동일 Clearcoat 실험 | 제거량 예측 무효 |
| 잔존 Clearcoat 안전한계 | 프로젝트 결정 필요 | 초기두께·공정·품질 요구 | burn-through 위험 |
| 온도 안전한계 | 미보정 | 재료·compound·패드 실험 | 열 손상 위험 |
| Removal absolute/relative tolerance | 프로젝트 결정 필요 | 측정기 오차·목표맵 검증 | 무작업 정책 통과 |
| Ra/Rz/잔존 scratch 기준 | 실측 보정 필요 | 동일 height map·품질 요구 | 결함 잔존 |
| 20° GU 보정식·합격기준 | 미보정 | 실제 gloss meter 교차검증 | 상대값을 GU로 오인 |

예시 레지스트리 파일:

```json
{
  "name": "policy_frequency_hz",
  "value": "REQUIRED_CONFIG",
  "unit": "Hz",
  "status": "unresolved",
  "source": null,
  "approved_by": null,
  "approved_date": null,
  "evidence_run_ids": [],
  "config_path": "configs/ppo.yaml"
}
```

---

## 29. 구현 순서와 완료 게이트

### Gate 1: 기준 소스와 단위 확정

- canonical commit 기록
- pad/robot/surface asset scale 확인
- 모든 공정변수 단위 명시
- 실행 mode와 Step-over 정의 확정
- resolved config 저장

통과 증거: `resolved_config.json`, 단위시험 결과, git commit.

### Gate 2: Isaac Lab 환경 이식

- observation/action 없이 기존 기준 제어기 실행
- reset 재현성 확보
- collision/contact/safety event 연결
- 매 스텝 raw logger 연결

통과 증거: v5와 action=0 baseline 비교표 및 동일 seed 영상.

### Gate 3: 평면 잔차 RL

- 기존 잔차 action 사양 확인
- action scale을 baseline 잔여오차로 결정
- 단계별 reward mask 적용
- reward hacking 회귀시험 통과

통과 증거: baseline 대비 고정 seed Force/Path/Contact 지표.

### Gate 4: 절차곡면 일반화

- surface generator와 split manifest 생성
- 국소 좌표·법선·주곡률 observation 연결
- 볼록/오목/곡률 전이 curriculum
- validation checkpoint 선택

통과 증거: 미학습 synthetic test 결과표와 worst-case 영상.

### Gate 5: 실제 mesh 일반화

- 새 mesh/point cloud 스캔
- 작업영역 분할과 경로 생성
- LocalSurfaceContext 변환
- planner/safety/RL 전체 연결

통과 증거: 학습에 사용하지 않은 실제 mesh 패널의 지표·로그·영상.

### Gate 6: BO와 recipe 연결

- 캘리브레이션 데이터 schema 검증
- BO search space와 hard constraint 확정
- recipe JSON export/import
- PPO episode에 recipe_id 연결

통과 증거: 동일 recipe를 action=0 baseline과 PPO가 각각 수행한 비교 결과.

### Gate 7: 품질 모델과 최종 평가

- removal/height/Ra/Rz 모델 독립 검증
- 정상부 과다제거와 온도 안전조건 적용
- 실제 GU 보정 완료 후 Gloss 연결
- BO 추천 recipe 실제 시편 재검증

통과 증거: `recipe_id + policy_checkpoint_id + specimen_id`로 추적 가능한 최종 품질 보고서.

각 Gate는 이전 Gate의 증거가 남아야 진행한다. 단순히 시뮬레이션 화면이 움직이거나 train
reward가 증가한 것만으로 완료 처리하지 않는다.
