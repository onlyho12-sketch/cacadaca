# Isaac Lab 잔차 RL 착수 가이드 — cacadaca

> 2026-08-26 작성. 선행 문서: `learning/HANDOFF.md`(BC 구현 현황), `learning/PLAN_BC_V2.md` 5~7장(RL 설계),
> `learning/handoff/README.md`(BC 인계 계약).
> 이 문서는 **그 설계를 이 머신의 Isaac Lab에 실제로 올리는 절차**만 다룬다.

---

## 0. 30초 요약

- 이 머신(`/home/eon`)의 Isaac Lab 3.0.0 + Isaac Sim 스택은 **동작 확인 완료**. 다만 실행 전
  **venv + setup_python_env.sh 를 둘 다 source** 해야 한다 (1장). 이걸 빼면 torch 없는 python이 잡힌다.
- BC 체크포인트(`learning/handoff/bc_mlp.pt`)는 이 venv의 **torch 2.10 + CUDA에서 로드·배치추론 검증 완료**.
  파라미터 동결도 확인했다 (1.3장).
- **착수 순서의 첫 단추는 접촉 모델 결정이다.** PLAN 7-8은 B안(재학습)을 권했지만,
  이 환경 조건에서는 **A안(가상 스프링 이식)을 권장**한다 — 근거는 3장. 이 결정이 바뀌면 뒤가 전부 바뀐다.
- GPU가 **RTX 5060 Laptop 8GB**다. PLAN의 "병렬 512~2048 env"는 이 머신에서 비현실적이다.
  **128~512 env(직접 환경, 렌더링 off)** 로 잡고 시작할 것 (1.2장).
- 마일스톤은 M0~M5, 각 단계에 **통과/실패 판정 기준**을 붙였다 (6장). M1(잔차=0 동등성)을 통과하기 전에는
  PPO를 돌리지 말 것.

---

## 1. 실행 환경 — 이 머신에서 검증된 사실

### 1.1 활성화 레시피 (이대로 해야 한다)

```bash
source ~/IsaacLab/.venv/bin/activate
source ~/isaacsim/setup_python_env.sh
```

**왜 두 줄인가:**
- `~/IsaacLab/isaaclab.sh`는 python 을 `$VIRTUAL_ENV` → `$CONDA_PREFIX` → `env_isaaclab/` → `_isaac_sim/python.sh`
  순으로 찾는다. **`.venv`는 이 목록에 없다.** 따라서 그냥 `./isaaclab.sh -p` 하면
  `_isaac_sim/python.sh`(kit 내장 python 3.12, **torch 없음**)로 떨어져서 `ModuleNotFoundError: torch`가 난다.
- `isaaclab.sh`는 Isaac Sim 환경변수를 `_isaac_sim/setup_conda_env.sh`에서 가져오려 하는데,
  설치된 Isaac Sim에는 그 파일이 없고 **`setup_python_env.sh`만 있다**
  (실행 시 `[WARNING] setup_conda_env.sh is missing` 경고가 이것이다).
  이 줄이 없으면 `import isaacsim`이 실패한다.

검증:
```
활성화 후 → isaacsim OK / torch 2.10.0+cu128 / AppLauncher OK
```

### 1.2 하드웨어와 병렬 규모

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop, **8151 MiB**, sm_120 |
| driver / CUDA | 580.173.02 / torch cu128, `torch.cuda.is_available()` = True |
| Warp | 1.13.0, cuda:0 인식 |

**⚠ PLAN 5장의 "병렬 512~2048"은 이 머신 기준이 아니다.** 8GB에서 렌더링까지 켜면 수백 env도 빠듯하다.
- 시작값: **`--num_envs 128`, `--headless`**
- 학습 안정화 후 256 → 512로 올리며 VRAM 확인
- 접촉 모델을 A안(해석식)으로 가면 PhysX 접촉 페어가 없어 메모리가 훨씬 가볍다 — 3장이 규모에도 유리하다

### 1.3 스택 동작 확인 (실제 실행함)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 --headless --num_envs 32 --max_iterations 3
# → exit 0, Training time 2.17s, Iteration time 0.23s
```

설치된 RL 라이브러리: **rsl_rl / skrl 2.1.0 / rl_games / stable_baselines3 2.9.0** 전부 사용 가능.
백엔드는 `isaaclab_newton 0.13.6`, `isaaclab_physx 1.1.3` 둘 다 설치돼 있고 위 스모크 테스트는 PhysX로 돌았다.

**BC 정책 로드 검증:**
```
BCPolicy(in=9, out=2, device=cuda, val_loss=0.2358)
(256, 9) → (256, 2) 배치 추론 정상, requires_grad = False (동결 확인)
```
→ `bc_policy.py`는 수정 없이 이 venv에서 그대로 쓸 수 있다.

**미설치:** `sklearn`. 곡률 PCA를 사전계산할 때 필요하면 넣거나, `numpy.linalg.eigh`로 대체할 것
(어차피 사전계산이라 성능 문제 없음).

### 1.4 알고리즘 선택

**rsl_rl 권장.** 이유: Isaac Lab이 1급으로 지원하고, 스모크 테스트가 통과한 경로이며,
PLAN 5장의 PPO 하이퍼파라미터(lr 3e-4, clip 0.2)를 그대로 얹을 수 있다.
skrl은 잔차 구조를 커스터마이즈하기 편하지만 지금은 검증된 경로를 먼저 쓰는 게 낫다.

---

## 2. 착수 전 게이트 — 순서가 PLAN 7-9와 다르다

PLAN 7-9는 *"새 환경에서 규칙 컨트롤러를 돌려 접촉력을 모아 BC 전이 가능성을 확인하라"* 고 되어 있다.
**그런데 새 환경에는 아직 규칙 컨트롤러가 없다.** 그러므로 실제 순서는:

```
[3장] 접촉 모델 A/B 결정  →  A면 전이가 구성상 보장됨 (검사는 M1에서 자동으로 됨)
                          →  B면 새 환경 로그 수집 → BC 재학습이 M2 앞에 추가로 붙음
```

A안을 택하면 `policy.check_transfer()`는 M1 단계에서 **가상 스프링 이식이 제대로 됐는지 검산하는 용도**로
쓰인다 (합격 기준: `compatible == True`, 즉 평균 힘 배율 0.4~2.5배 & 학습범위 0.27~6.92N 안 비율 > 50%).

---

## 3. 핵심 결정 #1 — 접촉 모델은 A안(가상 스프링 이식)을 권장

### 결론
PLAN 7-8은 **B안(새 환경 로그로 BC 재학습)** 을 권했다. 그 판단은 "이식이 비싸다"는 전제 위에 있었다.
**이 프로젝트 조건에서는 그 전제가 성립하지 않으므로 A안을 권장한다.**

### 근거

**(1) 이식할 것이 사실상 수식 두 줄이다.** 원 시뮬은 이미 강체 충돌을 끈 상태다
(`USE_PHYSICAL_CONTACT_SENSOR = False`) — 즉 힘은 물리엔진이 아니라 해석식이 만든다:

```python
# common.py: VIRTUAL_PAD_STIFFNESS = 350.0,  VIRTUAL_PAD_DAMPING = 35.0
pad_compression      = max(0.0, contact_dist - actual_clearance)
command_virtual_force = clip(350.0 * pad_compression - 35.0 * z_vel, 0.0, soft_limit)

# 어드미턴스 (ADMITTANCE_MASS=1.0, ADMITTANCE_DAMPING=50.0, ADMITTANCE_MAX_VEL=0.02)
accel     = (control_force - target_force - 50.0 * z_vel) / 1.0
z_vel    += accel * dt;  z_vel = clip(z_vel, ±0.02)
z_offset += z_vel * dt;  z_offset = clip(z_offset, press_min=0.012, press_max=0.080)
filtered  = 0.55 * raw + 0.45 * filtered_prev      # FORCE_FILTER_ALPHA = 0.55
```
이건 **완전히 벡터화 가능한 텐서 연산**이다. `(num_envs,)` 상태 3개(`z_offset`, `z_vel`, `filtered`)만 들고 있으면 된다.

**(2) 병렬화에 압도적으로 유리하다.** 강체 접촉을 쓰면 env마다 패드-차체 접촉 페어가 생겨
8GB에서 env 수가 급격히 제한되고, 60Hz 연마 접촉은 수치적으로도 불안정하다
(원 프로젝트가 "슬램 제거"를 이유로 충돌을 끈 이력이 그 증거다). 해석식은 그 비용이 0이다.

**(3) BC가 즉시 유효해진다.** 힘 분포(0.27~6.92N, 평균 3.12N)가 구성상 보존되므로
`prev_force` 정규화 통계가 어긋나지 않고, "BC 재학습 반나절"이 통째로 사라진다.

**(4) 배포 계약 검증(HANDOFF 3.5장)이 같은 환경에서 가능해진다.**
`z_offset` 포화율 65.7% 대비 개선 여부는 **동일한 압입 한계(`press_min/max`)** 아래에서만 비교 의미가 있다.
B안으로 물리를 갈아치우면 이 검증 항목 자체가 무의미해진다.

### A안의 한계 — 정직하게 표현할 것
가상 스프링은 **모델이지 물리 검증이 아니다.** 대외 표현은 여전히
*"규칙 기반 시연으로 부트스트랩되고, 해석적 접촉 모델 위에서 학습된 정책"* 이다.
실물/강체 접촉 검증은 별도 단계로 남는다 (9장 미결정 항목).

### B안으로 가야 하는 조건
Isaac Lab의 강체·순응 접촉을 **반드시 써야 하는 요구가 새로 생긴 경우**에만.
그때는 PLAN 7-8 B안대로 `extract_dataset.py → train.py → evaluate.py`를 그대로 재사용해 재학습한다
(파이프라인은 환경 비의존이다).

---

## 4. 환경 설계 — 구현 형태

### 4.1 관측 (10개) — BC 입력 9개 + 잔여량

PLAN 7-2 그대로. **순서를 절대 바꾸지 말 것** (앞 9개는 `bc_policy.STATE_COLUMNS`와 1:1이어야 한다):

```
0-2  normal_x, normal_y, normal_z
3    tilt_deg            = degrees(acos(clip(normal_z, -1, 1)))
4    curvature           = 국소 PCA λ_min / Σλ   ← 반드시 점군 KDTree PCA 방식 유지
5    progress            = 구간 내 경로 진행률 [0,1]
6    is_side             = (tilt_deg > 45.0)     ← 로봇 1대 환경 대체 정의 (PLAN 7-1 확정)
7    phase               = 0.0 고정
8    prev_force          = 직전 스텝 filtered 접촉력 [N]
9    remaining_removal   = 목표 − 누적 제거량 (패드 발자국 내 평균)   ← 잔차 전용
```

### 4.2 ★ 사전계산이 성능의 전부다 (KDTree 금지)

PLAN 7-9의 경고를 구체화한다. **런타임에 KDTree 쿼리를 돌리면 안 된다.** 경로는 주어지므로
(에이전트는 힘·속도만 조절, PLAN 7-1) **웨이포인트마다 필요한 값을 전부 미리 계산해 텐서로 들고 있는다:**

| 사전계산 텐서 | shape | 내용 |
|---|---|---|
| `wp_normal` | `(W, 3)` | 웨이포인트별 표면 법선 |
| `wp_curvature` | `(W,)` | 국소 PCA 곡률 |
| `wp_footprint_idx` | `(W, P)` | 패드 반경 안 점군 인덱스 (padding = -1) |
| `wp_footprint_mask` | `(W, P)` | 유효 마스크 |
| `wp_seg_id`, `wp_progress` | `(W,)` | 구간 소속 / 진행률 |

런타임의 발자국 조회는 **KDTree 쿼리가 아니라 gather 한 번**이 된다. 이 하나로 env 수 제약이 사라진다.

> 생성 스크립트는 `learning/bc/extract_dataset.py`의 법선·곡률 계산부와
> `build_segments`(빈 구간 스킵 규칙)를 재사용할 것. **seg 번호 규칙이 다르면 뒤가 전부 어긋난다**
> (HANDOFF 6장 함정 (2)).

### 4.3 Preston 제거 맵 (GPU)

PLAN 6장 그대로, `scatter_add_`로 벡터화:

```python
# K_PRESTON = 1.0 (임의 단위), V_SPIN = ω·(2R/3) — 하나 골라 고정하고 주석에 근거를 남길 것
pressure = force / pad_area                       # (E,)
delta    = K_PRESTON * pressure * V_SPIN * dt     # (E,)
idx      = wp_footprint_idx[cur_wp]               # (E, P)
removal.scatter_add_(1, idx.clamp(min=0), delta.unsqueeze(1) * mask)
```

**⚠ 착수 체크리스트 (PLAN 6-8)에서 반드시 먼저 정할 것:**
접촉 반경이 `POLISH_MARK_RADIUS = 0.038`과 `POLISHING_DISK_RADIUS = 0.055`로 **불일치한다.**
`P = F/A`의 `A`와 발자국 반경을 **같은 값으로** 맞추고 결정 근거를 코드 주석에 남길 것.

### 4.4 ★ BC는 환경 안에서 호출한다 (중요)

```python
# _pre_physics_step 내부
a_base = self.bc(obs[:, :9])                     # 동결, no_grad, raw 관측
delta  = torch.tanh(action) * self.delta_bound   # 정책 출력 = 잔차만
cmd    = a_base + delta                          # (target_force, feed_speed)
```

**왜 환경 안인가:** rsl_rl의 `empirical_normalization`이 켜지면 정책이 보는 관측은 러닝 정규화된 값이다.
BC는 **자체 학습 통계로 내부 정규화**하므로 raw 관측을 받아야 한다.
BC를 환경 밖(정책 네트워크 안)에 두면 이 두 정규화가 충돌해 조용히 망가진다.
행동공간을 잔차만으로 두면 이 문제가 원천적으로 없어진다.

잔차 bound (PLAN 7-4): **Δ힘 ±2.0 N / Δ속도 ±270 mm/s**. 속도 쪽은 과대 가능 — 튜닝 대상.

### 4.5 보상 (PLAN 7-5)

```
step:  -w1·(이번 제거량 − 이번 목표량)²
       -w2·max(0, 초과)²        ← 과연마
       -w3·max(0, 미달)²
       -w4·|Δa|                 ← 잔차 페널티. 없으면 잔차가 BC를 무시한다. 필수.
end:   -w6·std(최종 잔여량)
```
**`w2 ≫ w3` 비대칭이 핵심이다** — 덜 깎으면 한 번 더 돌면 되지만 더 깎으면 되돌릴 수 없다.
초기 가중치는 **운영점에서 각 항의 크기가 비슷해지도록 정규화한 뒤** 튜닝할 것 (PLAN 7-10).

### 4.6 에피소드 (PLAN 7-6)

```
1 에피소드 = 한 구간, 최대 3패스
종료: 전 지점 목표 도달(성공) / 3패스 초과 / 스텝 예산 초과(감점) / 힘 허용치 초과(즉시 종료 + 큰 감점)
리셋: 제거량 맵 초기화 + 목표 맵 랜덤 재생성 + 시작점 복귀
예산: 구간당 7.6s × 3패스 × 1.5 ≈ 34s (= 60Hz 기준 약 2040 스텝)
```
**목표 맵은 가우시안 블롭 랜덤 필드**로 매 에피소드 재생성한다.
**기하(곡률·기울기) 기반 생성은 금지** — 정책이 잔여량 대신 곡률을 보고 추측하게 되며,
이는 v1에서 `pos_z`를 목발 삼아 좌표를 외웠던 것과 **정확히 같은 실패 패턴**이다 (PLAN 7-3).

리셋 시 **구간도 랜덤으로 바꿀 것** (38구간 확보). env마다 다른 구간을 배정해야 다양성이 나온다.

---

## 5. 리포지토리 배치

### 원칙 — `scripts/`는 불가침
`learning/` 도입 때 지킨 원칙을 그대로 유지한다. **`scripts/` 아래는 읽기만 한다.**
가상 스프링 상수도 수정이 아니라 **복사 + 출처 주석**으로 가져올 것 (원본이 바뀌면 대조할 수 있게).

### 권장 구조
```
learning/rl/                      # PLAN에서 예고한 위치
├── env/
│   ├── polish_env.py             # DirectRLEnv 서브클래스
│   ├── polish_env_cfg.py         # 설정
│   ├── contact.py                # 가상 스프링 + 어드미턴스 (scripts/ 상수 복사, 출처 명시)
│   ├── removal.py                # Preston 제거 맵
│   └── precompute.py             # 4.2 텐서 생성 (extract_dataset.py 로직 재사용)
├── bc_policy.py                  # handoff/ 에서 복사 (수정 금지)
├── bc_mlp.pt
└── train_rsl_rl.py
```

Isaac Lab 3.0은 외부 태스크 프로젝트 템플릿을 만들어 준다 — 뼈대는 이걸로 뽑고 내용만 채우는 게 빠르다:
```bash
./isaaclab.sh --new       # (-n) 외부 태스크 프로젝트 생성
```
**API 시그니처는 반드시 생성된 템플릿을 기준으로 맞출 것.** 3.0은 Newton/PhysX 백엔드 분리 등
이전 버전과 달라진 부분이 있어, 웹의 오래된 예제를 그대로 옮기면 어긋난다.

---

## 6. 마일스톤 — 각 단계에 판정 기준이 있다

### M0. 환경 부팅
- [ ] 1.1 활성화 레시피로 빈 `PolishEnv`가 `--headless --num_envs 4`로 뜬다
- **판정:** reset/step이 예외 없이 100스텝 돈다

### M1. ★ 잔차 = 0 동등성 (여기를 통과하기 전 PPO 금지)
잔차를 0으로 고정하고 BC만으로 주행한다. 이 단계가 전체에서 가장 중요하다 —
여기서 걸러지지 않은 오류는 PPO 로그 안에 숨어서 며칠을 잡아먹는다.
- [ ] 접촉력 분포를 모아 `policy.check_transfer(forces)` 실행 → **`compatible == True`**
- [ ] BC 출력에 음수·발산 없음, 힘이 `PHYSICAL_FORCE_SOFT_LIMIT`(top 14N / side 6N) 안
- [ ] `curvature` 값 분포가 학습 데이터와 같은 자릿수 (평면 ≈ 0)
- **판정:** 위 3개 전부 통과. 실패 시 4.2 사전계산이나 3장 이식을 의심할 것 (PPO 탓이 아니다)

### M2. 배포 계약 검증 (HANDOFF 3.5장 숙제)
- [ ] BC 목표힘 투입 시 `z_offset` 포화율이 **65.7%보다 확실히 낮은가**
- [ ] 힘 추종 오차(목표 vs 실측)가 규칙 컨트롤러보다 작은가
- **판정:** 포화가 남으면 `PRESS_OFFSET_MAX` / 패드 강성 재검토 (계약 자체를 되돌리기 전에 이것부터)

### M3. 규칙 baseline 재측정
새 환경에서 규칙 컨트롤러(또는 BC 단독)로 1회 주행해 제거 균일도를 잰다.
- [ ] `p95/p5` 비율 산출 — 원 환경 값 **23.2배**와 같은 자릿수인가
- [ ] 미달 비율(원: 50%) / 과연마 비율(원: 34%) 산출
- **판정:** 이 숫자가 곧 "RL이 넘어야 할 선"이다. 없으면 개선 주장을 정량화할 수 없다

### M4. PPO 소규모 학습
- [ ] `--num_envs 128 --headless`, rsl_rl PPO (lr 3e-4, clip 0.2)
- [ ] 보상 항별 로그를 전부 남길 것 (w1~w6 중 하나가 지배하는지 확인)
- **판정:** 잔차 크기가 bound에 계속 붙어 있으면 → w4 부족 또는 bound 과소
       / 잔차가 0 근처에서 안 움직이면 → w4 과다

### M5. 규모 확대 + 비교
- [ ] 256 → 512 env, VRAM 모니터링
- [ ] M3 baseline 대비 **균일도 개선률 / 과연마 감소율**로 결과 표현
- **⚠ 절대 수치(Ra µm 등)는 주장하지 말 것** — `k=1.0`이 임의 단위이기 때문 (PLAN 6-3)

---

## 7. 함정 목록 (이 프로젝트 고유)

1. **활성화 두 줄** — 1.1. 가장 흔한 첫 삽질 지점이다.
2. **런타임 KDTree** — 4.2. 단일 env에선 안 보이다가 병렬에서 터진다.
3. **`is_side`는 `tilt_deg > 45`** — 로봇 1대 환경엔 측면/천장 구분이 없다. BC 재학습은 불필요.
4. **`phase`는 항상 0.0** — 죽은 입력이지만 자리를 채워야 차원이 맞는다.
5. **BC 입력 순서 고정** — `policy.state_columns`로 검증할 것. `bc_policy.py`가 순서 불일치 시 예외를 던진다.
6. **BC 정규화 이중 적용 금지** — 밖에서 따로 정규화하지 말 것. 4.4 참고.
7. **목표 맵을 기하와 상관시키지 말 것** — 4.6. v1 좌표 암기와 같은 실패 패턴.
8. **`w4`(잔차 페널티) 생략 금지** — 없으면 잔차가 BC를 무시하고 커져 "잔차 RL"이 아니게 된다.
9. **접촉 반경 불일치** — 4.3. `0.038` vs `0.055`를 먼저 정리하고 시작할 것.
10. **`seg` 인덱스 규칙** — 빈 구간을 건너뛴 뒤의 번호다. `build_segments`와 동일하게 재구성할 것.
11. **이송속도 이상치(최대 0.92 m/s)는 진짜 데이터다** — `POLISH_SPEED_SCALE = 3.0` 설계값. 필터링 금지.
12. **`scripts/` 수정 금지** — 5장.
13. **로그 CSV는 git에 있어 `git checkout`으로 덮어써진다** — 실행 후 즉시
    `learning/data/raw/<날짜>/`로 복사할 것 (HANDOFF 4장의 SL 유실 사고 원인).

---

## 8. 비교 기준 숫자 (원 환경 실측, 1737샘플 / 38구간)

| 항목 | 값 | 용도 |
|---|---|---|
| 제거량 `p95/p5` | **23.2배** | 균일도 개선 baseline |
| 1회 주행 미달 지점 | **50%** | 패스 설계 근거 |
| 1회 주행 과연마(2배↑) | **34%** | 비대칭 보상 근거 |
| 접촉력 분포 | 0.27~6.92 N (μ 3.12) | 전이 판정 |
| 압력 분포 | 59~1526 Pa (μ 688) | Preston 입력 |
| `z_offset` 포화율 | **65.7%** | M2 판정 기준 |
| 구간 1회 주행 시간 | 중앙값 7.6s (0.8~276) | 스텝 예산 |
| 구간당 스텝 | 중앙값 28 (4~184) | 에피소드 길이 |

---

## 9. 아직 정해지지 않은 것 (착수 전에 팀이 답해야 함)

- [ ] **접촉 모델 A/B 최종 확정** (3장). 이 문서는 A안을 권장하지만 결정은 팀 몫이다.
- [ ] 접촉 반경 `0.038` vs `0.055` (4.3)
- [ ] 목표 제거량 절대값 — 현 데이터 중앙값 **828**(임의 단위) 권장
- [ ] `V_SPIN` 정의 — 반경 중간(5.97 m/s) vs 면적가중(7.96 m/s). `k`에 흡수되므로 아무거나, **단 문서화**
- [ ] `w1~w6` 초기 가중치 (4.5)
- [ ] 차 1대 허용 시간 — 산업 기준이 없으면 "규칙 × 1.5" 기본값
- [ ] 잔차 속도 bound ±270 mm/s 가 적정한가 (과대 의심)
