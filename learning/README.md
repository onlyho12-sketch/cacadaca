# PolyTwin 학습 파이프라인 (learning/)

기존 시뮬레이션 워크스페이스(`scripts/`)와 분리된 학습 전용 공간.
설계 근거는 프로젝트 루트의 `PolyTwin_학습모델_설계_요약.md` 참고.

## 구조

```
learning/
├── config.py            # 경로/데이터 정의/하이퍼파라미터 (전 스크립트 공통)
├── bc/                  # 1단계: 모방학습 (Behavior Cloning)
│   ├── extract_dataset.py   # v5 로그 → (state, action) 데이터셋
│   ├── model.py             # MLP 정책 + 정규화기
│   ├── train.py             # 지도학습
│   └── evaluate.py          # 예측 vs 정답 시각화
├── data/processed/      # bc_dataset.npz (생성물)
├── checkpoints/         # bc_mlp.pt (생성물)
└── outputs/             # 평가 그래프 (생성물)
```

RL 잔차(PPO) 단계는 추후 `learning/rl/`로 추가 예정.

## 실행 (반드시 isaacsim_venv 파이썬 사용)

```bash
PY=~/isaacsim_venv/bin/python      # torch 2.7 + CUDA 포함
$PY learning/bc/extract_dataset.py # ① 데이터셋 추출
$PY learning/bc/train.py           # ② 학습
$PY learning/bc/evaluate.py        # ③ 평가 → outputs/bc_eval.png
```

## 데이터 정의

- **선생님(teacher)** = 기존 규칙 컨트롤러 + 물리엔진이 실제로 낸 결과
  (숙련공 데이터 미확보 → 규칙 시연 로그로 부트스트랩, 실력 개선은 이후 RL 잔차 담당)
- **상태(12)**: 표면 위치(3) · 법선(3) · 기울기 · 곡률 · 경로 진행률 · 작업면(top/side) ·
  단계(현재 0 고정) · **직전 스텝 실측 힘 filtered(t-1)** (누수 방지 1스텝 지연)
- **행동(2)**: 접촉력 [N] = 로그 실측 `filtered`(t)를 궤적별 스무딩(w=21≈0.35s)한 값,
  이송속도 [m/s] = path_idx 진행량 실측 역산 후 동일 스무딩. 회전속도는 고정이라 제외.
- 스텝 단위 힘/속도 요동은 물리 노이즈라 학습 불가 → 라벨은 부드러운 프로파일(a_base),
  순간 보정은 RL 잔차 담당. (config.LABEL_SMOOTH_WINDOW 주석 참고)
- 로그의 (seg, path_idx)를 좌표로 되돌릴 때 agent.py와 동일하게
  `CAR_LIFT_Z` 보정 + `filter_safe_waypoints` 필터링을 재현해야 seg 인덱스가 맞음 (extract_dataset.py 참고).

## 알아둘 것

- 이송속도는 이중 모드: 접촉력 정상 범위면 고속 전진(~500mm/s), 벗어나면 크리프(~30mm/s).
  prev_force 입력이 이 모드 전환의 단서다.
- train/val 분할은 (rail, seg) 그룹 단위 — 같은 구간이 양쪽에 섞이는 누수 방지.
- SL 레일 로그는 현재 비어 있음(헤더만). 시뮬레이션 재실행으로 로그가 쌓이면 자동 포함됨.
- 시뮬을 여러 번 돌려 로그를 누적하려면 실행 후 `scripts/force_log_rail_*.csv`를
  `learning/data/raw/<날짜>/`로 복사 — extract가 자동으로 전부 읽는다 (동일 파일은 중복 제거).
