# Gate A 결과 동결 및 기초 회귀 기록

- 생성 시각: 2026-08-30 23:06:30 KST
- 저장소: `/home/rokey/cacadaca`
- 브랜치: `lr_yj`
- HEAD: `1059b31b6c79e3ffb889bcc4e3d27fe8afa82fa7`
- origin/lr_yj: `1059b31b6c79e3ffb889bcc4e3d27fe8afa82fa7`
- 상태: **결과 동결 완료 / 재부팅 후 PhysX 1환경 회귀 PASS**

## 1. 동결 완료 항목

기존 파일은 이동·삭제·덮어쓰기하지 않았다. 현재 dirty 파일과 미완료 finish-grid CSV는
`snapshot/` 아래에 원래 상대경로를 유지한 복사본으로 보존했다.

- `git_status_before.txt`: Gate A 시작 전 Git 상태
- `working_tree_before.patch`: 추적 파일의 binary-safe diff
- `untracked_before.txt`: Gate A 시작 전 미추적 파일 목록
- `git_head.txt`, `git_origin_lr_yj.txt`: 로컬/원격 기준 커밋
- `snapshot/`: dirty 코드·문서·finish-grid CSV 12개 복사본

## 2. SHA-256 manifest

| Manifest | 범위 | 항목 수 | Manifest 자체 SHA-256 |
|---|---|---:|---|
| `checkpoints_csv_ppt_sha256.txt` | Gate A 폴더를 제외한 기존 checkpoint·CSV·PPT 후보 | 634 | `4b485a622fd7de350606c4283c33e23f05347977f0aa5b9d2239a9425bf7b85f` |
| `gloss_results_sha256.txt` | `gloss_test/results/` 전체 | 6,323 | `02ff122c17d54a6d40c1cf97cdac23a1d0c2e3b1fae56747f301507ed28a5ac4` |
| `dirty_snapshot_sha256.txt` | `snapshot/` 복사본 | 12 | `e7f10f42e8d2da9bf06ddcce0776eef2385f6a4e62263eaeef0c782c251f6ed3` |

생성 직후 세 manifest 모두 `sha256sum -c --quiet`로 재검증해 PASS했다.
`working_tree_before.patch`의 SHA-256은
`6abe5139eaaa851f2a7dee08b2588a21bf4abf4dd7abe2806ea7a12e58a6fd60`이다.

## 3. PhysX 1환경 회귀

### 3.1 최초 시도와 GPU 복구

기존 평면 챔피언을 12.7 mm/s, 물리 접촉, 1환경·1표면·최대 1 pass로 실행하려 했다.
결과 CSV는 이 디렉터리의 새 이름만 사용하도록 지정했다.

실행은 환경이나 정책 생성 전에 CUDA 초기화에서 중단됐다.

```text
IOptionalCuda: cuInit failed (CUresult 999) -- treating as CPU-only
RuntimeError: CUDA unknown error
```

후속 읽기 전용 진단 결과:

- `nvidia-smi`는 GPU를 표시하지만 Isaac Sim Python의 `torch.cuda.is_available()`은 `False`
- `torch.cuda.device_count()`는 1
- `nvidia-smi -q`의 `GPU Recovery Action`은 `Reboot`
- 회귀 결과·pass·tile CSV는 생성되지 않음
- 기존 체크포인트와 결과 파일은 변경되지 않음

재부팅은 Gate A 승인 범위를 넘는 시스템 변경이므로 수행하지 않았고, 같은 명령을 반복하지
않았다. 시도 결과는 `regression_attempts.csv`에 기록했다.

사용자가 재부팅한 뒤 다음 상태를 확인했다.

- `GPU Recovery Action: None`
- Isaac Sim Python `torch.cuda.is_available() == True`
- CUDA 장치: `NVIDIA GeForce RTX 5080 Laptop GPU`

### 3.2 재실행 결과

같은 명령을 재실행해 다음 새 파일을 생성했다.

- `regression_champion_1env.csv`: 시퀀스 결과 1행
- `regression_champion_1env_passes.csv`: pass 결과 1행
- `regression_champion_1env_tiles.csv`: 5×5 타일 결과 25행
- `regression_comparison.csv`: 기존 100환경 결과의 env 0/pass 1과 수치 대조
- `regression_outputs_sha256.txt`: 위 회귀 CSV와 시도 기록 5개의 SHA-256 manifest

| 항목 | Gate A 결과 |
|---|---:|
| 접촉 모드 | PhysX physical |
| 평균 명령 힘 | 7.4473 N |
| 평균 사용 힘 | 7.2806 N |
| 최대 사용 힘 | 8.6940 N |
| 14 N 초과 | 0 |
| sensor fallback | 0 |
| 평균 이송속도 | 6.6148 mm/s |
| GU proxy | 44.05 → 61.98 |
| Ra | 0.2087 μm |
| Rz | 1.9972 μm |
| 잔여 scratch | 1.3154 μm |
| 최소 clearcoat | 37.9401 μm |
| 최고온도 | 39.1502 °C |
| safety_ok | True |

종료 사유 `fail_max_passes`는 힘·온도·clearcoat 안전 실패가 아니다. Gate A가 의도적으로
`max_passes=1`을 사용했고 이 표면이 1 pass 후 GU 70 및 Ra 0.20 μm 동시 목표에 미달했기
때문이다. Rz는 2.0 μm 기준을 통과했다.

기존 `repolish_tesla12p7_bc_champion_100env_passes.csv`의 동일 env 0/sequence 0/pass 1과
비교한 12개 지표가 모두 지정 절대 허용차 안에 들었다. 예를 들어 평균 사용 힘 차이는
`0.0000076 N`, GU 차이는 `0.000145`, Ra 차이는 `0.0000018 μm`다. 따라서 물리 접촉,
정책, 품질·열 모델의 단일환경 회귀를 PASS로 판정한다.

`regression_outputs_sha256.txt` 자체의 SHA-256은
`2b177aa116e9894d4e7b2c2c857174003e39a1015521de9f1e533731764960b8`이며 생성 직후
`sha256sum -c --quiet` 재검증을 통과했다.

## 4. GPU 복구 후 재개 명령

재부팅 후 먼저 아래 두 조건을 확인한다.

```bash
nvidia-smi -q | rg -n "GPU Recovery Action"
/home/rokey/isaacsim-6.0.1/python.sh -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

CUDA가 정상일 때만 다음 회귀를 다시 실행한다.

```bash
/home/rokey/isaacsim-6.0.1/python.sh learning/rl/repolish_eval.py --headless \
  --checkpoint learning/rl/robot/champion/model_bc_robot_tesla12p7.pt \
  --policy_mode checkpoint --num_envs 1 --num_sequences 1 --max_passes 1 \
  --cooldown_s 20 --feed_speed_mm_s 12.7 \
  --out learning/rl/robot/results/gate_a_freeze_20260830_230630/regression_champion_1env.csv \
  --pass_out learning/rl/robot/results/gate_a_freeze_20260830_230630/regression_champion_1env_passes.csv \
  --tile_out learning/rl/robot/results/gate_a_freeze_20260830_230630/regression_champion_1env_tiles.csv
```

Gate B의 큰 연속 평판·중앙 ROI 구현은 별도 승인 후 시작한다.

## 5. 최종 보존 확인

- `snapshot/`의 12개 복사본과 원본을 byte 단위로 비교: PASS
- checkpoint·CSV manifest 재검증: PASS
- `gloss_test/results/` 6,323개 manifest 재검증: PASS
- 보호 파일 4개 범위의 Git diff: 없음
- `git_status_before.txt`와 `git_status_after.txt`: 동일
- 종료 시 실행 중인 Isaac Sim·학습·평가 프로세스: 없음
- 재부팅 후 PhysX 1환경 회귀 및 기존 env 0/pass 1 비교: PASS
