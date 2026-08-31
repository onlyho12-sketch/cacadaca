# Gate E1 BC 교사 정책 paired screen

## 판정

- Gate E1 물리 파일럿은 완료했다. BC 데이터 생성·BC 학습·PPO는 수행하지 않았다.
- 다음 단계의 **상대적으로 나은 교사 후보**는 `champion_then_finish`다.
- 현재 `spatial_guarded` 규칙은 탈락이다. 이는 `spatial120` 관측 자체의 탈락이
  아니라, PT-DESIGN 보호 우선순위/임계값이 지나치게 보수적이라는 판정이다.
- `champion_then_finish`도 legacy stress에서 0/8 성공이므로 검증된 완성 expert는
  아니다. 승인 없이 legacy 실패 label을 대규모 BC 데이터에 포함하면 안 된다.

## 실험 설계

- frozen teacher checkpoint:
  `learning/rl/robot/champion/model_bc_robot_tesla12p7.pt`
- checkpoint SHA-256:
  `5fa1a65a60a90ffca5116a8d75d20a34749ebd69d71bb6f27287e0957f49b14a`
- 관측: Gate D `spatial120`; frozen champion에는 prefix `base14`만 입력
- 프로필: `legacy_stress`, `factory_prepolish`,
  `factory_prepolish_deep_defect`, `factory_prepolish_deep_stress`
- 경로: step-over 0.40, `balanced_extend5`, 2.100 m의 `same_xx`와 `cross_xy`
- seed: 각 프로필에서 42, 139, 236, 333
- 물리 seed: 42로 명시
- 교사: `champion_then_finish`, `spatial_guarded`
- 총 64 executions = 4 profiles × 2 paths × 4 seeds × 2 teachers
- paired 비교 단위는 32 profile/path/seed 조건이다.
- factory profile과 공간 교사 규칙/임계값은 실측 생산분포가 아닌 **PT-DESIGN**이다.

두 교사는 pass 1에서 frozen champion으로 완전히 동일하다. pass 2 이상에서만 다르다.

- `champion_then_finish`: `[-1.0, -0.74015748]`
  (목표 feed 8.0 mm/s)의 보존 finish action
- `spatial_guarded`: cached 5×5 scratch/under/over와 clearcoat margin을 검사해
  protect, defect-champion, finish 중 선택

## 결과

| 교사 | 성공 | 안전 | sensor fault | 2-pass 조건 |
|---|---:|---:|---:|---:|
| champion_then_finish | 24/32 | 32/32 | 0 | 8 |
| spatial_guarded | 24/32 | 32/32 | 0 | 8 |

- factory 세 profile은 두 교사·두 경로에서 모두 1-pass 성공: 교사별 24/24.
- legacy stress는 두 교사 모두 2-pass 후 `fail_max_passes`: 교사별 0/8.
- 모든 종료는 안전 64/64이며 sensor fault step은 0이다.
- `spatial_guarded`는 legacy 후속 21,168 step 전부를 `later_protect`로 분류했다.
  `later_finish`와 `later_defect_champion`은 0이었다. 즉 보호 조건이 결함 조건을
  완전히 가린 상태다.

legacy 8조건 평균:

| 지표 | champion_then_finish | spatial_guarded | spatial - baseline |
|---|---:|---:|---:|
| GU | 65.866890 | 64.275785 | -1.591105 |
| Ra (μm) | 0.156875 | 0.150995 | -0.005880 |
| Rz (μm) | 2.227043 | 2.232257 | +0.005214 |
| 잔여 scratch (μm) | 1.452369 | 1.526470 | +0.074101 |
| clearcoat min (μm) | 41.128774 | 41.271848 | +0.143074 |

공간 보호 교사는 clearcoat를 평균 0.143 μm 더 남기고 Ra를 0.00588 μm 낮췄지만,
GU와 잔여 scratch가 모든 legacy paired 조건에서 악화됐다. 성공 수와 안전 수를
개선하지 못했으므로 현재 임계값은 채택할 근거가 없다.

## 산출물

- `all_sequences.csv`: 64개 실행 통합
- `teacher_profile_summary.csv`: 교사×경로×프로필 16개 집계
- `paired_teacher_differences.csv`: 32개 동일조건 차이
- `metadata.json`: 입력 run, seed, 경로, teacher provenance
- `artifact_checksums.csv`: 네 원본 run과 이 요약의 파일 체크섬

초기 물리 seed를 명시하기 전에 생성한
`gate_e1_champion_then_finish_samexx_16_20260831_024000/`은 삭제하지 않고 smoke
근거로만 보존했다. 최종 64-execution 집계에서는 제외했다.

## 다음 승인 전 중단점

Gate E2 대규모 데이터 수집은 시작하지 않는다. 다음 결정이 먼저 필요하다.

1. factory 1-pass 성공 label만 우선 수집하고 legacy는 평가용 regression guard로
   분리하거나,
2. legacy pass-2 expert를 별도 개선·재검증한 뒤 혼합 데이터에 넣는다.

현재 결과만으로는 legacy 실패 label 10%를 포함한 110만-sample BC를 승인하지 않는다.

