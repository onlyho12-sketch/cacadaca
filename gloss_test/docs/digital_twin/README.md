# PolyTwin 논문 기반 폴리싱 디지털 트윈 문서

이 폴더가 PolyTwin 폴리싱 강화학습과 20° Gloss 시뮬레이션의 **현재 기준 문서**다.

## 확정 전제

```text
실제 시편 실험             : 수행하지 않음
실제 Gloss Meter 보정       : 수행하지 않음
실제 장비 기반 계수 재보정 : 수행하지 않음

물리·품질 모델 근거         : 확보한 논문 직접값과 논문 기반 파생값
시뮬레이션 환경             : Isaac Lab + Isaac Sim RTX
최종 20° Gloss              : 논문 기반 simulation GU proxy
강화학습                    : 다양한 국소 곡면에서 폴리싱 제어와 품질 개선 학습
공정조건 최적화             : 디지털 트윈 내부에서 BO 수행
```

본 프로젝트에서 `GU`, `제거량`, `Ra/Rz`라고 표시하는 시뮬레이션 출력은 실제 계측값이 아니라
논문 근거 모델이 계산한 값이다. 결과 파일에는 다음 필드명을 사용한다.

```text
predicted_20deg_gu_literature_proxy
predicted_removal_um_literature_model
predicted_ra_um
predicted_rz_um
predicted_clearcoat_remaining_um
```

## 문서 읽는 순서

1. [00_프로젝트_정의.md](00_프로젝트_정의.md)
2. [01_논문근거_파라미터.md](01_논문근거_파라미터.md)
3. [02_표면상태_폴리싱물리모델.md](02_표면상태_폴리싱물리모델.md)
4. [03_20도_GU_광학모델.md](03_20도_GU_광학모델.md)
5. [04_IsaacLab_환경_RL학습.md](04_IsaacLab_환경_RL학습.md)
6. [05_BO_PPO_통합.md](05_BO_PPO_통합.md)
7. [06_평가_로그_검증.md](06_평가_로그_검증.md)
8. [07_구현_인수인계.md](07_구현_인수인계.md)
9. [08_RL_출력_GU연동규격.md](08_RL_출력_GU연동규격.md)

## 근거 수준 표기

| 태그 | 의미 | 사용 방법 |
|---|---|---|
| `L-DIRECT` | 논문에서 직접 측정·사용한 값 | 논문 조건과 함께 기록 |
| `L-DERIVED` | 논문값을 단위변환·정규화·결합한 값 | 파생식을 기록 |
| `PT-DESIGN` | PolyTwin이 시뮬레이션을 위해 정한 값 | 논문 직접값으로 표현 금지 |
| `SYNTHETIC` | 위 값을 사용해 모델이 생성한 결과 | 실제 계측값으로 표현 금지 |

## 시스템 구성

```text
논문 근거 파라미터
        ↓
폴리싱 표면상태 전이모델
        ↓
Removal / Height / Scratch / Ra / Rz / Clearcoat Map
        ↓
20° GU proxy 모델 + RTX 광학 시각화
        ↓
Isaac Lab PPO 접촉·경로·품질 개선 학습
        ↑
BO가 Force/RPM/Feed/Step-over 레시피 제공
```

## 구버전 문서

상위 폴더의 다음 문서는 실제 실험·실측 보정을 전제로 작성된 구버전이므로 현재 구현기준으로
사용하지 않는다.

- `../PolyTwin_RL_강화학습_구현_전달서.md`
- `../PolyTwin_RL_논문근거_검증_전달안.md`

논문 출처 추적이나 과거 의사결정 확인에만 사용한다.
