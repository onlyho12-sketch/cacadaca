# Gate E7 release/regression validation

## 판정

- 전체 release 판정: **PASS**
- 실패 기준: 없음
- champion은 교체하지 않았다.

## 미사용 seed 26000 same/cross 합계

- 안전: champion 64/64, candidate 64/64; candidate fault 0.
- 기존 품질: 48/64 → 48/64.
- all-4 면적: 93.7842% → 94.0283%.
- 평균 step: 3498.0 → 4040.7 (+15.51%).
- 타일 100/95/90/80%: 1141/1225/1273/1370 → 1149/1238/1286/1391.
- 동일 seed same_xx 반복 CSV 바이트 일치: True.

후보는 안전·품질·all-4·네 타일 기준·재현성을 통과했지만 평균 작업시간이
champion 대비 15.51% 증가해 사전 동결한 +40% 제한을 초과했다.
기준을 사후 완화하지 않으며 release candidate 승격을 보류한다.
