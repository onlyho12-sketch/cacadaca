# Gate E6 equal-sample PPO credit pilot

## 결론

- 48-step/λ0.95 control과 128-step/λ0.99 candidate는 각각 46,080 samples를 사용했다.
- screen 최우수는 `control_46080`(48-step/λ0.95 control 최종)였다. 긴 rollout 후보가 control을 이기지 못했다.
- 새 seed same/cross 합계에서 안전 통과는 champion 32/32, trained 32/32이고 품질 집계 통과는 각각 24/32, 24/32로 같다.
- 네 목표 동시 통과 면적은 champion 94.2172% → trained 94.4037% (Δ +0.1866%p)다.
- 평균 작업 step은 3487.3 → 4007.9 (Δ +520.6, +14.93%)로 늘었다.
- 따라서 champion을 승격/교체하지 않는다. Gate E6는 설정 선택 pilot이며 평면 작업 전체 완료가 아니다.

## 타일을 가장 많이 통과시킨 정책

same/cross 32표면, 정책당 800타일 기준:

- 내부 all-4 면적 100% 타일: champion 620, trained 620 — tie
- 내부 all-4 면적 ≥95% 타일: champion 642, trained 642 — tie
- 내부 all-4 면적 ≥90% 타일: champion 649, trained 651 — control_46080

엄격 100%와 95% 기준에서는 최다 정책이 챔피언과 공동 1위이고, 90% 기준에서만 trained가 더 많다. 타일 평균값 하나로 타일 전체를 통과 처리하지 않았다.

## 셀 내부 면적 정의

- 기존 2 mm ROI 셀마다 5×5-cell(10×10 mm) PT-DESIGN 국소창으로 GU/Ra/Rz를 평가했다.
- 품질 4개: local GU≥70, local Ra≤0.20 μm, local Rz≤2.0 μm, scratch가 초기보다 개선 또는 초기<0.05 μm.
- clearcoat/temperature/force-contact는 품질 4개와 섞지 않고 별도 안전으로 판정했다.
- 출력은 SYNTHETIC이며 실제 광택계/조도계 실측으로 표현하면 안 된다.

## 다음 작업(미승인)

1. all-4 면적 및 타일 100/95/90% 중 실제 승인 기준을 정한다.
2. 승인 후 선택 설정(현 결과는 48-step/λ0.95)으로 더 큰 seed/본 학습 Gate를 설계한다.
3. 평면 검증 후 곡면은 기존 곡면 자산을 동결한 별도 Gate에서 재현성부터 감사한다. 현재 base14에는 곡률/법선 구분 정보가 없어 바로 혼합 학습하지 않는다.

## 산출물

- `screen_sequence_summary.csv`, `screen_tile_counts.csv`
- `confirm_sequence_summary.csv`, `confirm_tile_counts.csv`, `confirm_profile_summary.csv`
- `paired_sequences.csv`, `paired_tiles.csv`, `decision.json`, `checksums.sha256`
