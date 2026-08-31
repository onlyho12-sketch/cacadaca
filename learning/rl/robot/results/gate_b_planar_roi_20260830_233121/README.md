# Gate B — legacy_stress 큰 평판/중앙 ROI 진단

## 판정

- 실행 완료: **4/4**, 모두 2 pass 후 `fail_quality_regression`
- 안전: **4/4 안전 유지**, fallback 0 step
- coverage: pass 1 평균 **99.825%**, pass 2 **100%**
- 결론: pass 2에서 미세 표면과 scratch는 개선됐지만 저주파 제거 waviness와 중앙 과제거가 더 크게 증가해 전체 Ra/Rz가 악화됐다. 현재 회귀의 주원인은 미접촉 영역보다 경로 누적 제거량 불균일이다.

이 결과는 사용자가 지정한 `legacy_stress` 기준이다. 기존 `make_flat_patch()`의 0.05–2.0 μm, 4–12 scratch 규칙을 변경하지 않았다. 다만 큰 연속맵에서는 중앙 ROI의 scratch seed를 유지하고 주변과 연속인 별도 support background seed를 사용하므로, 과거 120 mm 패치의 배열과 byte 단위로 동일하다는 뜻은 아니다. 그 재현성 검증은 Gate B2 완료조건으로 남긴다.

## 고정 조건

| 항목 | 값 |
|---|---:|
| checkpoint | `model_bc_robot_tesla12p7.pt` |
| checkpoint SHA-256 | `5fa1a65a60a90ffca5116a8d75d20a34749ebd69d71bb6f27287e0957f49b14a` |
| 정책 observation | 14차원, 기존 champion 그대로 |
| 접촉 | PhysX physical contact |
| 환경/seed 계열 | 4 env × 1 sequence |
| 물리 Workpiece | 360 × 360 mm |
| 품질 상태맵 | 320 × 320 mm, 2 mm/cell |
| 중앙 평가 ROI | 200 × 200 mm, 100 × 100 cells |
| pad | Ø110 mm |
| ROI 끝점에서 품질맵 footprint 여유 | 5 mm |
| ROI 끝점에서 물리평판 footprint 여유 | 25 mm |
| step-over | 0.184 × pad = 20.24 mm |
| raster line | 10 |
| state-machine 1 pass 경로 길이 | 4.000 m |
| 공칭 feed | 12.7 mm/s |
| fine/waviness 분리 | Gaussian σ=10 mm, **PT-DESIGN** |
| coverage 하한 | removal ≥0.1 μm, **PT-DESIGN** |

Gaussian 분리는 원인 진단용 합성 지표이며 ISO profilometer cutoff가 아니다. `total Ra - fine Ra`도 엄밀한 가산 분해가 아니라 저주파 영향의 진단값이다.

## pass 평균

| 지표 | pass 1 | pass 2 | 변화 |
|---|---:|---:|---:|
| GU proxy | 66.098 | 68.813 | +2.715 |
| 전체 Ra (μm) | 0.2440 | 0.4361 | **+0.1921** |
| 전체 Rz (μm) | 2.2173 | 2.7781 | **+0.5608** |
| fine Ra (μm) | 0.05611 | 0.04959 | **−0.00653** |
| fine Rz (μm) | 1.33059 | 1.01766 | **−0.31293** |
| Ra 저주파 영향 진단값 (μm) | 0.18785 | 0.38649 | **+0.19865** |
| removal waviness Ra (μm) | 0.19532 | 0.38329 | **+0.18797** |
| removal waviness std (μm) | 0.25408 | 0.52559 | **+0.27152** |
| scratch max (μm) | 1.2274 | 0.7354 | **−0.4920** |
| clearcoat min (μm) | 41.393 | 40.039 | −1.354 |
| 평균 제거량 (μm) | 0.8417 | 1.6853 | +0.8437 |
| 제거량 CV | 0.3414 | 0.3412 | −0.0002 |
| 중앙−가장자리 제거량 (μm) | 0.3505 | 0.7649 | **+0.4145** |
| 중앙/가장자리 제거량 비 | 1.491 | 1.519 | +0.027 |
| coverage | 99.825% | 100% | +0.175%p |

네 환경 모두 pass 1→2에서 GU 상승, scratch 감소, fine Ra/Rz 감소가 동시에 나타났다. 그러나 전체 Ra/Rz, 저주파 영향, removal waviness 및 중앙−가장자리 제거량 차이도 네 환경 모두 증가했다. 따라서 기존 전체 Ra 악화가 미세 표면 자체의 악화만을 의미하지 않으며, raster-scale 제거 지형이 점수에 크게 섞여 있음을 확인했다.

## 평균 5×5 제거량 지도 (μm)

행/열은 CSV의 `tile_x`, `tile_y` 순서다.

pass 1:

```text
0.333 0.634 0.758 0.754 0.538
0.577 0.875 1.016 1.088 0.857
0.757 1.030 1.106 1.149 0.911
0.895 1.098 1.093 1.016 0.786
0.728 0.894 0.841 0.745 0.560
```

pass 2:

```text
0.716 1.391 1.670 1.521 1.016
1.279 1.966 2.252 2.165 1.538
1.597 2.117 2.324 2.276 1.685
1.700 2.048 2.077 1.984 1.591
1.377 1.664 1.539 1.430 1.210
```

pass 1의 전체 100개 env×tile 중 coverage 최저는 87.5%였고 평균 지도에서는 좌상단 tile만 95.6%, 나머지는 100%였다. pass 2는 모든 tile이 100%였다. 반면 제거량은 중앙부가 일관되게 높고 경계, 특히 모서리가 낮다.

## 산출물

- `initial_diagnostics.csv`: 초기 4표면의 분리 지표
- `planar_roi_sequences.csv`: 시퀀스 종료 결과 4행
- `planar_roi_passes.csv`: pass 진단 8행
- `planar_roi_tiles.csv`: 초기 100 + pass 1 100 + pass 2 100 = 300행
- `pass_deltas.csv`: 동일 표면 pass 1→2 차이 4행
- `geometry.csv`: 평판/ROI/pad/step-over 고정값
- `metadata.json`: 실행 provenance
- `artifacts_sha256.txt`: 코드 및 산출물 checksum

모든 CSV에서 명시적 NaN/±Inf는 0건이다. Gate B2와 Gate C는 이 폴더를 덮어쓰지 않고 별도 출력 경로를 사용해야 한다.
