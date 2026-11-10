# NWP 일사량 MOS 보정 평가 (2026-11-10)

## 요약 (결론 먼저)

JMA MSM 예보 일사량에 MOS(Model Output Statistics) 선형 보정을 적용했으나,
**테스트(2024-11-07~12-31, C=359.4 MW)에서 유의미한 개선이 없었다.**
CV에서는 소폭 개선(8.686→8.611%)이 있었으나 최종 테스트에서는 모든 조합에서
보정 전보다 0.01~0.13%p 나쁘거나 비슷했다. 보정 전 수치를 유지하는 것이 낫다.

| 조건 | 변형 | 보정 전 FER | MOS 후 FER | Δ |
|---|---|---|---|---|
| day2 (48h 전, 주 평가) | island_mean | **8.531%** | 8.655% | +0.124%p (악화) |
| day1 (24h 전, 낙관적) | island_mean | **7.353%** | 7.359% | +0.006%p (동등) |
| day2 | stations_level | 8.647% | 8.764% | +0.117%p (악화) |
| day1 | stations_level | 7.202% | 7.334% | +0.132%p (악화) |

- 하루 전(day2) 조건의 현실적 상한은 여전히 **FER ≈ 8.5%** (관측기상 상한 4.18%).
- 48h persistence 16.38% 대비 스킬 점수도 보정 후 소폭 하락 (0.470 → 0.463).

## 방법

- 스크립트: `scripts/mos_correct_nwp.py`
  (`--select` = 방법 선택 CV, `--evaluate-test` = 사전 등록된 최종 비교 1회)
- 후보: `identity`(무보정), `linear`(obs = a·fcst + b), `linear_cloud`(＋운량 항), `quantile`(경험적 분위수 매핑)
- 적합: 관측소별, 적합 창 **2024-01-02 ~ 2024-10-31** (테스트 이전. 검증 후반부·테스트 미사용).
  낮(ASOS 관측 존재) 시간의 예보-관측 일사 쌍만 사용 (jeju 3,836 / gosan 3,856 / seogwipo 2,925 쌍).
  seongsan은 관측 일사가 없어 보정하지 않음 (island_mean 경로에서 사용되지 않음).
- 방법 선택: `train_solar_v2.py`와 동일한 월별 rolling-origin CV(5~10월, island_mean, day2)로 수행.
  CV FER: identity 8.686% / **linear 8.611%** / linear_cloud 8.707% / quantile 8.797% → linear 선택.
  선택 결과: `models/solar_v2_nwp_mos/mos_selection_lead2_island_mean.json`
- 최종 비교: 사전 등록 원칙에 따라 선택된 linear로 테스트를 딱 1회 평가 (variant × lead 4조합).
  테스트 데이터는 MOS 적합과 방법 선택에 일절 사용하지 않았다.

## 적합된 계수 (linear, day2 예보 기준)

| 관측소 | a (기울기) | b (절편, MJ/m²) | 쌍 수 |
|---|---|---|---|
| jeju | 0.9232 | 0.0444 | 3,836 |
| gosan | 0.9183 | 0.1706 | 3,856 |
| seogwipo | 0.9000 | 0.1506 | 2,925 |

해석: JMA MSM 일사는 평균적으로 관측보다 약 8~10% 과대(기울기 < 1) — 과대 추세는
CV에서 소폭의 개선으로 이어졌으나, 테스트 구간(늦가을)에서는 그 편향이 이미
LightGBM이 학습으로 흡수하고 있어 추가 이득이 없었다.

## 왜 개선되지 않았나 (가설)

1. 보정 이득이 작다: 예보-관측 선형 관계가 이미 R²이 높아 계수 보정만으로는 편향 제거 폭이 작다.
2. LightGBM이 계절별 편향을 이미 학습한다: 같은 학습 구간에서 예보 입력의 체계적 과대를
   모델이 흡수하므로, 사후 선형 보정은 중복되고 검증 구간 밖(11~12월)에서는 과적합 방향으로 작동.
3. 잔여 오차의 주성분이 일사량 크기 편향이 아니라 운량·강수 타이밍 오차(양자화된 MSM 그리드)라
   크기 보정으로는 잡히지 않는다.

## 후속 제안

- 단순 크기 보정보다 조건부 보정(운량·강수 예보 구간별 층화, 또는 예보 앙상블 스프레드 활용)
- 운량 자체의 MOS (MSM 운량은 3시간 간격 보간값으로 오차가 큼)
- 2025 KPX 데이터 확보 시 fresh hold-out으로 재검증

## 재현

```bash
.venv/bin/python scripts/mos_correct_nwp.py --select          # 방법 선택 (테스트 미사용)
.venv/bin/python scripts/mos_correct_nwp.py --evaluate-test   # 사전 등록 최종 비교
```

산출물: `models/solar_v2_nwp_mos/` (mos_evaluation.json, mos_selection_*.json,
`<variant>_<weather>_moslinear/` 모델·예측·지표 — gitignored)
