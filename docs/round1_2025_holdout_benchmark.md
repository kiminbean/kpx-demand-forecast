# 1차 라운드 — 2025 KPX hold-out 재평가 + 상용 벤치마크 TOP3 판정

> 날짜: 2026-11-14. 사전등록 스크립트: `scripts/evaluate_holdout_2025.py` (실행 전 작성).
> **원칙: 이 결과는 모델 선택·튜닝에 절대 재사용하지 않는다.** 2024-11-07~12-31 test window는 일체 사용하지 않았다.

## 배경

이전 사이클(라운드 1~3: docs/round1_dayahead_causality.md → round2_nwp_cv_rederivation.md → round3_wind_end_label_retrain.md)까지 모든 평가는 2024년 데이터 내부(2024-11~12 test window 또는 월별 CV)에서 이루어졌다. 이번 라운드에서는 2025년 KPX 실적 + 2025년 ASOS 관측을 새로 확보해 **완전한 시간 외 hold-out**으로 기존 프로덕션 모델을 재평가하고, 상용/공개 벤치마크와 비교해 현재 위치를 판정한다.

밴리스트(반복 금지 확인 완료):
- 2024-11~12 test window 재사용 금지 → 준수 (2025 전체를 hold-out으로 사용)
- MOS 채택 금지(개선 미미, round3 확정) → 재시도 없음
- 풍력 interval-START 라벨/브레이크 이전 학습 금지 → 준수 (phase5_tw0999 동결, interval-END, post-break만)

## 데이터 확보

| 항목 | 내용 |
|---|---|
| KPX 2025 실적 | data.go.kr 15065269, 2025-01-01~12-31, 166,440행 원본 (cp949). 2025-03부터 제주 풍력 라벨 붕괴(월 최대 13 MW 이하) |
| KMA ASOS 2025 | 4관측소(제주 184/고산 185/성산 188/서귀포 189) × 8,760시간, KMA ASOS Hourly API 직접 수집 (scripts/evaluate_holdout_2025.py와 동일 COLMAP 스키마) |
| 스키마 차이 | KPX 지역 라벨: 2024 파일 풍력='제주', 2025 파일은 태양광·풍력 모두='제주도'. 파일별 region 지정으로 처리 |
| 피처 | weather_5year와 동일한 4관측소 섬 평균 방식 (max abs diff로 기존 weather_5year_hourly.csv와 동일 구성임을 검증) |

## 재평가 설계 (사전등록, 2026-11-14)

- **태양광**: Solar V2 두 variant(island_mean/stations_level)를 2024 전체(8,784시간)로 재학습 → 2025 전체(8,760시간) hold-out 평가. **관측 기상 입력**이므로 D-1 10:00 하루 전 성능의 **상한**이다(실제 day-ahead는 NWP 난기류 오차가 더해짐).
- **풍력**: phase5_tw0999 하이퍼파라미터 동결, 2024-06-01 브레이크 후(post-break, 5,135시간)로 재학습 → 2025-01~02(1,416시간) 평가. 2025-03 이후는 라벨 붕괴로 평가 무의미(참고용 전년 FER 14.89%는 무효 표기).
- 벤치마크: 48시간 persistence, KPX 정산 구간(actual ≥ 10% × capacity) 동일 적용.
- 용량 민감도: 태양광 proxy_max_2025 320.6 / eff_359.4 / market 449.5 MW, 풍력 등록 104.9 / c_min 92.8 / c_max 106.7 / proxy 101.4 MW.

## 결과

### 태양광 (2025 hold-out, 관측기상 = 상한, proxy C=320.6 MW)

| 모델 | FER | MAE (MW) | R² | skill vs persistence48h |
|---|---|---|---|---|
| Solar V2 `stations_level` | **5.53%** | 7.4 | 0.96 | 0.68 |
| Solar V2 `island_mean` (API serving) | **7.32%** | 9.4 | 0.94 | 0.57 |
| persistence 48h | 17.17% | — | — | — |

용량 민감도: stations_level 5.53% (320.6) / 5.01% (359.4) / **4.12% (449.5)**; island_mean 7.32% / 6.66% / **5.48%**.

해석:
- 2024-11~12 test window에서의 4.16/4.18% 대비 열화. 원인: (1) 2024 후반 학습이 1년 지나 개념 노후, (2) 2025 커버리지·설비 변화, (3) 정산 조건(actual ≥ 10% × C)이 용량 정의에 따라 평가 시간 수가 달라짐.
- stations_level이 island_mean보다 1.79 pp 우위 — D-2 발전량 레벨(lvl_p) 피처가 hold-out에서도 유효. 단 API는 lvl_p를 서빙할 수 없어 island_mean이 프로덕션인 상태 유지.
- 월별 FER(island_mean): 3월 11.2%·1월 9.6%로 초봄·겨울이 열화, 9~12월 4.2% 이하로 하반기가 우수 — 학습이 2024년까지라 시즌 경향이 2025년 후반에 정합(겨울 하늘 상태 재현 양호), 2025년 3월 이상 기상이 어려웠음. stations_level은 동 패턴에서 전월대 0.5~5.5 pp 우위.

### 풍력 (2025-01~02 hold-out, C=104.9 MW 등록)

| 모델 | FER | MAE (MW) | R² | MBE |
|---|---|---|---|---|
| phase5_tw0999 (post-break 재학습) | **13.23%** | 14.64 | 0.718 | +6.11 MW |
| persistence 48h | 34.63% | — | — | — |

용량 민감도: 13.00% (106.7) / 13.23% (104.9) / 13.70% (101.4) / 15.09% (92.8). hit_4 15.1%, hit_6 24.5%.

해석:
- 이전 사이클의 역사적 test-window 수치(FER 17.61%, 2024-11~12) 대비 개선. persistence 대비 skill 0.62로 유의미한 예보력 유지.
- **+6.11 MW 과대예측 편향(MBE)** 이 FER의 절반 가까이 기여 — 2025년 초 커버리지가 2024 post-break보다 더 낮아진 것으로 추정. 편향 보정(MOS 유사한 선형 보정)만으로 FER 개선 여지가 있으나, 이는 결과 재사용 금지 원칙상 다음 라운드에서 새 사전등록으로 검증해야 함.
- 2025-03 이후 제주 풍력 라벨 붕괴(월 최대 13 MW) 확인 — AGENTS.md의 기존 경고가 재확인됨. 2025 풍력 hold-out은 1~2월이 한계.

## 상용 벤치마크 TOP3 판정

FER = |예측−실적|/용량 은 국제 표기로 cnMAE(capacity-normalized MAE)와 동일 정의. 아래 3개를 비교 기준으로 채택한다.

### TOP 1 — KPX 본토 태양광 참여자 평균 5.6% (2026 Q1)

- 같은 시장, 같은 지표, 같은 정산 규정이라 가장 직접적인 비교 기준.
- 판정: 관측기상 상한 기준 stations_level 5.53%는 참여자 평균 수준, island_mean 7.32%는 미달. **NWP day-ahead 실측(CV 8.50~8.75%)은 참여자 평균에 미달** — ≤6% 목표 미달 판정 유지.

### TOP 2 — 상용 태양광 24h 예보: SolarAnywhere rMAE 16.7~19.6% (GFS 26.9~30.1%)

- DNV SolarAnywhere 브로슈어(2026) 기준 24시간 GHI rMAE(지면 일사량 기준, 발전량 아님) — 지표 기반이 달라 직접 비교는 불가하나, 상용 24h 예보의 대략적 스케일 참고.
- 판정: 태양광 FER 5.5~7.3%는 일사량 rMAE 17~20% 상용 수치 대비 훨씬 우위로 보이나, 일사량→발전량 변환·정산 시간 필터 때문에 스케일이 다름. 참고용.

### TOP 3 — 풍력 day-ahead 단일 지역 cnMAE: Amperon 기준 ~11%, Nordic 지역 평균 5.7% (소규모 지역 >8%)

- Amperon: 업계 풍력 cnMAE 기준 ~11%, 자사 8~12%. Hauge et al. (Wind Energy, 2016, 10.1002/we.2073): Nordic day-ahead 지역별 MAE 평균 5.7% of capacity, 작은 지역은 8%+ (집단 응집 시 2.5%까지 개선).
- 판정: 제주 풍력 13.23%는 **단일 소규모 지역 업계 범위(8~13%)의 하한권 밖** — 다소 열위. 단 persistence 34.63% 대비 skill 0.62는 확보. 편향 보정으로 11% 내외 진입 가능성.

### 종합 판정

| 자산 | 우리 (hold-out) | TOP 벤치마크 | 판정 |
|---|---|---|---|
| 태양광 (day-ahead, CV) | 8.50~8.75% | KPX 참여자 5.6% | 미달 |
| 태양광 (관측기상 상한) | 5.53~7.32% | KPX 참여자 5.6% | stations_level은 도달, island_mean 미달 |
| 풍력 (2025-01~02) | 13.23% | 소규모 지역 8~13% | 하한권 밖, 개선 여지 |

## 산출물

- `scripts/evaluate_holdout_2025.py` — 사전등록 재평가 스크립트 (태양광 2 variant + 풍력 + persistence)
- `models/holdout_2025/` (gitignored) — holdout_2025_metrics.json + 예측 CSV 3종
- 2025 ASOS: `data/raw/weather/` 하위 2025 파일 (raw, gitignored)
- KPX 2025 원본: `data/raw/kpx/` 하위 2025 파일 (raw, gitignored)

## 다음 라운드 제안 (사전등록 필요)

1. 풍력 과대예측 편향 보정: post-break 학습분의 MBE를 이용한 상수/선형 보정을 **2024 내 CV**로 검증 후 2025-01~02에 1회 적용 (결과 재사용 금지 원칙 준수).
2. 태양광 2025 재학습 주기: 2024년만 학습한 모델이 1년 뒤 열화 — 2025 상반기까지 포함한 재학습이 다음 hold-out(2026)에서 유지되는지.
3. stations_level의 lvl_p를 API 서빙 가능 형태로 근사 (D-2 발전량 레벨을 API history로 대체 가능한지).
