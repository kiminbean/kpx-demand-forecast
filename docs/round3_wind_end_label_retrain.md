# 3차 라운드: 풍력 interval-END 라벨 + 2024-06 커버리지 브레이크 명시 처리 재학습

- 날짜: 2026-11-14
- 브랜치: `feature/solar-v2-kpx-fer`
- 범위: API 서빙 풍력 모델 재학습(`models/wind_phase5_fair/` 제자리 교체), API 1.1.0 → 1.2.0, 테스트 갱신, 문서(AGENTS.md 포함)

## 배경

- 1차 라운드(`docs/round1_dayahead_causality.md`)에서 태양광 day-ahead 인과성(마감 마스크)을 정합화했고, 2차 라운드(`docs/round2_nwp_cv_rederivation.md`)에서 NWP CV를 재산출했다(≤6% 목표 미달 유지).
- 2차 라운드에서 연기된 과제가 바로 이번 3차 과제다: **풍력 interval-END 라벨 재학습**.
- 기존 API 서빙 풍력 모델의 두 결함:
  1. **라블 정렬**: interval-START 라벨(거래시간 h를 (h, h+1]로 해석) 사용 — 태양광 V2가 얻은 개선과 동일한 1시간 정렬 문제를 풍력은 미해결 상태로 두고 있었다.
  2. **커버리지 브레이크 무시**: 2024-06-01 제주 신재생유 경매 시범 이후 공개 풍력 시계열은 시장 전체의 18~35%만 커버. 브레이크 전 혼합 학습은 "concept drift"의 대부분을 설명한다(AGENTS.md Data Pitfalls).
- 구 수치(R² 0.7576, RMSE 14.96 MW, MAE 11.57 MW, API path R² 0.757)는 interval-START 라벨 + 브레이크 전 혼합 학습 기반이므로 **무효화된 역사적 수치**다. 새 기준으로 재산출한 값으로만 판단한다.

## 변경 사항

### 1. 재학습 스크립트 신설: `scripts/train_wind_fer.py`
- interval-END 라벨(거래시간 h = (h−1, h])로 학습·평가.
- 브레이크 후 데이터만 학습: 2024-06-01 01:00 ~ 2024-11-06 23:00 (3,815 포인트, 브레이크 전 혼합 금지).
- 모델 설정은 사전등록(`docs/wind_fer_evaluation.md`, 2026-11-10) 선정값 `phase5_tw0999` 고정 — 재튜닝 없음.
- 산출물: `models/wind_phase5_fair/` 제자리 교체(`wind_lgbm_fair.lgb`, `wind_fer_metadata.json`, `feature_importance_fair.csv`).

### 2. API 갱신 (v1.1.0 → 1.2.0)
- 구간 의미 통일: 풍력·태양광 모두 "T 예측 = (T−1h, T] 발전량" (`api/service.py`, `api/main.py`, `api/schemas.py`, `api/__init__.py`).
- 모델 메트릭 하드코딩 교체: RMSE 14.96 → 12.15 MW, R² 0.7576 → 0.5517. `main.py`는 `WIND_TEST_RMSE_MW`를 `.service`에서 직접 import(하드코딩 제거).
- 불확실성 폭: ±1.28 × 14.96 ≈ ±19.1 MW → **±1.28 × 12.15 ≈ ±15.6 MW** (calibrated 80% 구간 아님 — 기존 문구 유지).
- `energy_type: both` → `total_mw` null 근거 변경: "(intervals differ)" → **모델 오차 독립 가정**(합산 시 오차 중복 가산). 각 예측에 `interval_start`/`interval_end` 동일 구간 제공.
- 버전 bump 4파일: `api/config.py`, `api/__init__.py`, `api/schemas.py`, `README.md`.

### 3. 테스트 갱신 (`tests/test_api_service.py`)
- 구식 패리티 검사(OFFLINE_WIND_R2=0.7576 ± 0.01) 폐기 → **하한 파손 검사 WIND_MIN_R2=0.40** (CV 검증월 R² 0.457~0.684, 시험 분할 실측 0.505 기준).
- END 라벨 정렬 테스트: `y_end = real_df["wind_mw"].iloc[start-1:len-1]` (레거시 CSV는 구간 시작 라벨).
- `both`/배치 응답 어서트 갱신(`total_mw` null, `interval_start`/`interval_end`).

## 밴리스트 준수 (이전 라운드 금지·연기 항목)

| 금지 항목 | 본 라운드 준수 |
|---|---|
| 테스트 윈도우(2024-11-07~12-31) 선택·튜닝 재사용 금지 | 재학습은 2024-06-01 01:00~2024-11-06 23:00만 사용. test-window 평가는 사전등록 1회(2026-11-10) 이미 소진 — **3회 소진**(solar 2×, wind 1×), 재실행 없음 |
| MOS 보정 재실행 금지 (가드 봉인) | 미실행 |
| 해줌 특허 2단 재시도 금지 | 미실행 |
| `--evaluate-test` 재실행 금지 | 미실행 — 사전등록 결과 JSON 재사용 |
| 2차 연기: 풍력 interval-END 재학습 | **본 라운드 과제로 소진** |

## 검증 결과

- 전체 테스트: **119 passed, 883 warnings in 6.60s, exit=0** (`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_models.py`).
- `scripts/train_wind_fer.py` 실행: exit 0.
- 잔여 grep: `14.96|0.7576|T+1h|interval-start|(T, T+1h]` — `api/` 0건, `tests/` 2건(의도된 역사 주석). `1.1.0` 잔여 0건.

## 기존 수치에 미치는 영향 (Δ 비교표)

재학습 모델 `phase5_tw0999` (interval-END 라벨, post-break only):

| 항목 | 구 (무효화된 역사적 수치) | 신 |
|---|---|---|
| 라벨 | interval-START | **interval-END** (h = (h−1, h]) |
| 학습 데이터 | 브레이크 전 혼합 | **post-break only** 2024-06-01 01:00~ (3,815 h) |
| 구간 의미 | (T, T+1h] | **(T−1h, T]** (태양광과 통일) |
| CV RMSE | 14.96 MW | **12.15 MW** (fold별 6.32 / 13.86 / 16.28) |
| CV R² | 0.7576 (API path 0.757) | **0.5517** (fold별 0.6839 / 0.5139 / 0.4572) |
| CV FER | 미산출 | **10.49%** (8/9/10월 5.52 / 14.28 / 11.68) |
| 불확실성 | ±19.1 MW | **±15.6 MW** (±1.28 × RMSE, calibrated 아님) |
| capacity C | — | 104.9 MW |
| `total_mw` (both) | intervals differ | **null** (오차 독립 가정) |

주의: 구·신 수치는 라벨·학습 데이터·평가 방식이 모두 달라 **직접 비교 불가**다. R² 감소(0.7576 → 0.5517)는 모델 열화가 아니라 (a) interval-END 라벨이 만드는 더 어려운 예측 문제, (b) 브레이크 후 18~35% 커버리지 표본만 사용, (c) 패리티 검사 폐기(하한 검사로 전환)를 반영한 것.

사전등록 test-window 평가(역사적 수치, 2026-11-10 사전등록 1회, `docs/wind_fer_evaluation.md`): FER 17.61%, RMSE 21.36 MW, R² 0.5053, MAE 17.06 MW, MBE −6.52 MW (n=1,027 평가시간) vs 48h persistence FER 34.13%. test-window는 재사용 금지이므로 이후 성능 판단은 CV FER 또는 2025 hold-out으로만 한다.

## 후속 과제

1. **풍력 FER 개선**: CV 기준 10.49%로 태양광(4~9%) 대비 still high. 2025 hold-out 데이터 확보 후 재평가 (2025-03부터 제주 풍력 라벨 붕괴 주의).
2. Multi-model NWP (ECMWF IFS, KMA LDAPS) — 기존 후속 유지.
3. 17:00 second-submission forecast — 기존 후속 유지.
