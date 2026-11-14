# J-REF (Jeju Renewable Energy Forecast) - Agent Instructions

> Renewable energy (solar/wind) generation forecasting system for Jeju Island, Korea
> Last updated: 2026-11-07

## Project Context

**Business Goal**: Forecast Jeju solar and wind generation (KPX market-traded volume) from weather data, evaluated with the KPX forecast error rate (FER). The day-ahead (D-1 10:00) evaluation with 48h-old JMA MSM forecasts is done: FER 8.53-8.65%, which misses the ≤6% internal target; the served API model (4.18% with observed weather) is an upper bound.

**Production path (LightGBM only)**:
- Solar: `models/solar_v2/island_mean` (V2 features, island-mean weather + previous 1 h weather). Served by API 1.2.0.
- Wind: `models/wind_phase5_fair/` (Phase-5 LightGBM, interval-END labels, post-break data only; retrained by `scripts/train_wind_fer.py`). Served by API 1.2.0.
- Bi-LSTM (`src/models/solar_lstm.py`) and Transformer (`src/models/wind_transformer.py`) are early experiments on synthetic data. They are NOT production and need PyTorch, which is not installed in `.venv`.

**Tech Stack**:
- Python 3.10+, LightGBM, pandas, numpy, scikit-learn, Optuna
- FastAPI (API server), Streamlit (dashboard)
- PyTorch only for the legacy DL experiments (optional)

**Layout**: `src/` library (data, features, models, evaluation) used by `scripts/` (CLI training/evaluation), `api/` (FastAPI) and `streamlit_app.py` (dashboard).

---

## Evaluation Standard (read before reporting any metric)

Full spec: `docs/solar_evaluation_criteria.md`, code: `src/evaluation/metrics.py`.

- **Solar/wind**: KPX forecast error rate **FER** = |forecast − actual| / capacity × 100, only hours with actual ≥ 10% of capacity (전력시장운영규칙 제14.1.1조, 별표2 Ⅴ). **Demand**: MAPE.
- Do NOT use generation MAPE, MAE/Mean or R² as the headline metric (MAPE explodes near zero output; R² depends on test variance).
- Settlement tiers: 2021~ ≤6% 4원, 6-8% 3원 (Jeju 2024 ×0.5 from 2024-06-01; Jeju forecast program ended 2025-06-01). 2025 revision (mainland): ≤4% 4원, 4-6% 3원. KPX mainland solar participant average ≈5.6% (2026 Q1).
- Test denominator C = **359.4 MW** (449.5 × 0.7996 Jun-Oct coverage); sensitivity range 342.3-449.5 MW.
- **Test-window rule**: 2024-11-07~12-31 has already been used three times, all pre-registered (solar 2×, wind 1× on 2026-11-10 — see docs/wind_fer_evaluation.md). NEVER use it again for model selection or tuning. Use monthly rolling-origin CV (`--cv`) or a 2025 hold-out.

---

## Operational Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt -r requirements_api.txt  # requirements.txt lacks lightgbm (requirements_api.txt, used by render.yaml, has it); torch is only for legacy DL
cp .env.example .env  # Add KMA_API_KEY
```

### Data Collection
```bash
python scripts/download_data.py --year 2024        # KMA ASOS hourly
python scripts/collect_5year_data.py               # 5-year weather patterns
python scripts/download_nwp_openmeteo.py           # JMA MSM previous-run forecasts → data/external/nwp/ (gitignored)
```

### Model Training (Solar V2)
```bash
python scripts/train_solar_v2.py --cv                                   # model selection: CV + validation, test untouched
python scripts/train_solar_v2.py --variant island_mean --evaluate-test  # train+val fit → test → saves models/solar_v2/<variant> (overwrites API model)
python scripts/train_solar_v2.py --weather nwp_msm_day2 --cv            # day-ahead condition (48h-old forecasts): CV + validation only, nothing saved
# Test FER reproduction (8.65% stations_level / 8.53% island_mean): --weather nwp_msm_day2 --evaluate-test → models/solar_v2_nwp/ (gitignored); test window already used, not for selection
```
`--weather` choices: `asos` (default, observed), `nwp_msm_day2` (primary day-ahead), `nwp_msm_day1` (optimistic). Do not run `--evaluate-test` again on the 2024-11~12 window for selection purposes (see test-window rule).

**Causality note (round 1, 2026-11-14, docs/round1_dayahead_causality.md; round 2 re-derivation docs/round2_nwp_cv_rederivation.md; round 3 MOS causal re-selection docs/mos_nwp_correction.md)**: training masks now use `training_mask_for_forecast` (interval END ≤ D-1 10:00 deadline) and the day2→day1 NWP fallback was removed. The recorded NWP day-ahead test FER 8.53-8.65% was produced under the pre-fix mask + fallback and is historical. Corrected causal CV (round 2, `--weather nwp_msm_day2 --cv`, test unused): FER 8.50% stations_level / 8.75% island_mean. Round 3: `mos_correct_nwp.py --select` was unblocked with real safety conditions (`_require_safe_inputs` verifies CV folds and input max index end before TEST_START and trims post-test rows) and causal rolling CV re-selected `linear_cloud` at 8.579% vs identity 8.752% — gain only −0.17 pp, MOS not adopted; the ≤6% target is still missed under the strict deadline mask.

Legacy DL experiments (torch required, not production): `python scripts/train.py --model solar --epochs 100`.

### Wind Retraining (API serving model)
```bash
python scripts/train_wind_fer.py   # interval-END labels, post-break only (2024-06-01 01:00~), phase5_tw0999 → models/wind_phase5_fair/ (in-place)
```
Test-window FER for this model was evaluated once (pre-registered, 2026-11-10, `scripts/evaluate_wind_fer.py`); do not re-run `--evaluate-test`.

### Testing
```bash
# Standard suite (47 passed as of 2026-11-07)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_models.py

# Single file
.venv/bin/python -m pytest tests/test_solar_v2.py -q
```
`tests/test_models.py` imports torch and is excluded because torch is not installed.

### Linting & Formatting
```bash
black src/ scripts/ api/ tests/
isort src/ scripts/ api/ tests/
flake8 src/ scripts/ api/
mypy src/ --ignore-missing-imports
```

### Running Services
```bash
# API server (v1.1.0)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
python run_api.py

# Dashboard (real KPX data)
streamlit run streamlit_app.py
```
`dashboard/app.py` is the legacy dashboard and reads the synthetic `data/processed/merged_dataset.csv`.

---

## Data Pitfalls (must know)

- **KPX target**: data.go.kr 15065269 "지역별 시간별 태양광 및 풍력 발전량" = market-traded volume (전력거래량). Excludes KEPCO-PPA and self-use plants, includes ESS charge/discharge. Only 2024 (8,784 h) is in this repo.
- **Interval-end time convention**: KPX 거래시간 h covers (h−1)~h시. ASOS hourly `solar_radiation` at t is accumulated over (t−1, t] (MJ/m²). V2 code labels every sample by interval END T.
- **1-hour misalignment**: the legacy `data/processed/kpx_real_merged.csv` labels generation by interval START, so it pairs generation (t, t+1] with radiation (t−1, t]. Correlation 0.894 → 0.948 when aligned. Most of the V2 gain came from fixing this. Never build new solar features on the legacy labels without re-aligning.
- **2024-06-01 coverage break** (Jeju renewable bidding pilot): from then the public series covers only ~80% of market solar output (Jun 0.81 → Oct 0.76) and 18-35% of wind. This explains most of the old wind "concept drift" (max 282 → 102 MW). Any model spanning this date must handle it.
- **ASOS radiation coverage**: jeju & gosan all year, seogwipo only from 2024-04-09, seongsan none.
- **Torch-free imports**: `src/models/__init__.py` loads model classes lazily (`__getattr__`), so importing `src.models.solar_v2` or the API does not require torch. Keep it that way; do not add eager torch imports to shared modules.
- **NWP data**: Open-Meteo JMA MSM, CC BY 4.0, non-commercial free tier. `previous_day2` = run issued ~48 h earlier (available at the D-1 10:00 deadline, primary); `previous_day1` = ~24 h earlier (may post-date the deadline, optimistic only).

---

## API 1.2.0 Behaviour (api/*)

- Solar at datetime T = generation over (T−1h, T] (KPX 거래시간 h = hour of T). Requires ≥1 h of `history` (400 otherwise).
- Daytime solar requires `solar_radiation` (MJ/m², ≤5.0; W/m² × 0.0036) and `cloud_cover` (400 if missing).
- Wind at T = generation over (T−1h, T] (interval-END labels, same convention as solar). Phase-5 model retrained post-break only (`scripts/train_wind_fer.py`): CV RMSE 12.15 MW, R² 0.5517.
- `energy_type: both` → `total_mw` is null (solar and wind prediction errors are assumed independent, so summing would double-count error); each prediction has `interval_start` / `interval_end` (both now share the same (T−1h, T] interval).
- Model not loaded → 500 (never a silent 0 MW). Timezone-aware inputs are normalised to KST.
- Bounds: ±1.28 × RMSE (normal-error approximation). Solar uses observed-weather test RMSE split by daylight (day 15.65 / night 0.58 MW → ±20.0 / ±0.74 MW); wind uses ±1.28 × 12.15 MW ≈ ±15.6 MW. NOT a calibrated 80% interval.

---

## Golden Rules (Immutable)

### Data Integrity
- **NEVER modify `data/raw/`** - Original data is sacred
- **NEVER leak future information** - Check for look-ahead bias in feature engineering
- **ALWAYS use 2020-2023 data** for pattern statistics (not 2024 test data)
- **ALWAYS maintain time-series order** - No shuffling in train/test split
- **ALWAYS label samples by interval end** for new solar work (see Data Pitfalls)
- **NEVER reuse the 2024-11-07~12-31 test window** for model selection

### Model Safety
- **NEVER suppress type errors** with `# type: ignore`
- **Do not commit new model artifacts.** `models/*` is gitignored except `models/solar_v2/`, `models/wind_phase5_fair/` and `models/real_baseline/` (tracked because the API deploy loads them); `models/solar_v2_nwp/` and `data/external/nwp/` stay ignored.
- **ALWAYS use `set_seed(42)`** for reproducibility
- **ALWAYS use `get_device()`** in the legacy PyTorch code (MPS priority)

### Known Data Leakage Variables (DO NOT USE as features)
```python
LEAKY_VARIABLES = [
    'solar_cf', 'wind_cf',          # Derived from target (CF = gen/capacity)
    'total_mw',                      # Sum of targets
    '*_mw_lag_*', '*_mw_ma_*',      # Target-based lag/rolling
    'wind_gen_change', 'is_curtailed'  # Target derivatives
]
```
Exception: Solar V2 `stations_level` uses a D-2 generation level (`lvl_p`), which is information available at the D-1 10:00 deadline. It needs per-station weather and generation history; the API accepts neither (one weather set + recent weather history), so it serves `island_mean`.

---

## Code Style Guidelines

### Imports (isort compatible)
```python
# Standard library
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import lightgbm as lgb
import numpy as np
import pandas as pd

# Local
from src.evaluation.metrics import evaluate_generation, kpx_error_rate
from src.models.solar_v2 import SolarV2Model
```

### Naming Conventions
- Variables/functions: `snake_case`; classes: `PascalCase`; constants: `SCREAMING_SNAKE_CASE`; private: leading underscore.

### Type Hints and Docstrings
- Type hints required on public functions. Google-style docstrings (Args / Returns / Raises).

### Error Handling
```python
# Explicit, informative errors (fail fast)
if not data_path.exists():
    raise FileNotFoundError(
        f"Data file not found: {data_path}\n"
        "Run 'python scripts/download_data.py --year 2024' first."
    )

import logging
logger = logging.getLogger(__name__)
logger.warning(f"Missing values in {col}: {missing_count}")
```
The API must return an error (400/500) rather than a fabricated value when inputs or models are missing.

---

## Project Structure

```
kpx-demand-forecast/
├── api/              # FastAPI server (v1.1.0)
│   ├── main.py       # Endpoints
│   ├── schemas.py    # Pydantic models
│   ├── service.py    # Business logic (solar V2 + wind Phase 5)
│   └── config.py     # API settings, model paths
├── streamlit_app.py  # Dashboard (real KPX data)
├── dashboard/        # Legacy Streamlit app (synthetic data)
├── data/
│   ├── raw/          # Original data (READ ONLY)
│   ├── processed/    # kpx_real_merged.csv (legacy interval-start labels)
│   ├── sample/       # sample_data.csv = full real 2024 series (cloud dashboard)
│   └── external/nwp/ # Open-Meteo NWP (gitignored)
├── models/           # models/* gitignored except solar_v2/, wind_phase5_fair/, real_baseline/ (solar_v2_nwp/ ignored)
├── scripts/          # CLI entry points (train_solar_v2.py, download_nwp_openmeteo.py, phase*.py)
├── src/
│   ├── config.py     # Global configuration
│   ├── data/         # Data loaders, API clients
│   ├── evaluation/   # metrics.py (FER)
│   ├── features/     # solar_features_v2.py, wind features
│   ├── models/       # solar_v2.py (LightGBM), legacy LSTM/Transformer (lazy imports)
│   ├── training/     # Legacy PyTorch training loop
│   └── inference/    # Legacy prediction logic
├── docs/             # solar_evaluation_criteria.md, PRD.md, DATA_COLLECTION_GUIDE.md
└── tests/            # pytest tests
```

---

## Key Implementation Patterns

### Feature Engineering (Prevent Leakage)
```python
# GOOD: Weather-based rolling (no target leakage)
df['wind_speed_rolling_mean_3h'] = df['wind_speed'].rolling(3).mean()
df['wind_speed_anomaly'] = df['wind_speed'] - pattern_stats['wind_speed_mean']

# BAD: Target-based features (DATA LEAKAGE!)
# df['wind_mw_lag_1h'] = df['wind_mw'].shift(1)  # NEVER DO THIS
```

### Solar V2 Physics
- Per-station tilted-plane irradiance (Erbs decomposition → Hay-Davies transposition, tilt 25°, S/SE/SW orientations → NOCT cell temperature), cloud trend, night ESS profile post-processing. `stations_level` adds a D-2 generation level (`lvl_p`) blend.

### Time-Weighted Training (Wind, Concept Drift)
```python
n_samples = len(X_train)
time_weights = np.array([0.999 ** (n_samples - i - 1) for i in range(n_samples)])
model.fit(X_train, y_train, sample_weight=time_weights)
```
Much of the wind "drift" is actually the 2024-06-01 coverage break; prefer handling the break explicitly when retraining. The API serving model now does this: `scripts/train_wind_fer.py` trains on post-break hours only (2024-06-01 01:00~, interval-END labels).

---

## Performance Benchmarks

Solar, test 2024-11-07~12-31 (370 evaluated hours), C = 359.4 MW. Observed-weather inputs are an upper bound for day-ahead performance.

| Model | Inputs | FER | MAE | R² | Status |
|-------|--------|-----|-----|-----|--------|
| Solar legacy `real_baseline` | observed | 6.93% | 9.70 MW | 0.913 | Retired |
| Solar V2 `stations_level` | observed | 4.16% | 5.16 MW | 0.967 | Research best |
| Solar V2 `island_mean` | observed | 4.18% | 5.25 MW | 0.969 | **Production (API)** |
| Solar V2 `stations_level` / `island_mean` | NWP 48h-old | 8.65% / 8.53% | - | - | Day-ahead primary |
| 48h persistence | - | 16.38% | - | - | Reference |

- V2 vs aligned-only (4.26%): −0.10 pp (95% CI −0.50~+0.36, not significant). Forecast error cost: +4.49 pp (CI +2.75~+5.42).
- Wind Phase-5 (interval-END labels, post-break only): CV(8/9/10월) FER 10.49% (5.52/14.28/11.68), RMSE 12.15 MW, R² 0.5517; test-window eval (역사적, 2024-11~12, 사전등록 1회): FER 17.61%, RMSE 21.36 MW, R² 0.5053 vs 48h persistence 34.13% — **Production (API)**. 구(무효화된 역사적 수치): interval-START 라벨 기반 R² 0.7576, RMSE 14.96 MW.

---

## Open Follow-ups

- Wind: FER 개선 — 2025-01~02 hold-out 사전등록 재평가 완료(2026-11-14, docs/round1_2025_holdout_benchmark.md): FER 13.23%, persistence 34.63%, R² 0.718, **MBE +6.11 MW 과대예측 편향** — 편향 보정(post-break CV 검증)이 다음 개선 후보. 상용 벤치마크(단일 소규모 지역 cnMAE 8~13%) 하한권 밖. 2025-03부터 제주 풍력 라벨 붕괴(월 최대 13 MW) 재확인
- Solar: 2025 전체 hold-out (관측기상=상한) island_mean 7.32% / stations_level 5.53% (proxy C=320.6) — KPX 본토 참여자 평균 5.6% 대비 island_mean 미달; NWP day-ahead 실측(8.50~8.75%)은 참여자 평균 미달 유지. 2024-only 학습의 1년 열화 확인(3월 11.2% 열화 vs 하반기 4% 이하)
- Multi-model NWP (ECMWF IFS from 2024-03, KMA LDAPS via KMA API Hub), 17:00 second-submission forecasts, MOS correction of forecast radiation
- KPX 2025 지역 라벨 스키마: 2024 파일 풍력='제주', 2025 파일 모두='제주도' — 파일별 region 지정 필요 (2025-03부터 제주 풍력 라벨 붕괴)

---

## Context Map

- **[Evaluation Criteria](./docs/solar_evaluation_criteria.md)** - FER definition, denominators, Solar V2 and day-ahead results
- **[API Development](./api/)** - FastAPI endpoints, Pydantic schemas
- **[Feature Engineering](./src/features/)** - Solar V2 physics, wind V2 features, leakage prevention
- **[Model Architecture](./src/models/)** - Solar V2 LightGBM, legacy LSTM/Transformer
- **[Data Collection](./docs/DATA_COLLECTION_GUIDE.md)** - KMA, KPX, NWP sources and pitfalls
- **[Dashboard](./streamlit_app.py)** - Streamlit visualization (real KPX data)
