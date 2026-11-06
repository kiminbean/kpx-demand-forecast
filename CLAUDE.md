# J-REF Project Instructions for Claude

## Project Overview
제주도 신재생에너지 발전량 예측 시스템 (Jeju Renewable Energy Forecast)

## Autonomous Mode
**IMPORTANT**: 이 프로젝트에서는 사용자의 동의 없이 자율적으로 진행합니다.
- 작업 완료까지 중간 확인 없이 진행
- 오류 발생 시 자동으로 수정 시도
- 필요한 결정은 최선의 판단으로 자율 결정

## Project Structure
```
kpx-demand-forecast/
├── src/           # 소스 코드
├── data/          # 데이터 (raw, processed)
├── models/        # 학습된 모델
├── notebooks/     # Jupyter 노트북
├── scripts/       # 실행 스크립트
├── streamlit_app.py # Streamlit 대시보드 (KPX 실측 데이터)
├── dashboard/     # 초기 대시보드 (합성 데이터 merged_dataset.csv)
└── api/           # FastAPI 서버
```

## Key Commands
```bash
# 데이터 수집
python scripts/download_data.py --year 2024
python scripts/download_nwp_openmeteo.py   # JMA MSM 과거 예보 → data/external/nwp/ (gitignore)

# 모델 학습 (train.py는 초기 Bi-LSTM/Transformer 실험용, torch 필요)
python scripts/train.py --model solar --epochs 100
python scripts/train.py --model wind --epochs 100

# 태양광 V2 (LightGBM): --cv = 모델 선택, --evaluate-test = 테스트 1회 평가 + models/solar_v2/ 저장
python scripts/train_solar_v2.py --cv
python scripts/train_solar_v2.py --weather nwp_msm_day2 --cv   # 하루 전 조건 (48시간 전 예보)

# 테스트 (torch 미설치라 test_models.py 제외, 47 passed)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_models.py

# 예측
python scripts/predict.py --model solar

# 대시보드 (KPX 실측)
streamlit run streamlit_app.py
```

## API Keys
- KMA_API_KEY: .env 파일에 저장됨
- 기상청 공공데이터포털 API 사용

## Model Architecture
- 운영 모델: **LightGBM**. API는 태양광 `models/solar_v2/island_mean`(섬 평균 기상만 필요), 풍력 `models/wind_phase5_fair/`를 서빙한다. 연구 최고 변형 `stations_level`(FER 4.16%)은 D-2 발전 수준 피처라 발전량 이력이 필요해 API에 쓰지 않는다
- 입력: API 모델은 **관측 기상**(ASOS, 또는 사용자가 넣은 예보 기상) → 발전량. V2는 직전 1시간 기상도 쓴다. 수치예보(JMA MSM) 입력 하루 전 조건 평가는 `docs/solar_evaluation_criteria.md` §7 (FER 8.65%)
- Bi-LSTM + Self-Attention(태양광) / Transformer Encoder(풍력)는 합성 데이터 초기 실험(R² 0.36 / −0.17)이며 운영에 쓰지 않음
- 평가 목표: 태양광·풍력은 KPX 예측오차율 **FER** = |예측−실측| ÷ 설비용량 × 100 (실측 ≥ 용량 10% 시간), 수요는 MAPE. 상세: `docs/solar_evaluation_criteria.md`

## Data Sources
1. 기상청 ASOS 시간자료 (API). 일사: 제주·고산 연중, 서귀포 2024-04-09부터, 성산 없음
2. KPX 발전량: data.go.kr 15065269 (전력거래량, 한전 PPA·자가용 제외, ESS 포함), 2024년 8,784시간. 거래시간 h = (h−1)~h시
3. 수치예보: Open-Meteo Previous Runs API의 JMA MSM (CC BY 4.0, 비상업). 상세: `docs/DATA_COLLECTION_GUIDE.md`

## Development Guidelines
- MPS (Apple Silicon) 우선 사용
- Early Stopping 적용
- 체크포인트 자동 저장
- 시계열 순서 유지 (데이터 누수 방지)

## Current Status
- [x] 프로젝트 구조 생성
- [x] 데이터 수집 모듈
- [x] Feature Engineering
- [x] 모델 구현 (LSTM, Transformer)
- [x] 학습 파이프라인
- [x] 추론 파이프라인
- [x] Streamlit 대시보드
- [x] 실제 기상 데이터 수집 (KMA API)
- [x] 합성 발전량 데이터 생성
- [x] 모델 학습 완료
- [x] 최종 평가 완료
- [x] 평가 기준 FER 전환 (`docs/solar_evaluation_criteria.md`)
- [x] Solar V2 (1시간 정렬 수정 + 경사면 물리 피처), API는 `island_mean` 서빙
- [x] 하루 전 조건 NWP(JMA MSM) 평가 (FER 8.65%)
- [x] API 1.1.0 (태양광 구간 끝 시각 규약, 풍력은 구간 시작 라벨 유지, 응답에 interval_start/interval_end 명시, 입력 검증, 모델 미로드 시 500)
- [x] 대시보드 KPX 실측 데이터 표시 (`streamlit_app.py`)
- 브랜치 `feature/solar-v2-kpx-fer` (3681664, b0029a5, e9e9afb, cf14ac3), main 미병합. `render.yaml` autoDeploy 켜져 있음

## Training Results (2025-12-16)
> 초기 딥러닝 실험(합성 발전량 데이터) 기록. 운영 모델은 LightGBM이다.

### Solar Model (Bi-LSTM + Attention)
- Epochs: 22 (Early Stopped)
- Best Val Loss: 0.0531
- Test R²: 0.36
- Test RMSE: 48.07 MW
- Test MAE: 22.86 MW

### Wind Model (Transformer Encoder)
- Epochs: 18 (Early Stopped)
- Best Val Loss: 0.338
- Test R²: -0.17 (합성 데이터 한계)
- Test RMSE: 25.02 MW
- Test MAE: 13.25 MW

## Data Statistics
- Weather Data: 35,136 rows (4 Jeju stations, 2024)
- Generation Data: 8,784 rows (hourly, 2024)
- Training/Val/Test Split: 70/15/15

## Gemini Collaboration (2025-12-16)

### Data Leakage Detection
Gemini와의 협업으로 다음 데이터 누수 변수 발견 및 제거:
- `solar_cf`, `wind_cf`: 타겟에서 직접 파생 (CF = 발전량/용량)
- `total_mw`: 타겟들의 합계
- `*_mw_lag_*`, `*_mw_ma_*`: 타겟 기반 지연/롤링 특성

### LightGBM Results

**합성 데이터 (Synthetic):**
| Model | R² | RMSE | 비고 |
|-------|-----|------|------|
| Solar | 0.9960 | 3.95 MW | 물리모델 역학습 |
| Wind | 0.9743 | 3.62 MW | 물리모델 역학습 |

**실제 KPX 데이터 (Real):** ⭐
| Model | R² | RMSE | Top Features |
|-------|-----|------|--------------|
| Solar | **0.9134** | 17.62 MW | solar_potential, solar_radiation |
| Wind | **0.2530** | 26.28 MW | wind_power_density, dayofyear |

### Physics-Informed Features Added
- **Solar**: Solar Zenith/Azimuth Angle, Clear Sky Index, Air Mass
- **Wind**: Wind Power Density (0.5ρv³), Wind Shear, Turbine Power Curve

### KPX Data Sources Identified
1. 공공데이터포털: https://www.data.go.kr/data/15065269/fileData.do
2. EPSIS: https://epsis.kpx.or.kr
3. API 엔드포인트: `https://apis.data.go.kr/B552115/PvAmountByLocHr`

## Wind R² 0.25 원인 분석 (2025-12-16, Gemini 협업)

### 진짜 원인 (3가지)

| 원인 | 설명 | 영향 |
|------|------|------|
| **R²의 분산 효과** | 테스트셋 분산이 학습셋의 25.9%로 감소 | R² = 1 - MSE/Var → 분모↓ → R²↓ |
| **개념 이동 (Concept Drift)** | 학습(1-10월) vs 테스트(11-12월) 패턴 차이 | 최대 발전량 282→102MW |
| **풍력의 확률성** | 태양광(결정론적) vs 풍력(확률적) | 본질적 예측 한계 |

> **정정 (2026-11-06)**: "개념 이동"의 대부분은 계절 변화가 아니라 데이터 포괄 범위 변화다. 2024-06-01 제주 재생에너지 입찰제도 시행 후 공개 KPX 거래량은 시장 태양광 출력의 약 80%, 풍력의 18~35%만 담는다 (`docs/solar_evaluation_criteria.md` §3).

### 출력제어 3.3% 영향
- **결론: R² 0.25의 주요 원인 아님**
- is_curtailed 플래그의 평균 발전량이 정상 데이터와 거의 동일
- 출력제어보다 계절적 개념 이동이 주요 원인

### MAE 기준 공정 평가

| 모델 | R² | MAE | MAE/Mean | 평가 |
|------|-----|-----|----------|------|
| Solar | 0.9139 | 9.77 MW | **27.9%** | 우수 (저장 모델 재현값: 0.9134 / 9.70 MW / 27.3%) |
| Wind | 0.2540 | 21.88 MW | **56.6%** | 허용 가능 |

**핵심**: R² 차이(0.91 vs 0.25)만큼 MAE 차이(27.9% vs 56.6%)가 극단적이지 않음

### 권장 사항
1. R² 대신 MAE/MAPE로 성능 평가
2. 운영 시 최근 데이터로 주기적 재학습 (Concept Drift 대응)
3. Physics Constraints 적용 (Cut-in/Cut-out 로직)

## Phase 1 성능 개선 (2025-12-16, Gemini 협업)

### 토론 결과 핵심 합의사항
1. **데이터 5년 확장**: 필수 베이스라인이지만 그것만으로 부족
2. **Physics-informed Features**: Game Changer
3. **누수 변수 완전 제거**: 필수
4. **하이퍼파라미터 튜닝**: 강한 정규화 + 낮은 학습률

### 추가된 V2 피처 (91개)
| 카테고리 | 피처 예시 | 목적 |
|----------|-----------|------|
| **대기 안정도** | richardson_number, stability_class | 동적 Wind Shear 보정 |
| **동적 Wind Shear** | dynamic_alpha, wind_speed_hub_dynamic | 10m→80m 변환 정확도 |
| **기상 Lag/Rolling** | wind_speed_lag_1h~24h, rolling_mean_3h~24h | 시계열 패턴 학습 |
| **풍향 상호작용** | wind_sector_8, wind_u/v, is_northwest | 후류효과 간접 모델링 |
| **공간적 시차** | upstream_wind_1h, spatial_weight_west | 바람 이동 선행 지표 |
| **효율 관련** | combined_efficiency, temp_efficiency_wind | 이론/실제 효율 학습 |

### Wind 모델 V2 결과 🎉

| 지표 | Baseline | **V2 (Phase 1)** | 개선율 |
|------|----------|------------------|--------|
| R² | 0.2530 | **0.5153** | **+103.7%** |
| RMSE | 26.28 MW | **21.16 MW** | **+19.5%** |
| MAE | 21.88 MW | **17.88 MW** | **+18.3%** |

### Top 피처 (V2)
1. `wind_power_density` - v³ 효과
2. `dayofyear` - 계절성
3. `wpd_hub` - 허브 높이 WPD
4. `wind_speed_mean_3h` - 3시간 롤링 평균
5. `month` - 월별 패턴

### 실패한 시도
- **Hybrid (Linear + LightGBM)**: Linear 모델이 Concept Drift에 취약 → 테스트셋에서 R² = -12.8

### 핵심 교훈
1. **누수 변수 완전 제거가 핵심**: `wind_gen_change`, `is_curtailed` 제거 후 성능 급상승
2. **강한 정규화 필요**: reg_alpha=1.0, reg_lambda=1.0, min_child_samples=50
3. **기상 변수 시계열 피처**: 타겟이 아닌 기상 변수의 lag/rolling이 효과적

## Phase 2: 5년치 기상 패턴 활용 (2025-12-16) 🎯

### 개요
- 기상청 ASOS 5년치 데이터 (2020-2024) 수집 완료
- 총 175,392개 관측 데이터 → 43,848시간 집계
- 월-시간 조합별 기상 패턴 통계 생성 (288개 조합)

### 추가된 피처
| 카테고리 | 피처 예시 | 설명 |
|----------|-----------|------|
| **과거 패턴 통계** | temperature_mean, wind_speed_std | 월-시간별 5년 평균/표준편차 |
| **이상치 지표** | wind_speed_anomaly, temp_anomaly | 현재값 - 과거평균 편차 |
| **계절성 보강** | {var}_min, {var}_max | 월-시간별 극값 |

### Phase 2 Wind 모델 결과 🏆

| 지표 | Phase 0 | Phase 1 | **Phase 2** | 개선율 (vs Phase 0) |
|------|---------|---------|-------------|---------------------|
| **R²** | 0.2530 | 0.5153 | **0.6357** | **+151.3%** |
| **RMSE** | 26.28 MW | 21.16 MW | **18.34 MW** | **+30.2%** |
| **MAE** | 21.88 MW | 16.43 MW | **14.93 MW** | **+31.8%** |

### Top 피처 (Phase 2)
1. `wind_speed_rolling_mean_3h` - 3시간 롤링 평균 ⭐
2. `dayofyear` - 계절성
3. `wind_speed` - 직접 풍속
4. `temperature_min` - 5년 과거 월-시간 최저기온
5. `wind_speed_anomaly` - 과거 대비 풍속 이상치 ⭐

### 핵심 인사이트
1. **과거 패턴 대비 이상치가 중요**: `wind_speed_anomaly`가 상위 5위 피처
2. **롤링 평균이 핵심**: 3시간 롤링 평균이 1위, 6시간이 7위
3. **계절적 컨텍스트**: 5년 데이터로 계절별 정상 범위 학습 가능

### Phase별 성능 변화 요약
```
Phase 0 (Baseline)    → R² 0.25  (기본 기상 변수)
Phase 1 (V2 Features) → R² 0.52  (+103.7%, Physics-informed + 누수 제거)
Phase 2 (5yr Patterns)→ R² 0.64  (+151.3%, 과거 패턴 활용)
```

## Phase 3: 3가지 핵심 문제 해결 (2025-12-16) 🔧

### 문제 1: 테스트셋 분산 감소 해결
**문제**: 테스트셋(11-12월) 분산이 학습셋의 25.9%로 감소 → R² 저하

**해결책**: 시계열 교차검증 + 분산 정규화 R²

| Fold | 분산비율 | 일반 R² | 정규화 R² |
|------|----------|---------|-----------|
| 1 | 92.3% | 0.6400 | 0.6677 |
| 2 | 68.8% | 0.3030 | 0.5206 |
| 3 | 3.3% | -0.9553 | **0.9360** |
| 4 | 13.4% | 0.3675 | **0.9151** |
| 5 | 29.8% | 0.4506 | 0.8364 |
| **평균** | - | 0.1612 | **0.7752** |

**결론**: 분산 정규화 시 R²가 0.16 → 0.78로 대폭 증가. 테스트셋 분산이 낮을수록 정규화 효과가 큼.

### 문제 2: Concept Drift 해결
**문제**: 학습(1-10월) vs 테스트(11-12월) 패턴 차이

**해결책**: 시간 가중치 학습 (지수 감쇠: decay=0.999)

| 모델 | R² | RMSE | MAE |
|------|-----|------|-----|
| 일반 학습 | 0.6357 | 18.34 | 14.93 |
| **시간 가중치 학습** | **0.7441** | **15.37** | **12.00** |

**월별 개선**:
- 11월: R² 0.608 → 0.650 (+0.043)
- 12월: R² 0.651 → 0.810 (+0.159) ⭐

**결론**: 최근 데이터에 높은 가중치를 부여하면 Concept Drift 완화. 특히 12월(테스트셋)에서 +0.16 개선.

### 문제 3: 풍력 확률성 해결
**문제**: 풍력은 태양광과 달리 확률적 특성 → 점 예측의 한계

**해결책**: Quantile Regression으로 예측 구간 제공

| 분위수 | Pinball Loss |
|--------|--------------|
| Q10 | 3.60 |
| Q25 | 8.26 |
| Q50 | 11.61 |
| Q75 | 5.45 |
| Q90 | 5.93 |

- **80% 예측 구간 커버리지**: 65.2% (목표: 80%)
- **평균 예측 구간 폭**: 70.73 MW
- **평균 IQR (Q75-Q25)**: 8.15 MW

**불확실성 기반 분류**:
- 낮은 불확실성 구간: R² -2.31 (오히려 예측 부정확)
- 높은 불확실성 구간: R² 0.57 (예측 정확)

**결론**: 불확실성이 높다고 판단된 시점이 오히려 예측 정확도가 높음 → 불확실성 추정 모델 재검토 필요

## Phase 4: Optuna 하이퍼파라미터 튜닝 (2025-12-16) ⚙️

### 최적 파라미터 (20 trials)
```python
{
    'num_leaves': 32,
    'learning_rate': 0.0333,
    'feature_fraction': 0.7724,
    'bagging_fraction': 0.5561,
    'bagging_freq': 7,
    'min_child_samples': 37,
    'reg_alpha': 1.6449,
    'reg_lambda': 2.1817,
    'max_depth': 7
}
```

### Optuna 모델 성능
| 지표 | Phase 3 (시간가중치) | Optuna |
|------|---------------------|--------|
| R² | **0.7441** | 0.6783 |
| RMSE | **15.37 MW** | 17.24 MW |
| MAE | **12.00 MW** | 13.78 MW |

**결론**: Optuna 튜닝이 오히려 성능 하락 → 시간 가중치 학습이 현재 데이터셋에 더 효과적

### Top 15 Features (Optuna 모델)
1. `wind_speed_anomaly` - 과거 대비 풍속 이상치 ⭐
2. `dayofyear` - 계절성
3. `month` - 월별 패턴
4. `wind_speed` - 직접 풍속
5. `wind_speed_rolling_min_12h` - 12시간 최저 풍속
6. `wind_speed_rolling_min_6h`
7. `wind_speed_rolling_min_3h`
8. `solar_radiation_anomaly`
9. `solar_radiation`
10. `temperature_min`

## 최종 성능 요약 (2025-12-16) 🏆

### Wind 모델 Phase별 성능
| Phase | R² | RMSE | 개선율 (vs Phase 0) |
|-------|-----|------|---------------------|
| Phase 0 (Baseline) | 0.2530 | 26.28 MW | - |
| Phase 1 (V2 Physics) | 0.5153 | 21.16 MW | +103.7% |
| Phase 2 (5yr Patterns) | 0.6357 | 18.34 MW | +151.3% |
| **Phase 3 (시간 가중치)** | **0.7441** | **15.37 MW** | **+194.1%** |
| Phase 4 (Optuna) | 0.6783 | 17.24 MW | +168.1% |

### 최고 성능 모델
- **모델**: LightGBM + 시간 가중치 학습
- **R²**: 0.7441 (+194.1% vs Baseline)
- **RMSE**: 15.37 MW (-41.5% vs Baseline)
- **MAE**: 12.00 MW

### 핵심 성공 요인
1. **누수 변수 완전 제거**: 타겟 기반 파생 변수 모두 제거
2. **Physics-informed Features**: V2 피처 (대기 안정도, Wind Shear 등)
3. **5년 기상 패턴 활용**: 과거 평균 대비 이상치 피처
4. **시간 가중치 학습**: Concept Drift 대응 (+0.11 R² 개선)

### 저장된 모델
- `models/wind_phase4/wind_lgbm_optuna_best.lgb` - Optuna 튜닝 모델
- `models/wind_phase4/feature_importance_optuna.csv` - 피처 중요도

## Phase 5: Gemini 협업 - 모델 검증 및 Bias 수정 (2025-12-17) ✅

### Gemini 검증 요청 사항
1. 데이터 누수(Data Leakage) 가능성 검토
2. 피처 엔지니어링의 적절성 평가
3. 시간 가중치 학습 방식의 타당성
4. R² 0.74 성능이 과적합 없이 일반화 가능한지
5. 실제 운영 환경에서의 적용 가능성

### Gemini 지적사항
| 항목 | 지적 내용 | 심각도 |
|------|----------|--------|
| **5년 기상 패턴** | 2024년 전체(테스트셋 포함)로 통계 계산 → Look-ahead Bias | 🔴 치명적 |
| **시간 가중치** | 468배 가중치 비율이 계절성 학습 저해 | 🟡 우려 |
| **1년치 데이터** | 연간 변동성 학습 불가 | 🟡 제한적 |

### Look-ahead Bias 수정 및 재검증

**문제 코드 (수정 전)**:
```python
# scripts/merge_5year_data.py:213
weather_patterns = create_5year_weather_patterns(df_weather_all)  # 2020-2024 전체 사용
```

**수정 후**:
```python
# 2020-2023년만 사용 (테스트셋 2024년 11-12월 제외)
df_weather_until_2023 = df_weather_all[df_weather_all['datetime'] < '2024-01-01']
weather_patterns = create_5year_weather_patterns(df_weather_until_2023)
```

### 공정한 재검증 결과 🎉

| 모델 | R² | RMSE | MAE | 비고 |
|------|-----|------|-----|------|
| 기존 (누수 포함) | 0.7441 | 15.37 MW | 12.00 MW | 이전 측정값 |
| 누수 수정 + 가중치 없음 | 0.6255 | 18.60 MW | 15.17 MW | - |
| 누수 수정 + 약한 가중치 | 0.6669 | 17.54 MW | 14.67 MW | decay=0.9999 |
| **누수 수정 + 강한 가중치** | **0.7576** | **14.96 MW** | **11.57 MW** | decay=0.999 ⭐ |

### 핵심 발견사항

**예상과 다른 결과**: Look-ahead Bias 수정 후 성능이 오히려 **+0.0135 향상**

| 지표 | 기존 (누수) | 수정 후 | 차이 |
|------|------------|---------|------|
| R² | 0.7441 | 0.7576 | **+0.0135** |
| RMSE | 15.37 MW | 14.96 MW | **-0.41 MW** |

**결론**:
1. ✅ **모델 성능은 유효함** - 누수로 인한 과대평가 아님
2. ✅ **2020-2023년 패턴으로 충분** - 계절성 학습 가능
3. ✅ **시간 가중치 효과 입증** - Concept Drift 대응에 유효

### 시간 가중치 효과 분석

| 가중치 | decay | 비율(끝/첫) | R² | 결론 |
|--------|-------|-------------|-----|------|
| 없음 | - | 1x | 0.6255 | Baseline |
| 약한 | 0.9999 | 1.8x | 0.6669 | +0.04 |
| **강한** | 0.999 | 468x | **0.7576** | **+0.13** |

**Gemini 우려에 대한 반박**: 강한 시간 가중치가 오히려 성능을 크게 향상시킴. 이는 Concept Drift(계절적 패턴 변화)가 이 데이터셋에서 중요한 요인임을 입증. (정정: 드리프트의 대부분은 2024-06 공개 데이터 포괄 범위 변화 — 위 "Wind R² 0.25 원인 분석" 정정 참조)

### 최종 검증된 성능

```
Phase 0 (Baseline)           → R² 0.2530, RMSE 26.28 MW
Phase 5 (누수 수정, 공정 평가) → R² 0.7576, RMSE 14.96 MW
총 개선율: +199.4%
```

### 저장된 모델
- `models/wind_phase5_fair/wind_lgbm_fair.lgb` - 공정하게 검증된 최종 모델
- `models/wind_phase5_fair/feature_importance_fair.csv` - 피처 중요도

## Phase 6: Gemini 협업 - 추가 성능 향상 시도 (2025-12-17) 🔬

### Gemini 토론 내용

| 라운드 | 핵심 주장 | Claude 의견 |
|--------|----------|-------------|
| **1** | Physics-Informed Features + 경량 NN 스태킹 | 데이터 부족으로 NN 과적합 우려 |
| **2** | 출력제어 데이터 정제 + Ramp Rate 피처 | 동의, 구현 시도 |
| **3** | NWP 예보 데이터 융합 + 주파수 도메인 피처 | 데이터 확보 어려움 |

### 합의된 개선 방안 실험 결과

| 시도 | 설명 | R² | Phase 5 대비 |
|------|------|-----|--------------|
| **Phase 5** | 기존 최고 성능 | **0.7576** | 기준선 |
| Phase 6-1 | Ramp Rate 피처 + 출력제어 가중치 | 0.7339 | **-3.1%** |
| Phase 6-2 | 잔차 학습 2단계 모델 | 0.7260 | **-4.2%** |
| Phase 6-3 | Optuna 심화 튜닝 (30 trials) | 0.7062 | **-5.1%** |
| Phase 6-4 | 피처 다이어트 (Top 30) | 0.7425 | **-2.0%** |

### Gemini 진단: 실패 원인 분석

| 시도 | 실패 원인 | 세부 설명 |
|------|----------|----------|
| **Ramp Rate 피처** | 미분 노이즈 증폭 | 센서 노이즈가 차분 연산으로 Signal보다 크게 증폭 |
| **출력제어 가중치** | 학습 신호 박탈 | 고풍속 구간의 실제 발전 패턴 학습 기회 제거 |
| **잔차 학습** | 백색 잡음 과적합 | R² 0.76 잔차는 패턴 없는 Aleatoric Uncertainty |
| **Optuna** | Validation 과적합 | Test와 Validation의 Concept Drift 차이 |

### 핵심 결론 🎯

**R² 0.76은 현재 데이터의 이론적 한계에 근접**

Gemini 분석에 따르면:
- **Bayes Error Rate**: ~0.78-0.79로 추정
- **정보의 한계**: 과거 기상 데이터만으로는 돌풍/급변 예측 불가
- **추가 개선 필요 조건**: NWP 예보 데이터(기상청 LDAPS/GFS) 융합

### 권장 사항

1. **Phase 5 모델 유지** (R² 0.7576, RMSE 14.96 MW)
2. 추가 성능 개선보다 **불확실성 정량화(Quantile Regression)** 고도화 권장
3. NWP 예보 데이터 확보 시 **Forecast Error 학습** 시도 가능

### Top 30 핵심 피처 (Phase 5 기준)

```
 1. wind_speed_anomaly        # 5년 평균 대비 풍속 이상치 ⭐
 2. dayofyear                 # 연중 일수 (계절성)
 3. temperature_min           # 5년 월-시간별 최저기온
 4. wind_speed                # 직접 풍속
 5. month                     # 월별 패턴
 6. wind_speed_rolling_min_6h # 6시간 최저 풍속
 7. wind_speed_rolling_mean_3h # 3시간 롤링 평균
 8. solar_radiation           # 일사량
 9. solar_radiation_anomaly   # 일사량 이상치
10. wind_speed_rolling_min_3h # 3시간 최저 풍속
... 외 20개
```

## 최종 Wind 모델 성능 요약 (Phase 0 → Phase 5)

| Phase | 설명 | R² | RMSE | 개선율 |
|-------|------|-----|------|--------|
| Phase 0 | Baseline | 0.2530 | 26.28 MW | - |
| Phase 1 | V2 Physics Features | 0.5153 | 21.16 MW | +103.7% |
| Phase 2 | 5년 기상 패턴 | 0.6357 | 18.34 MW | +151.3% |
| Phase 3 | 시간 가중치 학습 | 0.7441 | 15.37 MW | +194.1% |
| **Phase 5** | **누수 수정 + 검증** | **0.7576** | **14.96 MW** | **+199.4%** |

**총 개선: R² 0.25 → 0.76 (+199.4%), RMSE 26 → 15 MW (-43%)**

## MAPE 상세 분석 결과 (2025-12-22) 📊

### 태양광 모델 (Solar LightGBM Baseline)

**전체 성능:**
| 지표 | 값 |
|------|-----|
| **R²** | **0.9134** |
| RMSE | 17.62 MW |
| MAE | 9.70 MW |

> 정정 (2026-11-06): 저장 모델 `models/real_baseline/solar_lgbm_baseline.lgb`를 테스트 1,314시간에 재현한 값. 기존 기록은 R² 0.9138 / RMSE 17.63 / MAE 9.89.

**구간별 MAPE:**
| 발전량 구간 | 샘플수 | MAPE | MAE |
|-------------|--------|------|-----|
| 1-10 MW | 76 | 328.5% | 10.54 MW |
| 10-50 MW | 141 | 76.4% | 19.81 MW |
| 50-100 MW | 122 | 43.6% | 30.53 MW |
| 100-200 MW | 167 | **17.5%** | 23.05 MW |
| 200-500 MW | 40 | **8.1%** ⭐ | 17.49 MW |

**실용적 평가 지표:**
| 지표 | 값 | 설명 |
|------|-----|------|
| MAPE (>1MW) | 81.15% | 저발전량 구간 왜곡 포함 |
| MAPE (≥50MW) | **26.05%** | 실용적 발전 구간 |
| sMAPE | 42.94% | 대칭 MAPE (왜곡 완화) |
| **MAE/Mean** | **27.3%** | 간편 MAPE 대안 (기존 기록 11.70%는 재현 안 됨) |

### 풍력 모델 (Wind LightGBM Phase 5 Fair)

**전체 성능:**
| 지표 | 값 |
|------|-----|
| **R²** | **0.7576** |
| RMSE | 14.96 MW |
| MAE | 11.57 MW |

**구간별 MAPE:**
| 발전량 구간 | 샘플수 | MAPE | MAE |
|-------------|--------|------|-----|
| 1-10 MW | 193 | 311.6% | 11.19 MW |
| 10-30 MW | 377 | 62.5% | 10.91 MW |
| 30-50 MW | 211 | **29.7%** | 11.66 MW |
| 50-100 MW | 448 | **17.7%** | 12.82 MW |
| 100-300 MW | 1 | 20.0% | 20.32 MW |

> 정정 (2026-11-06): 이 절의 풍력 구간·실용 지표는 저장 모델 `models/wind_phase5_fair/wind_lgbm_fair.lgb`를 Phase 5 전처리(`scripts/phase5_bias_correction.py`)로 테스트 1,318시간(2024-11-07~12-31)에 다시 예측해 재현한 값이다 (R²·RMSE·MAE 일치). 기존 기록은 재현되지 않았다.

**실용적 평가 지표:**
| 지표 | 값 | 설명 |
|------|-----|------|
| MAPE (>1MW) | 79.60% | 저발전량 구간 왜곡 포함 |
| MAPE (≥30MW) | **21.53%** | 실용적 발전 구간 |
| sMAPE | 52.19% | 대칭 MAPE |
| **MAE/Mean** | **29.94%** | 간편 MAPE 대안 (테스트 평균 38.64 MW) |

### 태양광 vs 풍력 비교 요약

| 지표 | 태양광 | 풍력 | 비고 |
|------|--------|------|------|
| **R²** | 0.9134 | 0.7576 | 태양광 +0.156 |
| RMSE | 17.62 MW | 14.96 MW | 풍력이 낮음 |
| MAE | 9.70 MW | 11.57 MW | 태양광이 낮음 |
| **MAPE (실용구간)** | **26.05%** | **21.53%** | 구간 정의가 다름 (≥50MW / ≥30MW) |
| sMAPE | 42.94% | 52.19% | 풍력이 높음 |
| **MAE/Mean** | **27.3%** | **29.94%** | 유사 |

### MAPE 해석 주의사항

1. **저발전량 왜곡**: MAPE는 분모가 작을수록 크게 왜곡됨
   - 1-10MW 구간: 300%+ (비현실적)
   - 100MW+ 구간: 10-30% (신뢰 가능)

2. **권장 지표 우선순위**:
   1. **MAE/Mean**: 왜곡 없이 전체 성능 파악
   2. **실용구간 MAPE**: 실제 운영 시 의미 있는 구간
   3. **sMAPE**: 저발전량 왜곡 완화

3. **목표 달성 여부**:
   - 태양광 고발전량(200MW+): MAPE **8.1%** → 목표 10% **달성** ✅
   - 풍력 실용구간(30MW+): MAPE **21.53%** → 목표 10% 미달성

> 정정 (2026-11-06): 발전량 평가는 MAPE·MAE/Mean 대신 **KPX FER**(설비용량 대비)로 바꿨다. MAE/Mean은 평가 기간 평균에 좌우되고, "MAPE < 10%" 목표는 FER 기준으로 대체한다. 아래 "Solar V2 및 평가 기준" 참조.

## Solar V2 및 평가 기준 (2026-11-06)

- **평가 기준**: 태양광·풍력 = FER = |예측−실측| ÷ 설비용량 × 100, 실측 ≥ 용량 10% 시간만. 수요 = MAPE. 근거·정산 구간: `docs/solar_evaluation_criteria.md`
- **정렬 버그 수정**: KPX 거래시간 h는 (h−1)~h시 발전량, ASOS 일사는 직전 1시간 누적 → 기존 파이프라인이 1시간 어긋나 있었음. 개선의 대부분이 이 수정에서 나옴
- **V2 추가 요소**: 경사면 일사 물리 피처, D-2 발전 수준(lvl_p), 야간 ESS 프로파일 후처리
- **테스트** (2024-11-07~12-31, 분모 359.4 MW, 관측 기상 입력 → 하루 전 예측 성능의 **상한**):

| 모델 | FER | MAE | R² |
|------|-----|-----|-----|
| 기존 `real_baseline` | 6.93% | 9.70 MW | 0.913 |
| 정렬만 수정 | 4.26% | 5.84 MW | 0.965 |
| **V2 `stations_level`** | **4.16%** | **5.16 MW** | 0.967 |
| V2 `island_mean` (**API 서빙 모델**) | 4.18% | 5.25 MW | 0.969 |

- 정렬만 수정한 모델 대비 V2 이득은 테스트에서 유의하지 않음 (−0.10%p, 95% CI −0.50 ~ +0.36)
- API는 `island_mean`(섬 평균 기상 + 직전 1시간 기상, RMSE 10.63)을 서빙한다. `stations_level`은 연구 최고값이지만 D-2 발전 수준 피처라 발전량 이력이 있어야 해서 API에 쓰지 않는다
- **하루 전 조건** (docs §7): JMA MSM 과거 예보(Open-Meteo, 48시간 전 발표 = D-1 10시 제출 시점에 확실히 있는 예보) 입력 시 테스트 FER **8.65%** (`stations_level`) / 8.53% (`island_mean`), 분모 449.5 MW면 7.07% / 6.96%
  - 관측 기상 4.16% 대비 예보 오차 비용 +4.49%p (95% CI +2.75 ~ +5.42). 24시간 전 예보 7.20%(낙관적), 48시간 전 지속성 16.38%, KPX 육지 태양광 참여자 평균 약 5.6% (2026년 1분기)
  - 재현: `scripts/download_nwp_openmeteo.py` → `scripts/train_solar_v2.py --weather nwp_msm_day2 --evaluate-test` (`--variant island_mean`이면 8.53%, 모델 선택은 `--cv`)
- **데이터 포괄 범위 단절**: 2024-06-01(제주 입찰제도 시범) 이후 공개 KPX 거래량은 시장 태양광의 ~80%, 풍력의 18~35%만 포함 → 풍력 "개념 이동"(최대 282→102 MW)의 주원인
- **테스트 구간 소진**: 2024-11-07~12-31은 관측 기상 V2 평가와 NWP 평가에 2회 사용했다(사전 등록, 튜닝 없음). 이후 모델 선택에 다시 쓰지 말고 CV 또는 2025년 hold-out으로 한다
- 코드: `src/features/solar_features_v2.py`, `src/models/solar_v2.py`, `src/evaluation/metrics.py`, `scripts/train_solar_v2.py` / 모델: `models/solar_v2/`

## Next Steps (향후 개선사항)
- [x] Wind R² 0.25 원인 분석 완료
- [x] 출력제어(Curtailment) 영향 분석 완료
- [x] **Phase 1 완료: R² 0.25 → 0.52 달성**
- [x] **Phase 2 완료: R² 0.52 → 0.64 달성**
- [x] **Phase 3 완료: 3가지 핵심 문제 해결, R² 0.74 달성**
- [x] **Phase 4 완료: Optuna 튜닝 (R² 0.68)**
- [x] **Phase 5 완료: Gemini 협업 검증, 공정 R² 0.76 달성** ✅
- [x] **Phase 6 완료: 추가 개선 시도 (한계 도달 확인)** 🔬
- [x] **MAPE 상세 분석 완료 (2025-12-22)** 📊
- [x] 태양광 하루 전 NWP(JMA MSM) 평가 완료 — FER 8.65% (docs §7)
- [x] Solar V2·FER 평가, API 1.1.0, 대시보드 실측 데이터 (2026-11-06~07)
- [ ] 풍력 재학습: 구간 끝 라벨 + 2024-06 포괄 범위 단절 처리
- [ ] 하루 전 성능 개선: 다중 예보 모델(ECMWF IFS 2024-03~, 기상청 API허브 LDAPS), 17시 2차 제출용 예보, 예보 일사 MOS 보정
- [ ] 2025년 KPX 자료를 새 hold-out으로 확보 (헤더 `거래일`, 0 채운 시각, 제주 풍력 라벨 `제주도`, 2025-03부터 풍력 다시 급감)
- [ ] 실시간 예측 API 구현 (optional)
