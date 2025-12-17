# Jeju Renewable Energy Demand Forecast (J-REF)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

## Project Overview

제주도의 **태양광 및 풍력 발전량**을 딥러닝 기반으로 정밀하게 예측하는 시스템입니다.

기존의 통계적 기법이나 단순 시각화 서비스를 넘어, **제주도의 국지적 기상 특성(Micro-climate)을 반영한 고성능 예측 모델**을 구축하는 것을 목표로 합니다.

> **Target:** 시각화는 [햇빛바람지도](https://map.60hz.io/page/map)를 지향하되, 예측 성능(Accuracy)은 이를 초월합니다.

---

## Architecture Philosophy

**Clean Architecture** 원칙을 준수하여 유지보수성과 확장성을 확보합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                        │
│                   (Streamlit Dashboard, API)                     │
├─────────────────────────────────────────────────────────────────┤
│                        Application Layer                         │
│              (Training, Inference, Visualization)                │
├─────────────────────────────────────────────────────────────────┤
│                         Domain Layer                             │
│           (Solar-Net, Wind-Net, Ensemble Models)                 │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure Layer                        │
│            (Data Loaders, API Clients, File I/O)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principles

- **Modularity:** 데이터 전처리, 모델 학습, 추론 로직이 완벽히 분리
- **Strict Typing:** Python Type Hint를 적극 사용하여 코드 안정성 확보
- **Hybrid Modeling:** 태양광(Solar)과 풍력(Wind)의 물리적 특성 차이를 고려하여 별도의 모델을 학습 후 앙상블

---

## Directory Structure

```
kpx-demand-forecast/
├── .github/                   # CI/CD workflows
├── data/                      # 데이터 저장소 (Git ignore)
│   ├── raw/                   # 원천 데이터 (KPX, 기상청)
│   ├── processed/             # 전처리 완료 데이터
│   └── external/              # 외부 데이터 (휴일 정보 등)
├── docs/                      # 프로젝트 문서
│   └── PRD.md                 # 제품 요구사항 정의서
├── models/                    # 학습된 모델 체크포인트
├── notebooks/                 # EDA 및 프로토타이핑
├── src/                       # 핵심 소스 코드
│   ├── config.py              # 환경 설정
│   ├── data/                  # 데이터 로더
│   │   ├── weather_fetcher.py # 기상청 API 연동
│   │   └── kpx_loader.py      # KPX 데이터 로더
│   ├── features/              # Feature Engineering
│   ├── models/                # 모델 아키텍처
│   │   ├── solar_lstm.py      # 태양광 예측 모델
│   │   ├── wind_transformer.py# 풍력 예측 모델
│   │   └── ensemble.py        # 앙상블 모델
│   ├── training/              # 학습 루프
│   └── visualization/         # 시각화 유틸리티
├── api/                       # FastAPI 서버
├── tests/                     # 테스트 코드
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Model Architecture

### Dual-Stream Ensemble

태양광과 풍력은 물리적 특성이 완전히 다르므로, 각각 특화된 모델을 개발합니다.

```
                    Weather Features
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    ┌───────────┐                   ┌───────────┐
    │ Solar-Net │                   │ Wind-Net  │
    │  (LSTM)   │                   │  (TFT)    │
    └─────┬─────┘                   └─────┬─────┘
          │                               │
          └───────────┬───────────────────┘
                      │
                      ▼
               ┌─────────────┐
               │  Ensemble   │
               │   Layer     │
               └──────┬──────┘
                      │
                      ▼
              Final Prediction (MW)
```

| Model | Target | Architecture | Input Sequence |
|-------|--------|--------------|----------------|
| Solar-Net | 태양광 | Bi-LSTM + Attention | 24h |
| Wind-Net | 풍력 | Temporal Fusion Transformer | 48h |
| Ensemble | 통합 | Gradient Boosting | Model Outputs |

---

## 🚀 Quick Start

```bash
# 1. 저장소 클론
git clone https://github.com/kiminbean/kpx-demand-forecast.git
cd kpx-demand-forecast

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate  # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 설정 (.env 파일 생성)
echo "KMA_API_KEY=your_api_key_here" > .env

# 5. 대시보드 실행
streamlit run dashboard/app.py
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (MPS) 또는 NVIDIA GPU (CUDA)
- 기상청 공공데이터포털 API 키 ([data.go.kr](https://www.data.go.kr/))

### Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/kiminbean/kpx-demand-forecast.git
   cd kpx-demand-forecast
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**
   ```bash
   # .env 파일 생성
   cp .env.example .env
   # KMA_API_KEY 값을 본인의 API 키로 수정
   ```

5. **Download Data**
   ```bash
   # 기상청 ASOS 데이터 수집 (2024년)
   python scripts/download_data.py --year 2024
   ```

---

## Usage

### 1. 데이터 수집

```bash
# 기상청 ASOS 시간 데이터 수집
python scripts/download_data.py --year 2024

# 5년치 기상 패턴 데이터 수집 (2020-2024)
python scripts/download_5year_weather.py
```

### 2. 모델 학습

```bash
# 태양광 모델 학습 (LightGBM)
python scripts/train.py --model solar --epochs 100

# 풍력 모델 학습 (LightGBM)
python scripts/train.py --model wind --epochs 100
```

### 3. 예측

```bash
# 단일 날짜 예측
python scripts/predict.py --model solar --date 2024-12-17

# 배치 예측
python scripts/predict.py --model wind --start 2024-12-17 --end 2024-12-24
```

### 4. API 서버

```bash
# FastAPI 서버 실행
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# API 문서 확인: http://localhost:8000/docs
```

**API 엔드포인트:**
```bash
# 태양광 발전량 예측
curl "http://localhost:8000/predict/solar?date=2024-12-17"

# 풍력 발전량 예측
curl "http://localhost:8000/predict/wind?date=2024-12-17"

# 전체 발전량 예측
curl "http://localhost:8000/predict/all?date=2024-12-17"
```

### 5. 대시보드

```bash
# Streamlit 대시보드 실행
streamlit run dashboard/app.py

# 브라우저에서 http://localhost:8501 접속
```

---

## Data Sources

| Source | Data | URL |
|--------|------|-----|
| 기상청 | ASOS/AWS 관측 데이터 | [data.kma.go.kr](https://data.kma.go.kr/) |
| 기상청 | LDAPS 수치예보 | [data.kma.go.kr](https://data.kma.go.kr/) |
| KPX | 발전량 통계 | [epsis.kpx.or.kr](https://epsis.kpx.or.kr/) |

---

## 📊 Model Performance

### Current Results (2024-12-17)

| Model | R² | RMSE | MAE | Status |
|-------|-----|------|-----|--------|
| **Solar (LightGBM)** | 0.9134 | 17.63 MW | 9.77 MW | ✅ Production |
| **Wind (LightGBM)** | 0.7576 | 14.96 MW | 11.57 MW | ✅ Production |

### Wind Model Evolution

| Phase | Description | R² | Improvement |
|-------|-------------|-----|-------------|
| Phase 0 | Baseline | 0.2530 | - |
| Phase 1 | Physics-informed Features | 0.5153 | +103.7% |
| Phase 2 | 5-year Weather Patterns | 0.6357 | +151.3% |
| Phase 5 | Time-weighted + Fair Validation | **0.7576** | **+199.4%** |

### Key Features (Wind Model)

1. `wind_speed_anomaly` - 5년 평균 대비 풍속 이상치
2. `dayofyear` - 계절성
3. `temperature_min` - 월-시간별 최저기온
4. `wind_speed_rolling_mean_3h` - 3시간 롤링 평균
5. `wind_power_density` - 풍력 에너지 밀도 (½ρv³)

---

## Tech Stack

**Core**
- Python 3.10+
- PyTorch 2.0+
- Pandas, NumPy, Scikit-learn

**API & Visualization**
- FastAPI
- Streamlit
- Plotly, PyDeck

**Infrastructure**
- Docker
- Git

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

- ✅ 자유롭게 사용, 수정, 배포 가능
- ✅ 상업적 사용 가능
- ⚠️ 파생작도 GPL-3.0으로 공개 필수
- ⚠️ 소스코드 공개 의무

---

## Acknowledgments

- [햇빛바람지도 (60hz.io)](https://map.60hz.io/page/map) - 벤치마크 서비스
- [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) - 모델 참조
- 기상청 및 한국전력거래소 공공데이터 제공

---

*Developed with Claude Code*
