# 데이터 수집 가이드

J-REF 프로젝트에 필요한 데이터 소스와 수집 방법을 설명합니다.

---

## 1. 데이터 소스 개요

| 데이터 | 출처 | 용도 | 수집 방법 |
|--------|------|------|-----------|
| **ASOS** | 기상청 | 종관기상관측 (시간별) | **Selenium 크롤러** (권장) |
| **AWS** | 기상청 | 방재기상관측 (10분별) | Open API |
| **발전량** | 전력거래소(KPX), data.go.kr 15065269 | 시간별 태양광·풍력 거래량 (타겟) | 수동 다운로드 |
| **수치예보(NWP)** | Open-Meteo Previous Runs API (JMA MSM) | 하루 전 조건 학습·평가 | `scripts/download_nwp_openmeteo.py` |

---

## 2. 기상 데이터 수집 (권장: Selenium 크롤러)

### 2.1 의존성 설치

```bash
pip install selenium webdriver-manager
```

Chrome 브라우저가 설치되어 있어야 합니다.

### 2.2 크롤러 사용법

```python
from src.data import KMACrawler

# 크롤러 초기화
crawler = KMACrawler(
    headless=True,        # 브라우저 창 숨김
    download_dir="data/raw/weather",
)

# 단일 지점 다운로드
crawler.download(
    station="제주",           # 지점명 또는 코드 ("184")
    start_date="2024-01-01",  # 시작일
    end_date="2024-12-31",    # 종료일
    data_type="시간",         # "시간" 또는 "일"
)

# 연도별 일괄 다운로드
crawler.download_batch(
    station="제주",
    start_year=2020,
    end_year=2024,
    data_type="시간",
)
```

### 2.3 명령줄 실행

```bash
# 2024년 제주도 전체 관측소 데이터
python scripts/download_data.py --year 2024

# 특정 기간
python scripts/download_data.py --start 2024-01-01 --end 2024-06-30

# 브라우저 창 표시 (디버깅용)
python scripts/download_data.py --year 2024 --no-headless
```

### 2.4 제주도 관측소

| 지점번호 | 지점명 | 위도 | 경도 |
|----------|--------|------|------|
| 184 | 제주 | 33.5142 | 126.5294 |
| 185 | 고산 | 33.2939 | 126.1628 |
| 188 | 성산 | 33.3867 | 126.8803 |
| 189 | 서귀포 | 33.2464 | 126.5653 |

> 일사(`solar_radiation`) 관측 범위 (2024): 제주·고산은 연중, 서귀포는 2024-04-09부터, 성산은 없음.
> ASOS 시간 일사는 시각 t의 **직전 1시간 (t−1, t] 누적값**(MJ/m²)이다.

---

## 3. 기상 데이터 수집 (대안: API)

API 키가 있는 경우 더 안정적인 REST API를 사용할 수 있습니다.

### 3.1 API 키 발급

1. [공공데이터포털](https://www.data.go.kr/) 접속
2. 회원가입 및 로그인
3. "기상청_지상(종관, ASOS) 시간자료 조회서비스" 검색 후 활용 신청
4. 마이페이지 → 활용신청 현황 → **일반 인증키(Decoding)** 복사

### 3.2 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 편집
KMA_API_KEY=your_api_key_here
```

### 3.3 API 사용법

```python
from src.data import KMAAPI

api = KMAAPI(api_key="your_key")  # 또는 환경변수에서 자동 로드

# 시간별 데이터
df = api.get_hourly_data(
    station_id="184",  # 제주
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# 일별 데이터
df = api.get_daily_data(
    station_id="184",
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# CSV 저장
api.download(
    station="제주",
    start_date="2024-01-01",
    end_date="2024-12-31",
    data_type="시간",
    output_dir="data/raw/weather",
)
```

---

## 4. KPX 발전량 데이터

### 4.1 수동 다운로드

KPX 데이터는 현재 수동 다운로드가 필요합니다.

**타겟 데이터 (사용 중)**: 공공데이터포털 [15065269](https://www.data.go.kr/data/15065269/fileData.do) "한국전력거래소_지역별 시간별 태양광 및 풍력 발전량" (CSV, `data/raw/kpx/`)
- **전력시장 거래량**이다. 한전 직접거래(PPA)·자가용 설비는 빠지고, ESS 충방전은 포함된다.
- 이 저장소에는 2024년(8,784시간)만 학습·평가에 쓴다.

**참고 (EPSIS)**:

1. **전력통계정보시스템 (EPSIS)** 접속
   - URL: https://epsis.kpx.or.kr/

2. 메뉴 경로:
   - 전력통계 → 발전설비/발전량 → **지역별 발전량**

3. 조건 설정:
   - 지역: **제주**
   - 에너지원: **태양광**, **풍력**
   - 기간: 원하는 기간

4. **Excel 다운로드** 클릭

5. 파일 저장:
   ```
   data/raw/kpx/solar_generation_2024.csv
   data/raw/kpx/wind_generation_2024.csv
   ```

### 4.2 데이터 형식 (예상)

```csv
일시,지역,에너지원,발전량(MWh),설비용량(MW)
2024-01-01 00:00,제주,태양광,0.0,500.0
2024-01-01 01:00,제주,태양광,0.0,500.0
...
```

### 4.3 시각 규약과 1시간 어긋남 (중요)

- KPX **거래시간 h**는 (h−1)~h시 발전량이다. 예: 거래시간 13 = 12:00~13:00.
- ASOS 시간 일사도 시각 t에 (t−1, t] 누적값이다. 따라서 둘은 **구간 끝 시각**으로 맞춰야 한다.
- 기존 병합 파일 `data/processed/kpx_real_merged.csv`는 발전량을 **구간 시작** 시각으로 붙였다. 그래서 발전량 (t, t+1]과 일사 (t−1, t]가 짝지어져 1시간 어긋나 있다 (발전량-일사 상관 0.894 → 정렬 후 0.948).
- 태양광 V2 코드(`src/features/solar_features_v2.py`)와 API 1.1.0은 모든 표본을 구간 끝 시각 T로 라벨링한다. 풍력 Phase 5 모델과 API 풍력 예측은 아직 구간 시작 라벨((T, T+1h])이다.
- 새 피처를 `kpx_real_merged.csv` 위에 만들 때는 반드시 다시 정렬한다.

### 4.4 2024-06-01 포괄 범위 단절

- 2024-06-01 제주 재생에너지 입찰제도 시범사업 시행 이후, 공개 거래량은 시장 태양광 출력의 약 80%(6월 0.81 → 10월 0.76), 풍력의 18~35%만 포함한다. 1~5월에는 시장 태양광 출력과 일치했다.
- 기존 풍력 모델의 "개념 이동"(최대 발전량 282 → 102 MW)은 대부분 이 변화로 설명된다.
- 제주 시장참여 태양광 설비(KPX 제주본부, 2024 분기별): 445.7 / 445.9 / 447.3 / 449.5 MW. 제주 전체 태양광은 643.4 MW.
- 평가 분모(실효 설비용량) 정하는 법: [solar_evaluation_criteria.md](solar_evaluation_criteria.md) §3

### 4.5 2025년 파일 형식 변경

2025년 KPX 파일은 2024년과 형식이 다르므로 로더를 그대로 쓰지 말고 확인 후 변환한다.
- 날짜 헤더가 `거래일`이다.
- 시각 컬럼이 0으로 채운 두 자리 형식이다.
- 제주 풍력 지역 라벨이 `제주도`다.
- 2025-03부터 풍력 거래량이 다시 크게 줄어든다 (포괄 범위 변화 의심, 학습 전 확인 필요).

2025년 자료는 아직 쓰지 않은 **새 hold-out**으로 남겨 둔다 (2024-11~12 테스트 구간은 이미 사용함).

### 4.6 다운로드한 파일 로드

```python
from src.data import KPXScraper

scraper = KPXScraper()
df = scraper.load_downloaded_data(
    "data/raw/kpx/solar_generation_2024.xlsx",
    energy_type="solar"
)
```

---

## 5. 수치예보(NWP) 과거 예보 수집

하루 전 예측을 정직하게 평가하려면 관측 기상 대신 **그때 발표된 예보**가 필요하다.

```bash
# JMA MSM, 2024 전체 → data/external/nwp/openmeteo_previous_runs_jma_msm.csv (git 제외)
python scripts/download_nwp_openmeteo.py

# 다른 모델·기간
python scripts/download_nwp_openmeteo.py --model ecmwf_ifs025 --start 2024-03-01
```

- 출처: Open-Meteo Previous Runs API, **JMA MSM (5 km)**. 2024년 1월부터 일사·기온·습도·운량이 모두 있는 유일한 모델이다 (ECMWF IFS 0.25는 2024-03 이후만, KMA·GFS는 2024년 보관분 없음).
- `previous_day2` = 유효 시각보다 48시간 전 발표분. KPX 하루 전 10시 제출 시점에 확실히 있으므로 **주 지표**로 쓴다.
- `previous_day1` = 24시간 전 발표분. 제출 시점 이후 발표분이 섞일 수 있어 **낙관적 보조값**이다.
- 단위 변환: 일사 W/m²(직전 1시간 평균) × 0.0036 → MJ/m²(ASOS와 같은 직전 1시간 누적), 운량 % ÷ 10 → 0~10.
- 관측소 4곳(제주·고산·성산·서귀포) 좌표의 격자값을 받는다.
- 라이선스: Open-Meteo 무료 API는 **비상업** 용도, 데이터는 **CC BY 4.0** (출처 표기 필요).
- 모델 선택·CV: `python scripts/train_solar_v2.py --weather nwp_msm_day2 --cv` (§7 CV 8.5~8.7%, 저장 없음). 테스트 수치(8.65% / 섬 평균 8.53%) 재현: `--weather nwp_msm_day2 --evaluate-test [--variant island_mean]` → `models/solar_v2_nwp/` (테스트 구간은 이미 2회 사용했으므로 모델 선택에 다시 쓰지 않는다)

---

## 6. 샘플 데이터 생성 (테스트용)

API 키나 크롤러 없이 모델 개발을 테스트하려면:

```bash
# 샘플 데이터 생성
python scripts/download_data.py --sample --year 2024
```

또는 Python에서:

```python
from src.data import create_sample_weather_data, create_merged_sample_data
from datetime import datetime

# 기상 샘플
weather = create_sample_weather_data(
    datetime(2024, 1, 1),
    datetime(2024, 12, 31),
)

# 기상 + 발전량 통합 샘플
merged = create_merged_sample_data(
    datetime(2024, 1, 1),
    datetime(2024, 12, 31),
    solar_capacity=500.0,
    wind_capacity=300.0,
)
```

---

## 7. 데이터 품질 체크리스트

### 7.1 기상 데이터

- [ ] 결측치 비율 < 5%
- [ ] 기온 범위: -10°C ~ 40°C
- [ ] 풍속 범위: 0 ~ 50 m/s
- [ ] 일사량: 0 이상, 낮 시간대만 양수
- [ ] 시간 연속성 확인

### 7.2 발전량 데이터

- [ ] 음수 값 없음
- [ ] 설비용량 초과 없음
- [ ] 야간 태양광 = 0
- [ ] 시간대별 누락 없음
- [ ] 구간 끝 시각 라벨 확인 (일사와 시차 0에서 상관 최대)
- [ ] 2024-06-01 전후 포괄 범위 차이 반영

---

## 8. 디렉토리 구조

```
data/
├── raw/
│   ├── weather/
│   │   └── jeju_weather_20{20..24}_all_stations.csv   # ASOS 4개 관측소 시간자료
│   └── kpx/
│       └── 한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_*.csv   # 15065269 (타겟)
├── processed/
│   ├── kpx_real_merged.csv     # KPX 실측 + 기상 평균 (구간 시작 라벨, §4.3)
│   ├── merged_dataset.csv      # 합성 데이터 (초기 대시보드)
│   └── weather_5year_hourly.csv   # ASOS 5년 시간 집계 (풍력 평년값)
├── sample/
│   └── sample_data.csv          # 2024년 실측 전체 (클라우드 대시보드용)
└── external/
    ├── holidays_2024.csv
    └── nwp/                     # Open-Meteo 과거 예보 (git 제외)
```

---

## 9. 트러블슈팅

### Selenium 크롤러

**Chrome 버전 오류**
```
SessionNotCreatedException: This version of ChromeDriver only supports Chrome version XX
```
→ `webdriver-manager`가 자동으로 해결합니다. 안되면:
```bash
pip install --upgrade webdriver-manager
```

**헤드리스 모드 실패**
→ `--no-headless` 옵션으로 브라우저 창을 표시하여 디버깅

### API 오류

**SERVICE_KEY_IS_NOT_REGISTERED_ERROR**
→ API 키가 잘못되었거나 서비스 활용 신청 미승인

**LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR**
→ 일일 호출 한도 초과 (일반: 1,000회/일)

---

## 10. 참고 링크

- [기상자료개방포털](https://data.kma.go.kr/)
- [공공데이터포털](https://www.data.go.kr/)
- [전력통계정보시스템](https://epsis.kpx.or.kr/)
- [KPX 지역별 시간별 태양광·풍력 발전량 (data.go.kr 15065269)](https://www.data.go.kr/data/15065269/fileData.do)
- [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)

---

*Last Updated: 2026-11-07*
