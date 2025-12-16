# 데이터 수집 가이드

J-REF 프로젝트에 필요한 데이터 소스와 수집 방법을 설명합니다.

---

## 1. 데이터 소스 개요

| 데이터 | 출처 | 용도 | 수집 방법 |
|--------|------|------|-----------|
| **ASOS** | 기상청 | 종관기상관측 (시간별) | **Selenium 크롤러** (권장) |
| **AWS** | 기상청 | 방재기상관측 (10분별) | Open API |
| **발전량** | 전력거래소(KPX) | 시간별 발전량 | 수동 다운로드 |

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

### 4.3 다운로드한 파일 로드

```python
from src.data import KPXScraper

scraper = KPXScraper()
df = scraper.load_downloaded_data(
    "data/raw/kpx/solar_generation_2024.xlsx",
    energy_type="solar"
)
```

---

## 5. 샘플 데이터 생성 (테스트용)

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

## 6. 데이터 품질 체크리스트

### 6.1 기상 데이터

- [ ] 결측치 비율 < 5%
- [ ] 기온 범위: -10°C ~ 40°C
- [ ] 풍속 범위: 0 ~ 50 m/s
- [ ] 일사량: 0 이상, 낮 시간대만 양수
- [ ] 시간 연속성 확인

### 6.2 발전량 데이터

- [ ] 음수 값 없음
- [ ] 설비용량 초과 없음
- [ ] 야간 태양광 = 0
- [ ] 시간대별 누락 없음

---

## 7. 디렉토리 구조

```
data/
├── raw/
│   ├── weather/
│   │   ├── 제주_시간_20240101_20241231.csv
│   │   ├── 고산_시간_20240101_20241231.csv
│   │   └── ...
│   └── kpx/
│       ├── solar_generation_2024.csv
│       └── wind_generation_2024.csv
├── processed/
│   ├── solar_features.parquet
│   ├── wind_features.parquet
│   └── merged_dataset.parquet
└── external/
    └── holidays_2024.csv
```

---

## 8. 트러블슈팅

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

## 9. 참고 링크

- [기상자료개방포털](https://data.kma.go.kr/)
- [공공데이터포털](https://www.data.go.kr/)
- [전력통계정보시스템](https://epsis.kpx.or.kr/)

---

*Last Updated: 2024-12*
