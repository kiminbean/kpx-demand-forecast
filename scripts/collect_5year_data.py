"""
5년치 데이터 수집 스크립트 (2020-2024)

Phase 2: 데이터 확장

데이터 소스:
1. KPX 공공데이터포털: 지역별 시간별 태양광 및 풍력 발전량
   - https://www.data.go.kr/data/15065269/fileData.do

2. 기상청 ASOS API: 제주 지역 시간별 기상 데이터
   - https://www.data.go.kr/data/15057210/openapi.do
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# ========================
# KPX 데이터 수집
# ========================

class KPXDataCollector:
    """
    KPX 공공데이터포털에서 발전량 데이터 수집

    API 엔드포인트:
    - 파일 데이터: https://www.data.go.kr/data/15065269/fileData.do
    - OpenAPI 변환 가능
    """

    BASE_URL = "https://apis.data.go.kr/B553530/epower/rdw/rdwTp"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DATA_GO_KR_API_KEY") or os.environ.get("KMA_API_KEY")

    def download_from_portal(self, years: list = None) -> pd.DataFrame:
        """
        공공데이터포털에서 직접 파일 다운로드 (수동 필요)

        실제로는 공공데이터포털에서 CSV 파일을 다운로드해야 합니다.
        여기서는 기존 데이터를 로드하고 처리합니다.
        """
        years = years or [2020, 2021, 2022, 2023, 2024]

        # 기존 다운로드된 파일 확인
        kpx_dir = RAW_DATA_DIR / "kpx"

        all_files = list(kpx_dir.glob("*.csv"))
        print(f"Found {len(all_files)} CSV files in {kpx_dir}")

        # 가장 최근 KPX 데이터 파일 로드
        kpx_file = kpx_dir / "한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20241231_20251216_144253.csv"

        if kpx_file.exists():
            # 인코딩 자동 감지
            for encoding in ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']:
                try:
                    df = pd.read_csv(kpx_file, encoding=encoding)
                    print(f"Loaded with encoding: {encoding}")
                    break
                except:
                    continue
            else:
                print(f"Failed to load {kpx_file}")
                return None

            return df
        else:
            print(f"KPX file not found: {kpx_file}")
            return None


def process_kpx_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPX 데이터 전처리

    원본 컬럼: 거래일자, 거래시간, 지역, 발전원, 시간대거래량(MWh)
    """
    # 컬럼명 정리 (한글 인코딩 문제 대응)
    df.columns = ['date', 'hour', 'region', 'source', 'generation']

    # 제주 데이터만 필터링
    df_jeju = df[df['region'].str.contains('제주', na=False)].copy()

    if len(df_jeju) == 0:
        print("Warning: No Jeju data found, trying alternative filter")
        # 인코딩 문제로 'jeju' 또는 다른 패턴 시도
        df_jeju = df.copy()

    # datetime 생성
    df_jeju['datetime'] = pd.to_datetime(df_jeju['date'].astype(str)) + pd.to_timedelta(df_jeju['hour'].astype(int) - 1, unit='h')

    # 피벗: 태양광/풍력을 컬럼으로
    df_pivot = df_jeju.pivot_table(
        index='datetime',
        columns='source',
        values='generation',
        aggfunc='sum'
    ).reset_index()

    # 컬럼명 정리
    df_pivot.columns.name = None

    # 태양광/풍력 컬럼 찾기
    solar_col = [c for c in df_pivot.columns if '태양' in str(c) or 'solar' in str(c).lower()]
    wind_col = [c for c in df_pivot.columns if '풍력' in str(c) or 'wind' in str(c).lower()]

    result = pd.DataFrame({'datetime': df_pivot['datetime']})

    if solar_col:
        result['solar_mw'] = df_pivot[solar_col[0]].values
    else:
        result['solar_mw'] = 0

    if wind_col:
        result['wind_mw'] = df_pivot[wind_col[0]].values
    else:
        result['wind_mw'] = 0

    result['total_mw'] = result['solar_mw'] + result['wind_mw']

    return result.sort_values('datetime').reset_index(drop=True)


# ========================
# 기상청 ASOS 데이터 수집
# ========================

class KMAASOSCollector:
    """
    기상청 ASOS API를 통한 기상 데이터 수집
    """

    BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    # 제주 지역 관측소
    JEJU_STATIONS = {
        '184': '제주',
        '185': '고산',
        '188': '성산',
        '189': '서귀포',
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("KMA_API_KEY")

        if not self.api_key:
            raise ValueError("KMA_API_KEY 환경변수가 필요합니다.")

    def get_hourly_data(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        특정 관측소의 시간별 데이터 수집

        Args:
            station_id: 관측소 ID (예: '184' = 제주)
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
        """
        start_dt = start_date.replace("-", "")
        end_dt = end_date.replace("-", "")

        all_data = []
        page_no = 1
        num_of_rows = 999

        while True:
            params = {
                'serviceKey': self.api_key,
                'numOfRows': num_of_rows,
                'pageNo': page_no,
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'HR',
                'startDt': start_dt,
                'endDt': end_dt,
                'startHh': '00',
                'endHh': '23',
                'stnIds': station_id,
            }

            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                result_code = data.get('response', {}).get('header', {}).get('resultCode')
                if result_code != '00':
                    result_msg = data.get('response', {}).get('header', {}).get('resultMsg')
                    print(f"API Error: [{result_code}] {result_msg}")
                    break

                items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])

                if not items:
                    break

                all_data.extend(items)

                total_count = int(data.get('response', {}).get('body', {}).get('totalCount', 0))
                if page_no * num_of_rows >= total_count:
                    break

                page_no += 1
                time.sleep(0.5)  # API 호출 간격

            except Exception as e:
                print(f"Request error: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        return df

    def collect_jeju_data(
        self,
        start_year: int,
        end_year: int,
        save_intermediate: bool = True,
    ) -> pd.DataFrame:
        """
        제주 지역 전체 관측소 데이터 수집 (연도별)

        Args:
            start_year: 시작 연도
            end_year: 종료 연도
            save_intermediate: 중간 결과 저장 여부
        """
        all_data = []

        for year in range(start_year, end_year + 1):
            print(f"\n{'='*50}")
            print(f"Collecting {year} data...")
            print('='*50)

            year_file = RAW_DATA_DIR / "weather" / f"jeju_weather_{year}_all_stations.csv"

            # 이미 수집된 데이터가 있으면 로드
            if year_file.exists():
                print(f"Loading cached data: {year_file}")
                df_year = pd.read_csv(year_file)
                all_data.append(df_year)
                continue

            year_data = []

            for station_id, station_name in self.JEJU_STATIONS.items():
                print(f"  Station: {station_name} ({station_id})")

                # 월별로 수집 (API 제한 대응)
                for month in range(1, 13):
                    start_date = f"{year}-{month:02d}-01"

                    if month == 12:
                        end_date = f"{year}-12-31"
                    else:
                        next_month = datetime(year, month + 1, 1)
                        end_date = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")

                    print(f"    {start_date} ~ {end_date}...", end=" ")

                    df = self.get_hourly_data(station_id, start_date, end_date)

                    if len(df) > 0:
                        df['station_id'] = station_id
                        df['station_name'] = station_name
                        year_data.append(df)
                        print(f"OK ({len(df)} rows)")
                    else:
                        print("No data")

                    time.sleep(0.3)

            if year_data:
                df_year = pd.concat(year_data, ignore_index=True)

                # 중간 저장
                if save_intermediate:
                    year_file.parent.mkdir(parents=True, exist_ok=True)
                    df_year.to_csv(year_file, index=False)
                    print(f"Saved: {year_file}")

                all_data.append(df_year)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


def process_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    기상 데이터 전처리
    """
    # 컬럼명 매핑
    column_map = {
        'tm': 'datetime',
        'stnId': 'station_id',
        'stnNm': 'station_name',
        'ta': 'temperature',      # 기온
        'hm': 'humidity',         # 습도
        'ws': 'wind_speed',       # 풍속
        'wd': 'wind_direction',   # 풍향
        'pa': 'pressure',         # 기압
        'ps': 'sea_pressure',     # 해면기압
        'ss': 'sunshine_hours',   # 일조시간
        'icsr': 'solar_radiation',# 일사량
        'dc10Tca': 'cloud_cover', # 운량
        'rn': 'precipitation',    # 강수량
        'vs': 'visibility',       # 시정
    }

    # 존재하는 컬럼만 리네임
    rename_cols = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_cols)

    # datetime 변환
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 수치형 변환
    numeric_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                    'pressure', 'solar_radiation', 'cloud_cover', 'precipitation', 'visibility']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def aggregate_weather_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    여러 관측소 데이터를 시간별로 집계 (평균)
    """
    if 'datetime' not in df.columns:
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # datetime 기준 평균
    df_agg = df.groupby('datetime')[numeric_cols].mean().reset_index()

    return df_agg


# ========================
# 메인 실행
# ========================

def main():
    print("=" * 70)
    print("Phase 2: 5년치 데이터 수집 (2020-2024)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 기상 데이터 수집
    print("\n[1] Collecting Weather Data (KMA ASOS)...")

    try:
        kma = KMAASOSCollector()

        # 5년치 수집 (2020-2024)
        df_weather = kma.collect_jeju_data(2020, 2024, save_intermediate=True)

        if len(df_weather) > 0:
            print(f"\nTotal weather records: {len(df_weather)}")

            # 전처리
            df_weather = process_weather_data(df_weather)

            # 시간별 집계
            df_weather_hourly = aggregate_weather_by_hour(df_weather)

            # 저장
            output_file = RAW_DATA_DIR / "weather" / "jeju_weather_2020_2024_hourly.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_weather_hourly.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
            print(f"Weather data: {len(df_weather_hourly)} hourly records")
        else:
            print("No weather data collected")

    except Exception as e:
        print(f"Weather collection error: {e}")
        df_weather_hourly = None

    # 2. KPX 발전량 데이터 확인
    print("\n[2] Checking KPX Generation Data...")

    kpx = KPXDataCollector()
    df_kpx = kpx.download_from_portal()

    if df_kpx is not None:
        print(f"KPX data loaded: {len(df_kpx)} records")
        print(f"Columns: {df_kpx.columns.tolist()}")

        # 전처리
        df_kpx_processed = process_kpx_data(df_kpx)

        if df_kpx_processed is not None:
            print(f"Processed KPX data: {len(df_kpx_processed)} hourly records")

            # 저장
            output_file = RAW_DATA_DIR / "kpx" / "jeju_generation_2024_processed.csv"
            df_kpx_processed.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
    else:
        print("KPX data not available")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return df_weather_hourly, df_kpx


if __name__ == "__main__":
    df_weather, df_kpx = main()
