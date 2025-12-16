"""
KPX 5년치 발전량 데이터 수집 스크립트

공공데이터포털 API 활용:
https://www.data.go.kr/data/15065269/fileData.do
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class KPXAPICollector:
    """
    KPX 전력거래량 데이터 API 수집

    공공데이터포털 API:
    - 지역별 시간별 태양광 및 풍력 발전량
    """

    # 실제 API 엔드포인트 (공공데이터포털에서 확인 필요)
    BASE_URL = "https://apis.data.go.kr/B553530/epower/rdw/rdwTp"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DATA_GO_KR_API_KEY") or os.environ.get("KMA_API_KEY")

        if not self.api_key:
            print("Warning: No API key found. Using existing downloaded data.")

    def try_collect_api(self, year: int) -> pd.DataFrame:
        """API로 데이터 수집 시도"""
        if not self.api_key:
            return None

        # API 파라미터 (실제 API 스펙에 맞게 조정 필요)
        params = {
            'serviceKey': self.api_key,
            'numOfRows': 10000,
            'pageNo': 1,
            'dataType': 'JSON',
            'baseYr': str(year),
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                # 데이터 파싱 로직 (API 응답 구조에 따라)
                items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                if items:
                    return pd.DataFrame(items)
            else:
                print(f"API response: {response.status_code}")

        except Exception as e:
            print(f"API error: {e}")

        return None


def load_existing_kpx_data() -> pd.DataFrame:
    """기존 KPX 데이터 로드 (이미 다운로드된 파일)"""
    kpx_dir = RAW_DATA_DIR / "kpx"

    # 가장 큰 (가장 완전한) 파일 찾기
    csv_files = list(kpx_dir.glob("*.csv"))

    all_data = []

    for f in csv_files:
        # 한국전력거래소 파일 찾기
        if '한국전력거래소' in f.name or 'kpx' in f.name.lower():
            for encoding in ['cp949', 'utf-8', 'euc-kr', 'utf-8-sig']:
                try:
                    df = pd.read_csv(f, encoding=encoding)
                    print(f"Loaded {f.name} with {encoding}: {len(df)} rows")

                    # 컬럼명 정규화
                    if '거래일자' in df.columns:
                        all_data.append(df)
                        break
                except:
                    continue

    if all_data:
        return pd.concat(all_data, ignore_index=True).drop_duplicates()
    return None


def process_kpx_to_jeju_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPX 원본 데이터를 제주 시간별 발전량으로 변환

    원본 컬럼: 거래일자, 거래시간, 지역, 연료원, 전력거래량(MWh)
    """
    print(f"Input data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # 컬럼명 표준화
    df.columns = ['date', 'hour', 'region', 'source', 'generation']

    # 날짜 범위 확인
    df['date'] = pd.to_datetime(df['date'].astype(str))
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

    # 지역별 데이터 수
    print(f"\nRegion distribution:")
    print(df['region'].value_counts())

    # 연료원별 데이터 수
    print(f"\nSource distribution:")
    print(df['source'].value_counts())

    # 제주 데이터 필터링
    df_jeju = df[df['region'].str.contains('제주', na=False)].copy()
    print(f"\nJeju data: {len(df_jeju)} rows")

    if len(df_jeju) == 0:
        print("Warning: No Jeju data found!")
        return None

    # datetime 생성 (hour는 1-24, 0-23으로 변환)
    df_jeju['datetime'] = pd.to_datetime(df_jeju['date'].astype(str)) + \
                          pd.to_timedelta(df_jeju['hour'].astype(int) - 1, unit='h')

    # 태양광/풍력 분리
    df_solar = df_jeju[df_jeju['source'].str.contains('태양', na=False)].copy()
    df_wind = df_jeju[df_jeju['source'].str.contains('풍력', na=False)].copy()

    print(f"Solar: {len(df_solar)}, Wind: {len(df_wind)}")

    # 시간별 집계
    solar_hourly = df_solar.groupby('datetime')['generation'].sum().reset_index()
    solar_hourly.columns = ['datetime', 'solar_mw']

    wind_hourly = df_wind.groupby('datetime')['generation'].sum().reset_index()
    wind_hourly.columns = ['datetime', 'wind_mw']

    # 병합
    result = pd.merge(solar_hourly, wind_hourly, on='datetime', how='outer')
    result = result.fillna(0)
    result['total_mw'] = result['solar_mw'] + result['wind_mw']

    # 정렬
    result = result.sort_values('datetime').reset_index(drop=True)

    print(f"\nResult: {len(result)} hourly records")
    print(f"Date range: {result['datetime'].min()} ~ {result['datetime'].max()}")

    # 연도별 데이터 수
    result['year'] = result['datetime'].dt.year
    print(f"\nRecords by year:")
    print(result.groupby('year').size())

    return result.drop('year', axis=1)


def main():
    print("=" * 70)
    print("KPX 5년치 발전량 데이터 수집")
    print("=" * 70)

    # 1. 기존 데이터 로드
    print("\n[1] Loading existing KPX data...")
    df_raw = load_existing_kpx_data()

    if df_raw is not None:
        print(f"Loaded total: {len(df_raw)} rows")

        # 2. 전처리
        print("\n[2] Processing to Jeju hourly data...")
        df_jeju = process_kpx_to_jeju_hourly(df_raw)

        if df_jeju is not None:
            # 3. 저장
            output_file = RAW_DATA_DIR / "kpx" / "jeju_generation_all_years.csv"
            df_jeju.to_csv(output_file, index=False)
            print(f"\nSaved: {output_file}")

            # 통계
            print("\n" + "=" * 50)
            print("Summary Statistics")
            print("=" * 50)
            print(f"Total records: {len(df_jeju)}")
            print(f"Date range: {df_jeju['datetime'].min()} ~ {df_jeju['datetime'].max()}")
            print(f"\nSolar MW: min={df_jeju['solar_mw'].min():.2f}, max={df_jeju['solar_mw'].max():.2f}, mean={df_jeju['solar_mw'].mean():.2f}")
            print(f"Wind MW: min={df_jeju['wind_mw'].min():.2f}, max={df_jeju['wind_mw'].max():.2f}, mean={df_jeju['wind_mw'].mean():.2f}")

            return df_jeju
    else:
        print("No KPX data found. Please download from data.go.kr")

    return None


if __name__ == "__main__":
    df = main()
