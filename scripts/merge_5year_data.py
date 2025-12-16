"""
5년치 데이터 병합 및 전처리 스크립트

기상 데이터: 2020-2024 (5년)
KPX 발전량: 2024 (1년) - 추후 확장 가능

전략:
1. 5년치 기상 데이터 병합
2. 2024년 KPX 데이터와 매칭
3. Cross-Year Validation을 위한 기상 패턴 활용
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def normalize_weather_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """각 연도별 기상 데이터 컬럼 정규화"""
    df = df.copy()

    # 컬럼명 매핑 (Raw → Processed)
    column_map = {
        'tm': 'datetime',
        'ta': 'temperature',
        'hm': 'humidity',
        'ws': 'wind_speed',
        'wd': 'wind_direction',
        'pa': 'pressure',
        'ps': 'sea_pressure',
        'ss': 'sunshine_hours',
        'icsr': 'solar_radiation',
        'dc10Tca': 'cloud_cover',
        'rn': 'precipitation',
        'vs': 'visibility',
    }

    # 이미 전처리된 데이터인지 확인
    is_already_processed = 'datetime' in df.columns and 'temperature' in df.columns

    if not is_already_processed:
        # Raw 데이터 → 표준 컬럼명으로 변환
        df = df.rename(columns=column_map)

    # datetime 변환
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # 필요한 컬럼만 선택
    use_cols = ['datetime', 'temperature', 'humidity', 'wind_speed', 'wind_direction',
                'pressure', 'solar_radiation', 'cloud_cover', 'precipitation', 'visibility',
                'station_id', 'station_name']
    use_cols = [c for c in use_cols if c in df.columns]
    df = df[use_cols]

    df['year'] = year
    return df


def load_weather_data(years: list = None) -> pd.DataFrame:
    """5년치 기상 데이터 로드 및 병합 (각 연도별 정규화 후 병합)"""
    years = years or [2020, 2021, 2022, 2023, 2024]

    weather_dir = RAW_DATA_DIR / "weather"
    all_data = []

    for year in years:
        file_path = weather_dir / f"jeju_weather_{year}_all_stations.csv"

        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"Loaded {year}: {len(df)} rows, columns: {list(df.columns[:5])}...")

            # 각 연도별로 컬럼 정규화
            df_normalized = normalize_weather_columns(df, year)
            all_data.append(df_normalized)
        else:
            print(f"Warning: {file_path} not found")

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        print(f"Total weather records: {len(df)}")
        print(f"Date range: {df['datetime'].min()} ~ {df['datetime'].max()}")
        return df
    return None


def process_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """기상 데이터 시간별 집계

    load_weather_data에서 이미 컬럼 정규화가 완료됨
    여기서는 시간별 평균 집계만 수행
    """
    df = df.copy()

    # NaT datetime 제거
    df = df.dropna(subset=['datetime'])

    # 사용할 컬럼
    use_cols = ['datetime', 'temperature', 'humidity', 'wind_speed', 'wind_direction',
                'pressure', 'solar_radiation', 'cloud_cover', 'precipitation', 'visibility']
    use_cols = [c for c in use_cols if c in df.columns]

    # 시간별 평균 (여러 관측소)
    numeric_cols = [c for c in use_cols if c != 'datetime']
    df_hourly = df.groupby('datetime')[numeric_cols].mean().reset_index()

    # 결측치 처리
    df_hourly = df_hourly.fillna(method='ffill').fillna(method='bfill')

    # 정렬
    df_hourly = df_hourly.sort_values('datetime').reset_index(drop=True)

    return df_hourly


def load_kpx_data() -> pd.DataFrame:
    """KPX 발전량 데이터 로드"""
    kpx_file = RAW_DATA_DIR / "kpx" / "jeju_generation_all_years.csv"

    if kpx_file.exists():
        df = pd.read_csv(kpx_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"KPX data: {len(df)} rows")
        return df

    # 대안: 처리된 파일
    alt_file = RAW_DATA_DIR / "kpx" / "jeju_generation_2024_processed.csv"
    if alt_file.exists():
        df = pd.read_csv(alt_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"KPX data (alt): {len(df)} rows")
        return df

    return None


def merge_weather_kpx(df_weather: pd.DataFrame, df_kpx: pd.DataFrame) -> pd.DataFrame:
    """기상 데이터와 KPX 데이터 병합"""

    # KPX 데이터 날짜 범위
    kpx_start = df_kpx['datetime'].min()
    kpx_end = df_kpx['datetime'].max()
    print(f"KPX date range: {kpx_start} ~ {kpx_end}")

    # 기상 데이터 날짜 범위
    weather_start = df_weather['datetime'].min()
    weather_end = df_weather['datetime'].max()
    print(f"Weather date range: {weather_start} ~ {weather_end}")

    # KPX 범위에 맞는 기상 데이터만 사용 (2024년)
    df_weather_2024 = df_weather[
        (df_weather['datetime'] >= kpx_start) &
        (df_weather['datetime'] <= kpx_end)
    ].copy()

    print(f"Weather data for 2024: {len(df_weather_2024)} rows")

    # 병합
    df_merged = pd.merge(df_kpx, df_weather_2024, on='datetime', how='inner')
    print(f"Merged data: {len(df_merged)} rows")

    return df_merged


def create_5year_weather_patterns(df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    5년치 기상 패턴 통계 생성

    목적: 각 시간대/월별 기상 변동성을 학습에 활용
    """
    df = df_weather.copy()
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear

    # 시간-월 조합별 통계
    stats_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                  'pressure', 'solar_radiation', 'cloud_cover']
    stats_cols = [c for c in stats_cols if c in df.columns]

    # 그룹별 통계
    patterns = df.groupby(['month', 'hour'])[stats_cols].agg(['mean', 'std', 'min', 'max'])
    patterns.columns = ['_'.join(col).strip() for col in patterns.columns.values]
    patterns = patterns.reset_index()

    return patterns


def add_historical_weather_features(df: pd.DataFrame, df_weather_all: pd.DataFrame) -> pd.DataFrame:
    """
    과거 기상 패턴을 현재 데이터의 피처로 추가

    5년치 기상 데이터에서 추출한 통계를 피처로 사용
    """
    df = df.copy()

    # 기본 시간 정보
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear

    # 5년치 기상 패턴 통계
    weather_patterns = create_5year_weather_patterns(df_weather_all)

    # 병합 (month, hour 기준)
    df = pd.merge(df, weather_patterns, on=['month', 'hour'], how='left', suffixes=('', '_hist'))

    # 현재 값과 과거 평균의 편차
    if 'wind_speed' in df.columns and 'wind_speed_mean' in df.columns:
        df['wind_speed_anomaly'] = df['wind_speed'] - df['wind_speed_mean']

    if 'temperature' in df.columns and 'temperature_mean' in df.columns:
        df['temperature_anomaly'] = df['temperature'] - df['temperature_mean']

    if 'solar_radiation' in df.columns and 'solar_radiation_mean' in df.columns:
        df['solar_radiation_anomaly'] = df['solar_radiation'] - df['solar_radiation_mean']

    return df


def main():
    print("=" * 70)
    print("Phase 2: 5년치 데이터 병합 및 전처리")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 기상 데이터 로드 (5년치)
    print("\n[1] Loading 5-year weather data...")
    df_weather_raw = load_weather_data([2020, 2021, 2022, 2023, 2024])

    if df_weather_raw is None:
        print("Error: No weather data found")
        return None

    # 2. 기상 데이터 전처리
    print("\n[2] Processing weather data...")
    df_weather = process_weather_data(df_weather_raw)
    print(f"Processed weather: {len(df_weather)} hourly records")
    print(f"Date range: {df_weather['datetime'].min()} ~ {df_weather['datetime'].max()}")

    # 3. KPX 데이터 로드
    print("\n[3] Loading KPX generation data...")
    df_kpx = load_kpx_data()

    if df_kpx is None:
        print("Error: No KPX data found")
        return None

    # 4. 데이터 병합
    print("\n[4] Merging weather and KPX data...")
    df_merged = merge_weather_kpx(df_weather, df_kpx)

    # 5. 과거 기상 패턴 피처 추가
    print("\n[5] Adding historical weather pattern features...")
    df_final = add_historical_weather_features(df_merged, df_weather)

    # 6. 저장
    output_file = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # 5년치 기상 데이터 저장 (별도)
    weather_output = PROCESSED_DATA_DIR / "weather_5year_hourly.csv"
    df_weather.to_csv(weather_output, index=False)
    print(f"Saved: {weather_output}")

    # 기상 패턴 통계 저장
    patterns = create_5year_weather_patterns(df_weather)
    patterns_output = PROCESSED_DATA_DIR / "weather_5year_patterns.csv"
    patterns.to_csv(patterns_output, index=False)
    print(f"Saved: {patterns_output}")

    # 통계 요약
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Final dataset: {len(df_final)} rows, {len(df_final.columns)} columns")
    print(f"Weather patterns: {len(patterns)} combinations")

    print(f"\nTarget statistics:")
    print(f"  Solar MW: mean={df_final['solar_mw'].mean():.2f}, std={df_final['solar_mw'].std():.2f}")
    print(f"  Wind MW:  mean={df_final['wind_mw'].mean():.2f}, std={df_final['wind_mw'].std():.2f}")

    print(f"\nNew features added:")
    new_cols = [c for c in df_final.columns if 'anomaly' in c or '_hist' in c or
                c.endswith('_mean') or c.endswith('_std') or c.endswith('_min') or c.endswith('_max')]
    print(f"  {len(new_cols)} historical pattern features")

    return df_final


if __name__ == "__main__":
    df = main()
