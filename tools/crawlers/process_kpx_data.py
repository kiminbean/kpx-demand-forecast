"""
KPX 데이터 전처리 모듈

다운로드한 KPX 발전량 데이터를 학습용 형식으로 변환합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_kpx_generation(
    input_path: str,
    output_path: Optional[str] = None,
    region: str = "제주",
) -> pd.DataFrame:
    """
    KPX 발전량 데이터 전처리

    Args:
        input_path: 입력 CSV 경로
        output_path: 출력 CSV 경로 (None이면 저장 안함)
        region: 필터링할 지역 ("제주" 또는 "제주도")

    Returns:
        전처리된 DataFrame
    """
    logger.info(f"Processing: {input_path}")

    # 파일 읽기 (인코딩 시도)
    for encoding in ['euc-kr', 'cp949', 'utf-8']:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            logger.info(f"Loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Cannot decode file")

    logger.info(f"Raw data: {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")

    # 컬럼명 정규화
    column_map = {
        '거래일자': 'date',
        '거래시간': 'hour',
        '지역': 'region',
        '연료원': 'fuel_type',
        '전력거래량(MWh)': 'generation_mwh',
    }
    df = df.rename(columns=column_map)

    # 지역 필터링 (제주 또는 제주도)
    jeju_mask = df['region'].str.contains(region, na=False)
    jeju_df = df[jeju_mask].copy()
    logger.info(f"Jeju records: {len(jeju_df)}")

    if len(jeju_df) == 0:
        logger.warning(f"No data found for region: {region}")
        logger.info(f"Available regions: {df['region'].unique()}")
        return pd.DataFrame()

    # datetime 생성
    jeju_df['date'] = pd.to_datetime(jeju_df['date'])
    jeju_df['hour'] = jeju_df['hour'].astype(int)

    # 시간을 0-23 형식으로 (1-24 → 0-23)
    jeju_df['hour_adjusted'] = jeju_df['hour'] - 1

    jeju_df['datetime'] = pd.to_datetime(
        jeju_df['date'].astype(str) + ' ' +
        jeju_df['hour_adjusted'].astype(str).str.zfill(2) + ':00:00'
    )

    # 연료원별 분리 및 피벗
    solar_df = jeju_df[jeju_df['fuel_type'] == '태양광'][['datetime', 'generation_mwh']]
    solar_df = solar_df.rename(columns={'generation_mwh': 'solar_mwh'})

    wind_df = jeju_df[jeju_df['fuel_type'] == '풍력'][['datetime', 'generation_mwh']]
    wind_df = wind_df.rename(columns={'generation_mwh': 'wind_mwh'})

    logger.info(f"Solar records: {len(solar_df)}")
    logger.info(f"Wind records: {len(wind_df)}")

    # 병합
    if len(solar_df) > 0 and len(wind_df) > 0:
        result_df = solar_df.merge(wind_df, on='datetime', how='outer')
    elif len(solar_df) > 0:
        result_df = solar_df.copy()
        result_df['wind_mwh'] = 0
    elif len(wind_df) > 0:
        result_df = wind_df.copy()
        result_df['solar_mwh'] = 0
    else:
        return pd.DataFrame()

    # 정렬 및 결측치 처리
    result_df = result_df.sort_values('datetime').reset_index(drop=True)
    result_df = result_df.fillna(0)

    # MWh → MW 변환 (1시간 기준이면 동일)
    result_df['solar_mw'] = result_df['solar_mwh']
    result_df['wind_mw'] = result_df['wind_mwh']
    result_df['total_mw'] = result_df['solar_mw'] + result_df['wind_mw']

    # 최종 컬럼 선택
    result_df = result_df[['datetime', 'solar_mw', 'wind_mw', 'total_mw']]

    logger.info(f"Final data: {len(result_df)} rows")
    logger.info(f"Date range: {result_df['datetime'].min()} ~ {result_df['datetime'].max()}")
    logger.info(f"Solar range: {result_df['solar_mw'].min():.2f} ~ {result_df['solar_mw'].max():.2f} MW")
    logger.info(f"Wind range: {result_df['wind_mw'].min():.2f} ~ {result_df['wind_mw'].max():.2f} MW")

    # 저장
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path}")

    return result_df


def merge_with_weather(
    generation_df: pd.DataFrame,
    weather_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    발전량 데이터와 기상 데이터 병합

    Args:
        generation_df: 발전량 DataFrame
        weather_path: 기상 데이터 CSV 경로
        output_path: 출력 경로

    Returns:
        병합된 DataFrame
    """
    logger.info(f"Merging with weather: {weather_path}")

    # 기상 데이터 로드
    weather_df = pd.read_csv(weather_path)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

    logger.info(f"Weather data: {len(weather_df)} rows")

    # 병합
    merged_df = generation_df.merge(weather_df, on='datetime', how='inner')

    logger.info(f"Merged data: {len(merged_df)} rows")

    if output_path:
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path}")

    return merged_df


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="KPX 데이터 전처리")
    parser.add_argument("-i", "--input", type=str, help="입력 CSV 파일")
    parser.add_argument("-o", "--output", type=str, help="출력 CSV 파일")
    parser.add_argument("-r", "--region", type=str, default="제주", help="지역 필터")
    parser.add_argument("-w", "--weather", type=str, help="기상 데이터 CSV (병합용)")

    args = parser.parse_args()

    # 입력 파일 자동 탐색
    if not args.input:
        kpx_files = glob.glob("data/raw/kpx/한국전력거래소*.csv")
        if kpx_files:
            args.input = max(kpx_files, key=lambda x: Path(x).stat().st_mtime)
            print(f"Using latest file: {args.input}")
        else:
            print("No KPX file found. Run kpx_crawler.py first.")
            exit(1)

    # 출력 파일 자동 생성
    if not args.output:
        args.output = "data/processed/kpx_jeju_generation.csv"

    # 전처리 실행
    result_df = process_kpx_generation(
        input_path=args.input,
        output_path=args.output,
        region=args.region
    )

    # 기상 데이터와 병합 (옵션)
    if args.weather and len(result_df) > 0:
        merged_output = args.output.replace('.csv', '_merged.csv')
        merge_with_weather(result_df, args.weather, merged_output)

    print(f"\n✓ 전처리 완료: {len(result_df)} rows")
