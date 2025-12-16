#!/usr/bin/env python3
"""
KPX 실제 발전량 데이터 다운로드

공공데이터포털에서 한국전력거래소 태양광/풍력 발전량 데이터를 다운로드합니다.

데이터 소스:
- 파일: https://www.data.go.kr/data/15065269/fileData.do
- API: https://apis.data.go.kr/B552115/PvAmountByLocHr/getPvAmountByLocHr
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO
import time

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# 공공데이터포털 API 키
API_KEY = os.getenv("KMA_API_KEY", "")


def download_solar_generation_api(
    start_date: str = "20240101",
    end_date: str = "20241231",
    region: str = "제주",
) -> pd.DataFrame:
    """
    공공데이터포털 API로 태양광 발전량 데이터 다운로드

    Args:
        start_date: 시작일 (YYYYMMDD)
        end_date: 종료일 (YYYYMMDD)
        region: 지역명 (제주, 서울, 경기 등)

    Returns:
        발전량 DataFrame
    """
    base_url = "https://apis.data.go.kr/B552115/PvAmountByLocHr/getPvAmountByLocHr"

    all_data = []
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    print(f"Downloading solar generation data from API...")
    print(f"Period: {start_date} ~ {end_date}")
    print(f"Region filter: {region}")

    while current_date <= end_dt:
        date_str = current_date.strftime("%Y%m%d")

        params = {
            "serviceKey": API_KEY,
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "json",
            "tradeYmd": date_str,
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if "response" in data and "body" in data["response"]:
                    items = data["response"]["body"].get("items", {}).get("item", [])

                    if items:
                        if isinstance(items, dict):
                            items = [items]

                        # 지역 필터링
                        for item in items:
                            if region in item.get("regionNm", ""):
                                all_data.append({
                                    "date": item.get("tradeYmd"),
                                    "hour": item.get("tradeNo"),
                                    "region": item.get("regionNm"),
                                    "solar_mwh": float(item.get("amgo", 0)),
                                })

                        if current_date.day == 1:
                            print(f"  {date_str}: {len([i for i in items if region in i.get('regionNm', '')])} records")

            time.sleep(0.1)  # API 호출 간격

        except Exception as e:
            print(f"  Error on {date_str}: {e}")

        current_date += timedelta(days=1)

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"\nTotal records: {len(df)}")
        return df
    else:
        print("No data retrieved from API")
        return pd.DataFrame()


def download_file_data_csv():
    """
    공공데이터포털 파일데이터 자동변환 API를 통해 CSV 다운로드

    참고: 공공데이터포털은 파일데이터를 REST API로 제공합니다.
    """
    # 파일데이터 API 엔드포인트
    # 데이터셋 ID: 15065269
    file_api_url = "https://apis.data.go.kr/B552115/PvAmountByLocHr/getPvAmountByLocHr"

    print("Attempting to download via file data API...")
    print("Note: Large dataset - this may take several minutes")

    # API를 통한 전체 데이터 조회
    all_data = []
    page = 1
    total_count = None

    while True:
        params = {
            "serviceKey": API_KEY,
            "pageNo": page,
            "numOfRows": 1000,
            "dataType": "json",
        }

        try:
            response = requests.get(file_api_url, params=params, timeout=60)

            if response.status_code == 200:
                data = response.json()

                if "response" in data:
                    body = data["response"].get("body", {})

                    if total_count is None:
                        total_count = body.get("totalCount", 0)
                        print(f"Total records available: {total_count}")

                    items = body.get("items", {}).get("item", [])

                    if not items:
                        break

                    if isinstance(items, dict):
                        items = [items]

                    all_data.extend(items)

                    if page % 10 == 0:
                        print(f"  Downloaded page {page}, total: {len(all_data)} records")

                    if len(all_data) >= total_count:
                        break

                    page += 1
                    time.sleep(0.1)
                else:
                    break
            else:
                print(f"API error: {response.status_code}")
                break

        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

    if all_data:
        df = pd.DataFrame(all_data)
        return df

    return pd.DataFrame()


def create_sample_download_script():
    """
    수동 다운로드 안내 스크립트 생성

    공공데이터포털에서 직접 다운로드해야 하는 경우를 위한 안내
    """
    instructions = """
# KPX 발전량 데이터 수동 다운로드 가이드

## 1. 파일 데이터 (권장)
URL: https://www.data.go.kr/data/15065269/fileData.do

1. 위 URL 접속
2. "다운로드" 버튼 클릭 (로그인 불필요)
3. CSV 파일 다운로드
4. 다운로드한 파일을 다음 경로에 저장:
   data/raw/kpx/kpx_solar_wind_2024.csv

## 2. 오픈 API
URL: https://www.data.go.kr/data/15103243/openapi.do

1. 공공데이터포털 회원가입
2. API 활용 신청
3. 발급받은 API 키를 .env 파일에 추가:
   KPX_API_KEY=your_api_key_here

## 데이터 항목
- 거래일자 (tradeYmd)
- 거래시간 (tradeNo)
- 지역 (regionNm)
- 연료원 (fuelType)
- 발전량 (amgo, MWh 단위)

## 제주 데이터 필터링
- 태양광: regionNm == "제주"
- 풍력: regionNm == "제주" (2023.12 이후 육지/제주로 구분)
"""

    guide_path = RAW_DATA_DIR / "kpx" / "DOWNLOAD_GUIDE.md"
    guide_path.parent.mkdir(parents=True, exist_ok=True)

    with open(guide_path, "w") as f:
        f.write(instructions)

    print(f"Download guide saved to: {guide_path}")
    return guide_path


def process_kpx_data(input_path: Path) -> pd.DataFrame:
    """
    다운로드한 KPX 데이터 전처리

    Args:
        input_path: 원본 CSV 경로

    Returns:
        전처리된 DataFrame
    """
    print(f"Processing KPX data from: {input_path}")

    # CSV 로드 (인코딩 시도)
    for encoding in ["utf-8", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            print(f"  Loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Cannot decode CSV file")

    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # 컬럼명 정규화
    column_mapping = {
        "거래일자": "date",
        "tradeYmd": "date",
        "거래시간": "hour",
        "tradeNo": "hour",
        "지역": "region",
        "regionNm": "region",
        "연료원": "fuel_type",
        "fuelType": "fuel_type",
        "발전량": "generation_mwh",
        "전력거래량": "generation_mwh",
        "amgo": "generation_mwh",
    }

    df = df.rename(columns=column_mapping)

    # 제주 데이터 필터링
    if "region" in df.columns:
        jeju_df = df[df["region"].str.contains("제주", na=False)].copy()
        print(f"  Jeju records: {len(jeju_df)}")
    else:
        jeju_df = df.copy()

    # datetime 생성
    if "date" in jeju_df.columns and "hour" in jeju_df.columns:
        jeju_df["date"] = jeju_df["date"].astype(str)
        jeju_df["hour"] = jeju_df["hour"].astype(int)

        jeju_df["datetime"] = pd.to_datetime(
            jeju_df["date"] + " " + (jeju_df["hour"] - 1).astype(str).str.zfill(2) + ":00:00"
        )

    # 연료원별 분리
    if "fuel_type" in jeju_df.columns:
        solar_df = jeju_df[jeju_df["fuel_type"].str.contains("태양", na=False)].copy()
        wind_df = jeju_df[jeju_df["fuel_type"].str.contains("풍력", na=False)].copy()

        print(f"  Solar records: {len(solar_df)}")
        print(f"  Wind records: {len(wind_df)}")

        # 시간별 집계
        if len(solar_df) > 0:
            solar_hourly = solar_df.groupby("datetime")["generation_mwh"].sum().reset_index()
            solar_hourly.columns = ["datetime", "solar_mw"]
            # MWh -> MW 변환 (1시간 기준이면 동일)
        else:
            solar_hourly = pd.DataFrame(columns=["datetime", "solar_mw"])

        if len(wind_df) > 0:
            wind_hourly = wind_df.groupby("datetime")["generation_mwh"].sum().reset_index()
            wind_hourly.columns = ["datetime", "wind_mw"]
        else:
            wind_hourly = pd.DataFrame(columns=["datetime", "wind_mw"])

        # 병합
        if len(solar_hourly) > 0 and len(wind_hourly) > 0:
            result_df = solar_hourly.merge(wind_hourly, on="datetime", how="outer")
        elif len(solar_hourly) > 0:
            result_df = solar_hourly
            result_df["wind_mw"] = 0
        elif len(wind_hourly) > 0:
            result_df = wind_hourly
            result_df["solar_mw"] = 0
        else:
            result_df = pd.DataFrame()
    else:
        result_df = jeju_df

    result_df = result_df.sort_values("datetime").reset_index(drop=True)

    print(f"  Final shape: {result_df.shape}")

    return result_df


def main():
    print("=" * 60)
    print("KPX 실제 발전량 데이터 다운로드")
    print("=" * 60)

    output_dir = RAW_DATA_DIR / "kpx"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. API로 데이터 다운로드 시도
    if API_KEY:
        print("\n[1] Trying API download...")
        df = download_solar_generation_api(
            start_date="20240101",
            end_date="20241231",
            region="제주"
        )

        if len(df) > 0:
            output_path = output_dir / "kpx_jeju_solar_api_2024.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
    else:
        print("\n[1] No API key found, skipping API download")

    # 2. 다운로드 가이드 생성
    print("\n[2] Creating download guide...")
    create_sample_download_script()

    # 3. 기존 파일 확인 및 처리
    existing_files = list(output_dir.glob("*.csv"))
    print(f"\n[3] Checking existing files: {len(existing_files)} found")

    for file_path in existing_files:
        if "processed" not in file_path.name:
            try:
                processed_df = process_kpx_data(file_path)
                if len(processed_df) > 0:
                    output_path = output_dir / f"processed_{file_path.stem}.csv"
                    processed_df.to_csv(output_path, index=False)
                    print(f"Processed: {output_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
