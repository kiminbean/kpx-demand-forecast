"""
KPX Data Scraper

한국전력거래소(KPX) 발전량 데이터 수집

데이터 출처:
- 전력통계정보시스템 (EPSIS): https://epsis.kpx.or.kr/
- 전력시장 공개정보: https://oasis.kpx.or.kr/
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import warnings

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from src.config import RAW_DATA_DIR
except ImportError:
    RAW_DATA_DIR = Path("data/raw")


class KPXScraper:
    """
    KPX 발전량 데이터 스크래퍼

    전력통계정보시스템(EPSIS)에서 제주도 신재생에너지 발전량 데이터를 수집합니다.

    Note:
        EPSIS 사이트는 로봇 접근이 제한될 수 있습니다.
        대량 수집 시 적절한 딜레이를 적용하세요.
    """

    # EPSIS 기본 URL
    EPSIS_BASE_URL = "https://epsis.kpx.or.kr"

    # OASIS (전력시장 공개정보) URL
    OASIS_BASE_URL = "https://oasis.kpx.or.kr"

    # 지역 코드
    REGION_CODES = {
        "전국": "00",
        "서울": "01",
        "경기": "02",
        "인천": "03",
        "강원": "04",
        "충북": "05",
        "충남": "06",
        "대전": "07",
        "전북": "08",
        "광주": "09",
        "전남": "10",
        "경북": "11",
        "대구": "12",
        "울산": "13",
        "경남": "14",
        "부산": "15",
        "제주": "16",
        "세종": "17",
    }

    # 에너지원 코드
    ENERGY_CODES = {
        "태양광": "solar",
        "풍력": "wind",
        "수력": "hydro",
        "바이오": "bio",
        "해양": "ocean",
        "연료전지": "fuel_cell",
    }

    def __init__(self, delay: float = 1.0):
        """
        Args:
            delay: 요청 간 딜레이 (초)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        })

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> Optional[requests.Response]:
        """
        HTTP 요청 (재시도 로직 포함)
        """
        time.sleep(self.delay)

        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=30)
                else:
                    response = self.session.post(url, data=data, timeout=30)

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"❌ 요청 실패: {e}")
                return None

        return None

    def fetch_generation_summary(
        self,
        year: int,
        month: Optional[int] = None,
        region: str = "제주",
    ) -> pd.DataFrame:
        """
        월별 발전량 요약 데이터 조회

        Args:
            year: 연도
            month: 월 (None이면 전체 연도)
            region: 지역명

        Returns:
            발전량 요약 DataFrame
        """
        # 이 기능은 EPSIS API가 제공되면 구현 가능
        # 현재는 샘플 데이터 반환
        warnings.warn(
            "EPSIS 실시간 크롤링은 사이트 정책에 따라 제한될 수 있습니다. "
            "수동 다운로드를 권장합니다.",
            UserWarning
        )

        return pd.DataFrame()

    def fetch_hourly_generation(
        self,
        date: datetime,
        energy_type: str = "solar",
        region: str = "제주",
    ) -> pd.DataFrame:
        """
        시간별 발전량 데이터 조회

        Note:
            실제 구현은 EPSIS/OASIS API 또는 웹 크롤링 필요
            현재는 데이터 형식 예시만 제공

        Args:
            date: 조회 날짜
            energy_type: 에너지 유형 ("solar", "wind")
            region: 지역명

        Returns:
            시간별 발전량 DataFrame
        """
        # EPSIS에서 실제 데이터를 가져오려면 로그인 및 세션 관리 필요
        # 여기서는 예상 데이터 형식만 반환
        return pd.DataFrame()

    def load_downloaded_data(
        self,
        filepath: str,
        energy_type: str = "solar",
    ) -> pd.DataFrame:
        """
        수동 다운로드한 EPSIS 데이터 로드 및 정제

        EPSIS에서 수동으로 다운로드한 Excel/CSV 파일을 로드합니다.

        Args:
            filepath: 파일 경로
            energy_type: 에너지 유형

        Returns:
            정제된 DataFrame
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        # 파일 형식에 따라 로드
        if filepath.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        elif filepath.suffix == ".csv":
            # 인코딩 시도
            for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("파일 인코딩을 인식할 수 없습니다.")
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {filepath.suffix}")

        # 데이터 정제
        df = self._clean_epsis_data(df, energy_type)

        return df

    def _clean_epsis_data(
        self,
        df: pd.DataFrame,
        energy_type: str,
    ) -> pd.DataFrame:
        """
        EPSIS 데이터 정제

        Args:
            df: 원시 DataFrame
            energy_type: 에너지 유형

        Returns:
            정제된 DataFrame
        """
        df = df.copy()

        # 일반적인 EPSIS 형식 처리
        # (실제 형식은 다운로드한 파일에 따라 조정 필요)

        # 컬럼명 표준화
        column_mapping = {
            "일시": "datetime",
            "날짜": "date",
            "시간": "hour",
            "발전량": "generation_mw",
            "발전량(MWh)": "generation_mwh",
            "설비용량": "capacity_mw",
            "이용률": "capacity_factor",
        }

        df = df.rename(columns={
            k: v for k, v in column_mapping.items() if k in df.columns
        })

        # datetime 생성
        if "datetime" not in df.columns:
            if "date" in df.columns and "hour" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["hour"].astype(str) + ":00"
                )
            elif "date" in df.columns:
                df["datetime"] = pd.to_datetime(df["date"])

        # 에너지 유형 추가
        df["energy_type"] = energy_type

        # 수치형 변환
        numeric_cols = ["generation_mw", "generation_mwh", "capacity_mw", "capacity_factor"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


def create_sample_generation_data(
    start_date: datetime,
    end_date: datetime,
    energy_type: str = "solar",
    capacity_mw: float = 500.0,
) -> pd.DataFrame:
    """
    샘플 발전량 데이터 생성

    실제 데이터 없이 모델 개발/테스트용 샘플 데이터를 생성합니다.

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        energy_type: 에너지 유형 ("solar", "wind")
        capacity_mw: 설비용량 (MW)

    Returns:
        샘플 발전량 DataFrame
    """
    import numpy as np

    date_range = pd.date_range(start_date, end_date, freq="h")
    n = len(date_range)

    hours = np.array([d.hour for d in date_range])
    days = np.array([d.timetuple().tm_yday for d in date_range])

    if energy_type == "solar":
        # 태양광: 일출-일몰 패턴
        # 기본 일사량 패턴 (정오에 최대)
        solar_pattern = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))

        # 계절 변화 (여름에 높음)
        seasonal = 0.8 + 0.2 * np.sin(2 * np.pi * (days - 80) / 365)

        # 날씨 변동성 (구름 등)
        weather_factor = np.clip(
            0.7 + 0.3 * np.random.random(n) + 0.1 * np.random.randn(n),
            0.1, 1.0
        )

        # 최종 발전량
        generation = capacity_mw * solar_pattern * seasonal * weather_factor

        # 야간은 0
        generation = np.where((hours >= 6) & (hours <= 19), generation, 0)

    else:  # wind
        # 풍력: 랜덤 + 일부 패턴
        # 기본 풍속 패턴 (새벽/저녁에 약간 높음)
        base_pattern = 0.4 + 0.1 * np.sin(2 * np.pi * (hours - 3) / 24)

        # 계절 변화 (겨울에 바람 강함)
        seasonal = 0.7 + 0.3 * np.cos(2 * np.pi * (days - 15) / 365)

        # 날씨 변동성 (높음)
        weather_factor = np.clip(
            0.5 + 0.5 * np.random.random(n) + 0.2 * np.random.randn(n),
            0.05, 1.0
        )

        # 최종 발전량
        generation = capacity_mw * base_pattern * seasonal * weather_factor

    df = pd.DataFrame({
        "datetime": date_range,
        "energy_type": energy_type,
        "region": "제주",
        "generation_mw": generation.round(2),
        "capacity_mw": capacity_mw,
        "capacity_factor": (generation / capacity_mw).round(4),
    })

    return df


def create_merged_sample_data(
    start_date: datetime,
    end_date: datetime,
    solar_capacity: float = 500.0,
    wind_capacity: float = 300.0,
) -> pd.DataFrame:
    """
    태양광 + 풍력 통합 샘플 데이터 생성

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        solar_capacity: 태양광 설비용량 (MW)
        wind_capacity: 풍력 설비용량 (MW)

    Returns:
        통합 샘플 DataFrame
    """
    solar_df = create_sample_generation_data(
        start_date, end_date, "solar", solar_capacity
    )
    wind_df = create_sample_generation_data(
        start_date, end_date, "wind", wind_capacity
    )

    # 태양광/풍력 피벗
    solar_df = solar_df.rename(columns={
        "generation_mw": "solar_mw",
        "capacity_factor": "solar_cf",
    })
    wind_df = wind_df.rename(columns={
        "generation_mw": "wind_mw",
        "capacity_factor": "wind_cf",
    })

    merged = solar_df[["datetime", "solar_mw", "solar_cf"]].merge(
        wind_df[["datetime", "wind_mw", "wind_cf"]],
        on="datetime",
    )

    merged["total_mw"] = merged["solar_mw"] + merged["wind_mw"]
    merged["region"] = "제주"

    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("⚡ KPX Scraper 모듈 테스트")
    print("=" * 60)

    # 샘플 데이터 생성 테스트
    print("\n📊 태양광 샘플 데이터:")
    solar_sample = create_sample_generation_data(
        datetime(2024, 7, 1),
        datetime(2024, 7, 7),
        "solar",
    )
    print(solar_sample.head(24).to_string(index=False))

    print("\n📊 풍력 샘플 데이터:")
    wind_sample = create_sample_generation_data(
        datetime(2024, 1, 1),
        datetime(2024, 1, 7),
        "wind",
    )
    print(wind_sample.head(24).to_string(index=False))

    print("\n📊 통합 샘플 데이터:")
    merged = create_merged_sample_data(
        datetime(2024, 6, 1),
        datetime(2024, 6, 3),
    )
    print(merged.head(24).to_string(index=False))

    print(f"\n✅ 총 {len(merged)}시간 데이터 생성")
