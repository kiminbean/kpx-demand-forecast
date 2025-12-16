"""
Weather Data Fetcher

기상청 API를 통한 기상 데이터 수집

지원 API:
- ASOS (종관기상관측) 시간자료
- AWS (방재기상관측) 시간자료
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

try:
    from src.config import api_config, RAW_DATA_DIR
except ImportError:
    # 독립 실행 시
    from dotenv import load_dotenv
    load_dotenv()
    RAW_DATA_DIR = Path("data/raw")
    api_config = None


class WeatherFetcher:
    """
    기상청 데이터 수집기

    기상청 공공데이터포털 API를 통해 기상 관측 데이터를 수집합니다.

    Features:
    - ASOS (종관기상관측) 시간별 데이터
    - AWS (방재기상관측) 시간별 데이터
    - 자동 재시도 및 진행률 표시
    - 일별/월별 배치 수집
    """

    # 기상청 API 기본 URL
    BASE_URL = "http://apis.data.go.kr/1360000"

    # ASOS 관측소 (제주도)
    JEJU_ASOS_STATIONS = {
        "184": {"name": "제주", "lat": 33.5142, "lon": 126.5294},
        "185": {"name": "고산", "lat": 33.2939, "lon": 126.1628},
        "188": {"name": "성산", "lat": 33.3867, "lon": 126.8803},
        "189": {"name": "서귀포", "lat": 33.2464, "lon": 126.5653},
    }

    # AWS 관측소 (제주도 - 일부)
    JEJU_AWS_STATIONS = {
        "753": {"name": "제주(공)", "lat": 33.5114, "lon": 126.4929},
        "855": {"name": "성산포", "lat": 33.4389, "lon": 126.9106},
        "870": {"name": "한림", "lat": 33.3914, "lon": 126.2656},
        "885": {"name": "우도", "lat": 33.5056, "lon": 126.9522},
    }

    # 컬럼 매핑 (API 응답 → 표준 컬럼명)
    ASOS_COLUMN_MAP = {
        "tm": "datetime",
        "stnId": "station_id",
        "stnNm": "station_name",
        "ta": "temperature",           # 기온 (°C)
        "hm": "humidity",              # 상대습도 (%)
        "ws": "wind_speed",            # 풍속 (m/s)
        "wd": "wind_direction",        # 풍향 (deg)
        "pa": "pressure",              # 현지기압 (hPa)
        "ps": "sea_level_pressure",    # 해면기압 (hPa)
        "ss": "sunshine_hours",        # 일조시간 (hr)
        "icsr": "solar_radiation",     # 일사량 (MJ/m²)
        "dsnw": "snow_depth",          # 적설량 (cm)
        "dc10Tca": "cloud_cover",      # 전운량 (1/10)
        "dc10LmcsCa": "low_cloud",     # 하층운량
        "vs": "visibility",            # 시정 (10m)
        "ts": "ground_temp",           # 지면온도 (°C)
        "pv": "vapor_pressure",        # 수증기압 (hPa)
        "td": "dew_point",             # 이슬점온도 (°C)
        "rn": "precipitation",         # 강수량 (mm)
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: 기상청 API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv("KMA_API_KEY")

        if api_config and not self.api_key:
            self.api_key = api_config.kma_api_key

        if not self.api_key:
            print(
                "⚠️  경고: API 키가 설정되지 않았습니다.\n"
                "   KMA_API_KEY 환경변수를 설정하거나 api_key 파라미터를 전달하세요.\n"
                "   데모 모드로 실행됩니다 (실제 데이터 수집 불가)."
            )

        self._request_count = 0
        self._last_request_time = 0

    def _rate_limit(self) -> None:
        """API 호출 속도 제한 (초당 10회 제한 준수)"""
        min_interval = 0.15  # 최소 150ms 간격
        elapsed = time.time() - self._last_request_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()
        self._request_count += 1

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_retries: int = 3,
    ) -> Optional[Dict]:
        """
        API 요청 (재시도 로직 포함)

        Args:
            endpoint: API 엔드포인트
            params: 요청 파라미터
            max_retries: 최대 재시도 횟수

        Returns:
            API 응답 JSON 또는 None
        """
        if not self.api_key:
            return None

        self._rate_limit()

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    endpoint,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()

                data = response.json()

                # 응답 코드 확인
                result_code = (
                    data.get("response", {})
                    .get("header", {})
                    .get("resultCode", "")
                )

                if result_code == "00":  # 정상
                    return data
                elif result_code == "03":  # 데이터 없음
                    return None
                else:
                    result_msg = (
                        data.get("response", {})
                        .get("header", {})
                        .get("resultMsg", "Unknown error")
                    )
                    print(f"⚠️  API 오류: {result_msg}")
                    return None

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 지수 백오프
                    continue
                print(f"❌ 요청 타임아웃: {endpoint}")

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"❌ 요청 실패: {e}")

            except ValueError as e:
                print(f"❌ JSON 파싱 실패: {e}")
                return None

        return None

    def fetch_asos_hourly(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        ASOS 시간별 관측 데이터 조회

        Args:
            station_id: 관측소 ID
            start_date: 시작 날짜
            end_date: 종료 날짜
            show_progress: 진행률 표시 여부

        Returns:
            기상 관측 DataFrame
        """
        endpoint = f"{self.BASE_URL}/AsosHourlyInfoService/getWthrDataList"

        all_data = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1

        # 진행률 표시
        iterator = range(total_days)
        if show_progress:
            station_name = self.JEJU_ASOS_STATIONS.get(station_id, {}).get("name", station_id)
            iterator = tqdm(iterator, desc=f"📡 {station_name}", unit="day")

        for _ in iterator:
            if current_date > end_date:
                break

            params = {
                "serviceKey": self.api_key,
                "pageNo": 1,
                "numOfRows": 999,
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "HR",
                "startDt": current_date.strftime("%Y%m%d"),
                "startHh": "00",
                "endDt": current_date.strftime("%Y%m%d"),
                "endHh": "23",
                "stnIds": station_id,
            }

            data = self._make_request(endpoint, params)

            if data:
                items = (
                    data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
                )
                if items:
                    if isinstance(items, dict):
                        items = [items]
                    all_data.extend(items)

            current_date += timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = self._process_weather_data(df)

        return df

    def fetch_aws_hourly(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        AWS 시간별 관측 데이터 조회

        Args:
            station_id: 관측소 ID
            start_date: 시작 날짜
            end_date: 종료 날짜
            show_progress: 진행률 표시 여부

        Returns:
            기상 관측 DataFrame
        """
        endpoint = f"{self.BASE_URL}/AwsHourlyInfoService/getWthrDataList"

        all_data = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1

        iterator = range(total_days)
        if show_progress:
            station_name = self.JEJU_AWS_STATIONS.get(station_id, {}).get("name", station_id)
            iterator = tqdm(iterator, desc=f"📡 {station_name}", unit="day")

        for _ in iterator:
            if current_date > end_date:
                break

            params = {
                "serviceKey": self.api_key,
                "pageNo": 1,
                "numOfRows": 999,
                "dataType": "JSON",
                "dataCd": "AWS",
                "dateCd": "HR",
                "startDt": current_date.strftime("%Y%m%d"),
                "startHh": "00",
                "endDt": current_date.strftime("%Y%m%d"),
                "endHh": "23",
                "stnIds": station_id,
            }

            data = self._make_request(endpoint, params)

            if data:
                items = (
                    data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
                )
                if items:
                    if isinstance(items, dict):
                        items = [items]
                    all_data.extend(items)

            current_date += timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = self._process_weather_data(df)

        return df

    def _process_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기상 데이터 후처리

        Args:
            df: 원시 DataFrame

        Returns:
            정제된 DataFrame
        """
        if df.empty:
            return df

        # 날짜시간 컬럼 생성
        if "tm" in df.columns:
            df["datetime"] = pd.to_datetime(df["tm"], errors="coerce")
        else:
            return df

        # 컬럼 매핑 적용
        available_cols = ["datetime"] + [
            col for col in self.ASOS_COLUMN_MAP.keys()
            if col in df.columns and col != "tm"
        ]

        df = df[available_cols].copy()
        df = df.rename(columns=self.ASOS_COLUMN_MAP)

        # 수치형 변환
        numeric_cols = [
            "temperature", "humidity", "wind_speed", "wind_direction",
            "pressure", "sea_level_pressure", "sunshine_hours",
            "solar_radiation", "cloud_cover", "visibility",
            "ground_temp", "vapor_pressure", "dew_point", "precipitation"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 시간순 정렬
        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def fetch_jeju_weather(
        self,
        start_date: datetime,
        end_date: datetime,
        source: str = "asos",
        stations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        제주도 기상 데이터 일괄 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            source: 데이터 소스 ("asos" 또는 "aws")
            stations: 관측소 ID 목록 (None이면 전체)

        Returns:
            모든 관측소 데이터 (병합)
        """
        if source.lower() == "asos":
            station_dict = self.JEJU_ASOS_STATIONS
            fetch_func = self.fetch_asos_hourly
        elif source.lower() == "aws":
            station_dict = self.JEJU_AWS_STATIONS
            fetch_func = self.fetch_aws_hourly
        else:
            raise ValueError(f"지원하지 않는 소스: {source}")

        if stations:
            station_dict = {k: v for k, v in station_dict.items() if k in stations}

        print(f"\n🌤️  제주도 기상 데이터 수집 시작 ({source.upper()})")
        print(f"📅 기간: {start_date.date()} ~ {end_date.date()}")
        print(f"📍 관측소: {len(station_dict)}개\n")

        all_dfs = []

        for station_id in station_dict.keys():
            df = fetch_func(station_id, start_date, end_date)

            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            print("⚠️  수집된 데이터가 없습니다.")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)

        print(f"\n✅ 수집 완료: {len(combined_df):,}행")
        print(f"📊 기간: {combined_df['datetime'].min()} ~ {combined_df['datetime'].max()}")

        return combined_df

    def get_station_info(self, source: str = "asos") -> pd.DataFrame:
        """
        관측소 정보 조회

        Args:
            source: "asos" 또는 "aws"

        Returns:
            관측소 정보 DataFrame
        """
        if source.lower() == "asos":
            stations = self.JEJU_ASOS_STATIONS
        else:
            stations = self.JEJU_AWS_STATIONS

        data = []
        for station_id, info in stations.items():
            data.append({
                "station_id": station_id,
                "station_name": info["name"],
                "latitude": info["lat"],
                "longitude": info["lon"],
                "source": source.upper(),
            })

        return pd.DataFrame(data)

    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        데이터 저장

        Args:
            df: 저장할 DataFrame
            filename: 파일명
            output_dir: 저장 디렉토리

        Returns:
            저장된 파일 경로
        """
        output_dir = Path(output_dir) if output_dir else RAW_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        if filepath.suffix == ".parquet":
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

        print(f"💾 저장 완료: {filepath}")
        return filepath


def create_sample_weather_data(
    start_date: datetime,
    end_date: datetime,
    station_id: str = "184",
) -> pd.DataFrame:
    """
    샘플 기상 데이터 생성 (API 키 없이 테스트용)

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        station_id: 관측소 ID

    Returns:
        샘플 기상 DataFrame
    """
    import numpy as np

    date_range = pd.date_range(start_date, end_date, freq="h")

    # 시간에 따른 패턴 생성
    hours = np.array([d.hour for d in date_range])
    days = np.array([d.timetuple().tm_yday for d in date_range])

    # 기온: 일변화 + 계절변화 + 노이즈
    temp_daily = 5 * np.sin(2 * np.pi * (hours - 6) / 24)  # 일변화
    temp_seasonal = 10 * np.sin(2 * np.pi * (days - 80) / 365)  # 계절변화
    temperature = 15 + temp_daily + temp_seasonal + np.random.normal(0, 2, len(date_range))

    # 일사량: 낮에만 양수
    solar_base = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    solar_radiation = solar_base * (0.8 + 0.2 * np.random.random(len(date_range)))
    solar_radiation = np.where((hours >= 6) & (hours <= 18), solar_radiation, 0)

    # 풍속: 랜덤 + 약간의 일변화
    wind_speed = 3 + 2 * np.random.random(len(date_range)) + 0.5 * np.sin(2 * np.pi * hours / 24)

    df = pd.DataFrame({
        "datetime": date_range,
        "station_id": station_id,
        "station_name": "제주(샘플)",
        "temperature": temperature.round(1),
        "humidity": np.clip(60 + 20 * np.random.randn(len(date_range)), 20, 100).round(0),
        "wind_speed": wind_speed.round(1),
        "wind_direction": np.random.randint(0, 360, len(date_range)),
        "pressure": (1013 + 5 * np.random.randn(len(date_range))).round(1),
        "solar_radiation": solar_radiation.round(2),
        "cloud_cover": np.random.randint(0, 11, len(date_range)),
    })

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("🌤️  WeatherFetcher 모듈 테스트")
    print("=" * 60)

    # 관측소 정보 출력
    fetcher = WeatherFetcher()
    print("\n📍 ASOS 관측소:")
    print(fetcher.get_station_info("asos").to_string(index=False))

    print("\n📍 AWS 관측소:")
    print(fetcher.get_station_info("aws").to_string(index=False))

    # 샘플 데이터 생성 테스트
    print("\n📊 샘플 데이터 생성 테스트:")
    sample = create_sample_weather_data(
        datetime(2024, 1, 1),
        datetime(2024, 1, 7),
    )
    print(sample.head(10).to_string(index=False))
    print(f"\n총 {len(sample)}행 생성됨")
