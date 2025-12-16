"""
KMA Portal Direct Fetcher

기상자료개방포털(data.kma.go.kr)에서 직접 데이터 다운로드

API 키 없이 웹 크롤링을 통해 ASOS/AWS 데이터를 수집합니다.

Usage:
    from src.data.kma_portal_fetcher import KMAPortalFetcher

    fetcher = KMAPortalFetcher()
    df = fetcher.fetch_asos_hourly(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        station_ids=["184", "185"],  # 제주, 고산
    )
"""

import time
import io
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

try:
    from src.config import RAW_DATA_DIR
except ImportError:
    RAW_DATA_DIR = Path("data/raw")


class KMAPortalFetcher:
    """
    기상자료개방포털 데이터 수집기

    https://data.kma.go.kr 에서 직접 데이터를 다운로드합니다.
    API 키 없이 웹 인터페이스를 통해 데이터를 수집합니다.

    지원 데이터:
    - ASOS (종관기상관측) 시간자료
    - ASOS 일자료
    """

    BASE_URL = "https://data.kma.go.kr"

    # ASOS 시간자료 다운로드 엔드포인트
    ASOS_HOURLY_URL = f"{BASE_URL}/data/grnd/selectAsosRltmList.do"

    # CSV 다운로드 엔드포인트
    CSV_DOWNLOAD_URL = f"{BASE_URL}/download/getFileDown.do"

    # 제주도 ASOS 관측소
    JEJU_STATIONS = {
        "184": "제주",
        "185": "고산",
        "188": "성산",
        "189": "서귀포",
    }

    # 전체 ASOS 관측소 (주요)
    MAJOR_STATIONS = {
        "108": "서울",
        "112": "인천",
        "119": "수원",
        "133": "대전",
        "143": "대구",
        "156": "광주",
        "159": "부산",
        "184": "제주",
        "185": "고산",
        "188": "성산",
        "189": "서귀포",
    }

    # 데이터 요소 코드
    ELEMENTS = {
        "ta": "기온",
        "hm": "습도",
        "ws": "풍속",
        "wd": "풍향",
        "pa": "현지기압",
        "ps": "해면기압",
        "ss": "일조",
        "icsr": "일사",
        "dsnw": "적설",
        "dc10Tca": "전운량",
        "rn": "강수량",
        "ts": "지면온도",
    }

    def __init__(self, delay: float = 0.5):
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
            "Referer": self.ASOS_HOURLY_URL,
        })

    def fetch_asos_hourly(
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        ASOS 시간자료 조회

        기상자료개방포털에서 ASOS 시간별 관측 데이터를 다운로드합니다.

        Note:
            - 한 번에 최대 1년 데이터 조회 가능
            - 기간이 길면 월별로 분할하여 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            station_ids: 관측소 ID 목록 (None이면 제주도 전체)
            show_progress: 진행률 표시 여부

        Returns:
            ASOS 시간자료 DataFrame
        """
        if station_ids is None:
            station_ids = list(self.JEJU_STATIONS.keys())

        # 기간이 1년 초과면 분할
        total_days = (end_date - start_date).days
        if total_days > 365:
            return self._fetch_long_period(
                start_date, end_date, station_ids, show_progress
            )

        all_dfs = []

        for station_id in station_ids:
            station_name = self.JEJU_STATIONS.get(
                station_id,
                self.MAJOR_STATIONS.get(station_id, station_id)
            )

            if show_progress:
                print(f"📡 {station_name} ({station_id}) 데이터 수집 중...")

            df = self._fetch_station_data(
                station_id, start_date, end_date
            )

            if df is not None and not df.empty:
                df["station_id"] = station_id
                df["station_name"] = station_name
                all_dfs.append(df)
                print(f"   ✅ {len(df):,}행 수집")
            else:
                print(f"   ⚠️  데이터 없음")

            time.sleep(self.delay)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        return combined

    def _fetch_station_data(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        단일 관측소 데이터 조회

        Args:
            station_id: 관측소 ID
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            DataFrame 또는 None
        """
        # 기상자료개방포털 POST 요청 파라미터
        params = {
            "pgmNo": "36",
            "tabNo": "1",
            "menuNo": "3",
            "serviceSe": "asos",
            "stCd": station_id,
            "stnNm": "",
            "startDt": start_date.strftime("%Y%m%d"),
            "startHh": "00",
            "endDt": end_date.strftime("%Y%m%d"),
            "endHh": "23",
            "dataFormCd": "F00501",  # 시간자료
            "dataTypeCd": "standard",
            "selectType": "1",
            "mddlClssCd": "",
            "lctnNm": "",
        }

        try:
            # 먼저 페이지 접속하여 세션 초기화
            self.session.get(self.ASOS_HOURLY_URL, timeout=30)
            time.sleep(0.3)

            # 데이터 요청
            response = self.session.post(
                f"{self.BASE_URL}/data/grnd/selectAsosRltmList.do",
                data=params,
                timeout=60,
            )
            response.raise_for_status()

            # CSV 다운로드 시도
            csv_params = {
                **params,
                "fileType": "csv",
                "dataDown": "Y",
            }

            csv_response = self.session.post(
                f"{self.BASE_URL}/cmmn/getFileDown.do",
                data=csv_params,
                timeout=60,
            )

            if csv_response.status_code == 200:
                content_type = csv_response.headers.get("Content-Type", "")

                if "csv" in content_type or "octet-stream" in content_type:
                    # CSV 파싱
                    try:
                        df = pd.read_csv(
                            io.StringIO(csv_response.text),
                            encoding="utf-8",
                        )
                        return self._process_asos_data(df)
                    except Exception:
                        pass

                # 다른 형식 시도
                try:
                    df = pd.read_csv(
                        io.StringIO(csv_response.content.decode("euc-kr")),
                    )
                    return self._process_asos_data(df)
                except Exception:
                    pass

            # HTML 파싱 시도 (테이블에서 데이터 추출)
            return self._parse_html_table(response.text)

        except requests.exceptions.RequestException as e:
            print(f"   ❌ 요청 실패: {e}")
            return None

    def _parse_html_table(self, html: str) -> Optional[pd.DataFrame]:
        """
        HTML 테이블에서 데이터 추출

        Args:
            html: HTML 문자열

        Returns:
            DataFrame 또는 None
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # 데이터 테이블 찾기
            tables = soup.find_all("table")

            for table in tables:
                try:
                    df = pd.read_html(str(table))[0]
                    if len(df) > 0 and len(df.columns) > 5:
                        return self._process_asos_data(df)
                except Exception:
                    continue

            return None

        except ImportError:
            print("   ⚠️  beautifulsoup4가 필요합니다: pip install beautifulsoup4")
            return None
        except Exception:
            return None

    def _process_asos_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ASOS 데이터 후처리

        Args:
            df: 원시 DataFrame

        Returns:
            정제된 DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        # 컬럼명 정규화 (한글 → 영문)
        column_mapping = {
            "일시": "datetime",
            "지점": "station_id",
            "지점명": "station_name",
            "기온(°C)": "temperature",
            "기온": "temperature",
            "습도(%)": "humidity",
            "습도": "humidity",
            "풍속(m/s)": "wind_speed",
            "풍속": "wind_speed",
            "풍향(16방위)": "wind_direction",
            "풍향": "wind_direction",
            "현지기압(hPa)": "pressure",
            "현지기압": "pressure",
            "해면기압(hPa)": "sea_level_pressure",
            "해면기압": "sea_level_pressure",
            "일조(hr)": "sunshine_hours",
            "일조": "sunshine_hours",
            "일사(MJ/m2)": "solar_radiation",
            "일사": "solar_radiation",
            "적설(cm)": "snow_depth",
            "적설": "snow_depth",
            "전운량(10분위)": "cloud_cover",
            "전운량": "cloud_cover",
            "강수량(mm)": "precipitation",
            "강수량": "precipitation",
            "지면온도(°C)": "ground_temp",
            "지면온도": "ground_temp",
        }

        # 존재하는 컬럼만 매핑
        df = df.rename(columns={
            k: v for k, v in column_mapping.items()
            if k in df.columns
        })

        # datetime 처리
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # 수치형 변환
        numeric_cols = [
            "temperature", "humidity", "wind_speed", "pressure",
            "sea_level_pressure", "sunshine_hours", "solar_radiation",
            "snow_depth", "cloud_cover", "precipitation", "ground_temp"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 정렬
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def _fetch_long_period(
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: List[str],
        show_progress: bool,
    ) -> pd.DataFrame:
        """
        장기간 데이터 수집 (월별 분할)

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            station_ids: 관측소 목록
            show_progress: 진행률 표시

        Returns:
            병합된 DataFrame
        """
        all_dfs = []
        current_start = start_date

        # 월별 분할
        while current_start < end_date:
            # 해당 월의 마지막 날
            if current_start.month == 12:
                current_end = datetime(current_start.year + 1, 1, 1) - timedelta(days=1)
            else:
                current_end = datetime(current_start.year, current_start.month + 1, 1) - timedelta(days=1)

            # end_date를 초과하지 않도록
            current_end = min(current_end, end_date)

            if show_progress:
                print(f"\n📅 {current_start.strftime('%Y-%m')} 데이터 수집...")

            df = self.fetch_asos_hourly(
                current_start,
                current_end,
                station_ids,
                show_progress=False,
            )

            if not df.empty:
                all_dfs.append(df)

            # 다음 달로
            if current_start.month == 12:
                current_start = datetime(current_start.year + 1, 1, 1)
            else:
                current_start = datetime(current_start.year, current_start.month + 1, 1)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)

    def fetch_jeju_weather(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        제주도 전체 관측소 데이터 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            제주도 기상 데이터 DataFrame
        """
        print("\n" + "=" * 60)
        print("🌤️  제주도 기상 데이터 수집 (기상자료개방포털)")
        print("=" * 60)
        print(f"📅 기간: {start_date.date()} ~ {end_date.date()}")
        print(f"📍 관측소: {', '.join(self.JEJU_STATIONS.values())}")
        print()

        df = self.fetch_asos_hourly(
            start_date,
            end_date,
            station_ids=list(self.JEJU_STATIONS.keys()),
        )

        if not df.empty:
            print(f"\n✅ 수집 완료: 총 {len(df):,}행")

        return df

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
            output_dir: 출력 디렉토리

        Returns:
            저장된 파일 경로
        """
        output_dir = Path(output_dir) if output_dir else RAW_DATA_DIR / "weather"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        if filepath.suffix == ".parquet":
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

        print(f"💾 저장: {filepath}")
        return filepath


def download_jeju_weather(
    year: int,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    특정 연도의 제주도 기상 데이터 다운로드

    편의 함수로, 연도를 지정하면 해당 연도의 전체 데이터를 수집합니다.

    Args:
        year: 연도
        output_dir: 출력 디렉토리

    Returns:
        저장된 파일 경로
    """
    fetcher = KMAPortalFetcher()

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    df = fetcher.fetch_jeju_weather(start_date, end_date)

    if df.empty:
        print("⚠️  데이터를 수집하지 못했습니다.")
        return None

    filename = f"jeju_asos_{year}.csv"
    return fetcher.save_data(df, filename, output_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("🌤️  KMA Portal Fetcher 테스트")
    print("=" * 60)

    fetcher = KMAPortalFetcher()

    print("\n📍 제주도 ASOS 관측소:")
    for code, name in fetcher.JEJU_STATIONS.items():
        print(f"   {code}: {name}")

    # 짧은 기간 테스트
    print("\n📊 테스트 데이터 수집 (최근 3일)...")

    from datetime import datetime, timedelta
    end_date = datetime.now() - timedelta(days=1)  # 어제까지
    start_date = end_date - timedelta(days=2)

    df = fetcher.fetch_asos_hourly(
        start_date=start_date,
        end_date=end_date,
        station_ids=["184"],  # 제주만
    )

    if not df.empty:
        print(f"\n✅ 수집 성공: {len(df)}행")
        print("\n📊 샘플 데이터:")
        print(df.head(10).to_string(index=False))
    else:
        print("\n⚠️  데이터 수집 실패 - 수동 다운로드를 시도해주세요.")
        print("   URL: https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36")
