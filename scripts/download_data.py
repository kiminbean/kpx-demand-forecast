#!/usr/bin/env python3
"""
데이터 다운로드 스크립트

기상청 및 KPX 데이터를 자동으로 수집합니다.

Usage:
    # 전체 데이터 수집 (2024년)
    python scripts/download_data.py --year 2024

    # 특정 기간 수집
    python scripts/download_data.py --start 2024-01-01 --end 2024-06-30

    # 기상 데이터만 수집
    python scripts/download_data.py --source weather --year 2024

    # 샘플 데이터 생성 (API 키 없이 테스트)
    python scripts/download_data.py --sample --year 2024
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import argparse

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="J-REF 데이터 수집 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_data.py --year 2024
    python scripts/download_data.py --start 2024-01-01 --end 2024-06-30
    python scripts/download_data.py --source weather --year 2024
    python scripts/download_data.py --sample --year 2024
        """
    )

    parser.add_argument(
        "--year",
        type=int,
        help="수집할 연도 (예: 2024)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="시작 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="종료 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "weather", "kpx"],
        default="all",
        help="데이터 소스 (기본값: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="샘플 데이터 생성 모드 (API 키 불필요)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="기상청 API 키 (환경변수 대신 직접 지정)"
    )

    return parser.parse_args()


def download_weather_data(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    api_key: str = None,
    use_selenium: bool = True,
    headless: bool = True,
) -> Path:
    """
    기상청 데이터 다운로드

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        output_dir: 출력 디렉토리
        api_key: API 키 (공공데이터포털 API 사용 시)
        use_selenium: Selenium 크롤러 사용 여부 (API 키 불필요)
        headless: 헤드리스 모드 (브라우저 창 숨김)

    Returns:
        저장된 파일 경로
    """
    print("\n" + "=" * 60)
    print("🌤️  기상청 데이터 수집")
    print("=" * 60)

    weather_dir = output_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    # 방법 1: Selenium 크롤러 (API 키 불필요, 권장)
    if use_selenium:
        try:
            from src.data.kma_crawler import KMACrawler

            print("📡 Selenium 크롤러 사용 (API 키 불필요)")
            print(f"   헤드리스 모드: {'예' if headless else '아니오'}")

            crawler = KMACrawler(
                headless=headless,
                download_dir=str(weather_dir),
            )

            # 제주도 4개 관측소
            stations = ["제주", "고산", "성산", "서귀포"]

            for station in stations:
                print(f"\n📍 {station} 관측소 데이터 수집...")
                filepath = crawler.download(
                    station=station,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    data_type="시간",  # 시간별 데이터
                )

                if filepath:
                    downloaded_files.append(filepath)
                    print(f"   ✅ 완료: {filepath}")
                else:
                    print(f"   ⚠️ 실패")

            if downloaded_files:
                print(f"\n✅ 총 {len(downloaded_files)}개 파일 다운로드 완료")
                return Path(downloaded_files[0]).parent

        except ImportError:
            print("⚠️  Selenium이 설치되지 않았습니다.")
            print("   설치: pip install selenium webdriver-manager")
        except Exception as e:
            print(f"⚠️  Selenium 크롤링 실패: {e}")

    # 방법 2: 공공데이터포털 API (API 키 필요)
    if not downloaded_files and api_key:
        try:
            from src.data.kma_api import KMAAPI

            print("\n📡 공공데이터포털 API 사용")
            api = KMAAPI(api_key=api_key)

            stations = {"제주": "184", "고산": "185", "성산": "188", "서귀포": "189"}

            for name, code in stations.items():
                print(f"\n📍 {name} 관측소 데이터 수집...")
                filepath = api.download(
                    station=code,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    data_type="시간",
                    output_dir=str(weather_dir),
                )

                if filepath:
                    downloaded_files.append(filepath)

        except Exception as e:
            print(f"⚠️  API 수집 실패: {e}")

    if not downloaded_files:
        print("\n⚠️  수집된 데이터가 없습니다.")
        print("\n📋 수동 다운로드 안내:")
        print("   1. https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36 접속")
        print("   2. 지점 선택: 제주, 고산, 성산, 서귀포")
        print(f"   3. 기간: {start_date.date()} ~ {end_date.date()}")
        print("   4. 자료형태: 시간자료")
        print("   5. CSV 다운로드 후 data/raw/weather/ 에 저장")
        return None

    return weather_dir


def download_kpx_data(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> Path:
    """
    KPX 발전량 데이터 다운로드

    Note:
        현재 자동 크롤링은 제한적입니다.
        수동 다운로드 안내를 출력합니다.

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        output_dir: 출력 디렉토리

    Returns:
        None (수동 다운로드 필요)
    """
    print("\n" + "=" * 60)
    print("⚡ KPX 발전량 데이터")
    print("=" * 60)

    print("""
📋 KPX 데이터는 수동 다운로드가 필요합니다.

1. 전력통계정보시스템 (EPSIS) 접속
   URL: https://epsis.kpx.or.kr/

2. 데이터 다운로드 경로:
   - 전력통계 → 발전설비/발전량 → 지역별 발전량
   - 지역: 제주
   - 에너지원: 태양광, 풍력
   - 기간: {start} ~ {end}

3. 다운로드 후 파일 저장:
   - 태양광: {output_dir}/kpx/solar_generation.csv
   - 풍력: {output_dir}/kpx/wind_generation.csv

4. 파일 로드 예시:
   from src.data.kpx_scraper import KPXScraper
   scraper = KPXScraper()
   df = scraper.load_downloaded_data("data/raw/kpx/solar_generation.csv", "solar")
""".format(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        output_dir=output_dir,
    ))

    # KPX 디렉토리 생성
    kpx_dir = output_dir / "kpx"
    kpx_dir.mkdir(parents=True, exist_ok=True)

    return None


def generate_sample_data(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> tuple:
    """
    샘플 데이터 생성

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        output_dir: 출력 디렉토리

    Returns:
        (기상 파일 경로, 발전량 파일 경로)
    """
    from src.data.weather_fetcher import create_sample_weather_data
    from src.data.kpx_scraper import create_merged_sample_data

    print("\n" + "=" * 60)
    print("📊 샘플 데이터 생성")
    print("=" * 60)

    # 기상 샘플 데이터
    print("\n🌤️  기상 샘플 데이터 생성 중...")
    weather_df = create_sample_weather_data(start_date, end_date)

    weather_dir = output_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    weather_file = weather_dir / f"sample_weather_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    weather_df.to_csv(weather_file, index=False, encoding="utf-8-sig")
    print(f"   💾 저장: {weather_file}")
    print(f"   📈 {len(weather_df):,}행")

    # 발전량 샘플 데이터
    print("\n⚡ 발전량 샘플 데이터 생성 중...")
    generation_df = create_merged_sample_data(start_date, end_date)

    kpx_dir = output_dir / "kpx"
    kpx_dir.mkdir(parents=True, exist_ok=True)

    generation_file = kpx_dir / f"sample_generation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    generation_df.to_csv(generation_file, index=False, encoding="utf-8-sig")
    print(f"   💾 저장: {generation_file}")
    print(f"   📈 {len(generation_df):,}행")

    return weather_file, generation_file


def merge_datasets(
    weather_file: Path,
    generation_file: Path,
    output_dir: Path,
) -> Path:
    """
    기상 + 발전량 데이터 병합

    Args:
        weather_file: 기상 데이터 파일
        generation_file: 발전량 데이터 파일
        output_dir: 출력 디렉토리

    Returns:
        병합된 파일 경로
    """
    print("\n" + "=" * 60)
    print("🔗 데이터 병합")
    print("=" * 60)

    weather_df = pd.read_csv(weather_file)
    generation_df = pd.read_csv(generation_file)

    # datetime 파싱
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    generation_df["datetime"] = pd.to_datetime(generation_df["datetime"])

    # 병합 (기상 데이터를 제주(184) 관측소 기준으로 필터)
    if "station_id" in weather_df.columns:
        weather_df = weather_df[weather_df["station_id"] == "184"].copy()

    merged_df = generation_df.merge(
        weather_df.drop(columns=["station_id", "station_name"], errors="ignore"),
        on="datetime",
        how="left",
    )

    # 저장
    processed_dir = output_dir.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    merged_file = processed_dir / "merged_dataset.csv"
    merged_df.to_csv(merged_file, index=False, encoding="utf-8-sig")

    print(f"💾 저장: {merged_file}")
    print(f"📈 {len(merged_df):,}행, {len(merged_df.columns)}컬럼")

    return merged_file


def main():
    """메인 함수"""
    args = parse_args()

    print("=" * 60)
    print("🚀 J-REF 데이터 수집 시작")
    print("=" * 60)

    # 날짜 범위 설정
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)
    else:
        # 기본값: 최근 7일
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=7)

    print(f"\n📅 수집 기간: {start_date.date()} ~ {end_date.date()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 출력 디렉토리: {output_dir}")

    # 샘플 데이터 생성 모드
    if args.sample:
        weather_file, generation_file = generate_sample_data(
            start_date, end_date, output_dir
        )

        # 데이터 병합
        if weather_file and generation_file:
            merge_datasets(weather_file, generation_file, output_dir)

        print("\n" + "=" * 60)
        print("✅ 샘플 데이터 생성 완료!")
        print("=" * 60)
        return

    # 실제 데이터 수집
    weather_file = None
    generation_file = None

    if args.source in ["all", "weather"]:
        weather_file = download_weather_data(
            start_date, end_date, output_dir, args.api_key
        )

    if args.source in ["all", "kpx"]:
        download_kpx_data(start_date, end_date, output_dir)

    print("\n" + "=" * 60)
    print("✅ 데이터 수집 완료!")
    print("=" * 60)

    if not weather_file:
        print("""
📌 다음 단계:
   1. KPX 데이터 수동 다운로드 (위 안내 참조)
   2. 데이터 병합 실행:
      python scripts/merge_data.py

   또는 샘플 데이터로 테스트:
      python scripts/download_data.py --sample --year 2024
""")


if __name__ == "__main__":
    main()
