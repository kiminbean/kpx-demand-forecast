"""
Data Module

데이터 로딩, 수집, 전처리 모듈

기상 데이터 수집 방법:
1. KMACrawler - Selenium 기반 웹 크롤링 (API 키 불필요, 권장)
2. KMAAPI - 공공데이터포털 REST API (API 키 필요, 더 안정적)
3. WeatherFetcher - 공공데이터포털 API 래퍼
"""

from .kpx_loader import KPXDataLoader
from .weather_fetcher import WeatherFetcher, create_sample_weather_data
from .kpx_scraper import KPXScraper, create_sample_generation_data, create_merged_sample_data

# 기상청 크롤러 (Selenium 기반 - API 키 불필요)
try:
    from .kma_crawler import KMACrawler
except ImportError:
    KMACrawler = None  # selenium 미설치 시

# 기상청 API (공공데이터포털 - API 키 필요)
from .kma_api import KMAAPI

# 설정
from .kma_config import (
    STATION_CODES,
    ELEMENT_CODES,
    get_station_code,
    get_element_codes,
    list_stations,
    list_elements,
)

__all__ = [
    # 데이터 로더
    "KPXDataLoader",
    "KPXScraper",
    # 기상 데이터 수집기
    "KMACrawler",      # Selenium 크롤러 (API 키 불필요, 권장)
    "KMAAPI",          # 공공데이터포털 API (API 키 필요)
    "WeatherFetcher",  # API 래퍼
    # 설정 및 유틸
    "STATION_CODES",
    "ELEMENT_CODES",
    "get_station_code",
    "get_element_codes",
    "list_stations",
    "list_elements",
    # 샘플 데이터 생성
    "create_sample_weather_data",
    "create_sample_generation_data",
    "create_merged_sample_data",
]
