"""
한국전력거래소(KPX) 발전량 데이터 크롤러

공공데이터포털(data.go.kr)에서 신재생에너지 발전량 데이터를 다운로드합니다.

데이터셋:
- 한국전력거래소_지역별 시간별 태양광 및 풍력 발전량
- URL: https://www.data.go.kr/data/15065269/fileData.do

사용법:
    from tools.crawlers import KPXCrawler

    crawler = KPXCrawler()
    filepath = crawler.download(output_dir="data/raw/kpx")

    # 또는 간단히
    from tools.crawlers import download_kpx_generation
    filepath = download_kpx_generation()
"""

import os
import time
import glob
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
)

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False
    print("Warning: webdriver_manager not installed. Run: pip install webdriver-manager")


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KPXCrawler:
    """
    한국전력거래소 신재생에너지 발전량 데이터 크롤러

    공공데이터포털에서 CSV 파일을 자동으로 다운로드합니다.
    """

    # 데이터셋 URL
    DATASETS = {
        'solar_wind': {
            'id': '15065269',
            'name': '한국전력거래소_지역별 시간별 태양광 및 풍력 발전량',
            'url': 'https://www.data.go.kr/data/15065269/fileData.do',
            'description': '지역별 시간별 태양광/풍력 발전량 (MWh)',
        },
        'solar_hourly': {
            'id': '15103243',
            'name': '한국전력거래소_지역별 시간별 태양광 발전량 정보',
            'url': 'https://www.data.go.kr/data/15103243/openapi.do',
            'description': '태양광 발전량 API (별도 신청 필요)',
        },
        'generation_by_fuel': {
            'id': '15142651',
            'name': '한국전력거래소_발전원별 발전량 현황조회',
            'url': 'https://www.data.go.kr/data/15142651/openapi.do',
            'description': '발전원별 발전량 API',
        },
    }

    def __init__(
        self,
        headless: bool = True,
        download_dir: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Args:
            headless: 헤드리스 모드 (True: 브라우저 숨김)
            download_dir: 다운로드 디렉토리
            timeout: 대기 시간 (초)
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None

        # 다운로드 디렉토리 설정
        if download_dir:
            self.download_dir = Path(download_dir).resolve()
        else:
            self.download_dir = Path.cwd() / "data" / "raw" / "kpx"
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # 임시 다운로드 디렉토리 (Chrome 다운로드용)
        self.temp_download_dir = Path.cwd() / ".temp_downloads"
        self.temp_download_dir.mkdir(parents=True, exist_ok=True)

    def _init_driver(self):
        """Selenium WebDriver 초기화"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless=new")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--lang=ko-KR")

        # 다운로드 설정
        prefs = {
            "download.default_directory": str(self.temp_download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # WebDriver 초기화
        if USE_WEBDRIVER_MANAGER:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            self.driver = webdriver.Chrome(options=chrome_options)

        self.driver.implicitly_wait(10)
        logger.info("Chrome WebDriver initialized")

    def _wait_for_download(self, timeout: int = 120) -> Optional[Path]:
        """다운로드 완료 대기"""
        logger.info(f"Waiting for download (timeout: {timeout}s)...")

        end_time = time.time() + timeout
        while time.time() < end_time:
            # 다운로드 중인 파일 확인 (.crdownload)
            downloading = list(self.temp_download_dir.glob("*.crdownload"))
            csv_files = list(self.temp_download_dir.glob("*.csv"))
            zip_files = list(self.temp_download_dir.glob("*.zip"))

            if csv_files and not downloading:
                latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Download complete: {latest_file.name}")
                return latest_file

            if zip_files and not downloading:
                latest_file = max(zip_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Download complete (zip): {latest_file.name}")
                return latest_file

            time.sleep(1)

        logger.warning("Download timeout")
        return None

    def _move_downloaded_file(self, temp_file: Path, final_name: Optional[str] = None) -> Path:
        """다운로드된 파일을 최종 위치로 이동"""
        if final_name:
            final_path = self.download_dir / final_name
        else:
            # 타임스탬프 추가
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = temp_file.stem
            suffix = temp_file.suffix
            final_path = self.download_dir / f"{stem}_{timestamp}{suffix}"

        shutil.move(str(temp_file), str(final_path))
        logger.info(f"File moved to: {final_path}")
        return final_path

    def download(
        self,
        dataset: str = 'solar_wind',
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        KPX 데이터 다운로드

        Args:
            dataset: 데이터셋 키 ('solar_wind', 'solar_hourly', 등)
            output_dir: 출력 디렉토리 (None이면 기본값 사용)
            filename: 저장할 파일명 (None이면 자동 생성)

        Returns:
            다운로드된 파일 경로
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(self.DATASETS.keys())}")

        dataset_info = self.DATASETS[dataset]
        url = dataset_info['url']

        if output_dir:
            self.download_dir = Path(output_dir).resolve()
            self.download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading: {dataset_info['name']}")
        logger.info(f"URL: {url}")

        try:
            # WebDriver 초기화
            self._init_driver()

            # 페이지 로드
            logger.info("Loading page...")
            self.driver.get(url)
            time.sleep(3)  # 페이지 로딩 대기

            # 다운로드 버튼 찾기 및 클릭
            download_button = self._find_download_button()

            if download_button:
                logger.info("Clicking download button...")
                self.driver.execute_script("arguments[0].click();", download_button)
                time.sleep(2)

                # 다운로드 완료 대기
                downloaded_file = self._wait_for_download()

                if downloaded_file:
                    final_path = self._move_downloaded_file(downloaded_file, filename)
                    logger.info(f"Download successful: {final_path}")
                    return final_path
                else:
                    logger.error("Download failed: timeout")
                    return None
            else:
                logger.error("Download button not found")
                return None

        except Exception as e:
            logger.error(f"Download error: {e}")
            raise

        finally:
            self.close()

    def _find_download_button(self):
        """다운로드 버튼 찾기"""
        wait = WebDriverWait(self.driver, self.timeout)

        # 다양한 선택자 시도
        selectors = [
            # data.go.kr 파일데이터 다운로드 버튼
            (By.CSS_SELECTOR, "a.btn-down"),
            (By.CSS_SELECTOR, "button.btn-down"),
            (By.CSS_SELECTOR, "a[title*='다운로드']"),
            (By.CSS_SELECTOR, "button[title*='다운로드']"),
            (By.XPATH, "//a[contains(text(), '다운로드')]"),
            (By.XPATH, "//button[contains(text(), '다운로드')]"),
            (By.XPATH, "//a[contains(@class, 'download')]"),
            (By.XPATH, "//button[contains(@class, 'download')]"),
            # CSV 다운로드 버튼
            (By.XPATH, "//a[contains(text(), 'CSV')]"),
            (By.XPATH, "//button[contains(text(), 'CSV')]"),
            # 파일 다운로드 링크
            (By.CSS_SELECTOR, "a.file-download"),
            (By.CSS_SELECTOR, "a[href*='.csv']"),
            # data.go.kr 특정 버튼
            (By.CSS_SELECTOR, "#fileDownloadBtn"),
            (By.CSS_SELECTOR, ".btn_down"),
            (By.CSS_SELECTOR, ".down_btn"),
        ]

        for by, selector in selectors:
            try:
                element = wait.until(
                    EC.element_to_be_clickable((by, selector))
                )
                logger.info(f"Found download button: {selector}")
                return element
            except TimeoutException:
                continue
            except NoSuchElementException:
                continue

        # 마지막 시도: 모든 버튼/링크 검색
        try:
            all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
            all_links = self.driver.find_elements(By.TAG_NAME, "a")

            for elem in all_buttons + all_links:
                text = elem.text.lower()
                if '다운로드' in text or 'download' in text or 'csv' in text:
                    logger.info(f"Found button by text: {elem.text}")
                    return elem

        except Exception as e:
            logger.debug(f"Button search failed: {e}")

        return None

    def close(self):
        """WebDriver 종료"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("WebDriver closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def list_datasets(cls) -> dict:
        """사용 가능한 데이터셋 목록"""
        return cls.DATASETS


def download_kpx_generation(
    output_dir: Optional[str] = None,
    dataset: str = 'solar_wind',
    headless: bool = True,
) -> Optional[Path]:
    """
    KPX 신재생에너지 발전량 데이터 다운로드 (편의 함수)

    Args:
        output_dir: 출력 디렉토리
        dataset: 데이터셋 ('solar_wind')
        headless: 헤드리스 모드

    Returns:
        다운로드된 파일 경로

    Usage:
        from tools.crawlers import download_kpx_generation

        # 기본 다운로드
        filepath = download_kpx_generation()

        # 특정 디렉토리에 저장
        filepath = download_kpx_generation(output_dir="data/raw/kpx")
    """
    crawler = KPXCrawler(headless=headless, download_dir=output_dir)
    return crawler.download(dataset=dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KPX 신재생에너지 발전량 데이터 다운로드"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/raw/kpx",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="solar_wind",
        choices=list(KPXCrawler.DATASETS.keys()),
        help="데이터셋"
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="브라우저 표시 (디버깅용)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 데이터셋 목록"
    )

    args = parser.parse_args()

    if args.list:
        print("\n사용 가능한 데이터셋:")
        for key, info in KPXCrawler.list_datasets().items():
            print(f"\n  [{key}]")
            print(f"    이름: {info['name']}")
            print(f"    설명: {info['description']}")
            print(f"    URL: {info['url']}")
    else:
        print("=" * 60)
        print("KPX 데이터 크롤러")
        print("=" * 60)

        filepath = download_kpx_generation(
            output_dir=args.output,
            dataset=args.dataset,
            headless=not args.no_headless
        )

        if filepath:
            print(f"\n✓ 다운로드 완료: {filepath}")
        else:
            print("\n✗ 다운로드 실패")
