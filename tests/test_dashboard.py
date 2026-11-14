"""
대시보드 데이터 출처 검증 테스트
================================

대시보드가 합성 데이터가 아니라 KPX 실측 발전량을 보여주는지,
화면에 데이터 종류를 표시하는지 확인한다.
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app.py"
REAL_DATA = ROOT / "data" / "processed" / "kpx_real_merged.csv"
SAMPLE_DATA = ROOT / "data" / "sample" / "sample_data.csv"
APP_TIMEOUT_SEC = 120


def _metric(at: AppTest, label_part: str) -> str:
    return next(m.value for m in at.metric if label_part in m.label)


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=APP_TIMEOUT_SEC).run()
    assert not at.exception
    return at


def test_overview_shows_real_kpx_wind_mean(app):
    real = pd.read_csv(REAL_DATA)
    assert _metric(app, "풍력 평균") == f"{real['wind_mw'].mean():.1f} MW"
    assert _metric(app, "태양광 평균") == f"{real['solar_mw'].mean():.1f} MW"


def test_data_kind_is_labeled_on_screen(app):
    captions = " ".join(c.value for c in app.caption)
    assert "KPX 실측" in captions


def test_cloud_sample_is_real_kpx_data():
    """Streamlit Cloud에서 읽는 샘플 파일도 실측 데이터와 같아야 한다"""
    real = pd.read_csv(REAL_DATA, parse_dates=["datetime"])
    sample = pd.read_csv(SAMPLE_DATA, encoding="utf-8-sig", parse_dates=["datetime"])
    merged = sample.merge(real, on="datetime", suffixes=("_s", "_r"))
    assert len(merged) == len(sample)
    assert (merged["wind_mw_s"] - merged["wind_mw_r"]).abs().max() < 1e-6
    assert (merged["solar_mw_s"] - merged["solar_mw_r"]).abs().max() < 1e-6
