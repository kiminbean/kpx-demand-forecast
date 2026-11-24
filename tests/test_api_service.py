"""
API 서빙 경로 검증 테스트
=========================

학습 때와 같은 입력이 서빙 때도 모델에 들어가는지 확인한다.
- 시정 단위 (API는 km, 학습 데이터는 KMA 10m 단위)
- 운량 0(맑음)이 기본값으로 바뀌지 않는지
- 풍력: 과거 기상(history)을 받아 학습 파이프라인과 같은 피처를 만드는지
- 풍력: 평년값이 학습 때와 같은 2020-2023년 기준인지
- 태양광: 섬 평균 기상으로 V2 모델 테스트 예측(predictions_test.csv)을 재현하는지
- 태양광: 시점 T 예측이 구간 (T-1h, T] 일사(T 라벨)를 쓰는지, 직전 1시간·낮 일사·낮 운량 결측을 거부하는지
- 모델이 없을 때 0 MW 대신 오류를 내는지 (단일·배치), 구간 필드와 total_mw(항상 null)
- 시간대 표기(KST naive / +09:00 / Z)와 무관하게 같은 시각으로 예측하는지
"""

import asyncio
import warnings

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sklearn.metrics import r2_score

from api.schemas import BatchPredictionRequest, EnergyType, PredictionRequest, WeatherInput
from api.service import ModelUnavailableError, PredictionService

ROOT = Path(__file__).resolve().parent.parent
REAL_DATA = ROOT / "data" / "processed" / "kpx_real_merged.csv"
WEATHER_5Y = ROOT / "data" / "processed" / "weather_5year_hourly.csv"
WEATHER_2024 = ROOT / "data" / "raw" / "weather" / "jeju_weather_2024_all_stations.csv"
SOLAR_V2_PRED = ROOT / "models" / "solar_v2" / "island_mean" / "predictions_test.csv"

# 오프라인 평가(scripts/phase5_bias_correction.py)와 같은 시간순 분할 지점
TEST_START_RATIO = 0.85
# 예전 오프라인 패리티 기준(phase5_bias_correction.py, R² 0.7576)은 interval-START 라벨·
# 브레이크 전 혼합 학습 모델의 역사적 수치 — 2026-11-14 interval-END 재학습으로 무효화됐다.
# 재학습 모델의 정확한 CV 지표(RMSE 12.15, R² 0.5517)는 models/wind_phase5_fair/wind_fer_metadata.json이 기준.
WIND_MIN_R2 = 0.40  # 파손 감지 하한: CV 검증월 0.457~0.684, 시험 분할(Nov-Dec) 실측 0.505 — 하한 여유
HISTORY_HOURS = 24
BATCH_LIMIT = 168

# 학습 섬 평균 일사는 연중 일사를 관측하는 제주(184)·고산(185) 평균, 나머지 변수는 4개 관측소 평균
SOLAR_RAD_STATIONS = [184, 185]
ISLAND_MEAN_VARS = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure",
                    "cloud_cover", "precipitation", "visibility"]
SOLAR_PARITY_TOL = 1e-6
SOLAR_HISTORY_HOURS = 1


def _weather(dt: datetime, **overrides) -> WeatherInput:
    base = dict(
        datetime=dt, temperature=10.0, humidity=60.0, wind_speed=5.0,
        wind_direction=300.0, pressure=1015.0, solar_radiation=0.0,
        cloud_cover=5.0, visibility=20.0, precipitation=0.0,
    )
    base.update(overrides)
    return WeatherInput(**base)


def _rows_to_inputs(df: pd.DataFrame) -> list:
    """학습 데이터 행을 API 입력으로 변환 (시정은 10m → km)"""
    return [
        WeatherInput(
            datetime=r.datetime.to_pydatetime(), temperature=r.temperature,
            humidity=min(r.humidity, 100), wind_speed=r.wind_speed,
            wind_direction=r.wind_direction % 360, pressure=r.pressure,
            solar_radiation=r.solar_radiation, cloud_cover=r.cloud_cover,
            precipitation=r.precipitation, visibility=r.visibility / 100,
        )
        for r in df.itertuples()
    ]


@pytest.fixture(scope="module")
def loaded() -> PredictionService:
    """모델 로드는 한 번만 (PredictionService는 싱글턴). 모델 유무와 무관한 변환 테스트에도 쓴다

    이 모듈은 이전 서빙 모델(공개 거래량 라벨, SERVING_LABEL=public)의 동작을 검증한다.
    기본값인 계통 라벨 모델은 tests/test_api_system_models.py에서 본다.
    """
    from api.config import settings

    previous = settings.SERVING_LABEL
    settings.SERVING_LABEL = "public"
    svc = PredictionService()
    svc.load_models()
    yield svc
    settings.SERVING_LABEL = previous
    svc.load_models()


# 풍력 테스트는 풍력 모델이 없을 때만, 태양광 테스트는 태양광 모델이 없을 때만 건너뛴다
@pytest.fixture(scope="module")
def service(loaded) -> PredictionService:
    if loaded.wind_model is None:
        pytest.skip("풍력 모델 파일 없음")
    return loaded


@pytest.fixture(scope="module")
def solar_service(loaded) -> PredictionService:
    if loaded.solar_model is None:
        pytest.skip("태양광 모델 파일 없음")
    return loaded


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    if not REAL_DATA.exists():
        pytest.skip(f"{REAL_DATA} 없음")
    return pd.read_csv(REAL_DATA, parse_dates=["datetime"])


def test_visibility_km_converted_to_training_unit(loaded):
    df = loaded._weather_to_dataframe([_weather(datetime(2024, 12, 1, 12), visibility=20.0)])
    # KMA ASOS 시정은 10m 단위: 20km = 2000
    assert df["visibility"].iloc[0] == pytest.approx(2000.0)


def test_zero_cloud_cover_is_preserved(loaded):
    df = loaded._weather_to_dataframe([_weather(datetime(2024, 6, 1, 12), cloud_cover=0.0)])
    assert df["cloud_cover"].iloc[0] == 0.0


def test_align_features_preserves_order_and_values_without_fragmentation():
    service = PredictionService()
    columns = [f"feature_{i}" for i in range(150)]
    frame = pd.DataFrame([list(range(150)), list(range(150))], columns=pd.Index(columns), dtype=float)
    frame.loc[0, "feature_3"] = np.inf
    frame.loc[1, "feature_4"] = np.nan
    requested = ["missing", *reversed(columns)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        actual = service._align_features(frame, requested)
    expected = frame.reindex(columns=requested).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    np.testing.assert_array_equal(actual, expected)


def test_wind_patterns_use_2020_2023_only(service):
    if not WEATHER_5Y.exists():
        pytest.skip(f"{WEATHER_5Y} 없음")
    w = pd.read_csv(WEATHER_5Y, parse_dates=["datetime"])
    past = w[(w.datetime < "2024-01-01") & (w.datetime.dt.month == 12) & (w.datetime.dt.hour == 0)]
    p = service.weather_patterns
    row = p[(p["month"] == 12) & (p["hour"] == 0)].iloc[0]
    assert row["temperature_min"] == pytest.approx(past["temperature"].min())
    assert row["wind_speed_mean"] == pytest.approx(past["wind_speed"].mean())


def test_wind_accuracy_with_history(service, real_df):
    """24시간 history API 경로 R²가 interval-END 라벨 실측 대비 재학습 모델 수준인지 확인한다

    예전 '오프라인 패리티(R² 0.7576)'는 interval-START 라벨 기반 역사적 수치로 무효화됐다 —
    재학습 모델(phase5_tw0999, 2024-06 브레이크 후)의 기준은 CV 평균 RMSE 12.15 / R² 0.5517이며
    이 테스트는 파이프라인 파손을 잡는 하한 검사만 한다.
    레거시 CSV는 구간 시작 라벨이므로 END 라벨 = 파일값[T-1h]로 정렬해 비교한다.
    """
    start = int(len(real_df) * TEST_START_RATIO)
    test = real_df.iloc[start:].reset_index(drop=True)
    preds = []
    for i in range(0, len(test), BATCH_LIMIT):
        lo = start + i
        history = _rows_to_inputs(real_df.iloc[lo - HISTORY_HOURS:lo])
        targets = _rows_to_inputs(test.iloc[i:i + BATCH_LIMIT])
        out = service.predict_batch(targets, EnergyType.WIND, history=history)
        assert out.history_hours == HISTORY_HOURS
        preds += [p["predictions"]["wind"] for p in out.predictions]
    # END 라벨: 시각 T 실측 = 파일값[T-1h] → target 행들보다 한 행 앞의 값을 쓴다
    y_end = real_df["wind_mw"].iloc[start - 1:len(real_df) - 1].values
    r2 = r2_score(y_end, np.array(preds))
    assert r2 >= WIND_MIN_R2


def test_single_prediction_without_history_still_works(service):
    out = service.predict(_weather(datetime(2024, 12, 1, 3)), EnergyType.WIND)
    assert out.history_hours == 0
    assert out.predictions[0].prediction_mw >= 0


def test_single_prediction_uses_history(service):
    t = datetime(2024, 12, 1, 3)
    calm = [_weather(t - timedelta(hours=h), wind_speed=1.0) for h in range(HISTORY_HOURS, 0, -1)]
    windy = [_weather(t - timedelta(hours=h), wind_speed=12.0) for h in range(HISTORY_HOURS, 0, -1)]
    a = service.predict(_weather(t), EnergyType.WIND, history=calm)
    b = service.predict(_weather(t), EnergyType.WIND, history=windy)
    assert a.history_hours == b.history_hours == HISTORY_HOURS
    assert a.predictions[0].prediction_mw != b.predictions[0].prediction_mw


def test_history_with_gap_is_rejected(service):
    t = datetime(2024, 12, 1, 3)
    gappy = [_weather(t - timedelta(hours=5)), _weather(t - timedelta(hours=1))]
    with pytest.raises(ValueError, match="연속"):
        service.predict(_weather(t), EnergyType.WIND, history=gappy)


def _opt(value):
    """결측(NaN)은 API에서 '모름'인 None으로 보낸다"""
    return None if pd.isna(value) else float(value)


def _island_rows_to_inputs(df: pd.DataFrame) -> list:
    """섬 평균 기상(datetime index, 학습 단위) → API 입력 (시정 10m → km)"""
    return [
        WeatherInput(
            datetime=t.to_pydatetime(), temperature=r.temperature,
            humidity=min(r.humidity, 100), wind_speed=r.wind_speed,
            wind_direction=r.wind_direction % 360, pressure=r.pressure,
            solar_radiation=_opt(r.solar_radiation), cloud_cover=_opt(r.cloud_cover),
            precipitation=_opt(r.precipitation), visibility=_opt(r.visibility / 100),
        )
        for t, r in zip(df.index, df.itertuples())
    ]


@pytest.fixture(scope="module")
def island_weather() -> pd.DataFrame:
    if not WEATHER_2024.exists():
        pytest.skip(f"{WEATHER_2024} 없음")
    w = pd.read_csv(WEATHER_2024, encoding="utf-8-sig", parse_dates=["datetime"])
    island = w.groupby("datetime")[ISLAND_MEAN_VARS].mean()
    island["solar_radiation"] = (
        w[w["station_id"].isin(SOLAR_RAD_STATIONS)].groupby("datetime")["solar_radiation"].mean()
    )
    return island


def test_solar_v2_batch_reproduces_offline_test_predictions(solar_service, island_weather):
    """구간 시작 2024-11-07 06:00~12-31 23:00 (T = 07:00~2025-01-01 00:00) 예측이 학습 스크립트 결과와 같아야 한다"""
    if not SOLAR_V2_PRED.exists():
        pytest.skip(f"{SOLAR_V2_PRED} 없음")
    expected = pd.read_csv(SOLAR_V2_PRED, index_col=0, parse_dates=True)["pred_mw"]
    first = expected.index[0]
    times = pd.date_range(first - pd.Timedelta(hours=SOLAR_HISTORY_HOURS), expected.index[-1], freq="h")
    # 2024년 파일에 없는 마지막 T(2025-01-01 00:00)는 야간이라 예측이 야간 프로파일로 바뀌므로 직전 값을 이어 쓴다
    weather = island_weather.reindex(times).ffill()
    weather.loc[weather.index > island_weather.index.max(), "solar_radiation"] = np.nan
    inputs = _island_rows_to_inputs(weather)

    preds = []
    for i in range(SOLAR_HISTORY_HOURS, len(inputs), BATCH_LIMIT):
        history = inputs[i - SOLAR_HISTORY_HOURS:i]
        targets = inputs[i:i + BATCH_LIMIT]
        out = solar_service.predict_batch(targets, EnergyType.SOLAR, history=history)
        assert out.history_hours == SOLAR_HISTORY_HOURS
        assert [p["datetime"] for p in out.predictions] == [w.datetime.isoformat() for w in targets]
        preds += [p["predictions"]["solar"] for p in out.predictions]

    assert len(preds) == len(expected)
    np.testing.assert_allclose(np.array(preds), expected.values, rtol=0, atol=SOLAR_PARITY_TOL)


def test_solar_without_history_is_rejected(service, solar_service):
    t = datetime(2024, 6, 1, 12)
    with pytest.raises(ValueError, match="직전 1시간"):
        solar_service.predict(_weather(t, solar_radiation=2.0), EnergyType.SOLAR)
    with pytest.raises(ValueError, match="직전 1시간"):
        solar_service.predict_batch([_weather(t, solar_radiation=2.0)], EnergyType.BOTH)


def test_daytime_solar_without_radiation_is_rejected(service, solar_service):
    t = datetime(2024, 6, 1, 12)
    history = [_weather(t - timedelta(hours=1), solar_radiation=2.5)]
    with pytest.raises(ValueError, match="일사"):
        solar_service.predict(_weather(t, solar_radiation=None), EnergyType.SOLAR, history=history)
    # 풍력은 일사 결측을 기존처럼 기본값으로 채워 계속 예측한다
    out = service.predict(_weather(t, solar_radiation=None), EnergyType.WIND, history=history)
    assert out.predictions[0].prediction_mw >= 0


def test_solar_prediction_at_t_uses_radiation_labelled_t(solar_service):
    """T 예측은 (T-1h, T] 일사 = T 라벨 값을 써야 한다 (T-1 라벨 일사와 짝지으면 1시간 어긋남)"""
    t = datetime(2024, 6, 1, 12)

    def solar_at_t(rad_t: float, rad_prev: float) -> float:
        history = [_weather(t - timedelta(hours=1), solar_radiation=rad_prev)]
        out = solar_service.predict(_weather(t, solar_radiation=rad_t), EnergyType.SOLAR, history=history)
        return out.predictions[0].prediction_mw

    base = solar_at_t(rad_t=1.0, rad_prev=1.0)
    assert solar_at_t(rad_t=3.0, rad_prev=1.0) > base
    assert solar_at_t(rad_t=1.0, rad_prev=3.0) == pytest.approx(base)


# ---------------------------------------------------------------------------
# 모델 미로드, 구간 의미, 입력 검증, 시간대 정규화
# ---------------------------------------------------------------------------

DAY_T = datetime(2024, 6, 1, 12)


def _day_history(**overrides) -> list:
    return [_weather(DAY_T - timedelta(hours=1), solar_radiation=2.5, **overrides)]


def test_batch_with_solar_unavailable_errors_instead_of_zero(loaded, monkeypatch):
    """태양광 모델이 없으면 배치도 0 MW가 아니라 오류여야 한다 (서비스 계층·엔드포인트 모두)"""
    from api import main

    monkeypatch.setattr(loaded, "solar_model", None)
    with pytest.raises(ModelUnavailableError, match="Solar"):
        loaded.predict_batch([_weather(DAY_T, solar_radiation=2.0)], EnergyType.SOLAR, history=_day_history())

    request = BatchPredictionRequest(
        weather_data=[_weather(DAY_T, solar_radiation=2.0)], history=_day_history(), energy_type=EnergyType.BOTH,
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.predict_batch(request))
    assert exc.value.status_code == 500
    assert "solar" in exc.value.detail


def test_interval_fields_and_total_none_for_both(service, solar_service):
    """풍력·태양광 모두 T 예측 = (T-1h, T]. total_mw는 오차 독립 가정으로 내지 않는다"""
    out = solar_service.predict(_weather(DAY_T, solar_radiation=2.0), EnergyType.BOTH, history=_day_history())
    assert out.total_mw is None
    by_type = {p.energy_type: p for p in out.predictions}
    wind, solar = by_type[EnergyType.WIND], by_type[EnergyType.SOLAR]
    assert (wind.interval_start, wind.interval_end) == (DAY_T - timedelta(hours=1), DAY_T)
    assert (solar.interval_start, solar.interval_end) == (DAY_T - timedelta(hours=1), DAY_T)
    # 낮 RMSE 기반 폭이 붙는다 (테스트 예측 파일이 있을 때)
    if solar_service.solar_rmse_day is not None:
        assert solar.upper_bound_mw - solar.prediction_mw == pytest.approx(1.28 * solar_service.solar_rmse_day)
        assert solar_service.solar_rmse_day > solar_service.solar_rmse_night

    batch = solar_service.predict_batch(
        [_weather(DAY_T, solar_radiation=2.0)], EnergyType.BOTH, history=_day_history()
    )
    intervals = batch.predictions[0]["intervals"]
    assert intervals["wind"]["start"] == (DAY_T - timedelta(hours=1)).isoformat()
    assert intervals["wind"]["end"] == DAY_T.isoformat()
    assert intervals["solar"]["end"] == DAY_T.isoformat()


def test_radiation_in_w_per_m2_is_rejected_by_schema():
    """W/m²(예: 500)를 MJ/m²로 착각해 보내면 스키마 단계에서 거부한다"""
    with pytest.raises(ValidationError):
        _weather(DAY_T, solar_radiation=500.0)
    assert _weather(DAY_T, solar_radiation=500.0 * 0.0036).solar_radiation == pytest.approx(1.8)


def test_daytime_solar_without_cloud_cover_is_rejected(service, solar_service):
    """낮 운량 결측을 기본값으로 채우면 조용히 틀린 예측이 되므로 거부한다 (풍력은 기본값으로 계속)"""
    with pytest.raises(ValueError, match="결측"):
        solar_service.predict(
            _weather(DAY_T, solar_radiation=2.0, cloud_cover=None), EnergyType.SOLAR, history=_day_history()
        )
    out = service.predict(_weather(DAY_T, cloud_cover=None), EnergyType.WIND, history=_day_history())
    assert out.predictions[0].prediction_mw >= 0


def test_same_instant_in_any_timezone_gives_same_predictions(service, solar_service):
    """KST naive / +09:00 / Z 표기가 같은 시각이면 풍력·태양광 예측이 같아야 한다 (오프셋 혼합도 허용)"""
    def run(t_iso: str, prev_iso: str):
        weather = _weather(t_iso, solar_radiation=2.0)
        history = [_weather(prev_iso, solar_radiation=2.5)]
        out = solar_service.predict(weather, EnergyType.BOTH, history=history)
        return {p.energy_type: (p.prediction_mw, p.interval_start, p.interval_end) for p in out.predictions}

    naive = run("2024-06-01T12:00:00", "2024-06-01T11:00:00")
    assert run("2024-06-01T12:00:00+09:00", "2024-06-01T11:00:00+09:00") == naive
    assert run("2024-06-01T03:00:00Z", "2024-06-01T02:00:00Z") == naive
    assert run("2024-06-01T03:00:00Z", "2024-06-01T11:00:00+09:00") == naive


def test_request_examples_are_valid():
    """/docs 예시가 스키마를 통과하고 태양광에 필요한 직전 1시간 history를 담고 있어야 한다"""
    for model in (PredictionRequest, BatchPredictionRequest):
        example = model.model_json_schema()["example"]
        request = model.model_validate(example)
        assert len(request.history) == SOLAR_HISTORY_HOURS
