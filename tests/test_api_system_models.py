"""
API 기본 서빙 모델(계통 라벨) 테스트
===================================

- 설정 기본값이 계통 라벨 모델 디렉터리를 가리키는지, 풍력 지표를 메타데이터에서 읽는지
- 태양광: level_history가 충분하면 발전 수준 모델을, 없거나 모자라면 기상만 쓰는 모델을 쓰는지
- 발전 수준에 D-1 이후 기록이 쓰이지 않는지 (누수 방지)
"""

from datetime import datetime, timedelta

import pytest

from api.config import settings
from api.schemas import EnergyType, LevelRecord, PredictionRequest, WeatherInput
from api.service import PredictionService

TARGET = datetime(2026, 5, 20, 12)


def _weather(t: datetime, **over) -> WeatherInput:
    base = dict(datetime=t, temperature=20.0, humidity=60.0, wind_speed=5.0, wind_direction=270.0,
                pressure=1013.0, solar_radiation=2.5, cloud_cover=3.0)
    base.update(over)
    return WeatherInput(**base)


def _level_history(first_lag_days: int, last_lag_days: int, solar_mw: float = 200.0) -> list:
    """거래일 D 기준 D-first ~ D-last 일의 낮(09~16시) 기록"""
    out = []
    for lag in range(first_lag_days, last_lag_days - 1, -1):
        day = TARGET.replace(hour=0) - timedelta(days=lag)
        for hour in range(9, 17):
            out.append(LevelRecord(datetime=day + timedelta(hours=hour), temperature=20.0, humidity=60.0,
                                   solar_radiation=2.0, cloud_cover=3.0, solar_mw=solar_mw))
    return out


@pytest.fixture(scope="module")
def svc() -> PredictionService:
    assert settings.SERVING_LABEL == "system"
    service = PredictionService()
    service.load_models()
    if service.solar_model is None or service.solar_level_model is None or service.wind_model is None:
        pytest.skip("계통 라벨 서빙 모델 파일 없음 — python scripts/train_serving_models_system.py")
    return service


def _solar(svc: PredictionService, level_history=None):
    out = svc.predict(_weather(TARGET), EnergyType.SOLAR, history=[_weather(TARGET - timedelta(hours=1))],
                      level_history=level_history)
    return out.predictions[0]


def test_default_settings_point_to_system_label_models(svc):
    assert settings.solar_model_dir.parts[-2:] == ("solar_v2_system", "island_mean")
    assert settings.wind_model_path.parent.name == "wind_system"
    assert svc.wind_rmse > 20 and 0 < svc.wind_r2 < 1          # 메타데이터에서 읽은 값 (이전 모델 상수 12.15가 아님)
    assert svc.solar_r2 is not None and svc.solar_rmse_day is not None


def test_solar_without_level_history_uses_weather_only_model(svc):
    r = _solar(svc)
    assert r.level_used is False and r.prediction_mw > 50
    assert r.lower_bound_mw <= r.prediction_mw <= r.upper_bound_mw


def test_solar_level_history_changes_prediction_with_fleet_level(svc):
    low, high = _solar(svc, _level_history(8, 2, 150.0)), _solar(svc, _level_history(8, 2, 300.0))
    assert low.level_used is True and high.level_used is True
    assert high.prediction_mw > low.prediction_mw * 1.2         # 발전 수준이 두 배면 예측도 뚜렷이 커진다


def test_solar_level_falls_back_when_history_is_short_or_too_recent(svc):
    base = _solar(svc).prediction_mw
    short = _solar(svc, _level_history(3, 2))                   # 2일치 — 최소 4일 필요
    recent = _solar(svc, _level_history(1, 0, 999.0))           # D-1·D일 기록뿐 — 지연 규칙상 쓸 수 없다
    assert short.level_used is False and recent.level_used is False
    assert short.prediction_mw == pytest.approx(base) and recent.prediction_mw == pytest.approx(base)


def test_level_ignores_records_after_d_minus_2(svc):
    old = _level_history(8, 2, 200.0)
    leaky = old + _level_history(1, 1, 999.0)                   # D-1 기록을 더해도 결과가 같아야 한다
    assert _solar(svc, leaky).prediction_mw == pytest.approx(_solar(svc, old).prediction_mw)


def test_duplicate_level_history_times_are_rejected(svc):
    dup = _level_history(8, 2)
    with pytest.raises(ValueError, match="두 번"):
        _solar(svc, dup + dup[:1])


def test_daytime_solar_without_cloud_cover_is_rejected(svc):
    with pytest.raises(ValueError, match="결측"):
        svc.predict(_weather(TARGET, cloud_cover=None), EnergyType.SOLAR,
                    history=[_weather(TARGET - timedelta(hours=1))])


def test_request_schema_accepts_level_history_and_wind_predicts(svc):
    req = PredictionRequest(weather=_weather(TARGET), history=[_weather(TARGET - timedelta(hours=1))],
                            level_history=_level_history(8, 2), energy_type=EnergyType.BOTH)
    out = svc.predict(req.weather, req.energy_type, history=req.history, level_history=req.level_history)
    by_type = {p.energy_type: p for p in out.predictions}
    assert by_type[EnergyType.SOLAR].level_used is True
    assert by_type[EnergyType.WIND].prediction_mw >= 0 and by_type[EnergyType.WIND].level_used is None
