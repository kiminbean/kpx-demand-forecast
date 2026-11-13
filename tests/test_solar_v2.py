"""
태양광 V2 피처·모델 테스트
=========================

- 정렬: KPX 거래시간 h → 구간 끝 시각 T = 거래일 + h시간
- 태양 기하: 제주 하지 남중 시각·고도
- 누수: 발전 수준 피처는 D-2까지의 발전량만 쓴다
- 야간 후처리, 저장/불러오기 왕복
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import solar_features_v2 as S2
from src.models.solar_v2 import LEVEL_COL, SolarV2Model

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "models" / "solar_v2" / "stations_level"


def test_training_mask_respects_first_delivery_day_deadline():
    from scripts.train_solar_v2 import training_mask_for_forecast

    ends = pd.DatetimeIndex(["2024-04-30 09:00", "2024-04-30 10:00",
                             "2024-04-30 11:00", "2024-05-01 00:00"])
    assert training_mask_for_forecast(ends, pd.Timestamp("2024-05-01 12:00")).tolist() == [True, True, False, False]


def test_day2_weather_never_falls_back_to_day1(tmp_path):
    from scripts.train_solar_v2 import load_nwp_weather

    path = tmp_path / "nwp.csv"
    pd.DataFrame([
        {"datetime": "2024-05-01 13:00", "station": "jeju", "lead_days": 2,
         "temperature": 20.0, "humidity": 60.0, "solar_radiation": np.nan, "cloud_cover": 3.0},
        {"datetime": "2024-05-01 13:00", "station": "jeju", "lead_days": 1,
         "temperature": 21.0, "humidity": 61.0, "solar_radiation": 2.0, "cloud_cover": 4.0},
    ]).to_csv(path, index=False)
    day2 = load_nwp_weather(2, path)
    assert pd.isna(day2.loc["2024-05-01 13:00", "solar_radiation_jeju"])
    assert load_nwp_weather(1, path).loc["2024-05-01 13:00", "solar_radiation_jeju"] == 2.0


def _write_kpx(path: Path, date_col: str, hours, values, region="제주도", fuel="태양광"):
    df = pd.DataFrame({date_col: "2024-03-01", "거래시간": hours, "지역": region, "연료원": fuel,
                       "전력거래량(MWh)": values})
    df.to_csv(path, index=False, encoding="cp949")


def test_kpx_hour_maps_to_interval_end(tmp_path):
    p = tmp_path / "kpx.csv"
    _write_kpx(p, "거래일자", [1, 13, 24], [0.0, 120.0, 0.5])
    s = S2.load_kpx_hourly(p)
    assert list(s.index) == [pd.Timestamp("2024-03-01 01:00"), pd.Timestamp("2024-03-01 13:00"),
                             pd.Timestamp("2024-03-02 00:00")]
    assert s.iloc[1] == 120.0


def test_kpx_2025_schema_is_accepted(tmp_path):
    # 2025년 파일: 헤더 '거래일', 0으로 채운 시간 문자열
    p = tmp_path / "kpx2025.csv"
    _write_kpx(p, "거래일", ["01", "02"], [1.0, 2.0])
    s = S2.load_kpx_hourly(p)
    assert list(s.index.hour) == [1, 2]


def test_kpx_invalid_hour_fails_fast(tmp_path):
    p = tmp_path / "bad.csv"
    _write_kpx(p, "거래일자", [0, 25], [1.0, 2.0])
    with pytest.raises(ValueError):
        S2.load_kpx_hourly(p)


def test_solar_noon_and_altitude_at_summer_solstice():
    times = pd.date_range("2024-06-21 11:00", "2024-06-21 14:00", freq="min")
    geo = S2.solar_geometry(times)
    noon = geo["zenith"].idxmin()
    # 경도 126.55°E, KST(135°E 기준) → 남중 약 12:34, 균시차(하지 ≈ −1.7분) 반영
    assert abs((noon - pd.Timestamp("2024-06-21 12:36")).total_seconds()) <= 5 * 60
    # 남중 천정각 ≈ 위도 − 적위 = 33.38 − 23.44 ≈ 9.9°
    assert geo["zenith"].min() == pytest.approx(9.9, abs=0.5)


def test_interval_irradiance_night_and_clear_sky_bounds():
    t = pd.date_range("2024-06-21 01:00", "2024-06-22 00:00", freq="h")
    irr = S2.interval_irradiance(t)
    night = irr["daylight_frac"] == 0
    assert (irr.loc[night, ["H0", "Hcs"]] == 0).all().all()
    assert (irr["Hcs"] <= irr["H0"] + 1e-12).all()
    # 하지 정오 구간 (11~12시, 12~13시) 대기권 밖 수평 일사 ≈ 4.4~4.6 MJ/m²
    assert 4.2 < irr.loc[pd.Timestamp("2024-06-21 13:00"), "H0"] < 4.8


def _synthetic_frame(n_days=40, seed=0):
    """맑은하늘 일사에 구름 잡음을 섞은 관측소 자료와 그에 비례하는 발전량"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-04-01 01:00", periods=24 * n_days, freq="h")
    irr = S2.interval_irradiance(idx)
    wide = pd.DataFrame(index=idx.union(idx - pd.Timedelta(hours=1)))
    cloud = pd.Series(rng.uniform(0.3, 1.0, len(wide)), index=wide.index)
    hcs_all = S2.interval_irradiance(wide.index)["Hcs"]
    for st in S2.ALL_STATIONS:
        for v in S2.WEATHER_VARS:
            wide[f"{v}_{st}"] = 15.0
        wide[f"cloud_cover_{st}"] = 10 * (1 - cloud)
        wide[f"solar_radiation_{st}"] = hcs_all * cloud if st in S2.RAD_STATIONS else np.nan
    frame = S2.build_solar_frame(wide, index=idx)
    gen = pd.Series(80.0 * irr["Hcs"].values * cloud.reindex(idx).values, index=idx)
    return frame, gen


def test_clear_sky_ratio_is_one_for_clear_sky_input():
    frame, _ = _synthetic_frame(3)
    for st in S2.RAD_STATIONS:
        frame[f"solar_radiation_{st}"] = frame["Hcs"]
    phys = S2.physics_features(frame)
    r = phys["p_ratio_cs"].dropna()
    assert len(r) > 0
    np.testing.assert_allclose(r.values, 1.0, atol=1e-9)


def test_generation_level_uses_only_d_minus_2():
    frame, gen = _synthetic_frame(30)
    p_mean = S2.physics_features(frame)["p_mean"]
    lvl = S2.generation_level(frame, gen, p_mean)
    # 4/15 발전량을 바꾸면 4/17 거래일부터 값이 바뀌어야 한다 (D-2 컷오프)
    gen2 = gen.copy()
    day = (gen2.index - pd.Timedelta(hours=1)).normalize()
    gen2[day == pd.Timestamp("2024-04-15")] *= 3
    lvl2 = S2.generation_level(frame, gen2, p_mean)
    changed = (lvl2 - lvl).abs() > 1e-12
    first = lvl.index[changed.values][0]
    assert first == pd.Timestamp("2024-04-17 01:00")
    # 미래 발전량이 결측이어도 과거 값은 그대로 (운영 시점 재현)
    gen3 = gen.where(gen.index < pd.Timestamp("2024-04-20"))
    lvl3 = S2.generation_level(frame, gen3, p_mean)
    early = lvl.index < pd.Timestamp("2024-04-21 01:00")
    np.testing.assert_allclose(lvl3[early].values, lvl[early].values, equal_nan=True)


def test_night_profile_replaces_after_sunset_only():
    idx = pd.date_range("2024-05-01 01:00", periods=48, freq="h")
    dl = S2.interval_irradiance(idx)["daylight_frac"]
    gen = pd.Series(np.where(dl.values == 0, 3.0, 50.0), index=idx)
    prof = S2.night_profile(gen, dl)
    pred = S2.apply_night_profile(np.full(48, -1.0), idx, dl.values, prof)
    assert np.all(pred[dl.values == 0] == 3.0)
    assert np.all(pred[dl.values > 0] == 0.0)  # 음수 예측은 0으로


def test_model_round_trip(tmp_path):
    frame, gen = _synthetic_frame(40)
    X = S2.create_solar_features_v2(frame)
    X[LEVEL_COL] = S2.generation_level(frame, gen, S2.physics_features(frame)["p_mean"])
    model = SolarV2Model(seeds=(1,)).fit(X, gen, frame["daylight_frac"])
    Xp = X[X[LEVEL_COL].notna()]  # 수준 이력이 쌓인 뒤 구간만 예측 (첫 며칠은 lvl_p 없음 → 실패가 정상)
    p1 = model.predict(Xp, frame["daylight_frac"])
    loaded = SolarV2Model.load(model.save(tmp_path / "m"))
    np.testing.assert_allclose(loaded.predict(Xp, frame["daylight_frac"]), p1)
    assert loaded.nan_allowed == model.nan_allowed and loaded.last_level == model.last_level
    assert np.corrcoef(p1, gen[Xp.index])[0, 1] > 0.9


@pytest.fixture(scope="module")
def fitted():
    frame, gen = _synthetic_frame(40)
    X = S2.create_solar_features_v2(frame)
    X[LEVEL_COL] = S2.generation_level(frame, gen, S2.physics_features(frame)["p_mean"])
    model = SolarV2Model(seeds=(1,)).fit(X, gen, frame["daylight_frac"])
    return frame, X, model


def test_predict_fails_on_missing_daytime_radiation(fitted):
    frame, X, model = fitted
    X2 = X.copy()
    day = frame["daylight_frac"] > 0
    X2.loc[day, "p_mean"] = np.nan
    with pytest.raises(ValueError, match="일사"):
        model.predict(X2, frame["daylight_frac"])


def test_predict_requires_level_unless_stale_allowed(fitted):
    frame, X, model = fitted
    X2 = X.copy()
    X2[LEVEL_COL] = np.nan
    with pytest.raises(ValueError, match=LEVEL_COL):
        model.predict(X2, frame["daylight_frac"])
    with pytest.warns(RuntimeWarning):
        p = model.predict(X2, frame["daylight_frac"], allow_stale_level=True)
    X3 = X.copy()
    X3[LEVEL_COL] = model.last_level
    np.testing.assert_allclose(p, model.predict(X3, frame["daylight_frac"]))


def test_predict_rejects_nan_in_feature_never_missing_in_training(fitted):
    frame, X, model = fitted
    assert model.nan_allowed["cloud_mean"] is False
    X2 = X.copy()
    first_day = X2.index[frame["daylight_frac"].values > 0][0]
    X2.loc[first_day, "cloud_mean"] = np.nan
    with pytest.raises(ValueError, match="cloud_mean"):
        model.predict(X2, frame["daylight_frac"])


def test_predict_rejects_missing_daylight_rows(fitted):
    frame, X, model = fitted
    with pytest.raises(ValueError, match="daylight_frac"):
        model.predict(X, frame["daylight_frac"].iloc[1:])


def test_tz_aware_times_are_converted_to_kst():
    naive = pd.date_range("2024-06-21 10:00", periods=5, freq="h")
    aware = naive.tz_localize("Asia/Seoul").tz_convert("UTC")
    pd.testing.assert_frame_equal(S2.solar_geometry(aware), S2.solar_geometry(naive), check_freq=False)


def test_generation_level_goes_missing_after_long_outage():
    frame, gen = _synthetic_frame(40)
    p_mean = S2.physics_features(frame)["p_mean"]
    # 4/10 이후 발전량 공급이 끊긴 상황: 창(14일)에 실제 자료가 4일 미만이 되면 NaN (낡은 값을 이어 쓰지 않음)
    gen_cut = gen.where(gen.index < pd.Timestamp("2024-04-10"))
    lvl = S2.generation_level(frame, gen_cut, p_mean)
    assert lvl.loc["2024-04-14"].notna().all()     # D-2 = 4/12, 창 안에 실제 자료 충분
    assert lvl.loc["2024-04-25"].isna().all()      # D-2 = 4/23, 창(4/10~4/23)에 실제 자료 0일


def test_island_mean_overwrites_station_columns():
    frame, _ = _synthetic_frame(2)
    frame["temperature_jeju"] = 10.0
    m = S2.to_island_mean(frame)
    expected = frame[[f"temperature_{st}" for st in S2.ALL_STATIONS]].mean(axis=1)
    for st in S2.ALL_STATIONS:
        np.testing.assert_allclose(m[f"temperature_{st}"].values, expected.values)
    assert m["solar_radiation_seogwipo"].isna().all()


@pytest.mark.skipif(not SAVED.exists(), reason="학습된 models/solar_v2/stations_level 없음")
def test_saved_model_reproduces_reported_test_metrics():
    import json

    import scripts.train_solar_v2 as T

    frame, X, y = T.build_dataset("stations_level")
    s = pd.Series(frame.index - pd.Timedelta(hours=1), index=frame.index)
    te = ((s >= T.TEST_START) & (s <= T.LAST_START)).values
    pred = SolarV2Model.load(SAVED).predict(X[te], frame["daylight_frac"])
    reported = json.loads((SAVED / "metrics_test.json").read_text(encoding="utf-8"))
    from src.evaluation import evaluate_generation

    cap = T.CAPACITY_TEST[T.HEADLINE_CAPACITY]
    got = evaluate_generation(y[te], pred, cap)["fer_mean_pct"]
    expected = reported["by_capacity"][T.HEADLINE_CAPACITY]["mainland_2025_benchmark"]["fer_mean_pct"]
    assert got == pytest.approx(expected, abs=1e-9)
