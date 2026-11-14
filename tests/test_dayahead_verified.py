"""
하루 전 재검증 실험(GFS 예보 수집 + walk-forward) 테스트
=====================================================

- GFS: idx 파싱, 쌍선형 가중치, 6시간 평균 일사를 1시간 값으로 푸는 계산, 마감 전 업로드 판정
- 실험: 재학습 일정, 학습 컷오프(공개 지연 포함), 보정이 미래 관측을 보지 않는지, 차이 신뢰구간
"""

import numpy as np
import pandas as pd
import pytest

import scripts.download_nwp_gfs_aws as G
import scripts.experiment_solar_dayahead_verified as E
from src.features import solar_features_v2 as S2


# ---------------------------------------------------------------------------
# GFS 수집
# ---------------------------------------------------------------------------

def test_parse_idx_gives_byte_ranges():
    text = ("1:0:d=2024050118:PRMSL:mean sea level:24 hour fcst:\n"
            "2:990253:d=2024050118:DSWRF:surface:18-24 hour ave fcst:\n"
            "3:1954616:d=2024050118:DLWRF:surface:18-24 hour ave fcst:\n")
    recs = G._parse_idx(text)
    assert recs[1] == ("DSWRF", "surface", "18-24 hour ave fcst", 990253, 1954615)
    assert recs[2][4] is None  # 마지막 메시지는 끝 위치를 모른다


def test_bilinear_weights_sum_to_one_and_hit_grid_points():
    for lat, lon in G.STATION_COORDS.values():
        assert sum(G._bilinear_weights(lat, lon).values()) == pytest.approx(1.0)
    w = G._bilinear_weights(33.5, 126.5)
    assert w["g_33.50_126.50"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        G._bilinear_weights(34.2, 126.5)


def _write_raw_run(raw_dir, init: pd.Timestamp, hourly: dict, posted: str) -> None:
    """f025부터의 1시간 평균 일사(hourly[f])로 GFS식 6시간 구간 평균을 만들어 raw CSV로 쓴다"""
    cfg = G.SCENARIOS["d1_10kst"]
    rows = []
    for lead in range(cfg["lead_first"], cfg["lead_last"] + 1):
        start = ((lead - 1) // G.DSWRF_BUCKET_HOURS) * G.DSWRF_BUCKET_HOURS
        avg = np.mean([hourly[f] for f in range(start + 1, lead + 1)])
        for var, val, step in (("dswrf_avg", avg, f"{start}-{lead}"), ("tcdc", 50.0, str(lead)),
                               ("tmp2m", 293.15, str(lead)), ("rh2m", 60.0, str(lead))):
            row = {"init_utc": init, "lead_h": lead, "var": var, "step_range": step, "posted_utc": posted}
            row.update({f"g_{la:.2f}_{lo:.2f}": val for la in G.BOX_LATS for lo in G.BOX_LONS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(raw_dir / f"{init:%Y%m%d}.csv", index=False)


def test_build_tidy_recovers_hourly_radiation_and_checks_deadline(tmp_path):
    hourly = {f: float(10 * (f - 24)) for f in range(25, 43)}  # f025=10, f026=20, … W/m²
    _write_raw_run(tmp_path, pd.Timestamp("2024-05-01 18:00"), hourly, "2024-05-01 21:45:00")
    _write_raw_run(tmp_path, pd.Timestamp("2024-05-02 18:00"), hourly, "2024-05-03 02:00:00")  # 마감(01:00 UTC) 뒤
    tidy = G.build_tidy("d1_10kst", tmp_path)
    jeju = tidy[tidy["station"] == "jeju"].set_index("datetime")

    # 초기화 05-01 18 UTC + f027 = 05-02 21 UTC = 05-03 06:00 KST, 구간 (05, 06]
    assert jeju.loc["2024-05-03 06:00", "solar_radiation"] == pytest.approx(hourly[27] * G.W_TO_MJ_PER_HOUR)
    # 6시간 구간이 바뀌는 f031(= 24-30 평균 다음의 30-31 평균)도 그대로 풀린다
    assert jeju.loc["2024-05-03 10:00", "solar_radiation"] == pytest.approx(hourly[31] * G.W_TO_MJ_PER_HOUR)
    # 첫 예보시간(f026)은 직전 값이 없어 풀 수 없다 (KST 05시, 야간)
    assert np.isnan(jeju.loc["2024-05-03 05:00", "solar_radiation"])
    assert jeju.loc["2024-05-03 06:00", "temperature"] == pytest.approx(20.0)
    assert jeju.loc["2024-05-03 06:00", "cloud_cover"] == pytest.approx(5.0)
    assert jeju.loc["2024-05-03 06:00":"2024-05-03 20:00", "deadline_ok"].all()
    assert not jeju.loc["2024-05-04 06:00":"2024-05-04 20:00", "deadline_ok"].any()


def test_scenarios_cover_same_local_hours():
    # 두 시나리오 모두 거래일 D의 05:00~20:00 KST를 덮어야 한다
    for cfg in G.SCENARIOS.values():
        init_kst_hour = cfg["cycle"] + 9 + 24 * cfg["init_day_offset"]  # 거래일 D 00시 기준 시각
        assert init_kst_hour + cfg["lead_first"] == 5
        assert init_kst_hour + cfg["lead_last"] == 20


# ---------------------------------------------------------------------------
# 실험 일정 · 누수 방지
# ---------------------------------------------------------------------------

def test_origins_match_existing_cv_folds_and_never_reach_test_window():
    for key in ("A", "As"):
        stage = E.STAGES[key]
        monthly = E.origins_for(stage, "monthly")
        assert [str(d.date()) for d in monthly] == [fs for fs, _ in E.T.CV_FOLDS]
        weekly = E.origins_for(stage, "weekly")
        assert weekly[0] == monthly[0] and max(weekly) < E.TEST_START
        assert E.origins_for(stage, "frozen") == [monthly[0]]
    assert len(E.origins_for(E.STAGES["B"], "monthly")) == 12
    assert len(E.origins_for(E.STAGES["D"], "monthly")) == 10
    for key in ("B", "C", "D"):
        stage = E.STAGES[key]
        assert max(E.origins_for(stage, "weekly")) < stage.eval_end
        assert E.origins_for(stage, "frozen") == [stage.eval_start]


def test_no_stage_evaluates_the_exhausted_test_window():
    lo, hi = E.TEST_START, pd.Timestamp("2025-01-01")
    for stage in E.STAGES.values():
        assert stage.eval_end <= lo or stage.eval_start >= hi
        assert stage.train_start < stage.eval_start
        if not stage.selection:
            assert E.STAGES[stage.decided_by].selection and E.STAGES[stage.decided_by].label == stage.label


def test_training_mask_applies_publication_lag():
    ends = pd.date_range("2024-04-20 10:00", "2024-05-01 00:00", freq="h")
    origin, start = pd.Timestamp("2024-05-01"), pd.Timestamp("2024-01-02")
    assert ends[E.training_mask(ends, origin, start)].max() == pd.Timestamp("2024-04-30 10:00")
    assert ends[E.training_mask(ends, origin, start, label_lag_days=7)].max() == pd.Timestamp("2024-04-23 10:00")
    late_start = pd.Timestamp("2024-04-25")
    assert ends[E.training_mask(ends, origin, late_start)].min() == pd.Timestamp("2024-04-25 01:00")


def test_stage_configs_have_unique_names_and_comparisons_resolve():
    for configs in (E.SELECTION_CONFIGS, E.CONFIRM_CONFIGS):
        names = [c.name for c in configs]
        assert len(names) == len(set(names)) and E.BASELINE in names
    known = {c.name for c in E.CONFIRM_CONFIGS} | set(E.ENSEMBLES)
    for members in E.ENSEMBLES.values():
        assert all(m in known for m in members)
    for _, a, b in E.CONFIRM_COMPARISONS:
        assert a in known and b in known


def _toy_weather(n_days: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-02-01 01:00", periods=24 * n_days, freq="h")
    rng = np.random.default_rng(0)
    w = pd.DataFrame(index=idx)
    for st in S2.ALL_STATIONS:
        w[f"solar_radiation_{st}"] = rng.uniform(0, 3, len(idx))
        w[f"cloud_cover_{st}"] = rng.uniform(0, 10, len(idx))
    return w


def test_mos_linear_recovers_mapping_and_ignores_later_observations():
    fc = _toy_weather()
    obs = fc.copy()
    for st in S2.RAD_STATIONS:
        obs[f"solar_radiation_{st}"] = 0.8 * fc[f"solar_radiation_{st}"] + 0.1
    fit_start, fit_end = pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-20 00:00")
    c1 = E.fit_mos(fc, obs, ["rad_linear"], fit_start, fit_end)
    assert c1["solar_radiation_jeju"]["a"] == pytest.approx(0.8)
    assert c1["solar_radiation_jeju"]["b"] == pytest.approx(0.1)

    later = obs.copy()
    later.loc[later.index > fit_end, [f"solar_radiation_{st}" for st in S2.RAD_STATIONS]] = 99.0
    assert E.fit_mos(fc, later, ["rad_linear"], fit_start, fit_end) == c1  # 적합 창 뒤 관측은 계수에 영향이 없다

    out = E.apply_mos(fc, c1)
    assert np.allclose(out["solar_radiation_jeju"], obs["solar_radiation_jeju"])
    assert out["cloud_cover_jeju"].equals(fc["cloud_cover_jeju"])  # 일사 보정은 운량을 건드리지 않는다


def test_mos_cloud_is_clipped_to_valid_range():
    fc = _toy_weather()
    obs = fc.copy()
    for st in S2.ALL_STATIONS:
        obs[f"cloud_cover_{st}"] = 2.0 * fc[f"cloud_cover_{st}"]
    out = E.apply_mos(fc, E.fit_mos(fc, obs, ["cloud_linear"], fc.index.min(), fc.index.max()))
    assert out["cloud_cover_jeju"].between(0, E.CLOUD_MAX).all()


def test_generation_level_lag_shifts_by_whole_days_and_rejects_lookahead():
    idx = pd.date_range("2024-03-01 01:00", periods=24 * 40, freq="h")
    frame = pd.DataFrame(index=idx)
    day = (idx - pd.Timedelta(hours=1)).normalize()
    gen = pd.Series(np.where((idx.hour >= 10) & (idx.hour <= 15), 100.0 + day.dayofyear, 0.0), index=idx)
    p = pd.Series(np.where((idx.hour >= 10) & (idx.hour <= 15), 500.0, 0.0), index=idx)
    lvl2 = S2.generation_level(frame, gen, p)
    lvl3 = S2.generation_level(frame, gen, p, lag_days=3)
    assert np.allclose(lvl3.iloc[24:].values, lvl2.iloc[:-24].values, equal_nan=True)
    with pytest.raises(ValueError):
        S2.generation_level(frame, gen, p, lag_days=1)


def test_paired_fer_diff_sign_and_zero():
    idx = pd.date_range("2025-01-01 01:00", periods=24 * 60, freq="h")
    y = np.full(len(idx), 100.0)
    good, bad = y + 5.0, y + 20.0
    same = E.paired_fer_diff(y, good, good, idx, cap=300.0)
    assert same["diff_pp"] == 0.0 and same["ci_lo"] == 0.0 and same["ci_hi"] == 0.0
    r = E.paired_fer_diff(y, good, bad, idx, cap=300.0)
    assert r["diff_pp"] == pytest.approx(-5.0)       # (5 − 20) / 300 × 100
    assert r["ci_hi"] < 0 and r["n_eval_hours"] == len(idx)


# ---------------------------------------------------------------------------
# 라벨
# ---------------------------------------------------------------------------

def test_system_hourly_shifts_solar_by_display_lag(tmp_path):
    import scripts.download_kpx_jeju_system as K

    times = pd.date_range("2026-05-01 00:05", "2026-05-01 03:00", freq="5min")
    solar = np.arange(len(times), dtype=float)
    lines = ["기준일시,공급능력(MW),현재수요(MW),신재생총합(MW),신재생태양광(MW),신재생풍력(MW)"]
    lines += [f"{t:%Y%m%d%H%M%S},1000,600,{v + 5},{v},5" for t, v in zip(times, solar)]
    (tmp_path / "2026-05.csv").write_text("\n".join(lines), encoding="utf-8")
    h = K.build_hourly(tmp_path)
    # 표시 그대로: (01:00, 02:00] → 표본 12~23의 평균. 지연 보정: 10분 뒤 표본 14~25의 평균
    assert h.loc["2026-05-01 02:00", "solar_raw_mw"] == pytest.approx(solar[12:24].mean())
    assert h.loc["2026-05-01 02:00", "solar_mw"] == pytest.approx(solar[14:26].mean())
    assert h.loc["2026-05-01 02:00", "wind_mw"] == pytest.approx(5.0)
    # 마지막 시간은 당겨 올 표본이 모자라 결측이어야 한다
    assert np.isnan(h.loc["2026-05-01 03:00", "solar_mw"]) or h.loc["2026-05-01 03:00", "n_samples"] == 12


def test_public_label_matches_fuel_zip_solar_for_overlap_years():
    if not (E.KPX_FUEL_ZIP.exists() and E.KPX_2024.exists() and E.KPX_2025.exists()):
        pytest.skip("원본 자료 없음")
    for year, path, tol in ((2025, E.KPX_2025, 1e-6), (2024, E.KPX_2024, 1.0)):
        zip_solar = E.load_fuel_zip_solar(years=(year,))
        public = S2.load_kpx_hourly(path).sort_index()
        assert zip_solar.index.equals(public.index)
        # 2025년은 같은 값. 2024년 파일은 2025-12에 받은 것이라 이후 수정정산분(연 합계 0.03%)만큼 다르다
        assert np.abs(zip_solar.values - public.values).max() < tol
        assert abs(zip_solar.sum() / public.sum() - 1) < 1e-3


def test_ifs_three_hourly_energy_is_split_by_clear_sky_and_conserved(tmp_path):
    import scripts.download_nwp_ecmwf_aws as I

    cfg = I.SCENARIOS["d1_10kst"]
    init = pd.Timestamp("2024-05-01 12:00")
    ssrd = np.cumsum([0.0, 0.0, 3.0e6, 9.0e6, 9.0e6, 3.0e6, 0.0]) + 5.0e7   # 3시간 구간 에너지 0·3·9·9·3·0 MJ
    rows = []
    for k, step in enumerate(cfg["steps"]):
        for var, val in (("ssrd", ssrd[k]), ("t2m", 283.15 + k), ("d2m", 283.15 + k)):
            row = {"init_utc": init, "step_h": step, "var": var, "posted_utc": "2024-05-01 19:55:00"}
            row.update({f"g_{la:.2f}_{lo:.2f}": val for la in I.BOX_LATS for lo in I.BOX_LONS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "20240501.csv", index=False)
    tidy = I.build_tidy("d1_10kst", tmp_path)
    jeju = tidy[tidy["station"] == "jeju"].set_index("datetime")

    assert jeju.index.min() == pd.Timestamp("2024-05-03 04:00") and jeju.index.max() == pd.Timestamp("2024-05-03 21:00")
    # 구간 (09, 12] KST의 9 MJ가 3시간에 나뉘고 합은 보존된다. 정오에 가까울수록 몫이 크다
    blk = jeju.loc["2024-05-03 10:00":"2024-05-03 12:00", "solar_radiation"]
    assert blk.sum() == pytest.approx(9.0) and blk.iloc[0] < blk.iloc[2]
    assert jeju["solar_radiation"].sum() == pytest.approx(24.0)
    # 기온은 3시간 사이를 선형 보간한다: 09시 12 ℃(k=2), 12시 13 ℃ → 10시 12.33 ℃. 이슬점 = 기온이면 습도 100%
    assert jeju.loc["2024-05-03 10:00", "temperature"] == pytest.approx(12 + 1 / 3)
    assert jeju["humidity"].round(6).eq(100.0).all()
    assert jeju["cloud_cover"].isna().all() and jeju["deadline_ok"].all()
    assert I.relative_humidity(np.array([20.0]), np.array([10.0]))[0] == pytest.approx(52.5, abs=0.5)


def test_dayahead_pipeline_sources_match_downloader_deadlines():
    import scripts.predict_dayahead as P

    for submission, sources in P.SOURCES.items():
        for name, module, scenario in sources:
            assert scenario in module.SCENARIOS and name in E.FORECAST_CSV
            # 10시 제출에는 10시 마감 전 공개가 확인되는 초기화분만, 17시 제출에는 17시 전 공개분까지 쓴다
            assert module.SCENARIOS[scenario]["deadline_kst_hour"] <= submission
    assert [n for n, _, _ in P.SOURCES[10]] == E.ENSEMBLES["ens_gfs10_ifs10"][0].split("_")[1:2] + ["ifs10"]


# ---------------------------------------------------------------------------
# 풍력 하루 전 (GFS 풍속)
# ---------------------------------------------------------------------------

def _write_wind_run(raw_dir, init: pd.Timestamp, posted: str, u: float = 3.0, v: float = 4.0) -> None:
    import scripts.download_nwp_gfs_wind as GW

    rows = []
    for lead in range(GW.LEAD_FIRST, GW.LEAD_LAST + 1):
        for var, val in (("u10", u / 2), ("v10", v / 2), ("u100", u + (lead - GW.LEAD_FIRST)), ("v100", v)):
            row = {"init_utc": init, "lead_h": lead, "var": var, "posted_utc": posted}
            row.update({c: val for c in GW.GRID_COLS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(raw_dir / f"{init:%Y%m%d}.csv", index=False)


def test_gfs_wind_tidy_covers_trading_day_and_flags_late_runs(tmp_path):
    import scripts.download_nwp_gfs_wind as GW

    _write_wind_run(tmp_path, pd.Timestamp("2025-06-01 18:00"), "2025-06-01 21:50:00")
    _write_wind_run(tmp_path, pd.Timestamp("2025-06-02 18:00"), "2025-06-03 03:00:00")   # 마감(01:00 UTC) 뒤
    tidy = GW.build_tidy(tmp_path).set_index("datetime")
    day1 = tidy.loc["2025-06-03 01:00":"2025-06-04 00:00"]
    assert len(day1) == 24 and day1["deadline_ok"].all()             # 거래일 D의 01~24시 전부
    assert not tidy.loc["2025-06-04 01:00":"2025-06-05 00:00", "deadline_ok"].any()
    assert day1["ws100_mean"].iloc[0] == pytest.approx(5.0)          # sqrt(3² + 4²)
    assert day1["ws10_mean"].iloc[0] == pytest.approx(2.5)
    # u = 3, v = 4 (남서풍 성분) → 불어오는 방향은 남서쪽: atan2(-3, -4) = 216.87°
    assert day1["wd100_deg"].iloc[0] == pytest.approx(216.87, abs=0.01)


def test_wind_features_use_only_forecast_values_and_blank_late_runs(tmp_path):
    import scripts.download_nwp_gfs_wind as GW
    import scripts.evaluate_wind_dayahead_gfs as WD

    _write_wind_run(tmp_path, pd.Timestamp("2025-06-01 18:00"), "2025-06-01 21:50:00")
    _write_wind_run(tmp_path, pd.Timestamp("2025-06-02 18:00"), "2025-06-03 03:00:00")
    f = WD.build_features(GW.build_tidy(tmp_path).reset_index(drop=True))
    assert WD.TARGET not in f.columns
    t = pd.Timestamp("2025-06-03 12:00")
    m = f["ws100_mean"]
    assert f.loc[t, "ws100_interval"] == pytest.approx((m[t] + m[t - pd.Timedelta(hours=1)]) / 2)
    assert f.loc[t, "ws100_p3h"] == pytest.approx(m[t + pd.Timedelta(hours=3)])
    assert f.loc[t, "ws100_cubed"] == pytest.approx(m[t] ** 3)
    assert f.loc["2025-06-04 01:00":"2025-06-05 00:00", "ws100_mean"].isna().all()   # 마감 뒤 업로드분은 결측


def test_supplement_run_is_limited_to_the_forecast_collected_late():
    assert E.SUPPLEMENT_WEATHER == "ifs17"
    late = [c.name for c in E.CONFIRM_CONFIGS if c.weather == E.SUPPLEMENT_WEATHER]
    assert late == ["im_ifs17_monthly"] and "ens_gfs17_ifs17" in E.ENSEMBLES
