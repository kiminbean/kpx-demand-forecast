"""2025 KPX hold-out 1회 재평가 (사전등록, 관측 기상 입력).

2026-11-14 사전등록. 목적:
- 태양광: 2024 전체로 학습 → 2025 hold-out 평가 (2024-11~12 테스트 윈도우 미사용)
- 풍력: 2024-06 브레이크 후 데이터로 재학습(phase5_tw0999 동결 설정) → 2025 1~2월 평가
  (2025-03부터 라벨 붕괴로 3월 이후는 참고용)

이 결과는 모델 선택·튜닝에 절대 재사용하지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_wind_fer import (  # noqa: E402
    BREAK_START,
    C_MAIN,
    C_MAX,
    C_MIN,
    PHASE5_PARAMS,
    create_weather_patterns_2020_2023,
    fit_lgbm,
    get_clean_features,
    time_weights,
)
from src.evaluation.metrics import evaluate_generation, kpx_error_rate, skill_score  # noqa: E402
from src.features.solar_features_v2 import (  # noqa: E402
    build_solar_frame,
    create_island_mean_features,
    create_solar_features_v2,
    generation_level,
    load_asos_stations,
    load_kpx_hourly,
    physics_features,
)
from src.models.solar_v2 import SolarV2Model  # noqa: E402

KPX_2024 = ROOT / "data/raw/kpx/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20241231_20251216_144253.csv"
KPX_2025 = ROOT / "data/raw/kpx/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20251231_20261114_agent.csv"
WEATHER_2024 = ROOT / "data/raw/weather/jeju_weather_2024_all_stations.csv"
WEATHER_2025 = ROOT / "data/raw/weather/jeju_weather_2025_all_stations.csv"
WEATHER_5Y = ROOT / "data/processed/weather_5year_hourly.csv"
OUT_DIR = ROOT / "models" / "holdout_2025"

HOLDOUT_START = pd.Timestamp("2025-01-01 00:00")
HOLDOUT_END = pd.Timestamp("2025-12-31 23:00")
WIND_VALID_END = pd.Timestamp("2025-02-28 23:00")  # 2025-03 라벨 붕괴 전 마지막 END 시각

# 태양광 분모: 2025 관측 최대 시간 발전량(선택용 대용값) / 2024 실효 용량 / 시장 용량
CAP_SOLAR = {
    "proxy_max_2025": 320.6,
    "eff_359_2024": 359.4,
    "market_449.5": 449.5,
}
HEADLINE_SOLAR_CAP = "proxy_max_2025"

WIND_CONFIG = "phase5_tw0999"


def _starts(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index - pd.Timedelta(hours=1), index=index)


def usable_rows(frame: pd.DataFrame, rad_stations) -> pd.Series:
    rad = frame[[f"solar_radiation_{st}" for st in rad_stations]]
    return ~((frame["daylight_frac"] > 0) & rad.isna().all(axis=1))


def build_solar_dataset(variant: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    gen = pd.concat([load_kpx_hourly(KPX_2024), load_kpx_hourly(KPX_2025)]).sort_index()
    if not gen.index.is_unique:
        raise ValueError("KPX 2024+2025 시계열에 중복 시각이 있습니다")
    wx = pd.concat([load_asos_stations(WEATHER_2024), load_asos_stations(WEATHER_2025)])
    wx = wx[~wx.index.duplicated()].sort_index().reindex(
        pd.date_range(wx.index.min(), wx.index.max(), freq="h"))
    frame = build_solar_frame(wx, generation=gen)
    if variant == "island_mean":
        X = create_island_mean_features(frame)
    elif variant == "stations_level":
        X = create_solar_features_v2(frame)
        lvl = generation_level(frame, frame["generation_mw"], physics_features(frame)["p_mean"])
        X = pd.concat([X, lvl], axis=1)
    else:
        raise ValueError(f"알 수 없는 변형: {variant}")
    return frame, X, frame["generation_mw"]


def eval_solar(variant: str) -> Dict:
    frame, X, y = build_solar_dataset(variant)
    s = _starts(frame.index)
    train_mask = ((s >= pd.Timestamp("2024-01-01 00:00")) & (s < HOLDOUT_START)).values
    eval_mask = ((s >= HOLDOUT_START) & (s <= HOLDOUT_END)).values
    train_mask &= usable_rows(frame, ["jeju", "gosan", "seogwipo"]).values

    model = SolarV2Model().fit(X[train_mask], y[train_mask], frame["daylight_frac"][train_mask])
    pred = model.predict(X[eval_mask], frame["daylight_frac"])
    y_te = y[eval_mask]

    persist48 = y.shift(48, freq="h").reindex(y_te.index).values
    cap = CAP_SOLAR[HEADLINE_SOLAR_CAP]

    monthly = {}
    for m, idx in y_te.groupby(y_te.index.month).groups.items():
        pos = y_te.index.get_indexer(idx)
        monthly[str(m)] = evaluate_generation(y_te.iloc[pos], pred[pos], cap)["fer_mean_pct"]

    by_capacity = {}
    for name, c in CAP_SOLAR.items():
        by_capacity[name] = evaluate_generation(y_te, pred, c)
    out = {
        "model": variant,
        "weather": "asos (관측, 수치예보 아님 → 하루 전 성능의 상한)",
        "train_period": "2024-01-01 ~ 2024-12-31 (interval 시작 기준, 구간 끝 라벨)",
        "holdout_period": "2025-01-01 ~ 2025-12-31",
        "n_eval_hours": int(eval_mask.sum()),
        "n_train_hours": int(train_mask.sum()),
        "headline_capacity": HEADLINE_SOLAR_CAP,
        "capacity_mw": cap,
        "by_capacity": by_capacity,
        "monthly_fer_pct": monthly,
        "persistence_48h_fer_pct": evaluate_generation(y_te, persist48, cap)["fer_mean_pct"],
        "skill_vs_persistence_48h": skill_score(y_te.values, pred, persist48),
        "actual_max_mw": float(y_te.max()),
    }
    d = OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual_mw": y_te, "pred_mw": pred}).to_csv(d / f"solar_{variant}_holdout_predictions.csv")
    e = by_capacity[HEADLINE_SOLAR_CAP]
    print(f"[태양광 {variant}] 2025 hold-out FER {e['fer_mean_pct']:.2f}% (C={cap} MW, {e['n_eval_hours']}h) "
          f"MAE {e['mae_mw']:.2f} RMSE {e['rmse_mw']:.2f} R2 {e['r2']:.4f} "
          f"| 지속성48h {out['persistence_48h_fer_pct']:.2f}% skill {out['skill_vs_persistence_48h']:.3f}")
    print("   분모별: " + " | ".join(
        f"{k}({v_cap}): {by_capacity[k]['fer_mean_pct']:.2f}%" for k, v_cap in CAP_SOLAR.items()))
    print("   월별 FER: " + " ".join(f"{k}:{v:.1f}" for k, v in monthly.items()))
    return out


def build_wind_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    weather = pd.read_csv(WEATHER_5Y, parse_dates=["datetime"])
    w25 = pd.read_csv(WEATHER_2025)
    cols = weather.columns.tolist()
    am25 = w25.groupby("datetime")[cols[1:]].mean().reset_index()
    am25["datetime"] = pd.to_datetime(am25["datetime"])
    weather_all = pd.concat([weather, am25[cols]]).sort_values("datetime").reset_index(drop=True)

    # 2024 파일은 풍력 지역 라벨이 '제주', 2025 파일부터 '제주도' (스키마 변경)
    gen = pd.concat([
        load_kpx_hourly(KPX_2024, region="제주", fuel="풍력"),
        load_kpx_hourly(KPX_2025, region="제주도", fuel="풍력"),
    ]).sort_index()
    gen = gen[~gen.index.duplicated()]
    # interval-END 라벨: END[T] = 파일값[T-1h]
    g = gen.to_frame("wind_mw")
    g["wind_mw"] = g["wind_mw"].shift(1)
    g = g.dropna(subset=["wind_mw"])

    df = g.join(weather_all.set_index("datetime"), how="inner")
    patterns = create_weather_patterns_2020_2023(weather_all)
    df.index.name = "datetime"
    df = df.reset_index()
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df = pd.merge(df, patterns, on=["month", "hour"], how="left", suffixes=("", "_hist"))
    if "wind_speed" in df.columns and "wind_speed_mean" in df.columns:
        df["wind_speed_anomaly"] = df["wind_speed"] - df["wind_speed_mean"]
    if "temperature" in df.columns and "temperature_mean" in df.columns:
        df["temperature_anomaly"] = df["temperature"] - df["temperature_mean"]
    if "solar_radiation" in df.columns and "solar_radiation_mean" in df.columns:
        df["solar_radiation_anomaly"] = df["solar_radiation"] - df["solar_radiation_mean"]
    df = df.set_index("datetime")

    from src.features.wind_features_v2 import create_wind_features_v2
    df = create_wind_features_v2(df)
    feature_cols = get_clean_features(df, "wind_mw")
    return df, feature_cols


def eval_wind() -> Dict:
    df, feature_cols = build_wind_dataset()
    tr = df[(df.index >= BREAK_START) & (df.index <= pd.Timestamp("2024-12-31 23:00"))]
    booster = fit_lgbm(tr[feature_cols], tr["wind_mw"].values, WIND_CONFIG)

    valid = df[(df.index >= HOLDOUT_START) & (df.index <= WIND_VALID_END)]
    pred = booster.predict(valid[feature_cols])
    y = valid["wind_mw"].values

    caps = {"registered_104.9": C_MAIN, "c_min_92.8": C_MIN, "c_max_106.7": C_MAX,
            "proxy_max_janfeb_2025": round(float(y.max()), 1)}
    by_capacity = {k: evaluate_generation(y, pred, c) for k, c in caps.items()}
    persist48 = df["wind_mw"].shift(48, freq="h").reindex(valid.index).values
    cap = caps["registered_104.9"]

    # 참고용: 3월 이후는 라벨 붕괴(최대 13 MW)라 무의미
    full = df[(df.index >= HOLDOUT_START) & (df.index <= HOLDOUT_END)]
    full_pred = booster.predict(full[feature_cols])
    full_fer = evaluate_generation(full["wind_mw"].values, full_pred, cap)["fer_mean_pct"]

    out = {
        "model": f"wind phase5 ({WIND_CONFIG})",
        "train_period": f"{BREAK_START} ~ 2024-12-31 23:00 (interval-END 라벨, 브레이크 후)",
        "n_train_hours": len(tr),
        "holdout_period": "2025-01-01 ~ 2025-02-28 (2025-03 라벨 붕괴 전)",
        "n_eval_hours": len(valid),
        "capacity_mw": cap,
        "by_capacity": by_capacity,
        "fer_pct": by_capacity["registered_104.9"]["fer_mean_pct"],
        "persistence_48h_fer_pct": evaluate_generation(y, persist48, cap)["fer_mean_pct"],
        "full_year_fer_pct_invalid": full_fer,
        "actual_max_mw_janfeb": float(y.max()),
        "label_collapse_note": "2025-03부터 월 최대 13 MW 이하 — 전년 대비 커버리지 붕괴, 3월 이후 평가 무의미",
    }
    d = OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual_mw": y, "pred_mw": pred}, index=valid.index).to_csv(
        d / "wind_holdout_predictions.csv")
    e = by_capacity["registered_104.9"]
    print(f"[풍력 {WIND_CONFIG}] 2025 1~2월 FER {e['fer_mean_pct']:.2f}% (C={cap} MW, {e['n_eval_hours']}h) "
          f"MAE {e['mae_mw']:.2f} RMSE {e['rmse_mw']:.2f} R2 {e['r2']:.4f} "
          f"| 지속성48h {out['persistence_48h_fer_pct']:.2f}%")
    print("   분모별: " + " | ".join(
        f"{k}({v_cap}): {by_capacity[k]['fer_mean_pct']:.2f}%" for k, v_cap in caps.items()))
    return out


def main() -> None:
    print("=== 2025 KPX hold-out 1회 재평가 (사전등록 2026-11-14) ===")
    results = {
        "solar": {v: eval_solar(v) for v in ["island_mean", "stations_level"]},
        "wind": eval_wind(),
    }
    (OUT_DIR / "holdout_2025_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[저장] {OUT_DIR / 'holdout_2025_metrics.json'}")


if __name__ == "__main__":
    main()
