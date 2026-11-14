#!/usr/bin/env python3
"""
제주 전체 풍력(계통운영정보 라벨) 재평가 — 관측 기상 입력, 상한
==============================================================

사전등록 (2026-11-14, 실행 전에 이 docstring으로 고정)
--------------------------------------------------
배경: 공개 거래량의 제주 `풍력`은 입찰제도 등록 자원이 빠져 2024-06-01과 2025-03-01에 포괄 범위가 줄었다
(docs/wind_2025_coverage_source_check.md). 지금 API 풍력 모델은 그 좁은 대상(2024-06~2025-02 약 100 MW)으로
학습했다. KPX 제주계통운영정보의 풍력은 입찰자원을 포함해 단절이 없으므로, 같은 모델 구성을 이 라벨로
다시 학습해 "제주 전체 풍력"에 대한 성능을 처음으로 잰다.

1. 라벨: data/processed/kpx_jeju_system_hourly.csv 의 wind_mw (5분 순시값의 구간 (T-1h, T] 평균, MW).
   시장 거래량과 표시 지연 0분, 2021년 상관 0.9985. 출력제어가 반영된 실적이다.
2. 피처·모델: scripts/evaluate_wind_fer.py 와 같다 — 섬 평균 ASOS 관측, 2020~2023 월×시 패턴,
   create_wind_features_v2, 설정 phase5_tw0999 고정. 튜닝·선택 없음.
3. 평가 구간 (각각 1회, 모델 선택에 쓰지 않는다)
   P1: 학습 2024-01-01 ~ 2024-12-31 → 평가 2025-01-01 ~ 2025-12-31
   P2: 학습 2024-01-01 ~ 2025-12-31 → 평가 2026-01-01 ~ 2026-10-31
   고정(구간 시작 전 1회 학습)과 월별 재학습(확장 창, 각 월 첫날의 전날 10시까지 자료)을 함께 낸다.
4. 입력은 같은 시각의 관측 기상이다. 하루 전에는 쓸 수 없으므로 결과는 하루 전 성능의 **상한**이다.
   풍력용 수치예보(풍속)는 아직 받지 않았다.
5. 지표: KPX FER (실측 ≥ 분모의 10%인 시간). 분모는 평가 구간의 최대 시간 발전량(주)과
   KPX 운영실적의 풍력 설비용량 422 MW(감도). 기준 예측은 48시간 지속성.
6. 2024-11-07 ~ 12-31 은 평가하지 않는다 (학습 자료로만 쓴다). 운영 모델·API는 바꾸지 않는다.

사용법
  python scripts/evaluate_wind_system_label.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_wind_fer import (  # noqa: E402
    create_weather_patterns_2020_2023,
    fit_lgbm,
    get_clean_features,
)
from scripts.experiment_solar_dayahead_verified import paired_fer_diff  # noqa: E402
from src.evaluation.metrics import evaluate_generation, skill_score  # noqa: E402
from src.features.wind_features_v2 import create_wind_features_v2  # noqa: E402

SYSTEM_HOURLY = ROOT / "data/processed/kpx_jeju_system_hourly.csv"
WEATHER_5Y = ROOT / "data/processed/weather_5year_hourly.csv"          # 섬 평균 2020~2024
WEATHER_YEARS = [ROOT / f"data/raw/weather/jeju_weather_{y}_all_stations.csv" for y in (2025, 2026)]
OUT_DIR = ROOT / "models" / "wind_system_label"

WIND_CONFIG = "phase5_tw0999"
TRAIN_START = pd.Timestamp("2024-01-01 01:00")
PERIODS: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {       # 평가 구간 [시작, 끝) — 구간 시작 시각 기준
    "P1_2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    "P2_2026": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-01")),
}
INSTALLED_CAPACITY_MW = 422.0     # KPX 2025년 제주지역 전력계통 운영실적 (2024·2025년 풍력 설비용량)
DEADLINE_HOUR_KST = 10
PERSISTENCE_HOURS = 48


def load_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """계통 풍력 라벨 + 누수 없는 피처 (구간 끝 시각 인덱스)"""
    weather = pd.read_csv(WEATHER_5Y, parse_dates=["datetime"])
    cols = weather.columns.tolist()
    parts = [weather]
    for path in WEATHER_YEARS:
        raw = pd.read_csv(path, parse_dates=["datetime"], low_memory=False)
        parts.append(raw.groupby("datetime")[cols[1:]].mean().reset_index()[cols])   # 4개 관측소 평균
    weather_all = pd.concat(parts).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

    label = pd.read_csv(SYSTEM_HOURLY, parse_dates=["datetime"], index_col="datetime")["wind_mw"].dropna()
    df = label.to_frame("wind_mw").join(weather_all.set_index("datetime"), how="inner")
    patterns = create_weather_patterns_2020_2023(weather_all)
    df.index.name = "datetime"
    df = df.reset_index()
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df = pd.merge(df, patterns, on=["month", "hour"], how="left", suffixes=("", "_hist"))
    for var in ("wind_speed", "temperature", "solar_radiation"):
        if var in df.columns and f"{var}_mean" in df.columns:
            df[f"{var}_anomaly"] = df[var] - df[f"{var}_mean"]
    df = create_wind_features_v2(df.set_index("datetime"))
    df = df[df.index >= TRAIN_START]
    return df, get_clean_features(df, "wind_mw")


def walk_forward(df: pd.DataFrame, features: List[str], start: pd.Timestamp, end: pd.Timestamp,
                 monthly: bool) -> pd.Series:
    """평가 구간 예측. monthly=False면 구간 시작 전에 한 번만 학습한다."""
    starts = df.index - pd.Timedelta(hours=1)
    origins = [d for d in pd.date_range(start, end, freq="MS") if d < end] if monthly else [start]
    bounds = origins[1:] + [end]
    pred = pd.Series(np.nan, index=df.index[(starts >= start) & (starts < end)])
    for origin, nxt in zip(origins, bounds):
        deadline = origin - pd.Timedelta(days=1) + pd.Timedelta(hours=DEADLINE_HOUR_KST)
        tr = df[df.index <= deadline]
        booster = fit_lgbm(tr[features], tr["wind_mw"].values, WIND_CONFIG)
        ev = np.asarray((starts >= origin) & (starts < nxt))
        pred[df.index[ev]] = np.clip(booster.predict(df.loc[ev, features]), 0, None)
    return pred


def main() -> None:
    print("=== 제주 전체 풍력(계통 라벨) 재평가 — 관측 기상 상한 (사전등록 2026-11-14) ===")
    df, features = load_dataset()
    print(f"자료 {df.index.min()} ~ {df.index.max()}, {len(df)}시간, 피처 {len(features)}개")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    for name, (start, end) in PERIODS.items():
        frozen = walk_forward(df, features, start, end, monthly=False)
        monthly = walk_forward(df, features, start, end, monthly=True)
        y = df["wind_mw"].reindex(frozen.index)
        persist = df["wind_mw"].shift(PERSISTENCE_HOURS, freq="h").reindex(frozen.index)
        ok = y.notna() & frozen.notna() & monthly.notna() & persist.notna()
        yv, idx = y[ok].values, y.index[ok]
        caps = {"proxy_max_eval": float(yv.max()), "installed_422": INSTALLED_CAPACITY_MW}
        res = {"period": [str(start), str(end)], "n_hours": int(ok.sum()), "capacities": caps, "models": {}}
        for label, p in (("frozen", frozen), ("monthly", monthly), ("persistence48", persist)):
            res["models"][label] = {k: evaluate_generation(yv, p[ok].values, c) for k, c in caps.items()}
        res["skill_vs_persistence48"] = {m: skill_score(yv, (frozen if m == "frozen" else monthly)[ok].values,
                                                        persist[ok].values) for m in ("frozen", "monthly")}
        res["monthly_minus_frozen"] = paired_fer_diff(yv, monthly[ok].values, frozen[ok].values, idx,
                                                      caps["proxy_max_eval"])
        results[name] = res
        pd.DataFrame({"actual_mw": y, "frozen_mw": frozen, "monthly_mw": monthly,
                      "persistence48_mw": persist}).to_csv(OUT_DIR / f"{name}_predictions.csv")
        for label in ("frozen", "monthly", "persistence48"):
            e, e2 = res["models"][label]["proxy_max_eval"], res["models"][label]["installed_422"]
            print(f"[{name}] {label:14s} FER {e['fer_mean_pct']:.2f}% (C={caps['proxy_max_eval']:.1f}, {e['n_eval_hours']}h) "
                  f"| C=422: {e2['fer_mean_pct']:.2f}% | MAE {e['mae_mw']:.2f} MBE {e['mbe_mw']:+.2f} R2 {e['r2']:.3f}")
        d = res["monthly_minus_frozen"]
        print(f"[{name}] 월별 − 고정: {d['diff_pp']:+.3f}%p [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] "
              f"| 스킬(48h 지속성 대비) 고정 {res['skill_vs_persistence48']['frozen']:.3f} "
              f"월별 {res['skill_vs_persistence48']['monthly']:.3f}")
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[저장] {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
