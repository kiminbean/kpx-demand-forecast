#!/usr/bin/env python3
"""
제주 전체 풍력 하루 전 평가 — GFS 풍속 예보 입력 (계통운영정보 라벨)
==================================================================

사전등록 (2026-11-14, 실행 전에 이 docstring으로 고정. 1회 실행, 모델 선택에 쓰지 않는다)
-------------------------------------------------------------------------------------
목적: `evaluate_wind_system_label.py`의 관측 기상 결과(2025년 11.8%, 2026-01~10 9.5%)는 상한이다.
      제출 마감 전에 공개된 풍속 예보를 넣었을 때의 하루 전 성능을 처음 잰다.

1. 라벨: data/processed/kpx_jeju_system_hourly.csv 의 wind_mw (구간 (T-1h, T] 평균).
2. 입력: NOAA GFS 0.25° D-2 18 UTC 초기화의 10 m·100 m 바람 (scripts/download_nwp_gfs_wind.py).
   제주 격자 20점. 업로드 시각이 D-1 10:00 KST 뒤인 초기화분은 쓰지 않고 그 날은 평가에서 뺀다.
3. 피처 (미리 고정, 고르지 않는다)
   - 격자별 100 m 풍속 20개, 격자 평균·최대·최소 100 m 풍속, 격자 평균 10 m 풍속, 100 m/10 m 비
   - 평균 100 m 풍속의 세제곱(출력 밀도 대용), 풍향 sin·cos
   - 라벨이 구간 평균이므로 T와 T-1h 값의 평균, 그리고 같은 예보 안의 ±1·±3시간 값과 변화량
   - 시각·계절: hour, month, dayofyear sin·cos, 예보시간
4. 모델: 관측 기상 평가와 같은 LightGBM 설정 phase5_tw0999 고정. 튜닝 없음.
5. 구간: 관측 기상 평가와 같다 — P1 학습 2024년 → 평가 2025년, P2 학습 2024~2025년 → 평가 2026-01~10.
   월별 재학습(각 월 첫날의 전날 10시까지 자료)을 주 결과로, 고정(1회 학습)을 함께 낸다.
6. 지표: KPX FER (실측 ≥ 분모의 10%). 분모는 구간 최대 시간 발전량(주), 422 MW(감도).
   기준 예측: 48시간 지속성. 관측 기상 모델(월별)과는 같은 시간에서 비교해 예보 오차 비용을 낸다.
7. 2024-11-07 ~ 12-31은 평가하지 않는다. 운영 모델·API는 바꾸지 않는다.

사용법
  python scripts/evaluate_wind_dayahead_gfs.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.evaluate_wind_system_label as W  # noqa: E402
from scripts.download_nwp_gfs_wind import TIDY_CSV  # noqa: E402
from scripts.evaluate_wind_fer import fit_lgbm  # noqa: E402
from scripts.experiment_solar_dayahead_verified import paired_fer_diff  # noqa: E402
from src.evaluation.metrics import evaluate_generation, skill_score  # noqa: E402

OUT_DIR = ROOT / "models" / "wind_dayahead_gfs"
OBS_DIR = W.OUT_DIR                      # 관측 기상 평가의 예측 (같은 시간 비교용)
NEIGHBOR_HOURS = (1, 3)
TARGET = "wind_mw"


def build_features(tidy: pd.DataFrame) -> pd.DataFrame:
    """GFS 풍속 tidy → 유효 시각(KST) 인덱스의 피처. 마감 뒤 업로드분은 결측으로 둔다."""
    t = tidy.set_index("datetime").sort_index()
    grid = [c for c in t.columns if c.startswith("ws100_") and c[6].isdigit()]
    base = ["ws100_mean", "ws100_max", "ws100_min", "ws10_mean"]
    f = t[grid + base].copy()
    f[~t["deadline_ok"].astype(bool)] = np.nan
    f = f.reindex(pd.date_range(f.index.min(), f.index.max(), freq="h"))
    m = f["ws100_mean"]
    f["shear_ratio"] = m / f["ws10_mean"].where(f["ws10_mean"] > 0.1)
    f["ws100_cubed"] = m ** 3
    wd = np.radians(t["wd100_deg"].reindex(f.index))
    f["wd_sin"], f["wd_cos"] = np.sin(wd), np.cos(wd)
    f["ws100_interval"] = (m + m.shift(1)) / 2            # 라벨 구간 (T-1h, T]의 평균에 대응
    for h in NEIGHBOR_HOURS:
        f[f"ws100_m{h}h"], f[f"ws100_p{h}h"] = m.shift(h), m.shift(-h)
        f[f"ws100_d{h}h"] = m - m.shift(h)
    f["lead_h"] = t["lead_h"].reindex(f.index)
    doy = f.index.dayofyear
    f["hour"], f["month"] = f.index.hour, f.index.month
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366)
    return f


def load_dataset() -> Tuple[pd.DataFrame, List[str]]:
    feats = build_features(pd.read_csv(TIDY_CSV, parse_dates=["datetime"]))
    label = pd.read_csv(W.SYSTEM_HOURLY, parse_dates=["datetime"], index_col="datetime")[TARGET].dropna()
    df = feats.join(label, how="inner")
    df = df[df.index >= W.TRAIN_START]
    return df, [c for c in df.columns if c != TARGET]


def walk_forward(df: pd.DataFrame, features: List[str], start: pd.Timestamp, end: pd.Timestamp,
                 monthly: bool) -> pd.Series:
    """평가 구간 예측. 예보가 없는 시간(ws100_mean 결측)은 학습에서 빼고 예측도 NaN으로 둔다."""
    starts = df.index - pd.Timedelta(hours=1)
    has_fc = df["ws100_mean"].notna().values
    origins = [d for d in pd.date_range(start, end, freq="MS") if d < end] if monthly else [start]
    pred = pd.Series(np.nan, index=df.index[(starts >= start) & (starts < end)])
    for origin, nxt in zip(origins, origins[1:] + [end]):
        deadline = origin - pd.Timedelta(days=1) + pd.Timedelta(hours=W.DEADLINE_HOUR_KST)
        tr = df[(df.index <= deadline) & has_fc]
        booster = fit_lgbm(tr[features], tr[TARGET].values, W.WIND_CONFIG)
        ev = np.asarray((starts >= origin) & (starts < nxt)) & has_fc
        pred[df.index[ev]] = np.clip(booster.predict(df.loc[ev, features]), 0, None)
    return pred


def main() -> None:
    print("=== 제주 전체 풍력 하루 전 평가 — GFS 풍속 입력 (사전등록 2026-11-14) ===")
    df, features = load_dataset()
    print(f"자료 {df.index.min()} ~ {df.index.max()}, {len(df)}시간, 피처 {len(features)}개, "
          f"예보 없는 시간 {int(df['ws100_mean'].isna().sum())}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    for name, (start, end) in W.PERIODS.items():
        frozen = walk_forward(df, features, start, end, monthly=False)
        monthly = walk_forward(df, features, start, end, monthly=True)
        obs = pd.read_csv(OBS_DIR / f"{name}_predictions.csv", index_col=0, parse_dates=True)
        y = df[TARGET].reindex(frozen.index)
        table = pd.DataFrame({"actual": y, "gfs_frozen": frozen, "gfs_monthly": monthly,
                              "obs_monthly": obs["monthly_mw"].reindex(frozen.index),
                              "persistence48": obs["persistence48_mw"].reindex(frozen.index)})
        ok = table.notna().all(axis=1)
        tb, yv = table[ok], table.loc[ok, "actual"].values
        caps = {"proxy_max_eval": float(yv.max()), "installed_422": W.INSTALLED_CAPACITY_MW}
        res = {"period": [str(start), str(end)], "n_hours": int(ok.sum()), "n_dropped": int((~ok).sum()),
               "capacities": caps, "models": {}}
        for col in ("gfs_monthly", "gfs_frozen", "obs_monthly", "persistence48"):
            res["models"][col] = {k: evaluate_generation(yv, tb[col].values, c) for k, c in caps.items()}
        cap = caps["proxy_max_eval"]
        res["skill_vs_persistence48"] = skill_score(yv, tb["gfs_monthly"].values, tb["persistence48"].values)
        res["forecast_cost_gfs_minus_obs"] = paired_fer_diff(yv, tb["gfs_monthly"].values, tb["obs_monthly"].values, tb.index, cap)
        res["monthly_minus_frozen"] = paired_fer_diff(yv, tb["gfs_monthly"].values, tb["gfs_frozen"].values, tb.index, cap)
        results[name] = res
        table.to_csv(OUT_DIR / f"{name}_predictions.csv")
        for col in ("gfs_monthly", "gfs_frozen", "obs_monthly", "persistence48"):
            e, e2 = res["models"][col]["proxy_max_eval"], res["models"][col]["installed_422"]
            print(f"[{name}] {col:14s} FER {e['fer_mean_pct']:.2f}% (C={cap:.1f}, {e['n_eval_hours']}h) | C=422: "
                  f"{e2['fer_mean_pct']:.2f}% | MAE {e['mae_mw']:.2f} MBE {e['mbe_mw']:+.2f} R2 {e['r2']:.3f}")
        c, d = res["forecast_cost_gfs_minus_obs"], res["monthly_minus_frozen"]
        print(f"[{name}] GFS − 관측: {c['diff_pp']:+.3f}%p [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | 월별 − 고정: "
              f"{d['diff_pp']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | 스킬 {res['skill_vs_persistence48']:.3f} "
              f"| 공통 {res['n_hours']}시간, 제외 {res['n_dropped']}")
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[저장] {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
