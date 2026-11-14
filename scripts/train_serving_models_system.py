#!/usr/bin/env python3
"""
API 서빙 모델 학습 — KPX 제주계통운영정보 라벨 (태양광·풍력)
==========================================================

공개 거래량의 `태양광`·`풍력`은 입찰제도 등록 자원이 빠져 포괄 범위가 계단식으로 줄었다
(docs/wind_2025_coverage_source_check.md). API가 예측하는 대상을 입찰자원을 포함하는
제주계통운영정보(5분값의 시간 평균)로 바꾸고, 2024-01 ~ 2026-10 자료로 다시 학습한다.

- 태양광: SolarV2Model 기본값, 관측 기상(ASOS) 입력. 두 변형을 저장한다.
    island_mean  : 기상만 쓰는 형태 (발전량 이력이 없을 때)
    island_level : + 발전 수준 lvl_p (요청에 최근 발전량·기상 이력이 있을 때)
  근거: docs/solar_dayahead_verified_2026-11-14.md (관측 기상·고정 조건에서 수준 피처가 세 구간 모두 0.8~1.6%p 낮음)
- 풍력: Phase-5 구성(phase5_tw0999) 그대로. 근거: docs/wind_system_label_eval_2026-11-14.md
- 구성은 고르지 않는다 (이미 평가한 구성 그대로). 불확실성 폭에 쓰는 지표는 이미 한 번 실행한
  2026-01 ~ 10 walk-forward(월별 재학습, 관측 기상) 예측에서 가져온다. 새 평가는 하지 않는다.

출력 (API가 읽는 디렉터리, git 추적)
  models/solar_v2_system/{island_mean,island_level}/   모델 + metrics_test.json + predictions_test.csv
  models/wind_system/                                   wind_lgbm_fair.lgb + 평년값 + wind_serving_metadata.json

사용법
  python scripts/train_serving_models_system.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.evaluate_wind_system_label as W  # noqa: E402
import scripts.experiment_solar_dayahead_verified as E  # noqa: E402
from scripts.evaluate_wind_fer import fit_lgbm  # noqa: E402
from src.evaluation.metrics import evaluate_generation  # noqa: E402
from src.models.solar_v2 import SolarV2Model  # noqa: E402

TRAIN_START = pd.Timestamp("2024-01-02 00:00")
TRAIN_END = pd.Timestamp("2026-11-01 00:00")          # 마지막 구간 끝 (2026-10-31 24시)
SOLAR_DIR = ROOT / "models" / "solar_v2_system"
WIND_DIR = ROOT / "models" / "wind_system"
LEGACY_WIND_DIR = ROOT / "models" / "wind_phase5_fair"
BACKTEST_DIR = E.OUT_DIR / "D"                        # 2026-01 ~ 10 walk-forward 예측 (확인 단계 D)
BACKTEST = {"island_mean": "im_obs_monthly", "island_level": "il_obs_monthly"}
WIND_BACKTEST = ROOT / "models" / "wind_system_label" / "metrics.json"
PATTERNS_FILE = "weather_patterns_2020_2023.csv"
BACKTEST_NOTE = "2026-01~10 walk-forward(월별 재학습, 관측 기상 입력). 하루 전 예보 입력이 아니다"


def train_solar(variant: str) -> None:
    gen = E.load_generation("system")
    gen = gen[(gen.index - pd.Timedelta(hours=1) >= TRAIN_START) & (gen.index <= TRAIN_END)]
    frame, X, y = E.build_features(E.load_observed(), gen, variant)
    ok = E.T.usable_rows(frame).values
    model = SolarV2Model().fit(X[ok], y[ok], frame["daylight_frac"][ok])
    out = model.save(SOLAR_DIR / variant)

    back = pd.read_csv(BACKTEST_DIR / f"{BACKTEST[variant]}.csv", parse_dates=["datetime"], index_col="datetime")
    df = pd.DataFrame({"actual_mw": E.load_generation("system").reindex(back.index), "pred_mw": back["pred_mw"]}).dropna()
    cap = float(df["actual_mw"].max())
    m = evaluate_generation(df["actual_mw"], df["pred_mw"], cap)
    df.to_csv(out / "predictions_test.csv")
    (out / "metrics_test.json").write_text(json.dumps({
        "label": "KPX 제주계통운영정보 태양광 (5분값의 시간 평균, 표시 지연 10분 보정)",
        "train_period": [str(TRAIN_START), str(TRAIN_END)], "n_train_hours": int(ok.sum()),
        "serving": {"r2": m["r2"], "rmse_mw": m["rmse_mw"], "fer_mean_pct": m["fer_mean_pct"],
                    "capacity_mw": cap, "note": BACKTEST_NOTE},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[태양광 {variant}] 학습 {int(ok.sum())}시간 → {out} | 참고 지표 FER {m['fer_mean_pct']:.2f}% "
          f"RMSE {m['rmse_mw']:.1f} MW (C={cap:.0f})")


def train_wind() -> None:
    df, features = W.load_dataset()
    tr = df[df.index <= TRAIN_END]
    booster = fit_lgbm(tr[features], tr["wind_mw"].values, W.WIND_CONFIG)
    WIND_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(WIND_DIR / "wind_lgbm_fair.lgb"), num_iteration=booster.best_iteration)
    shutil.copy(LEGACY_WIND_DIR / PATTERNS_FILE, WIND_DIR / PATTERNS_FILE)   # 2020~2023 평년값은 라벨과 무관하다
    back = json.loads(WIND_BACKTEST.read_text(encoding="utf-8"))["P2_2026"]["models"]["monthly"]["proxy_max_eval"]
    (WIND_DIR / "wind_serving_metadata.json").write_text(json.dumps({
        "label": "KPX 제주계통운영정보 풍력 (5분값의 구간 (T-1h, T] 평균)", "config": W.WIND_CONFIG,
        "train_period": [str(tr.index.min()), str(tr.index.max())], "n_train_hours": int(len(tr)),
        "n_features": len(features), "r2": back["r2"], "rmse_mw": back["rmse_mw"],
        "fer_mean_pct": back["fer_mean_pct"], "capacity_mw": back["capacity_mw"], "note": BACKTEST_NOTE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy = LEGACY_WIND_DIR / "wind_lgbm_fair.lgb"
    if legacy.exists():   # API 피처 생성 코드가 그대로 맞는지: 기존 서빙 모델과 피처 이름이 같아야 한다
        import lightgbm as lgb
        same = lgb.Booster(model_file=str(legacy)).feature_name() == booster.feature_name()
        print(f"[풍력] 기존 서빙 모델과 피처 이름 일치: {same}")
        if not same:
            raise RuntimeError("풍력 피처 구성이 기존 서빙 모델과 다릅니다 — API 피처 생성 코드를 확인하세요")
    print(f"[풍력] 학습 {len(tr)}시간, {booster.best_iteration}회 → {WIND_DIR} | 참고 지표 FER "
          f"{back['fer_mean_pct']:.2f}% RMSE {back['rmse_mw']:.1f} MW")


def main() -> None:
    np.random.seed(42)
    for variant in BACKTEST:
        train_solar(variant)
    train_wind()


if __name__ == "__main__":
    main()
