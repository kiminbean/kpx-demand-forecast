"""
API 서빙 풍력 모델 재학습 (interval-END 라벨 + 2024-06 커버리지 브레이크 명시 처리)
==================================================================================

3차 라운드(docs/round3_wind_end_label_retrain.md) 프로덕션 작업.

기존 API 서빙 모델(models/wind_phase5_fair/wind_lgbm_fair.lgb)은 legacy
interval-START 라벨 + 브레이크 전 데이터 혼합으로 학습돼 있었다. 이 스크립트는
scripts/evaluate_wind_fer.py의 사전 등록된 절차(2026-11-10)를 그대로 따르되
테스트 윈도우를 전혀 사용하지 않는 프로덕션 재학습이다.

- 라벨: interval-END (KPX 거래시간 h = (h-1, h]). load_end_label_dataset 재사용.
- 브레이크 처리: 2024-06-01 01:00 이후 데이터만 학습 (스케일이 다른 브레이크 전 혼합 금지).
- 설정: phase5_tw0999 — 사전 등록 CV(검증월 8/9/10)에서 FER 평균 최소로 이미 선택된 값.
  재선택하지 않는다 (models/wind_fer/wind_fer_results.json 참조).
- 학습 데이터: 2024-06-01 01:00 ~ 2024-11-06 23:00 (테스트 윈도우 11-07~12-31 미포함).
  테스트 윈도우는 태양광 2회 + 풍력 1회(2026-11-10 사전 등록 최종 평가)로 소진됐다.
- 검증: CV 검증월 RMSE·R²를 다시 계산해 API 불확실성 폭(±1.28·RMSE)의 근거로 저장.

사용법
------
    .venv/bin/python scripts/train_wind_fer.py

출력: models/wind_phase5_fair/ (API 서빙 디렉터리, git 추적됨)
  - wind_lgbm_fair.lgb            재학습된 모델 (기존 파일 교체)
  - feature_importance_fair.csv   재학습 모델의 gain 중요도
  - wind_fer_metadata.json        재학습 메타데이터 (학습 창·CV 지표·라벨 규약)
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))  # noqa: E402

from evaluate_wind_fer import (  # noqa: E402
    BREAK_START,
    C_MAIN,
    CV_VAL_MONTHS,
    TEST_START,
    create_weather_patterns_2020_2023,
    fit_lgbm,
    load_end_label_dataset,
)

warnings.filterwarnings("ignore")

SERVE_DIR = BASE_DIR / "models" / "wind_phase5_fair"
CONFIG = "phase5_tw0999"  # 사전 등록 CV 선택 (2026-11-10) — 재선택 금지


def cv_metrics(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
    """검증월 8/9/10 CV에서 phase5_tw0999의 전체 시간 RMSE·R²를 계산한다.

    evaluate_wind_fer.run_cv가 저장한 것은 FER과 RMSE뿐이라, API 메타데이터에
    쓸 R²와 설정 고정 CV를 여기서 다시 계산한다. 테스트 윈도우는 사용하지 않는다.
    """
    post = df[df.index >= BREAK_START]
    rows: List[Dict] = []
    for month in CV_VAL_MONTHS:
        val_mask = post.index.month == month
        tr, va = post[~val_mask], post[val_mask]
        model = fit_lgbm(tr[feature_cols], tr["wind_mw"].values, CONFIG)
        pred = np.clip(model.predict(va[feature_cols]), 0.0, None)
        y = va["wind_mw"].values
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append({
            "val_month": month,
            "rmse_mw": round(float(np.sqrt(ss_res / len(y))), 2),
            "r2": round(1.0 - ss_res / ss_tot, 4),
            "n_hours": int(len(y)),
        })
    return rows


def main() -> None:
    print("[1/4] interval-END 라벨 데이터 로드 (브레이크 후 학습, 테스트 윈도우 미사용)")
    df, feature_cols = load_end_label_dataset()

    print("[2/4] CV 검증월 RMSE·R² 재계산 (설정 고정: %s)" % CONFIG)
    cv_rows = cv_metrics(df, feature_cols)
    mean_rmse = float(np.mean([r["rmse_mw"] for r in cv_rows]))
    mean_r2 = float(np.mean([r["r2"] for r in cv_rows]))
    for r in cv_rows:
        print(f"      {r['val_month']}월: RMSE={r['rmse_mw']:.2f} MW, R²={r['r2']:.4f} ({r['n_hours']}h)")
    print(f"      평균: RMSE={mean_rmse:.2f} MW, R²={mean_r2:.4f}")

    print("[3/4] 최종 학습: 브레이크 후 ~ 2024-11-06 23:00")
    post = df[df.index >= BREAK_START]
    tr = post[post.index < TEST_START]
    X_tr, y_tr = tr[feature_cols], tr["wind_mw"].values
    model = fit_lgbm(X_tr, y_tr, CONFIG)
    print(f"      학습 {len(X_tr)}시간 ({tr.index.min()} ~ {tr.index.max()}), "
          f"피처 {len(feature_cols)}개")

    print("[4/4] 서빙 디렉터리 저장: %s" % SERVE_DIR)
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SERVE_DIR / "wind_lgbm_fair.lgb"
    model.save_model(str(model_path), num_iteration=model.best_iteration)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(SERVE_DIR / "feature_importance_fair.csv", index=False)

    # 학습 때 쓴 평년값과 서빙 파일이 동일 소스(2020-2023)인지 보장하기 위해 재기록한다
    create_weather_patterns_2020_2023(
        pd.read_csv(BASE_DIR / "data" / "processed" / "weather_5year_hourly.csv",
                    parse_dates=["datetime"])
    ).to_csv(SERVE_DIR / "weather_patterns_2020_2023.csv", index=False)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/train_wind_fer.py",
        "purpose": "API 서빙 풍력 모델 재학습 (interval-END 라벨 + 2024-06 브레이크 후 학습)",
        "config": CONFIG,
        "config_selection": "사전 등록 CV FER 평균 최소 (2026-11-10, models/wind_fer/wind_fer_results.json)",
        "label_convention": "interval-END (KPX 거래시간 h = (h-1, h]); END[T] = 파일값[T-1h]",
        "break_handling": "2024-06-01 01:00 이후 데이터만 학습 (브레이크 전 스케일 혼합 금지)",
        "train_window": [str(tr.index.min()), str(tr.index.max()), int(len(tr))],
        "test_window_used": False,
        "capacity_c_main_mw": C_MAIN,
        "cv_metrics": {
            "val_months": CV_VAL_MONTHS,
            "rows": cv_rows,
            "mean_rmse_mw": round(mean_rmse, 2),
            "mean_r2": round(mean_r2, 4),
        },
        "n_features": len(feature_cols),
        "best_iteration": int(model.best_iteration),
        "api_bounds_note": "±1.28·CV 평균 RMSE (정규근사, 캘리브레이션된 구간 아님)",
    }
    meta_path = SERVE_DIR / "wind_fer_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[완료] %s" % model_path)
    print("[완료] %s" % meta_path)
    print("[완료] API 경계 상수: WIND_TEST_RMSE_MW = %.2f (CV 평균), wind_r2 = %.4f (CV 평균)"
          % (mean_rmse, mean_r2))


if __name__ == "__main__":
    main()
