"""
풍력 발전량 예측 KPX FER(예측오차율) 평가
==========================================

목적
----
기존 풍력 모델(Phase-5, legacy interval-START 라벨, R²/RMSE 평가)을
KPX FER 체계로 재평가한다. 태양광 문서(docs/solar_evaluation_criteria.md)와
동일한 절차를 따른다.

사전 등록 (2026-11-10, 테스트 실행 전 작성 — 이 스크립트의 docstring이 등록본)
------------------------------------------------------------------------------
1. 라벨: KPX 거래시간 h는 (h-1, h] 구간. 공개 파일(kpx_jeju_generation.csv)은
   거래시간 h를 date+(h-1)로 기록하므로 interval-START 라벨이다.
   실측 상관 검증(2026-11-10): solar_mw[T] vs 일사[T]=0.894, vs 일사[T+1]=0.948 →
   파일 라벨은 START가 확실. interval-END 라벨로 통일하기 위해
   wind_mw_END[T] = wind_mw_file[T-1h] (pandas shift(1)) 로 변환한다.
   마지막 시각(12-31 23:00, 구간 (23:00, 24:00])은 END 라벨 슬롯이 2025년이라 제외(1시간).
2. 브레이크 처리: 2024-06-01 제주 입찰제도 시행으로 공개 풍력 계열이 제주 전체의
   18~35%만 담긴다(최대 282 → 102 MW). 브레이크 후(2024-06-01 01:00~) 데이터만으로
   학습·평가한다(브레이크 전 데이터는 분모·포괄률 추정에만 사용).
   브레이크 전 학습 데이터와 브레이크 후 계열은 스케일이 달라 혼합 학습이 불가능하고,
   사후 스케일링보다 "브레이크 후만 학습"이 단순·검증 가능하다.
3. 모델 선택: 월별 rolling-origin CV(검증 월 8/9/10월, 학습은 검증월 직전까지).
   후보 4개(기본 LightGBM / Phase-5 파라미터 / Phase-5+시간가중 0.999 / +0.9995),
   선택 기준 = CV FER 평균 최소(동률 시 단순한 설정). 테스트 윈도우는 선택에 절대 미사용.
4. 최종 평가: 2024-11-07 00:00 ~ 2024-12-31 23:00(END 라벨) 을 딱 한 번 실행.
   이 윈도우는 태양광에서 이미 2회 사용됐으므로(§6·§7), 풍력에서도 재사용 금지 —
   본 스크립트의 1회 실행이 최종 기록이며, 재튜닝·재선택 후 재실행하지 않는다.
5. 분모: 아래 CAPACITY 섹션 참조. 민감도 FER은 [C_max(분모 최대→FER 최소),
   C_min(분모 최소→FER 최대)] 로 보고.

분모(실횀 설비용량) 산정 — 태양광 문서 §3 방법의 풍력 적용
----------------------------------------------------------
- 저장소에 제주 풍력의 공식 시장참여 용량 등록값은 없다(KPX 제주본부 운영실적 미보유).
- 앵커 C_pre = 282.13 MW = 2024-01-01~05-31 관측 최대. 이 시기는 공개 계열이 시장
  전체와 일치했음이 태양광에서 확인됨(포괄률 1.00~1.015, 문서 §3) — 풍력도 동일
  시기 전체 포괄으로 가정. 관측 최대이므로 진위 용량의 하한이다.
- 포괄률 r = 브레이크 후 / 브레이크 전 상위분위수 비율. 평균 비율(0.329)은
  1~5월(강풍기) vs 6~12월의 계절 편향으로 과소 추정이므로, 용량에 제한되는
  상위 분위수 비율을 주값으로 쓴다: r_main = p99 비율 = 0.3719.
- 주 분모 C_main = 282.13 × 0.3719 = 104.9 MW.
- 민감도: C_min = 282.13 × 0.3290(발전량 가중 비율) = 92.8 MW ~
  C_max = 282.13 × 0.3782(p99.5 비율) = 106.7 MW.
- 교차검증: 테스트 윈도우(11~12월) 실측 최대 101.83 MW, p99.9 = 99.9 MW →
  계열 plateau ~102-105 MW 로 C_main=104.9 MW 와 일관.

사용법
------
    .venv/bin/python scripts/evaluate_wind_fer.py            # CV 선택 + 테스트 1회
    .venv/bin/python scripts/evaluate_wind_fer.py --mode cv  # 모델 선택만 (테스트 미사용)

출력: models/wind_fer/wind_fer_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # noqa: E402

from src.evaluation.metrics import (  # noqa: E402
    SETTLEMENT_TIERS_2021,
    evaluate_generation,
    kpx_error_rate,
)
from src.features.wind_features_v2 import create_wind_features_v2  # noqa: E402

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_OUT_DIR = BASE_DIR / "models" / "wind_fer"

BREAK_START = pd.Timestamp("2024-06-01 01:00")  # END 라벨 기준 브레이크 후 첫 시각
TEST_START = pd.Timestamp("2024-11-07 00:00")
TEST_END = pd.Timestamp("2024-12-31 23:00")

# --- 분모(사전 등록, 위 docstring 참조) ---
C_PRE = 282.13          # 2024 1~5월 관측 최대 (전체 포괄 시기)
R_MAIN = 0.3719         # p99 포괄률 (post/pre 상위분위수 비율)
R_MIN = 0.3290          # 발전량 가중(평균) 포괄률 — 계절 편향으로 과소
R_MAX = 0.3782          # p99.5 포괄률
C_MAIN = round(C_PRE * R_MAIN, 1)   # 104.9 MW
C_MIN = round(C_PRE * R_MIN, 1)     # 92.8 MW
C_MAX = round(C_PRE * R_MAX, 1)     # 106.7 MW

CV_VAL_MONTHS = [8, 9, 10]

PHASE5_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 50,
    "learning_rate": 0.02,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "max_depth": 8,
    "verbose": -1,
    "seed": 42,
}
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "seed": 42,
}


def get_clean_features(df: pd.DataFrame, target_col: str = "wind_mw") -> List[str]:
    """누수 없는 클린 피처 선택 (Phase-5와 동일)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_patterns = [
        "solar_mw", "wind_mw", "total_mw",
        "solar_cf", "wind_cf",
        "output_ratio", "expected_max",
        "is_curtailed", "is_decoupling",
        "_gen_change", "_mw_lag", "_mw_ma", "_mw_std",
    ]
    feature_cols = []
    for col in numeric_cols:
        if col == target_col:
            continue
        if any(p in col for p in exclude_patterns):
            continue
        feature_cols.append(col)
    return feature_cols


def create_weather_patterns_2020_2023(weather_all: pd.DataFrame) -> pd.DataFrame:
    """2020-2023 기상만으로 월×시간 패턴 통계 (2024 테스트 데이터 미사용 → 누출 없음)."""
    past = weather_all[weather_all["datetime"] < "2024-01-01"].copy()
    past["hour"] = past["datetime"].dt.hour
    past["month"] = past["datetime"].dt.month
    stats_cols = [
        c for c in ["temperature", "humidity", "wind_speed", "wind_direction",
                    "pressure", "solar_radiation", "cloud_cover"]
        if c in past.columns
    ]
    patterns = past.groupby(["month", "hour"])[stats_cols].agg(["mean", "std", "min", "max"])
    patterns.columns = ["_".join(col).strip() for col in patterns.columns.values]
    return patterns.reset_index()


def load_end_label_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """interval-END 라벨 풍력 데이터 + 누출 없는 피처를 만든다.

    Returns:
        (df, feature_cols): df는 datetime 인덱스, wind_mw가 END 라벨 실측.
    """
    weather = pd.read_csv(DATA_DIR / "weather_5year_hourly.csv", parse_dates=["datetime"])
    gen = pd.read_csv(DATA_DIR / "kpx_jeju_generation.csv", parse_dates=["datetime"])

    # interval-END 라벨 변환: END[T] = 파일값[T-1h] (거래시간 h=(h-1,h] 를 h로 라벨링)
    gen = gen.sort_values("datetime").set_index("datetime")
    gen["wind_mw"] = gen["wind_mw"].shift(1)
    gen = gen.dropna(subset=["wind_mw"])
    df = gen[["wind_mw"]].join(weather.set_index("datetime"), how="inner")

    # 2020-2023 패턴 피처 (누출 없음)
    patterns = create_weather_patterns_2020_2023(weather)
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
    df = create_wind_features_v2(df)
    feature_cols = get_clean_features(df, "wind_mw")
    return df, feature_cols


def time_weights(n: int, decay: float) -> np.ndarray:
    """학습 시간가중 (Phase-5 방식, 합=1 스케일)."""
    w = np.array([decay ** (n - i - 1) for i in range(n)])
    return w / w.sum() * n


def fit_lgbm(X_tr: pd.DataFrame, y_tr: np.ndarray, config: str) -> lgb.Booster:
    """설정 문자열에 따라 LightGBM 학습. 내부 시간순 80/20 분할로 early stopping.

    내부 검증은 테스트 윈도우가 아닌 학습 구간 말단 20%다 (테스트 무사용).
    """
    n = len(X_tr)
    cut = int(n * 0.8)
    X_in, y_in = X_tr.iloc[:cut], y_tr[:cut]
    X_es, y_es = X_tr.iloc[cut:], y_tr[cut:]
    weight = None
    params: Dict = dict(DEFAULT_PARAMS)
    num_boost_round = 2000
    if config == "default":
        num_boost_round = 500
    else:
        params = dict(PHASE5_PARAMS)
        if config == "phase5_tw0999":
            weight = time_weights(len(X_in), 0.999)
        elif config == "phase5_tw09995":
            weight = time_weights(len(X_in), 0.9995)
        elif config != "phase5":
            raise ValueError(f"알 수 없는 설정: {config}")
    train_set = lgb.Dataset(X_in, label=y_in, weight=weight, free_raw_data=False)
    val_set = lgb.Dataset(X_es, label=y_es, reference=train_set, free_raw_data=False)
    model = lgb.train(
        params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def fer_of(y_true: np.ndarray, y_pred: np.ndarray, capacity: float) -> Tuple[float, int]:
    """평가 대상 시간(실측 >= 10% 용량)의 평균 FER(%)과 시간 수."""
    pred = np.clip(y_pred, 0.0, None)
    fer, mask = kpx_error_rate(y_true, pred, capacity)
    return float(fer.mean()) if fer.size else float("nan"), int(mask.sum())


def run_cv(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[str, pd.DataFrame]:
    """월별 rolling-origin CV (검증월 8/9/10). 테스트 윈도우 미사용."""
    post = df[df.index >= BREAK_START]
    configs = ["default", "phase5", "phase5_tw0999", "phase5_tw09995"]
    rows = []
    for month in CV_VAL_MONTHS:
        val_mask = post.index.month == month
        tr = post[~val_mask]
        va = post[val_mask]
        X_tr, y_tr = tr[feature_cols], tr["wind_mw"].values
        X_va, y_va = va[feature_cols], va["wind_mw"].values
        for config in configs:
            model = fit_lgbm(X_tr, y_tr, config)
            pred = model.predict(X_va)
            fer, n_eval = fer_of(y_va, pred, C_MAIN)
            rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
            rows.append({"val_month": month, "config": config,
                         "fer_pct": round(fer, 3), "n_eval_hours": n_eval,
                         "rmse_mw": round(rmse, 2)})
    cv_df = pd.DataFrame(rows)
    summary = cv_df.groupby("config")["fer_pct"].mean().sort_values()
    best = str(summary.index[0])
    print("\n[CV] 월별 rolling-origin (검증월 8/9/10, 분모 %.1f MW)" % C_MAIN)
    print(cv_df.to_string(index=False))
    print("\n[CV] 설정별 평균 FER:")
    print(summary.round(3).to_string())
    print(f"[CV] 선택: {best} (사전 등록 기준: CV FER 평균 최소)")
    return best, cv_df


def run_test(df: pd.DataFrame, feature_cols: List[str], config: str) -> Dict:
    """사전 등록된 최종 평가 (테스트 윈도우 1회 실행)."""
    post = df[df.index >= BREAK_START]
    tr = post[post.index < TEST_START]
    te = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    X_tr, y_tr = tr[feature_cols], tr["wind_mw"].values
    X_te, y_te = te[feature_cols], te["wind_mw"].values
    print(f"\n[최종] 설정={config}, 학습 {len(X_tr)}시간 ({tr.index.min()} ~ {tr.index.max()}), "
          f"테스트 {len(X_te)}시간 ({te.index.min()} ~ {te.index.max()})")

    model = fit_lgbm(X_tr, y_tr, config)
    pred = np.clip(model.predict(X_te), 0.0, None)

    # 48시간 전 지속성 기준 (END 라벨 실측을 48h shift)
    full = df["wind_mw"]
    pers = full.shift(48).reindex(te.index)
    ok = pers.notna().values
    pers_fer, pers_n = fer_of(y_te[ok], pers.values[ok], C_MAIN)

    metrics = evaluate_generation(y_te, pred, C_MAIN, tiers=SETTLEMENT_TIERS_2021, jfpf=0.5)

    fer_lo, _ = fer_of(y_te, pred, C_MAX)
    fer_hi, _ = fer_of(y_te, pred, C_MIN)

    print(f"[최종] FER = {metrics['fer_mean_pct']:.2f}% (C={C_MAIN} MW, "
          f"평가 {metrics['n_eval_hours']}h / 전체 {metrics['n_hours']}h)")
    print(f"[최종] 민감도: C={C_MAX} → {fer_lo:.2f}% ~ C={C_MIN} → {fer_hi:.2f}%")
    print(f"[최종] 적중률: <=6% {metrics['hit_6_pct']:.1f}% / <=8% {metrics['hit_8_pct']:.1f}%")
    print(f"[최종] MAE={metrics['mae_mw']:.2f} MW, RMSE={metrics['rmse_mw']:.2f} MW, R2={metrics['r2']:.3f}")
    print(f"[기준] 48h 지속성 FER = {pers_fer:.2f}% (평가 {pers_n}h)")

    return {
        "config": config,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "fer_sensitivity": {"fer_pct_at_c_max": round(fer_lo, 3),
                            "fer_pct_at_c_min": round(fer_hi, 3)},
        "persistence_48h_fer_pct": round(pers_fer, 3),
        "capacity": {"c_pre_mw": C_PRE, "coverage_ratio": {"p99": R_MAIN, "mean": R_MIN, "p995": R_MAX},
                     "c_main_mw": C_MAIN, "c_min_mw": C_MIN, "c_max_mw": C_MAX},
        "train_window": [str(tr.index.min()), str(tr.index.max()), int(len(tr))],
        "test_window": [str(te.index.min()), str(te.index.max()), int(len(te))],
        "label_convention": "interval-END (KPX 거래시간 h = (h-1, h])",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="풍력 KPX FER 평가 (interval-END 라벨)")
    parser.add_argument("--mode", choices=["all", "cv"], default="all",
                        help="cv=모델 선택만(테스트 미사용), all=선택+최종 평가 1회")
    args = parser.parse_args()

    print("=" * 70)
    print("풍력 KPX FER 평가 (interval-END 라벨, 브레이크 후 학습)")
    print(f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}, 분모 C={C_MAIN} MW "
          f"(민감도 {C_MIN}~{C_MAX} MW)")
    print("=" * 70)

    df, feature_cols = load_end_label_dataset()
    print(f"데이터: {len(df)}시간 ({df.index.min()} ~ {df.index.max()}), "
          f"피처 {len(feature_cols)}개")
    post = df[df.index >= BREAK_START]
    print(f"브레이크 후(>= {BREAK_START}): {len(post)}시간")

    best_config, cv_df = run_cv(df, feature_cols)

    result: Dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/evaluate_wind_fer.py",
        "selection_cv": {"val_months": CV_VAL_MONTHS, "best_config": best_config,
                         "detail": cv_df.to_dict(orient="records")},
        "break_handling": "브레이크 후(2024-06-01 01:00~) 데이터만 학습·평가; "
                          "브레이크 전 데이터는 분모·포괄률 추정에만 사용",
        "test_window_rule": "2024-11-07~12-31 은 사전 등록된 최종 평가로 딱 한 번 실행 "
                            "(선택·튜닝 미사용)",
    }

    if args.mode == "all":
        result["final"] = run_test(df, feature_cols, best_config)

    MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_OUT_DIR / "wind_fer_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
