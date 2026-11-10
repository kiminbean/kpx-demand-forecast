#!/usr/bin/env python3
"""
NWP 일사량 MOS(Model Output Statistics) 보정
=============================================

JMA MSM 과거 예보(previous_day1/2)의 일사량을 관측 일사량(ASOS)에 맞게 통계 보정한다.
보정 계수는 학습 구간(2024-01-02 ~ 2024-10-31, 테스트 이전)에서만 적합하고,
테스트 구간(2024-11-07~12-31) 데이터는 적합·방법 선택에 절대 사용하지 않는다.

방법 선택은 월별 rolling-origin CV(테스트 미사용)로 하고, 선택된 방법으로
테스트를 딱 한 번 평가한다(사전 등록된 최종 비교).

사용법
  # 1) 보정 방법 선택: 후보 방법별 CV FER 비교 → models/solar_v2_nwp_mos/mos_selection.json
  python scripts/mos_correct_nwp.py --select

  # 2) 최종 비교(사전 등록): 선택된 방법으로 테스트 1회 평가 → mos_evaluation.json
  python scripts/mos_correct_nwp.py --evaluate-test --method linear

보정 후보 방법 (관측소별, 낮·관측 있는 시간만 적합)
  identity      : 보정 없음 (기준)
  linear        : obs = a·fcst + b (clip ≥ 0)
  linear_cloud  : obs = a·fcst + b + c·운량예보
  quantile      : 경험적 분위수 매핑 (적합 구간 분위수 대응, clip [0, obs 최대])

적용 대상 열: solar_radiation_{jeju,gosan,seogwipo} (+prev_*). seongsan은 관측이 없어 보정하지 않는다.
시각 규칙·피처·평가는 scripts/train_solar_v2.py와 동일하다 (구간 끝 T 라벨, FER, C=359.4 MW).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_solar_v2 as T  # noqa: E402  (분할·평가·피처 파이프라인 재사용)
from src.evaluation import JEJU_TRANSITION_JFPF, SETTLEMENT_TIERS_2021, evaluate_generation, skill_score  # noqa: E402
from src.features.solar_features_v2 import RAD_STATIONS, build_solar_frame, load_asos_stations, load_kpx_hourly  # noqa: E402
from src.models.solar_v2 import SolarV2Model  # noqa: E402

MOS_OUT_DIR = ROOT / "models" / "solar_v2_nwp_mos"
WEATHER_2024 = T.WEATHER_2024
METHODS = ["identity", "linear", "linear_cloud", "quantile"]
# 사전 등록: MOS 적합 구간 = 학습 구간 초반 ~ 2024-10-31 (테스트 2024-11-07 이전, 검증 후반부 제외)
FIT_START = pd.Timestamp("2024-01-02 00:00")
FIT_END = pd.Timestamp("2024-10-31 23:00")
QUANTILES = np.linspace(0.02, 0.98, 25)
# 사전 등록: 최종 비교는 island_mean(운영) + stations_level(연구) 변형, day1/day2 조건 각 1회
FINAL_VARIANTS = ["island_mean", "stations_level"]
FINAL_WEATHERS = ["nwp_msm_day2", "nwp_msm_day1"]


# ---------------------------------------------------------------------------
# MOS 적합 · 적용
# ---------------------------------------------------------------------------

def load_observed_weather(path: Path = WEATHER_2024) -> pd.DataFrame:
    """ASOS 관측 wide 형식 (예보와 같은 열 구성)"""
    return load_asos_stations(path)


def _fit_pairs(forecast: pd.DataFrame, observed: pd.DataFrame, station: str):
    """적합 구간에서 예보-관측 일사 쌍 (낮 = ASOS 관측 존재 시간)"""
    idx = forecast.index[(forecast.index >= FIT_START) & (forecast.index <= FIT_END)]
    f = forecast.loc[idx, f"solar_radiation_{station}"]
    o = observed.loc[idx, f"solar_radiation_{station}"]
    ok = f.notna() & o.notna()
    return f[ok].to_numpy(dtype=float), o[ok].to_numpy(dtype=float)


def fit_mos(forecast: pd.DataFrame, observed: pd.DataFrame, method: str) -> dict:
    """적합 구간에서만 예보→관측 일사 매핑을 적합한다. 관측 없는 관측소는 identity."""
    if method not in METHODS:
        raise ValueError(f"알 수 없는 MOS 방법: {method}")
    coeffs: dict = {"method": method, "fit_window": [str(FIT_START), str(FIT_END)], "stations": {}}
    for st in RAD_STATIONS:
        f, o = _fit_pairs(forecast, observed, st)
        entry: dict = {"type": "identity", "n_pairs": int(len(f))}
        if len(f) >= 100 and method != "identity":
            if method == "linear":
                a, b = np.polyfit(f, o, 1)
                entry = {"type": "linear", "a": float(a), "b": float(b), "n_pairs": int(len(f))}
            elif method == "linear_cloud":
                idx = forecast.index[(forecast.index >= FIT_START) & (forecast.index <= FIT_END)]
                fs_ = forecast.loc[idx, f"solar_radiation_{st}"]
                os_ = observed.loc[idx, f"solar_radiation_{st}"]
                fc = forecast.loc[idx, f"cloud_cover_{st}"]
                ok = (fs_.notna() & os_.notna() & fc.notna()).to_numpy()
                A = np.column_stack([fs_.to_numpy(dtype=float)[ok], fc.to_numpy(dtype=float)[ok],
                                     np.ones(int(ok.sum()))])
                (a, c, b), *_ = np.linalg.lstsq(A, os_.to_numpy(dtype=float)[ok], rcond=None)
                entry = {"type": "linear_cloud", "a": float(a), "b": float(b), "c": float(c), "n_pairs": int(ok.sum())}
            elif method == "quantile":
                fq = np.quantile(f, QUANTILES)
                oq = np.quantile(o, QUANTILES)
                fq_u, keep = np.unique(fq, return_index=True)
                entry = {"type": "quantile", "fcst_q": fq_u.tolist(), "obs_q": oq[keep].tolist(),
                         "obs_max": float(o.max()), "n_pairs": int(len(f))}
        coeffs["stations"][st] = entry
    return coeffs


def apply_mos(forecast: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
    """예보 wide 프레임의 일사 열만 보정해 반환 (입력은 수정하지 않음)"""
    out = forecast.copy()
    for st, e in coeffs["stations"].items():
        col = f"solar_radiation_{st}"
        if col not in out.columns or e["type"] == "identity":
            continue
        v = out[col].to_numpy(dtype=float)
        if e["type"] == "linear":
            cv = np.maximum(v * e["a"] + e["b"], 0.0)
        elif e["type"] == "linear_cloud":
            cc = out[f"cloud_cover_{st}"].to_numpy(dtype=float)
            cv = np.maximum(v * e["a"] + cc * e["c"] + e["b"], 0.0)
        else:  # quantile
            cv = np.interp(v, e["fcst_q"], e["obs_q"])
            cv = np.where(np.isnan(v), np.nan, np.clip(cv, 0.0, e["obs_max"]))
        out[col] = cv
    return out


# ---------------------------------------------------------------------------
# 데이터셋 · 평가 (train_solar_v2와 동일 규칙)
# ---------------------------------------------------------------------------

def build_dataset_corrected(variant: str, weather_wide: pd.DataFrame):
    gen = load_kpx_hourly(T.KPX_RAW)
    frame = build_solar_frame(weather_wide, generation=gen)
    if variant == "island_mean":
        X = T.create_island_mean_features(frame)
    elif variant == "stations_level":
        X = T.create_solar_features_v2(frame)
        lvl = T.generation_level(frame, frame["generation_mw"], T.physics_features(frame)["p_mean"])
        X = pd.concat([X, lvl], axis=1)
    else:
        raise ValueError(f"알 수 없는 변형: {variant}")
    return frame, X, frame["generation_mw"]


def cv_fer(variant: str, weather_wide: pd.DataFrame) -> dict:
    """테스트 미사용 월별 rolling-origin CV (방법 선택용)"""
    frame, X, y = build_dataset_corrected(variant, weather_wide)
    s = T._starts(frame.index)
    oof = pd.Series(np.nan, index=frame.index)
    folds = {}
    for fs, fe in T.CV_FOLDS:
        fs, fe = pd.Timestamp(fs), pd.Timestamp(fe)
        tr = ((s >= T.TRAIN_START) & (s < fs)).values
        ev = ((s >= fs) & (s < fe)).values
        oof[ev] = T.fit_predict(X, y, frame, tr, ev)
        folds[f"{fs:%m-%d}~{fe:%m-%d}"] = evaluate_generation(y[ev], oof[ev], T.CAPACITY_SELECTION)["fer_mean_pct"]
    m = oof.notna().values
    cv = evaluate_generation(y[m], oof[m], T.CAPACITY_SELECTION)
    return {"cv_fer_mean_pct": cv["fer_mean_pct"], "cv_mae_mw": cv["mae_mw"], "cv_folds_fer": folds}


def run_selection(lead_days: int, variant: str = "island_mean") -> dict:
    """후보 MOS 방법별 CV FER 비교 → 최적 방법 선택 (테스트 미사용)"""
    raw = T.load_nwp_weather(lead_days)
    obs = load_observed_weather()
    results = {}
    for method in METHODS:
        corrected = raw if method == "identity" else apply_mos(raw, fit_mos(raw, obs, method))
        r = cv_fer(variant, corrected)
        results[method] = r
        print(f"[lead{lead_days}/{variant}] MOS={method:12s} CV FER {r['cv_fer_mean_pct']:.3f}% "
              f"MAE {r['cv_mae_mw']:.2f} MW")
    best = min(results, key=lambda k: results[k]["cv_fer_mean_pct"])
    out = {"lead_days": lead_days, "variant": variant, "selection_metric": "cv_fer_mean_pct",
           "methods": results, "selected": best}
    MOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = MOS_OUT_DIR / f"mos_selection_lead{lead_days}_{variant}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lead{lead_days}] 선택된 MOS 방법: {best} → {path}")
    return out


def evaluate_test_corrected(variant: str, weather: str, method: str, out_dir: Path) -> dict:
    """선택된 MOS 방법으로 train+val 학습 → 테스트(2024-11-07~12-31) 1회 평가"""
    lead_days = T.WEATHER_SOURCES[weather]
    raw = T.load_nwp_weather(lead_days)
    coeffs = fit_mos(raw, load_observed_weather(), method)
    corrected = apply_mos(raw, coeffs)
    frame, X, y = build_dataset_corrected(variant, corrected)
    s = T._starts(frame.index)
    trv = ((s >= T.TRAIN_START) & (s < T.TEST_START)).values & T.usable_rows(frame).values
    te = ((s >= T.TEST_START) & (s <= T.LAST_START)).values
    model = SolarV2Model().fit(X[trv], y[trv], frame["daylight_frac"][trv])
    pred = model.predict(X[te], frame["daylight_frac"])
    y_te = y[te]
    persist48 = y.shift(48, freq="h").reindex(y_te.index).values
    cap = T.CAPACITY_TEST[T.HEADLINE_CAPACITY]
    e = evaluate_generation(y_te, pred, cap, SETTLEMENT_TIERS_2021, jfpf=JEJU_TRANSITION_JFPF)
    metrics = {
        "variant": variant,
        "weather": weather,
        "mos_method": method,
        "capacity_mw": cap,
        "test_period": [str(T.TEST_START), str(T.LAST_START)],
        "n_test_hours": int(te.sum()),
        "skill_vs_persistence_48h": skill_score(y_te.values, pred, persist48),
        "jeju_2024_rule": e,
        "mos_coefficients": coeffs,
        "note": "사전 등록된 최종 비교: MOS 방법은 CV로 사전 선택, 테스트 구간은 적합·선택에 미사용",
    }
    d = model.save(out_dir)
    (d / "metrics_test.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"actual_mw": y_te, "pred_mw": pred}).to_csv(d / "predictions_test.csv", encoding="utf-8")
    print(f"[{variant}/{weather}/MOS:{method}] TEST FER {e['fer_mean_pct']:.3f}% "
          f"(C={cap}MW, {e['n_eval_hours']}h) MAE {e['mae_mw']:.2f} RMSE {e['rmse_mw']:.2f} R2 {e['r2']:.4f} → {d}")
    return metrics


def run_final_test(method: str) -> dict:
    """사전 등록된 최종 비교: 보정 전/후 FER 표를 mos_evaluation.json에 저장"""
    comparison = {"test_period": [str(T.TEST_START), str(T.LAST_START)],
                  "capacity_mw": T.CAPACITY_TEST[T.HEADLINE_CAPACITY],
                  "headline_capacity": T.HEADLINE_CAPACITY,
                  "mos_method": method, "baselines": {}, "mos": {}}
    for variant in FINAL_VARIANTS:
        for weather in FINAL_WEATHERS:
            base_dir = T.NWP_OUT_DIR / f"{variant}_{weather}"
            base_path = base_dir / "metrics_test.json"
            if base_path.exists():
                bm = json.loads(base_path.read_text(encoding="utf-8"))
                b = bm["by_capacity"][T.HEADLINE_CAPACITY]["jeju_2024_rule"]
                comparison["baselines"][f"{variant}_{weather}"] = b
            m = evaluate_test_corrected(variant, weather, method,
                                        MOS_OUT_DIR / f"{variant}_{weather}_mos{method}")
            comparison["mos"][f"{variant}_{weather}"] = m["jeju_2024_rule"]
    path = MOS_OUT_DIR / "mos_evaluation.json"
    path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"최종 비교 저장 → {path}")
    return comparison


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true", help="MOS 방법 후보별 CV 비교 → 선택 (테스트 미사용)")
    ap.add_argument("--evaluate-test", action="store_true", help="선택된 방법으로 테스트 1회 평가 (사전 등록 최종 비교)")
    ap.add_argument("--method", choices=METHODS, default=None, help="evaluate-test에 쓸 MOS 방법 (기본: 선택 결과)")
    ap.add_argument("--lead", type=int, choices=[1, 2], default=2, help="select 대상 예보 lead (기본 2 = 48시간 전)")
    args = ap.parse_args()
    if not (args.select or args.evaluate_test):
        ap.error("--select 또는 --evaluate-test 중 하나 이상을 지정하세요")
    if args.select:
        run_selection(args.lead)
    if args.evaluate_test:
        method = args.method
        if method is None:
            sel_path = MOS_OUT_DIR / f"mos_selection_lead{args.lead}_island_mean.json"
            if sel_path.exists():
                method = json.loads(sel_path.read_text(encoding="utf-8"))["selected"]
                print(f"선택 결과 재사용: {method} ({sel_path})")
            else:
                ap.error("--method을 지정하거나 먼저 --select를 실행하세요")
        run_final_test(method)


if __name__ == "__main__":
    main()
