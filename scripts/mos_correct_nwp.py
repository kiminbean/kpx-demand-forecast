#!/usr/bin/env python3
"""
NWP 일사량 MOS(Model Output Statistics) 보정
=============================================

JMA MSM 과거 예보(previous_day1/2)의 일사량을 관측 일사량(ASOS)에 맞게 통계 보정한다.
보정 계수는 학습 구간(2024-01-02 ~ 2024-10-31, 테스트 이전)에서만 적합하고,
테스트 구간(2024-11-07~12-31) 데이터는 적합·방법 선택에 절대 사용하지 않는다.

방법 선택은 폴드 첫 거래일 D-1 10:00까지의 자료만 사용하는 rolling-origin CV로 한다.
발전량 라벨과 ASOS 관측이 구간 종료 즉시 제공된다고 가정하며, 실제 공개 지연은 미반영이다.
기존 테스트 구간은 이미 사용됐으므로 추가 선택·평가에 사용하지 않는다.

사용법
  # 보정 방법 선택: 후보 방법별 CV FER 비교 → mos_selection_lead{lead}_island_mean.json
  python scripts/mos_correct_nwp.py --select

  # 소진된 테스트 구간의 추가 평가는 허용되지 않는다.

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
from src.evaluation import evaluate_generation  # noqa: E402
from src.features.solar_features_v2 import RAD_STATIONS, build_solar_frame, load_asos_stations, load_kpx_hourly  # noqa: E402

MOS_OUT_DIR = ROOT / "models" / "solar_v2_nwp_mos"
WEATHER_2024 = T.WEATHER_2024
METHODS = ["identity", "linear", "linear_cloud", "quantile"]
# 사전 등록: MOS 적합 구간 = 학습 구간 초반 ~ 2024-10-31 (테스트 2024-11-07 이전, 검증 후반부 제외)
FIT_START = pd.Timestamp("2024-01-02 00:00")
FIT_END = pd.Timestamp("2024-10-31 23:00")
QUANTILES = np.linspace(0.02, 0.98, 25)


# ---------------------------------------------------------------------------
# MOS 적합 · 적용
# ---------------------------------------------------------------------------

def load_observed_weather(path: Path = WEATHER_2024) -> pd.DataFrame:
    """ASOS 관측 wide 형식 (예보와 같은 열 구성)"""
    _require_safe_inputs()
    return load_asos_stations(path)


def _fit_pairs(forecast: pd.DataFrame, observed: pd.DataFrame, station: str):
    """적합 구간에서 예보-관측 일사 쌍 (낮 = ASOS 관측 존재 시간)"""
    idx = forecast.index[(forecast.index >= FIT_START) & (forecast.index <= FIT_END)]
    f = forecast.loc[idx, f"solar_radiation_{station}"]
    o = observed.reindex(idx)[f"solar_radiation_{station}"]
    ok = np.isfinite(f) & np.isfinite(o)
    return f[ok].to_numpy(dtype=float), o[ok].to_numpy(dtype=float)


def _validate_quantile_entry(entry: dict, station: str):
    """검증된 분위수 계수를 반환하며 적용할 수 없는 계수는 거부한다."""
    try:
        fq = np.asarray(entry["fcst_q"], dtype=float)
        oq = np.asarray(entry["obs_q"], dtype=float)
        obs_max = float(entry["obs_max"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{station}: 잘못된 quantile MOS 계수") from exc
    if (fq.ndim != 1 or oq.ndim != 1 or not fq.size or fq.size != oq.size
            or not np.isfinite(fq).all() or not np.isfinite(oq).all()
            or not np.isfinite(obs_max) or obs_max < 0
            or np.any(fq[1:] <= fq[:-1]) or np.any(oq[1:] < oq[:-1])):
        raise ValueError(f"{station}: 잘못된 quantile MOS 계수")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dx, dy = np.diff(fq), np.diff(oq)
        if not (np.isfinite(dx).all() and np.isfinite(dy).all()):
            raise ValueError(f"{station}: quantile MOS 구간이 유한하지 않습니다")
        slope = dy / dx
        if not np.isfinite(slope).all() or np.any((dy != 0) & (slope == 0)):
            raise ValueError(f"{station}: quantile MOS 기울기를 표현할 수 없습니다")
    return fq, oq, obs_max


def fit_mos(forecast: pd.DataFrame, observed: pd.DataFrame, method: str) -> dict:
    """적합 구간에서만 예보→관측 일사 매핑을 적합한다. 관측 없는 관측소는 identity."""
    if method not in METHODS:
        raise ValueError(f"알 수 없는 MOS 방법: {method}")
    coeffs: dict = {"method": method, "fit_window": [str(FIT_START), str(FIT_END)], "stations": {}}
    for st in RAD_STATIONS:
        f, o = _fit_pairs(forecast, observed, st)
        entry: dict = {"type": "identity", "n_pairs": int(len(f))}
        if (len(f) >= 100 and method != "identity") or method == "linear_cloud":
            if method == "linear":
                a, b = np.polyfit(f, o, 1)
                entry = {"type": "linear", "a": float(a), "b": float(b), "n_pairs": int(len(f))}
            elif method == "linear_cloud":
                idx = forecast.index[(forecast.index >= FIT_START) & (forecast.index <= FIT_END)]
                fs_ = forecast.loc[idx, f"solar_radiation_{st}"]
                os_ = observed.reindex(idx)[f"solar_radiation_{st}"]
                fc = forecast.reindex(index=idx, columns=[f"cloud_cover_{st}"])[f"cloud_cover_{st}"]
                ok = (np.isfinite(fs_) & np.isfinite(os_) & np.isfinite(fc)).to_numpy()
                n_pairs = int(ok.sum())
                entry = {"type": "identity", "n_pairs": n_pairs}
                if n_pairs >= 100:
                    A = np.column_stack([fs_.to_numpy(dtype=float)[ok], fc.to_numpy(dtype=float)[ok],
                                         np.ones(n_pairs)])
                    (a, c, b), *_ = np.linalg.lstsq(A, os_.to_numpy(dtype=float)[ok], rcond=None)
                    entry = {"type": "linear_cloud", "a": float(a), "b": float(b), "c": float(c), "n_pairs": n_pairs}
            elif method == "quantile":
                with np.errstate(over="ignore", invalid="ignore"):
                    fq = np.quantile(f, QUANTILES)
                    oq = np.quantile(o, QUANTILES)
                if not (np.isfinite(fq).all() and np.isfinite(oq).all()):
                    raise ValueError(f"{st}: quantile MOS 적합 결과가 유한하지 않습니다")
                fq_u, keep = np.unique(fq, return_index=True)
                entry = {"type": "quantile", "fcst_q": fq_u.tolist(), "obs_q": oq[keep].tolist(),
                         "obs_max": float(o.max()), "n_pairs": int(len(f))}
                _validate_quantile_entry(entry, st)
        coeffs["stations"][st] = entry
    return coeffs


def apply_mos(forecast: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
    """예보 wide 프레임의 일사 열만 보정해 반환 (입력은 수정하지 않음)"""
    out = forecast.copy()
    for st, e in coeffs["stations"].items():
        col = f"solar_radiation_{st}"
        if e["type"] not in ("identity", "linear", "linear_cloud", "quantile"):
            raise ValueError(f"{st}: 알 수 없는 MOS 유형: {e['type']}")
        if col not in out.columns:
            continue
        v = out[col].to_numpy(dtype=float)
        if np.isinf(v).any():
            raise ValueError(f"{st}: 일사 예보에 무한대 값이 있습니다")
        if e["type"] == "identity":
            continue
        if e["type"] in ("linear", "linear_cloud"):
            keys = ("a", "b", "c") if e["type"] == "linear_cloud" else ("a", "b")
            if not all(np.isfinite(e[key]) for key in keys):
                raise ValueError(f"{st}: MOS 계수가 유한하지 않습니다")
        if e["type"] == "linear":
            with np.errstate(over="ignore", invalid="ignore"):
                cv = v * e["a"] + e["b"]
        elif e["type"] == "linear_cloud":
            cloud_col = f"cloud_cover_{st}"
            cc = out.reindex(columns=[cloud_col])[cloud_col].to_numpy(dtype=float)
            invalid = np.isfinite(v) & ~np.isfinite(cc)
            if invalid.any():
                raise ValueError(f"{cloud_col}: 일사 예보가 있는 {invalid.sum()}개 행의 운량이 없거나 유한하지 않습니다")
            with np.errstate(over="ignore", invalid="ignore"):
                cv = v * e["a"] + cc * e["c"] + e["b"]
        else:  # quantile
            fq, oq, obs_max = _validate_quantile_entry(e, st)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                cv = np.interp(v, fq, oq)
            if (np.isfinite(v) & ~np.isfinite(cv)).any():
                raise ValueError(f"{st}: MOS 보정 결과가 유한하지 않습니다")
            cv = np.where(np.isnan(v), np.nan, np.clip(cv, 0.0, obs_max))
        if e["type"] in ("linear", "linear_cloud"):
            if (np.isfinite(v) & ~np.isfinite(cv)).any():
                raise ValueError(f"{st}: MOS 보정 결과가 유한하지 않습니다")
            cv = np.maximum(cv, 0.0)
        out[col] = cv
    return out


# ---------------------------------------------------------------------------
# 데이터셋 · 평가 (train_solar_v2와 동일 규칙)
# ---------------------------------------------------------------------------

def _require_safe_inputs() -> None:
    """안전한 제한 기간 입력이 보장되기 전에는 원본 접근을 막는다."""
    raise RuntimeError("안전한 입력이 보장되지 않아 MOS 선택을 차단합니다.")


def build_dataset_corrected(variant: str, weather_wide: pd.DataFrame):
    _require_safe_inputs()
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


def cv_fer(variant: str, weather_wide: pd.DataFrame, observed: pd.DataFrame, method: str) -> dict:
    """폴드별 마감까지 MOS 적합 후 동일 계수를 과거 전체와 평가에 적용한다.

    라벨과 관측은 구간 종료 즉시 사용 가능하다고 가정한다 (공개 지연 미반영).
    """
    _require_safe_inputs()
    weather_wide = weather_wide.sort_index()
    observed = observed.sort_index()
    predictions = []
    actuals = []
    folds = {}
    for fs, fe in T.CV_FOLDS:
        fs, fe = pd.Timestamp(fs), pd.Timestamp(fe)
        mos_train = T.training_mask_for_forecast(weather_wide.index, fs)
        coeffs = fit_mos(weather_wide.loc[mos_train], observed, method)
        corrected = apply_mos(weather_wide.loc[weather_wide.index <= fe], coeffs)
        frame, X, y = build_dataset_corrected(variant, corrected)
        s = T._starts(frame.index)
        tr = T.training_mask_for_forecast(frame.index, fs)
        ev = ((s >= fs) & (s < fe)).values
        pred = T.fit_predict(X, y, frame, tr, ev)
        predictions.append(pd.Series(pred, index=frame.index[ev]))
        actuals.append(y[ev])
        folds[f"{fs:%m-%d}~{fe:%m-%d}"] = evaluate_generation(y[ev], pred, T.CAPACITY_SELECTION)["fer_mean_pct"]
    cv = evaluate_generation(pd.concat(actuals), pd.concat(predictions), T.CAPACITY_SELECTION)
    return {"cv_fer_mean_pct": cv["fer_mean_pct"], "cv_mae_mw": cv["mae_mw"], "cv_folds_fer": folds}


def run_selection(lead_days: int, variant: str = "island_mean") -> dict:
    """후보 MOS 방법별 CV FER 비교 → 최적 방법 선택 (테스트 미사용)"""
    _require_safe_inputs()
    raw = T.load_nwp_weather(lead_days)
    obs = load_observed_weather()
    results = {}
    for method in METHODS:
        r = cv_fer(variant, raw, obs, method)
        results[method] = r
        print(f"[lead{lead_days}/{variant}] MOS={method:12s} CV FER {r['cv_fer_mean_pct']:.3f}% "
              f"MAE {r['cv_mae_mw']:.2f} MW")
    best = min(results, key=lambda k: results[k]["cv_fer_mean_pct"])
    out = {"lead_days": lead_days, "variant": variant, "selection_metric": "cv_fer_mean_pct",
           "methods": results, "selected": best,
           "availability_assumption": "발전량 라벨과 ASOS 관측은 구간 종료 즉시 제공; 실제 공개 지연 미반영"}
    MOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = MOS_OUT_DIR / f"mos_selection_lead{lead_days}_{variant}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lead{lead_days}] 선택된 MOS 방법: {best} → {path}")
    return out


def evaluate_test_corrected(variant: str, weather: str, method: str, out_dir: Path) -> dict:
    """소진된 테스트 구간 평가는 허용되지 않는다."""
    raise RuntimeError("소진된 테스트 구간의 추가 평가는 금지됩니다.")


def run_final_test(method: str) -> dict:
    """소진된 테스트 구간의 최종 비교는 허용되지 않는다."""
    raise RuntimeError("소진된 테스트 구간의 추가 평가는 금지됩니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true", help="사용 불가: 안전한 입력이 보장되지 않아 선택 차단")
    ap.add_argument("--evaluate-test", action="store_true", help="사용 불가: 소진된 테스트 구간 평가 금지")
    ap.add_argument("--method", choices=METHODS, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--lead", type=int, choices=[1, 2], default=2, help="select 대상 예보 lead (기본 2 = 48시간 전)")
    args = ap.parse_args()
    if args.evaluate_test:
        ap.error("소진된 테스트 구간의 추가 평가는 금지됩니다.")
    if not args.select:
        ap.error("--select를 지정하세요")
    run_selection(args.lead)


if __name__ == "__main__":
    main()
