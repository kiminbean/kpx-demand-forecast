"""
풍력 예측 편향(MBE) 보정 — 2차 라운드 (사전등록 절차)
=====================================================

배경: docs/round1_2025_holdout_benchmark.md가 2025-01~02 hold-out에서
MBE +6.11 MW 과대예측 편향을 확인하고 "post-break 학습분의 MBE를 이용한
상수/선형 보정을 2024 내 CV로 검증 후 2025-01~02에 1회 적용"을 다음 라운드
과제로 사전등록했다. 이 스크립트가 그 절차를 그대로 수행한다.

설계 (사전등록 — 스크립트 실행 전에 고정):
- 모델/설정: phase5_tw0999 고정 (재선택 금지, models/wind_fer/wind_fer_results.json).
- 검증: 2024 브레이크 후 rolling CV, 검증월 8/9/10 (evaluate_wind_fer.run_cv와
  동일 fold 정의). fold별 out-of-fold(OOF) 예측 잔차만 보정 계수 산출에 사용.
- 보정 후보 (모두 2024 OOF 잔차에서만 계수 산출):
    constant: pred + c,  c = mean(y_oof - pred_oof)
    linear:   a·pred + b, (a, b) = OLS(y_oof ~ pred_oof)
- 공정 평가(leave-one-fold-out, LOFO): 각 fold의 보정 계수는 나머지 2개 fold의
  OOF 잔차에서 산출해 해당 fold에 적용 → 자기 잔차로 자기 평가 금지.
- 채택 기준 (사전등록): LOFO CV 평균 FER이 raw 대비 ≥ 0.5 pp 개선하고
  3개 fold 중 ≥ 2개에서 개선. 두 후보 모두 통과 시 CV 평균이 더 좋은 쪽,
  동률이면 단순한 constant 채택.
- 미채택 시 hold-out 적용 없음 (1회 적용 원칙 — hold-out을 보정 선택에 재사용 금지).
- 채택 시에만 최종 모델(브레이크 후 2024 전체 학습, evaluate_holdout_2025와 동일)을
  다시 학습하고, 풀링된 3-fold OOF 잔차로 계수를 산출해 2025-01-01~02-28 예측에
  한 번 적용한다. 2025 실측은 계수 산출에 절대 사용하지 않는다.

사용법
------
    .venv/bin/python scripts/wind_bias_correction_2025.py

출력: models/holdout_2025/wind_bias_correction_metrics.json (gitignored)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))  # noqa: E402

from evaluate_holdout_2025 import (  # noqa: E402
    HOLDOUT_START,
    WIND_CONFIG,
    WIND_VALID_END,
    build_wind_dataset,
)
from evaluate_wind_fer import (  # noqa: E402
    BREAK_START,
    C_MAIN,
    C_MAX,
    C_MIN,
    CV_VAL_MONTHS,
    TEST_START,
    fit_lgbm,
)
from src.evaluation.metrics import evaluate_generation  # noqa: E402

OUT_PATH = BASE_DIR / "models" / "holdout_2025" / "wind_bias_correction_metrics.json"

# 사전등록 채택 기준
MIN_FER_GAIN_PP = 0.5
MIN_FOLDS_IMPROVED = 2


def run_cv_oof(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """브레이크 후 검증월 8/9/10 rolling CV의 OOF 예측을 반환한다.

    fold 정의는 train_wind_fer.cv_metrics / evaluate_wind_fer.run_cv와 동일:
    학습 = 브레이크 후 전체 중 검증월 제외, 검증 = 해당 월.
    2024-12-31까지로 제한한다 — build_wind_dataset은 2025 데이터를 포함하므로
    월 번호 매칭만으로는 2025 검증월(그리고 hold-out 기간)이 CV에 들어간다.
    """
    post = df[(df.index >= BREAK_START) & (df.index <= pd.Timestamp("2024-12-31 23:00"))]
    rows: List[pd.DataFrame] = []
    for month in CV_VAL_MONTHS:
        val_mask = post.index.month == month
        tr, va = post[~val_mask], post[val_mask]
        model = fit_lgbm(tr[feature_cols], tr["wind_mw"].values, WIND_CONFIG)
        pred = np.clip(model.predict(va[feature_cols]), 0.0, None)
        fold = pd.DataFrame(
            {"wind_mw": va["wind_mw"].values, "pred_raw": pred},
            index=va.index,
        )
        if fold.index.max() >= TEST_START:
            raise RuntimeError(
                f"CV 검증 인덱스가 소진된 테스트 윈도우({TEST_START})에 들어갔습니다"
            )
        rows.append(fold)
        print(f"      {month}월 fold: {len(fold)}h")
    return pd.concat(rows).sort_index()


def fit_constant(resid: np.ndarray) -> float:
    """상수 보정 계수: OOF 잔차(실측 − 예측)의 평균."""
    return float(np.mean(resid))


def fit_linear(pred_oof: np.ndarray, y_oof: np.ndarray) -> Tuple[float, float]:
    """선형 보정 계수: OLS y ~ a·pred + b (OOF 쌍에 적합)."""
    a, b = np.polyfit(pred_oof, y_oof, 1)
    return float(a), float(b)


def fer_of(y: np.ndarray, pred: np.ndarray, capacity_mw: float) -> float:
    return float(evaluate_generation(y, pred, capacity_mw)["fer_mean_pct"])


def lofo_evaluate(oof: pd.DataFrame, capacity_mw: float) -> Dict[str, Dict[str, float]]:
    """leave-one-fold-out으로 raw/constant/linear 보정 FER을 산출한다.

    각 fold(검증월)의 보정 계수는 나머지 fold들의 OOF 잔차에서만 산출한다.
    """
    months = sorted(set(oof.index.month))
    per_variant: Dict[str, List[float]] = {"raw": [], "constant": [], "linear": []}
    fold_rows: List[Dict] = []
    for m in months:
        va = oof[oof.index.month == m]
        cal = oof[oof.index.month != m]
        resid_cal = cal["wind_mw"].values - cal["pred_raw"].values
        c = fit_constant(resid_cal)
        a, b = fit_linear(cal["pred_raw"].values, cal["wind_mw"].values)
        y = va["wind_mw"].values
        preds = {
            "raw": va["pred_raw"].values,
            "constant": np.clip(va["pred_raw"].values + c, 0.0, None),
            "linear": np.clip(a * va["pred_raw"].values + b, 0.0, None),
        }
        row: Dict = {"val_month": m, "n_hours": int(len(y)), "c_const": round(c, 4),
                     "a_lin": round(a, 4), "b_lin": round(b, 4)}
        for name, p in preds.items():
            fer = fer_of(y, p, capacity_mw)
            per_variant[name].append(fer)
            row[f"fer_{name}_pct"] = round(fer, 3)
        fold_rows.append(row)
    summary = {
        name: {
            "fold_fer_pct": [round(f, 3) for f in fers],
            "mean_fer_pct": round(float(np.mean(fers)), 3),
        }
        for name, fers in per_variant.items()
    }
    return {"folds": fold_rows, "summary": summary}


def decide(lofo: Dict[str, Dict[str, float]]) -> Tuple[str, Dict]:
    """사전등록 기준에 따라 채택 여부와 후보를 결정한다."""
    raw = lofo["summary"]["raw"]["mean_fer_pct"]
    verdict: Dict = {"raw_mean_fer_pct": raw, "min_fer_gain_pp": MIN_FER_GAIN_PP,
                     "min_folds_improved": MIN_FOLDS_IMPROVED, "candidates": {}}
    qualified: List[str] = []
    for name in ("constant", "linear"):
        s = lofo["summary"][name]
        gain = raw - s["mean_fer_pct"]
        n_better = sum(
            1 for f_raw, f_c in zip(lofo["summary"]["raw"]["fold_fer_pct"], s["fold_fer_pct"])
            if f_c < f_raw
        )
        ok = gain >= MIN_FER_GAIN_PP and n_better >= MIN_FOLDS_IMPROVED
        verdict["candidates"][name] = {
            "mean_fer_pct": s["mean_fer_pct"],
            "gain_pp": round(gain, 3),
            "folds_improved": n_better,
            "qualified": ok,
        }
        if ok:
            qualified.append(name)
    if not qualified:
        verdict["adopted"] = None
        return "none", verdict
    if len(qualified) == 2:
        g_const = verdict["candidates"]["constant"]["gain_pp"]
        g_lin = verdict["candidates"]["linear"]["gain_pp"]
        best = "constant" if g_const >= g_lin else "linear"
    else:
        best = qualified[0]
    verdict["adopted"] = best
    return best, verdict


def apply_to_holdout(df: pd.DataFrame, feature_cols: List[str],
                     method: str, oof: pd.DataFrame) -> Dict:
    """채택된 보정을 2025-01~02 hold-out 예측에 1회 적용한다.

    계수는 2024 OOF 잔차에서만 산출한다 (2025 실측 미사용).
    """
    tr = df[(df.index >= BREAK_START) & (df.index <= pd.Timestamp("2024-12-31 23:00"))]
    model = fit_lgbm(tr[feature_cols], tr["wind_mw"].values, WIND_CONFIG)
    ho = df[(df.index >= HOLDOUT_START) & (df.index <= WIND_VALID_END)]
    pred_raw = np.clip(model.predict(ho[feature_cols]), 0.0, None)
    y = ho["wind_mw"].values

    resid = oof["wind_mw"].values - oof["pred_raw"].values
    c = fit_constant(resid)
    a, b = fit_linear(oof["pred_raw"].values, oof["wind_mw"].values)
    pred_corr = (np.clip(pred_raw + c, 0.0, None) if method == "constant"
                 else np.clip(a * pred_raw + b, 0.0, None))

    actual_max = float(np.nanmax(y))
    caps = {
        "registered_104.9": C_MAIN,
        "c_min_92.8": C_MIN,
        "c_max_106.7": C_MAX,
        "proxy_actual_max_janfeb": round(actual_max, 1),
    }
    by_cap: Dict[str, Dict] = {}
    for name, cap in caps.items():
        raw_m = evaluate_generation(y, pred_raw, cap)
        corr_m = evaluate_generation(y, pred_corr, cap)
        by_cap[name] = {
            "capacity_mw": cap,
            "fer_raw_pct": round(raw_m["fer_mean_pct"], 3),
            "fer_corrected_pct": round(corr_m["fer_mean_pct"], 3),
            "fer_gain_pp": round(raw_m["fer_mean_pct"] - corr_m["fer_mean_pct"], 3),
            "mbe_raw_mw": round(raw_m["mbe_mw"], 3),
            "mbe_corrected_mw": round(corr_m["mbe_mw"], 3),
            "rmse_raw_mw": round(raw_m["rmse_mw"], 2),
            "rmse_corrected_mw": round(corr_m["rmse_mw"], 2),
            "r2_raw": round(raw_m["r2"], 4),
            "r2_corrected": round(corr_m["r2"], 4),
        }
    return {
        "holdout_period": f"{HOLDOUT_START.date()} ~ {WIND_VALID_END.date()}",
        "n_holdout_hours": int(len(y)),
        "correction_method": method,
        "correction_coefficients": {
            "constant_c_mw": round(c, 4),
            "linear_a": round(a, 4),
            "linear_b_mw": round(b, 4),
            "source": "2024 post-break 3-fold OOF 잔차 풀링 (2025 실측 미사용)",
        },
        "by_capacity": by_cap,
    }


def main() -> None:
    print("[1/4] interval-END 라벨 풍력 데이터 로드")
    df, feature_cols = build_wind_dataset()

    print("[2/4] 2024 post-break rolling CV OOF 예측 (검증월 8/9/10, 설정 고정 %s)" % WIND_CONFIG)
    oof = run_cv_oof(df, feature_cols)
    print(f"      OOF 총 {len(oof)}h")

    print("[3/4] LOFO 편향 보정 검증 (분모 %.1f MW)" % C_MAIN)
    lofo = lofo_evaluate(oof, C_MAIN)
    raw = lofo["summary"]["raw"]["mean_fer_pct"]
    print(f"      raw      평균 FER {raw:.3f}%  fold={lofo['summary']['raw']['fold_fer_pct']}")
    for name in ("constant", "linear"):
        s = lofo["summary"][name]
        print(f"      {name:<9}평균 FER {s['mean_fer_pct']:.3f}%  fold={s['fold_fer_pct']}")

    method, verdict = decide(lofo)
    print(f"      채택 판정: {method}")
    for name, cand in verdict["candidates"].items():
        print(f"        {name}: 개선 {cand['gain_pp']:+.3f} pp, "
              f"fold 개선 {cand['folds_improved']}/3, "
              f"통과={cand['qualified']}")

    result: Dict = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "script": "scripts/wind_bias_correction_2025.py",
        "pre_registered_in": "docs/round1_2025_holdout_benchmark.md (다음 라운드 제안 1번)",
        "model_config": WIND_CONFIG,
        "cv_val_months": CV_VAL_MONTHS,
        "decision_criteria": {"min_fer_gain_pp": MIN_FER_GAIN_PP,
                              "min_folds_improved": MIN_FOLDS_IMPROVED},
        "lofo": lofo,
        "decision": verdict,
        "holdout_applied": False,
    }

    if method == "none":
        print("[4/4] 채택 기준 미달 → hold-out 적용 없음 (1회 적용 원칙)")
    else:
        print(f"[4/4] 채택된 보정({method})을 2025-01~02 hold-out에 1회 적용")
        result["holdout"] = apply_to_holdout(df, feature_cols, method, oof)
        result["holdout_applied"] = True
        main_cap = result["holdout"]["by_capacity"]["registered_104.9"]
        print(f"      FER {main_cap['fer_raw_pct']:.2f}% → {main_cap['fer_corrected_pct']:.2f}% "
              f"({main_cap['fer_gain_pp']:+.2f} pp), "
              f"MBE {main_cap['mbe_raw_mw']:+.2f} → {main_cap['mbe_corrected_mw']:+.2f} MW")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[완료] {OUT_PATH}")


if __name__ == "__main__":
    main()
