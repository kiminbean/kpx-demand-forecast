"""2025-01~02 풍력 용량·분모(C) 가정 검증 (3차 라운드, 사전등록).

2026-11-14 사전등록(실행 전 작성). 2차 라운드
(docs/round2_wind_bias_correction.md) "다음 라운드 제안 1번"을 실행한다:
2025-01~02 hold-out의 +6.11 MW 과대예측(MBE)이 분모(용량) 가정
92.8~106.7 MW의 붕괴 기인인지 정량 확인한다.

절차 (실행 전 고정):
1. 모델/설정 동결 — 재학습·재선택 없음. 1차 라운드가 저장한
   models/holdout_2025/wind_holdout_predictions.csv (phase5_tw0999,
   interval-END 라벨)의 실측/예측을 그대로 재사용한다. hold-out 결과를
   모델 선택·튜닝에 쓰지 않는 1회 재평가 원칙 유지.
2. 라벨 스케일 증거: 2024-01~2025-12 월별 공개 라벨 최대/p95/평균.
   2025-03 라벨 붕괴의 전환 시점을 일별 최대로 확인.
3. 풍속 정규화 스케일 비교: 섬 평균 풍속 1 m/s 빈별 평균 출력을
   기준기간(2024-08-01~12-31 END, 브레이크 후 동일 플릿)과
   대상기간(2025-01-01~02-28 END)에서 비교. 스케일 비율
   r = (대상기간 빈별 평균 / 기준기간 빈별 평균)을 기준기간 빈 도수로
   가중 평균. 암시 용량 = r × 104.9 MW. n≥20인 빈만 사용.
   참고로 2024-01~05(브레이크 전, 더 큰 플릿)도 같은 방식으로 산출해
   방법이 플릿 규모 차이를 잡아내는지 검증한다.
4. 판정 기준 (사전등록):
   - r ≥ 0.90 → 분모 가정 유지, 13.23% (C=104.9) 기록 유지.
   - r < 0.90 → 암시 용량 = r × 104.9 MW로 분모 갱신, FER을
     MAE/암시용량으로 재산정해 기록 갱신.
   - 편향 보정(2차 라운드 미채택)은 이 검증 결과와 무관하게 재개하지
     않는다 — 2024 CV 기반 미채택 판정은 불변.
5. 민감도: 2025-02-25~28(라벨 전환 계단 구간) 제외 FER/MBE를 참고용
   산출. 판정·모델 선택에는 사용하지 않는다.

이 스크립트는 새로운 학습을 하지 않으며 2024-11~12 test window를
사용하지 않는다.
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

from scripts.evaluate_holdout_2025 import (  # noqa: E402
    KPX_2024,
    KPX_2025,
    OUT_DIR,
    WEATHER_2025,
    WEATHER_5Y,
)
from src.features.solar_features_v2 import load_kpx_hourly  # noqa: E402

C_REGISTERED = 104.9  # 등록 용량 (1차 라운드 기록 분모)
C_MIN, C_MAX = 92.8, 106.7
REF_START, REF_END = pd.Timestamp("2024-08-01 00:00"), pd.Timestamp("2024-12-31 23:00")
PRE_BREAK_START, PRE_BREAK_END = pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-05-31 23:00")
TARGET_START, TARGET_END = pd.Timestamp("2025-01-01 00:00"), pd.Timestamp("2025-02-28 23:00")
COLLAPSE_START = pd.Timestamp("2025-02-25 00:00")  # 일별 최대 계단 전환 시작
BIN_EDGES = np.arange(0.0, 17.0, 1.0)
MIN_BIN_N = 20
R_KEEP_THRESHOLD = 0.90  # 이 이상이면 분모 가정 유지


def load_wind_generation() -> pd.Series:
    """공개 라벨 풍력 시계열 (index = interval START, 파일 원값)."""
    gen = pd.concat([
        load_kpx_hourly(KPX_2024, region="제주", fuel="풍력"),
        load_kpx_hourly(KPX_2025, region="제주도", fuel="풍력"),
    ]).sort_index()
    if gen.index.duplicated().any():
        raise ValueError("KPX 풍력 시계열에 중복 시각이 있습니다")
    return gen


def monthly_label_stats(gen: pd.Series) -> Dict:
    agg = gen.resample("MS").agg(["mean", "max"])
    agg["p95"] = gen.resample("MS").quantile(0.95)
    agg = agg.round(2)
    out: Dict[str, Dict[str, float]] = {}
    for ts, row in agg.iterrows():
        out[str(pd.Timestamp(ts).date())] = {"mean": float(row["mean"]),
                                             "p95": float(row["p95"]),
                                             "max": float(row["max"])}
    return out


def daily_max_window(gen: pd.Series, start: str, end: str) -> Dict:
    dm = gen.resample("D").max()
    return {str(k.date()): float(v) for k, v in dm[start:end].items()}


def island_wind_speed() -> pd.Series:
    """2024까지는 5년 섬평균, 2025는 관측치 섬평균 (build_wind_dataset과 동일)."""
    weather = pd.read_csv(WEATHER_5Y, parse_dates=["datetime"])
    w25 = pd.read_csv(WEATHER_2025)
    cols = weather.columns.tolist()
    am25 = w25.groupby("datetime")[cols[1:]].mean().reset_index()
    am25["datetime"] = pd.to_datetime(am25["datetime"])
    weather_all = pd.concat([weather, am25[cols]]).sort_values("datetime")
    ws = weather_all.set_index("datetime")["wind_speed"].sort_index()
    return ws.rename("wind_speed")


def scale_ratio(gen: pd.Series, ws: pd.Series,
                ref: Tuple[pd.Timestamp, pd.Timestamp],
                target: Tuple[pd.Timestamp, pd.Timestamp]) -> Dict:
    """풍속 빈별 평균 출력 비율 r (기준기간 대비 대상기간)."""

    def bin_means(window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
        s = gen[(gen.index >= window[0]) & (gen.index <= window[1])]
        # END 라벨 정합: 파일값(START t)은 구간 (t, t+1h] → 출력을 END 시각으로 이동
        s_end = s.copy(); s_end.index = s.index + pd.Timedelta(hours=1)
        j = pd.concat([s_end.rename("out"), ws.rename("spd")], axis=1).dropna()
        b = pd.cut(j["spd"], BIN_EDGES, right=False)
        g = j.groupby(b, observed=False)["out"]
        out = pd.DataFrame({"mean": g.mean(), "n": g.size()})
        out["center"] = [iv.left for iv in out.index]
        return out

    ref_b = bin_means(ref)
    tgt_b = bin_means(target)
    centers = sorted(set(ref_b["center"]) & set(tgt_b["center"]))
    ref_b = ref_b.set_index("center").loc[centers]
    tgt_b = tgt_b.set_index("center").loc[centers]
    m_ref, n_ref = ref_b["mean"].values, ref_b["n"].values
    m_tgt, n_tgt = tgt_b["mean"].values, tgt_b["n"].values
    mask = (n_ref >= MIN_BIN_N) & (n_tgt >= MIN_BIN_N)
    ratios = m_tgt[mask] / m_ref[mask]
    r = float(np.average(ratios, weights=n_ref[mask]))
    return {
        "scale_ratio_r": round(r, 4),
        "implied_capacity_mw": round(r * C_REGISTERED, 1),
        "bins_used": int(mask.sum()),
        "bins": [{"wind_speed": float(centers[i]),
                  "ref_mean_mw": round(float(m_ref[i]), 2),
                  "target_mean_mw": round(float(m_tgt[i]), 2),
                  "ratio": round(float(m_tgt[i] / m_ref[i]), 3),
                  "ref_n": int(n_ref[i]), "target_n": int(n_tgt[i])}
                 for i in np.where(mask)[0]],
    }


def holdout_sensitivity(pred_csv: Path, implied_cap: float) -> Dict:
    p = pd.read_csv(pred_csv, index_col=0, parse_dates=True)
    y, pred = p["actual_mw"].values, p["pred_mw"].values
    idx = p.index
    main = (idx >= TARGET_START) & (idx <= TARGET_END)
    keep = main & (idx < COLLAPSE_START)
    collapse = main & (idx >= COLLAPSE_START)

    def fer(a: np.ndarray, f: np.ndarray, c: float) -> float:
        return float(np.mean(np.abs(a - f)) / c * 100)

    def mbe(a: np.ndarray, f: np.ndarray) -> float:
        return float(np.mean(f - a))

    return {
        "full_janfeb_fer_pct_c104_9": round(fer(y[main], pred[main], C_REGISTERED), 2),
        "full_janfeb_mbe_mw": round(mbe(y[main], pred[main]), 2),
        "collapse_window_fer_pct_c104_9": round(fer(y[collapse], pred[collapse], C_REGISTERED), 2),
        "collapse_window_mbe_mw": round(mbe(y[collapse], pred[collapse]), 2),
        "collapse_window_hours": int(collapse.sum()),
        "pre_collapse_fer_pct_c104_9": round(fer(y[keep], pred[keep], C_REGISTERED), 2),
        "pre_collapse_mbe_mw": round(mbe(y[keep], pred[keep]), 2),
        "fer_pct_implied_capacity": round(fer(y[main], pred[main], implied_cap), 2)
        if implied_cap else None,
        "persistence_note": "재산출 없음 — 1차 라운드 기록 유지",
    }


def main() -> None:
    print("=== 2025-01~02 풍력 용량·분모 검증 (사전등록 2026-11-14, 3차 라운드) ===")
    gen = load_wind_generation()
    ws = island_wind_speed()

    monthly = monthly_label_stats(gen)
    print("[월별 라벨 통계] 최대 MW:")
    print("  " + " | ".join(f"{k[2:7]}:{v['max']:.0f}" for k, v in monthly.items()
                            if k < "2026"))
    transition = daily_max_window(gen, "2025-02-20", "2025-03-10")
    print(f"[2025-02~03 전환] 일별 최대: {transition}")

    r_post = scale_ratio(gen, ws, (REF_START, REF_END), (TARGET_START, TARGET_END))
    r_pre = scale_ratio(gen, ws, (REF_START, REF_END), (PRE_BREAK_START, PRE_BREAK_END))
    print(f"[풍속 정규화] 2025-01~02 / 2024-08~12: r={r_post['scale_ratio_r']}, "
          f"암시 용량 {r_post['implied_capacity_mw']} MW ({r_post['bins_used']} bins)")
    print(f"[방법 검증] 2024-01~05(브레이크 전 대형 플릿) / 2024-08~12: "
          f"r={r_pre['scale_ratio_r']} — 1보다 크게 나와야 정상")

    keep = r_post["scale_ratio_r"] >= R_KEEP_THRESHOLD
    implied = None if keep else r_post["implied_capacity_mw"]
    sens = holdout_sensitivity(OUT_DIR / "wind_holdout_predictions.csv", implied)
    print(f"[민감도] 전체 FER {sens['full_janfeb_fer_pct_c104_9']}% (C=104.9), "
          f"전환구간(02-25~28) FER {sens['collapse_window_fer_pct_c104_9']}% "
          f"MBE {sens['collapse_window_mbe_mw']} MW, "
          f"전환 전 FER {sens['pre_collapse_fer_pct_c104_9']}% "
          f"MBE {sens['pre_collapse_mbe_mw']} MW")
    verdict = ("분모 가정 유지 — 13.23% (C=104.9 MW) 기록 유효" if keep
               else f"분모 갱신 — 암시 용량 {implied} MW, FER {sens['fer_pct_implied_capacity']}%")
    print(f"[판정] {verdict}")

    out = {
        "preregistered": "2026-11-14 (3차 라운드, 실행 전 docstring 고정)",
        "purpose": "2025-01~02 풍력 +6.11 MW MBE가 분모 붕괴 기인인지 검증",
        "monthly_label_stats": monthly,
        "collapse_transition_daily_max": transition,
        "scale_ratio_target_2025_janfeb": r_post,
        "scale_ratio_prebreak_2024_janmay": r_pre,
        "verdict_keep_denominator": bool(keep),
        "implied_capacity_mw": implied,
        "holdout_sensitivity": sens,
        "bias_correction_status": "2차 라운드 미채택 불변 (2024 CV 기반)",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "wind_denominator_2025_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[저장] {OUT_DIR / 'wind_denominator_2025_metrics.json'}")


if __name__ == "__main__":
    main()
