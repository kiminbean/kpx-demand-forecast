#!/usr/bin/env python3
"""
태양광 V2 학습·평가
===================

물리 피처(경사면 일사) + 발전 수준 정규화 블렌드 + 야간 ESS 후처리. 평가 지표는 KPX 예측오차율.

사용법
  # 모델 선택 근거 재현: 월별 rolling-origin CV(5~10월) + 고정 검증. 테스트 구간은 쓰지 않는다.
  python scripts/train_solar_v2.py --cv

  # train+val(2024-01-02~11-07)로 학습 → 테스트(11-07~12-31) 1회 평가 → models/solar_v2/ 저장
  python scripts/train_solar_v2.py --evaluate-test

  # 섬 평균 기상 1세트만 쓰는 형태 (발전 이력 불필요)
  python scripts/train_solar_v2.py --variant island_mean --evaluate-test

변형
  stations_level : 관측소별 물리 피처 + 발전 수준(lvl_p, D-2 컷오프) 블렌드 (기본, 최고 성능)
                   예측 시 관측소별 기상과 D-2까지의 발전량 이력이 필요하다.
  island_mean    : 섬 평균 기상 물리 피처 + 운량 추세. 발전 이력은 필요 없지만 직전 1시간 관측이 필요하다.

시각 규칙: 인덱스 T는 발전 구간 (T-1h, T]의 끝 (KPX 거래시간 h = T의 시). 기존 API(구간 시작 라벨)와 1시간 다르다.
입력 정보 주의: 기상은 대상 시각의 '관측값'이다. 실제 하루 전 예측에서는 수치예보(NWP)로 바꿔야 하며,
그때 오차는 여기 수치보다 커진다. 테스트 지표는 '관측 기상을 넣었을 때의 상한'으로 읽어야 한다.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import (  # noqa: E402
    JEJU_TRANSITION_JFPF,
    SETTLEMENT_TIERS_2021,
    SETTLEMENT_TIERS_2025,
    evaluate_generation,
    skill_score,
)
from src.features.solar_features_v2 import (  # noqa: E402
    ALL_STATIONS,
    RAD_STATIONS,
    WEATHER_VARS,
    build_solar_frame,
    create_island_mean_features,
    create_solar_features_v2,
    generation_level,
    load_asos_stations,
    load_kpx_hourly,
    physics_features,
)
from src.models.solar_v2 import SolarV2Model  # noqa: E402

KPX_RAW = ROOT / "data/raw/kpx/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20241231_20251216_144253.csv"
WEATHER_2024 = ROOT / "data/raw/weather/jeju_weather_2024_all_stations.csv"
NWP_CSV = ROOT / "data/external/nwp/openmeteo_previous_runs_jma_msm.csv"  # scripts/download_nwp_openmeteo.py
OUT_DIR = ROOT / "models" / "solar_v2"
NWP_OUT_DIR = ROOT / "models" / "solar_v2_nwp"
# day2는 약 48시간 전 예보 대용. 원자료에 실제 발표·입수 시각은 없어 제출 시점 가용성은 미검증.
WEATHER_SOURCES = {"asos": None, "nwp_msm_day2": 2, "nwp_msm_day1": 1}

# 기존 solar_lgbm_baseline과 같은 분할 경계 (구간 시작 시각 기준)
TRAIN_START = pd.Timestamp("2024-01-02 00:00")
VAL_START = pd.Timestamp("2024-09-13 12:00")
TEST_START = pd.Timestamp("2024-11-07 06:00")
LAST_START = pd.Timestamp("2024-12-31 23:00")
CV_FOLDS = [("2024-05-01", "2024-06-01"), ("2024-06-01", "2024-07-01"), ("2024-07-01", "2024-08-01"),
            ("2024-08-01", "2024-09-01"), ("2024-09-01", "2024-10-01"), ("2024-10-01", str(TEST_START))]

# 설비용량 분모 (MW)
# - 선택용 대용값: 학습 기간 최대 시간 발전량 (모델 비교 전용, 상대 비교에만 의미)
# - 테스트 실효 용량: 시장참여 태양광 449.5 MW(KPX 제주본부 2024 4분기) × 공개 거래량 포괄률.
#   포괄률은 테스트 이전 달(2024-06~10)만으로 정한다: 발전량 가중 0.7996 → 359.4 MW (10월만 0.7615 → 342.3 MW)
# - 공식 시장참여 용량: 449.5 MW (분모가 커서 오차율 하한)
CAPACITY_SELECTION = 384.87
CAPACITY_TEST = {"eff_359": 359.4, "eff_oct_342": 342.3, "market_449.5": 449.5, "proxy_384.87": 384.87}
HEADLINE_CAPACITY = "eff_359"


def load_nwp_weather(lead_days: int, path: Path = NWP_CSV) -> pd.DataFrame:
    """과거 예보(tidy CSV) → ASOS와 같은 wide 형식 ({변수}_{관측소}).

    day2 결측은 그대로 둔다. day1 예보는 D-1 10:00 이후 발표됐을 수 있다.
    예보에 없는 변수(풍속·기압 등)는 결측 — 태양광 피처는 일사·기온·습도·운량만 쓴다.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 python scripts/download_nwp_openmeteo.py 실행")
    df = pd.read_csv(path, parse_dates=["datetime"])
    cols = ["temperature", "humidity", "solar_radiation", "cloud_cover"]

    def wide(lead: int) -> pd.DataFrame:
        w = df[df["lead_days"] == lead].pivot(index="datetime", columns="station", values=cols)
        w.columns = [f"{v}_{st}" for v, st in w.columns]
        return w

    w = wide(lead_days)

    for v in WEATHER_VARS:
        for st in ALL_STATIONS:
            if f"{v}_{st}" not in w:
                w[f"{v}_{st}"] = np.nan
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))


def usable_rows(frame: pd.DataFrame) -> pd.Series:
    """낮인데 모든 관측소 일사가 비어 있는 행(예보 보관분 공백)은 학습에서 뺀다"""
    rad = frame[[f"solar_radiation_{st}" for st in RAD_STATIONS]]
    return ~((frame["daylight_frac"] > 0) & rad.isna().all(axis=1))


def build_dataset(variant: str, weather: str = "asos"):
    """구간 끝 T 인덱스의 (frame, X, y). y 결측 없음."""
    if weather not in WEATHER_SOURCES:
        raise ValueError(f"알 수 없는 기상 입력: {weather}")
    gen = load_kpx_hourly(KPX_RAW)
    wx = load_asos_stations(WEATHER_2024) if weather == "asos" else load_nwp_weather(WEATHER_SOURCES[weather])
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


def _starts(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index - pd.Timedelta(hours=1), index=index)


def training_mask_for_forecast(index: pd.DatetimeIndex, first_start: pd.Timestamp) -> np.ndarray:
    """첫 거래일의 전날 10시까지 종료된 구간만 사용 (실적 공개 지연은 미반영)."""
    deadline = first_start.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
    return np.asarray((index - pd.Timedelta(hours=1) >= TRAIN_START) & (index <= deadline))


def fit_predict(X, y, frame, train_mask, eval_mask) -> np.ndarray:
    train_mask = train_mask & usable_rows(frame).values
    model = SolarV2Model().fit(X[train_mask], y[train_mask], frame["daylight_frac"][train_mask])
    return model.predict(X[eval_mask], frame["daylight_frac"])


def run_cv(variant: str, weather: str = "asos") -> dict:
    frame, X, y = build_dataset(variant, weather)
    s = _starts(frame.index)
    oof = pd.Series(np.nan, index=frame.index)
    folds = {}
    for fs, fe in CV_FOLDS:
        fs, fe = pd.Timestamp(fs), pd.Timestamp(fe)
        tr = training_mask_for_forecast(frame.index, fs)
        ev = ((s >= fs) & (s < fe)).values
        oof[ev] = fit_predict(X, y, frame, tr, ev)
        folds[f"{fs:%m-%d}~{fe:%m-%d}"] = evaluate_generation(y[ev], oof[ev], CAPACITY_SELECTION)["fer_mean_pct"]
    m = oof.notna().values
    cv = evaluate_generation(y[m], oof[m], CAPACITY_SELECTION)
    tr = training_mask_for_forecast(frame.index, VAL_START)
    va = ((s >= VAL_START) & (s < TEST_START)).values
    val = evaluate_generation(y[va], fit_predict(X, y, frame, tr, va), CAPACITY_SELECTION)
    out = {"variant": variant, "weather": weather, "capacity_mw": CAPACITY_SELECTION, "cv": cv,
           "cv_folds_fer": folds, "val": val}
    print(f"[{variant}/{weather}] CV FER {cv['fer_mean_pct']:.3f}% MAE {cv['mae_mw']:.2f} RMSE {cv['rmse_mw']:.2f} "
          f"| VAL FER {val['fer_mean_pct']:.3f}% MAE {val['mae_mw']:.2f} RMSE {val['rmse_mw']:.2f}")
    print("   folds FER: " + " ".join(f"{k}:{v:.2f}" for k, v in folds.items()))
    return out


def run_test(variant: str, out_dir: Path, weather: str = "asos") -> dict:
    """train+val로 학습 → 테스트 1회 평가 → 저장"""
    frame, X, y = build_dataset(variant, weather)
    s = _starts(frame.index)
    trv = training_mask_for_forecast(frame.index, TEST_START) & usable_rows(frame).values
    te = ((s >= TEST_START) & (s <= LAST_START)).values
    model = SolarV2Model().fit(X[trv], y[trv], frame["daylight_frac"][trv])
    pred = model.predict(X[te], frame["daylight_frac"])
    y_te = y[te]

    # 스킬 점수 기준 예측
    # - 48h 지속성(D-2 같은 시각): 모델과 같은 정보 규칙(D-2)을 지키는 기준
    # - 24h 지속성(D-1 같은 시각): D-1 10시 제출 시점에는 아직 모르는 값이라 실제로는 불가능한, 더 강한 기준
    persist48 = y.shift(48, freq="h").reindex(y_te.index).values
    persist24 = y.shift(24, freq="h").reindex(y_te.index).values
    metrics = {
        "variant": variant,
        "weather": weather,
        "note": ("관측 기상 입력(수치예보 아님) → 하루 전 예측 성능의 상한" if weather == "asos"
                 else f"JMA MSM 과거 예보 입력({weather}) → KPX 하루 전 제출 조건에 가까운 성능"),
        "test_period": [str(TEST_START), str(LAST_START)],
        "n_test_hours": int(te.sum()),
        "skill_vs_persistence_48h": skill_score(y_te.values, pred, persist48),
        "skill_vs_persistence_24h_infeasible": skill_score(y_te.values, pred, persist24),
        "by_capacity": {},
    }
    for name, cap in CAPACITY_TEST.items():
        metrics["by_capacity"][name] = {
            # 제주 2024년 실제 규칙: 개정 전 구간 × JFPF 0.5 (테스트 기간 전체가 2024-06-01 이후)
            "jeju_2024_rule": evaluate_generation(y_te, pred, cap, SETTLEMENT_TIERS_2021, jfpf=JEJU_TRANSITION_JFPF),
            # 육지 현행(2025 개정) 구간으로 본 비교값 — 제주에는 적용된 적 없음
            "mainland_2025_benchmark": evaluate_generation(y_te, pred, cap, SETTLEMENT_TIERS_2025),
        }
    d = model.save(out_dir)
    (d / "metrics_test.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"actual_mw": y_te, "pred_mw": pred}).to_csv(d / "predictions_test.csv", encoding="utf-8")
    e = metrics["by_capacity"][HEADLINE_CAPACITY]["mainland_2025_benchmark"]
    print(f"[{variant}/{weather}] TEST FER {e['fer_mean_pct']:.3f}% (C={CAPACITY_TEST[HEADLINE_CAPACITY]}MW, {e['n_eval_hours']}h) "
          f"≤4% {e['hit_4_pct']:.1f}% ≤6% {e['hit_6_pct']:.1f}% | MAE {e['mae_mw']:.2f} RMSE {e['rmse_mw']:.2f} "
          f"R2 {e['r2']:.4f} | skill48 {metrics['skill_vs_persistence_48h']:.3f} → {d}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["stations_level", "island_mean"], default="stations_level")
    ap.add_argument("--weather", choices=list(WEATHER_SOURCES), default="asos",
                    help="asos(관측) | nwp_msm_day2(48시간 전 예보, 하루 전 제출 조건) | nwp_msm_day1(24시간 전, 낙관적)")
    ap.add_argument("--cv", action="store_true", help="CV + 검증 (테스트 미사용)")
    ap.add_argument("--evaluate-test", action="store_true", help="train+val 학습 → 테스트 평가 → 저장")
    ap.add_argument("--out", type=Path, default=None,
                    help="저장 경로 (기본 models/solar_v2/<variant>, 예보 입력은 models/solar_v2_nwp/<variant>_<weather>)")
    args = ap.parse_args()
    if not (args.cv or args.evaluate_test):
        ap.error("--cv 또는 --evaluate-test 중 하나 이상을 지정하세요")
    if args.cv:
        run_cv(args.variant, args.weather)
    if args.evaluate_test:
        default = OUT_DIR / args.variant if args.weather == "asos" else NWP_OUT_DIR / f"{args.variant}_{args.weather}"
        run_test(args.variant, args.out or default, args.weather)


if __name__ == "__main__":
    main()
