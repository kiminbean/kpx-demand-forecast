"""특허 KR101808047B1(해줌, 10-2016-0099850) 2단 구조 반영 실험.

특허 핵심 기법과 본 프로젝트의 대응:
  NN1(적정 발전량 함수) ≈ SolarV2Model을 '관측 기상(ASOS)'으로 학습 — 관측 입력 상한 성능(CV 4.541%).
  NN2(기상예보 보정 네트워크) ≈ 예보(JMA MSM day2) → 관측(ASOS) 변수별 보정기.
    - 특허는 NN2가 NN1의 1층 활성값을 목표로 학습하지만, 본 스택에서는
      관측소·변수별 선형 보정(obs = a·fcst + b)으로 구현한다(단순 MOS와 달리
      발전모델 "앞에" 놓아 관측 학습 모델과 합성하는 것이 특허 구조의 본질).
    - 기존 MOS 실험(scripts/mos_correct_nwp.py)은 예보 학습 LightGBM "뒤에서"
      일사만 사후 보정했고 네거티브(8.53→8.66%)였다. 이 실험은 다른 구조다.
  거리가중 보간/시간 보간/발전소별 미세조정은 섬 전체 집계+단일 데이터라 해당 없음.

사전 등록 프로토콜(테스트 창 2024-11-07~12-31 사용 금지 규칙 준수):
  - 선택은 월별 rolling-origin CV(scripts/train_solar_v2.py와 동일 CV_FOLDS)와
    검증 구간으로만 한다. 테스트 창은 이 실험에서 평가하지 않는다.
  - 보정기는 각 폴드 시작 이전 데이터로만 적합한다(누출 없음).
  - 베이스라인(동일 CV): island_mean day2 CV FER 8.686%, asos(상한) 4.541%.

사용: .venv/bin/python scripts/patent_two_stage.py [--variant island_mean]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_solar_v2 import (  # noqa: E402
    CAPACITY_SELECTION,
    CV_FOLDS,
    LAST_START,
    TEST_START,
    TRAIN_START,
    VAL_START,
    build_dataset,
    load_asos_stations,
    load_kpx_hourly,
    load_nwp_weather,
    usable_rows,
    WEATHER_2024,
)
from src.evaluation.metrics import evaluate_generation  # noqa: E402
from src.features.solar_features_v2 import (  # noqa: E402
    ALL_STATIONS,
    WEATHER_VARS,
    build_solar_frame,
    create_island_mean_features,
)
from src.models.solar_v2 import SolarV2Model  # noqa: E402

KPX_RAW = ROOT / "data/raw/kpx/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20241231_20251216_144253.csv"
OUT_DIR = ROOT / "models" / "solar_v2_nwp_patent"


def fit_calibrators(wx_obs: pd.DataFrame, wx_fcst: pd.DataFrame,
                    train_end: pd.Timestamp, multivariate: bool = False) -> dict:
    """학습 구간 (예보, 관측) 쌍으로 관측소·변수별 선형 보정기 적합.

    Args:
        multivariate: True면 같은 관측소의 4개 예보 변수 전부를 회귀 변수로 사용
            (특허 NN2가 다변량 기상예보를 입력으로 받는 것에 대응).

    Returns:
        {(var, station): (coef_vec, intercept, n_pairs)} — 항등은 coef=[1].
    """
    cal = {}
    for v in WEATHER_VARS:
        for st in ALL_STATIONS:
            ob = wx_obs[f"{v}_{st}"]
            if multivariate:
                fc_df = wx_fcst[[f"{vv}_{st}" for vv in WEATHER_VARS]].dropna()
                pair = fc_df.join(ob.rename("ob")).dropna()
            else:
                pair = pd.concat([wx_fcst[f"{v}_{st}"], ob], axis=1,
                                 keys=["fc", "ob"]).dropna()
            pair = pair[pair.index < train_end]
            n = len(pair)
            if multivariate:
                X = pair[wx_fcst.columns.intersection(
                    [f"{vv}_{st}" for vv in WEATHER_VARS])].values.astype(float)
            else:
                X = pair[["fc"]].values.astype(float)
            if n < 200 or np.linalg.matrix_rank(X) < X.shape[1]:
                coef = np.array([1.0] if not multivariate else [1.0] + [0.0] * (X.shape[1] - 1))
                cal[(v, st)] = (coef, 0.0, n, X.shape[1])
                continue
            A = np.hstack([X, np.ones((n, 1))])
            sol, *_ = np.linalg.lstsq(A, pair["ob"].values.astype(float), rcond=None)
            cal[(v, st)] = (sol[:-1], float(sol[-1]), n, X.shape[1])
    return cal


def apply_calibrators(wx_fcst: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """예보 와이드 프레임에 보정기를 적용해 '의사 관측' 와이드 프레임 생성."""
    out = wx_fcst.copy()
    for (v, st), (coef, b, _, k) in cal.items():
        col = f"{v}_{st}"
        if k == 1:
            mapped = out[col] * float(coef[0]) + b
        else:
            preds = wx_fcst[[f"{vv}_{st}" for vv in WEATHER_VARS]].values.astype(float)
            mapped = pd.Series(preds @ coef + b, index=out.index)
        # 보정 예측이 NaN이면 원 예보값으로 폴백 (쌍이 없는 구간 대비)
        out[col] = mapped.where(mapped.notna(), out[col])
    # 물리 범위 클립: 일사 ≥ 0 (발전량·맑은하늘 지수가 음수 일사를 만들지 않도록)
    for (v, st) in cal:
        col = f"{v}_{st}"
        if v == "solar_radiation":
            out[col] = out[col].clip(lower=0.0)
    return out


def _starts(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index - pd.Timedelta(hours=1), index=index)


def run_two_stage_cv(multivariate: bool = False) -> dict:
    gen = load_kpx_hourly(KPX_RAW)
    wx_obs = load_asos_stations(WEATHER_2024)
    wx_fcst = load_nwp_weather(2)

    # 관측 학습용 프레임/피처 (NN1 학습 입력) — 폴드 마스크만 바꿔 재사용
    frame_obs = build_solar_frame(wx_obs, generation=gen)
    X_obs = create_island_mean_features(frame_obs)
    y = frame_obs["generation_mw"]
    s = _starts(frame_obs.index)

    # 예보 기하/prev_* 구조가 필요한 평가용 프레임은 폴드별 보정기로 재구성한다
    oof = pd.Series(np.nan, index=frame_obs.index)
    folds = {}
    cal_diag = {}
    for fs, fe in CV_FOLDS:
        fs, fe = pd.Timestamp(fs), pd.Timestamp(fe)
        cal = fit_calibrators(wx_obs, wx_fcst, train_end=fs, multivariate=multivariate)
        wx_cal = apply_calibrators(wx_fcst, cal)
        frame_cal = build_solar_frame(wx_cal, generation=gen)
        X_cal = create_island_mean_features(frame_cal)

        tr = ((s >= TRAIN_START) & (s < fs)).values & usable_rows(frame_obs).values
        model = SolarV2Model().fit(X_obs[tr], y[tr], frame_obs["daylight_frac"][tr])
        ev = ((s >= fs) & (s < fe)).values
        oof[ev] = model.predict(X_cal[ev], frame_cal["daylight_frac"])
        folds[f"{fs:%m-%d}~{fe:%m-%d}"] = evaluate_generation(
            y[ev], oof[ev], CAPACITY_SELECTION)["fer_mean_pct"]
        rad_coef = {st: round(float(cal[("solar_radiation", st)][0][0]), 3) for st in ALL_STATIONS}
        cal_diag[f"{fs:%m-%d}"] = {"rad_coef1": rad_coef}

    m = oof.notna().values
    cv = evaluate_generation(y[m], oof[m], CAPACITY_SELECTION)

    # 검증 구간: 학습 < VAL_START, 평가 [VAL_START, TEST_START)
    cal = fit_calibrators(wx_obs, wx_fcst, train_end=VAL_START, multivariate=multivariate)
    wx_cal = apply_calibrators(wx_fcst, cal)
    frame_cal = build_solar_frame(wx_cal, generation=gen)
    X_cal = create_island_mean_features(frame_cal)
    tr = ((s >= TRAIN_START) & (s < VAL_START)).values & usable_rows(frame_obs).values
    model = SolarV2Model().fit(X_obs[tr], y[tr], frame_obs["daylight_frac"][tr])
    va = ((s >= VAL_START) & (s < TEST_START)).values
    val = evaluate_generation(y[va], model.predict(X_cal[va], frame_cal["daylight_frac"]),
                              CAPACITY_SELECTION)

    mapper = "mv_linear" if multivariate else "uni_linear"
    out = {"method": f"patent_two_stage_{mapper}", "variant": "island_mean",
           "weather": "nwp_msm_day2 -> asos-calibrated", "capacity_mw": CAPACITY_SELECTION,
           "cv": cv, "cv_folds_fer": folds, "calibration_diag": cal_diag, "val": val}
    print(f"[two-stage/{mapper}] CV FER {cv['fer_mean_pct']:.3f}% MAE {cv['mae_mw']:.2f} "
          f"RMSE {cv['rmse_mw']:.2f} | VAL FER {val['fer_mean_pct']:.3f}% "
          f"MAE {val['mae_mw']:.2f} RMSE {val['rmse_mw']:.2f}")
    print("   folds FER: " + " ".join(f"{k}:{v:.2f}" for k, v in folds.items()))
    return out


def run_two_stage_test(multivariate: bool = False) -> dict:
    """사전 등록된 테스트 1회 평가: 학습(train+val) < TEST_START, 평가 [TEST_START, LAST_START).

    테스트 창(2024-11-07~12-31) 재사용 금지 규칙 준수 — 선택은 CV·검증으로 완료됨.
    """
    gen = load_kpx_hourly(KPX_RAW)
    wx_obs = load_asos_stations(WEATHER_2024)
    wx_fcst = load_nwp_weather(2)

    frame_obs = build_solar_frame(wx_obs, generation=gen)
    X_obs = create_island_mean_features(frame_obs)
    y = frame_obs["generation_mw"]
    s = _starts(frame_obs.index)

    cal = fit_calibrators(wx_obs, wx_fcst, train_end=TEST_START, multivariate=multivariate)
    wx_cal = apply_calibrators(wx_fcst, cal)
    frame_cal = build_solar_frame(wx_cal, generation=gen)
    X_cal = create_island_mean_features(frame_cal)

    tr = ((s >= TRAIN_START) & (s < TEST_START)).values & usable_rows(frame_obs).values
    model = SolarV2Model().fit(X_obs[tr], y[tr], frame_obs["daylight_frac"][tr])
    te = ((s >= TEST_START) & (s < LAST_START)).values
    preds = pd.Series(
        model.predict(X_cal[te], frame_cal["daylight_frac"]), index=X_cal.index[te])
    test = evaluate_generation(y[te], preds, CAPACITY_SELECTION)

    mapper = "mv_linear" if multivariate else "uni_linear"
    out = {"method": f"patent_two_stage_{mapper}", "variant": "island_mean",
           "weather": "nwp_msm_day2 -> asos-calibrated", "capacity_mw": CAPACITY_SELECTION,
           "test_window": f"{TEST_START:%Y-%m-%d}~{LAST_START:%Y-%m-%d}",
           "test": test,
           "baseline_test_fer_pct": 8.53}
    print(f"[two-stage/{mapper}] TEST FER {test['fer_mean_pct']:.3f}% "
          f"MAE {test['mae_mw']:.2f} RMSE {test['rmse_mw']:.2f} "
          f"(베이스라인 day2 8.53%)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapper", choices=["uni", "mv"], default="uni",
                    help="uni=변수별 단변량 선형, mv=관측소별 다변량 선형(특허 NN2 대응)")
    ap.add_argument("--evaluate-test", action="store_true",
                    help="사전 등록된 테스트 창 1회 평가 (선택용 아님 — 재실행 금지)")
    ap.add_argument("--save", action="store_true", help="결과를 models/solar_v2_nwp_patent/에 저장")
    args = ap.parse_args()
    mv = args.mapper == "mv"
    if args.evaluate_test:
        out = run_two_stage_test(mv)
        suffix = f"test_{args.mapper}"
    else:
        out = run_two_stage_cv(mv)
        suffix = f"cv_{args.mapper}"
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"two_stage_{suffix}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
