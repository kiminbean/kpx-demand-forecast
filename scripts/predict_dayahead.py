#!/usr/bin/env python3
"""
태양광 하루 전 예측 — 제출 마감 전에 공개된 예보만 쓰는 운영 파이프라인
======================================================================

거래일 D의 시간별 제주 계통 태양광(MW)을 예측한다. 근거: docs/solar_dayahead_verified_2026-11-14.md

  --submission 10 (기본)  D-1 10:00 제출용. GFS D-2 18 UTC 입력 모델과 ECMWF IFS D-2 12 UTC 입력 모델의 예측 평균
                          (선택·확인 단계를 모두 통과한 구성)
  --submission 17         D-1 17:00 제출용. GFS D-1 00 UTC 입력 모델 (17시용 다중 예보는 평가하지 못해 쓰지 않는다)

절차
  1. 필요한 예보 초기화분을 내려받는다 (이미 있으면 건너뜀). 업로드 시각이 마감 뒤면 실패시킨다.
  2. 라벨(KPX 제주계통운영정보 태양광)과 과거 예보로 예보원별 SolarV2Model(island_mean)을 다시 학습한다.
     학습에는 제출 마감 시각까지 끝난 구간만 쓴다.
  3. 거래일 D의 24시간을 예측해 CSV로 쓴다.

먼저 필요한 것
  python scripts/download_kpx_jeju_system.py                 # 라벨 최신화 (이 스크립트의 --refresh-label로도 가능)
  과거 예보 tidy (scripts/download_nwp_gfs_aws.py, download_nwp_ecmwf_aws.py) — 학습용

사용법
  python scripts/predict_dayahead.py                          # 내일(KST), 10시 제출용
  python scripts/predict_dayahead.py --date 2026-11-16 --submission 17 --refresh-label
출력: models/dayahead_forecasts/forecast_{D}_s{10|17}.csv (gitignore 대상)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.download_kpx_jeju_system as K  # noqa: E402
import scripts.download_nwp_ecmwf_aws as IFS  # noqa: E402
import scripts.download_nwp_gfs_aws as GFS  # noqa: E402
import scripts.experiment_solar_dayahead_verified as E  # noqa: E402
from src.models.solar_v2 import SolarV2Model  # noqa: E402

OUT_DIR = ROOT / "models" / "dayahead_forecasts"
VARIANT = "island_mean"
# 제출 차수 → (예보원 이름, 내려받기 모듈, 시나리오)
SOURCES: Dict[int, List[tuple]] = {
    10: [("gfs10", GFS, "d1_10kst"), ("ifs10", IFS, "d1_10kst")],
    17: [("gfs17", GFS, "d1_17kst")],
}
HOURS_PER_DAY = 24


def fetch_run(module, scenario: str, day: pd.Timestamp) -> None:
    """거래일 day에 쓰는 초기화 1회분을 받고 tidy를 다시 만든다"""
    cfg = module.SCENARIOS[scenario]
    init_date = f"{day + pd.Timedelta(days=cfg['init_day_offset']):%Y%m%d}"
    is_gfs = module is GFS
    raw_dir = GFS.OUT_DIR / "raw" / ("gfs_aws" if is_gfs else "ecmwf_aws") / f"t{cfg['cycle']:02d}z"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if is_gfs:
        _, status = GFS.download_run(init_date, cfg["cycle"], cfg["lead_first"], cfg["lead_last"], str(raw_dir))
    else:
        _, status = IFS.download_run(init_date, scenario, str(raw_dir), list(IFS.MIRRORS))
    if status not in ("ok", "cached"):
        raise RuntimeError(f"{'GFS' if is_gfs else 'IFS'} {init_date} {cfg['cycle']:02d} UTC 초기화분을 받지 못했습니다: {status}")
    if status == "ok":   # 새로 받았을 때만 tidy를 다시 만든다
        prefix = "gfs_aws" if is_gfs else "ecmwf_aws"
        module.build_tidy(scenario, raw_dir).to_csv(GFS.OUT_DIR / f"{prefix}_{scenario}.csv", index=False)


def forecast_one(source: str, day: pd.Timestamp, deadline: pd.Timestamp) -> pd.Series:
    """예보원 1개로 학습·예측 (인덱스 = 거래일의 구간 끝 시각 24개)"""
    targets = pd.date_range(day + pd.Timedelta(hours=1), periods=HOURS_PER_DAY, freq="h")
    gen = E.load_generation("system")
    gen = gen[(gen.index - pd.Timedelta(hours=1) >= E.T.TRAIN_START) & (gen.index <= deadline)]
    if gen.index.max() < deadline - pd.Timedelta(days=3):
        raise RuntimeError(f"라벨이 {gen.index.max()}까지뿐입니다 — --refresh-label 또는 download_kpx_jeju_system.py 실행")
    weather = E.load_weather(source)
    day_rows = weather.reindex(targets)[[f"solar_radiation_{st}" for st in ("jeju", "gosan")]]
    if day_rows.loc[targets[5]:targets[19]].isna().any().any():   # 06~20시 예보가 비어 있으면 마감 전 공개분이 없는 것
        raise RuntimeError(f"{source}: {day:%Y-%m-%d} 예보가 없거나 마감 뒤에 올라왔습니다")
    frame, X, y = E.build_features(weather, gen.reindex(gen.index.union(targets)), VARIANT)
    train = (y.notna() & E.T.usable_rows(frame)).values
    model = SolarV2Model().fit(X[train], y[train], frame["daylight_frac"][train])
    pred = E._predict_rows(model, X, frame, np.asarray(frame.index.isin(targets)))
    return pd.Series(pred, index=targets, name=source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="거래일 D (YYYY-MM-DD, 기본: 내일)")
    ap.add_argument("--submission", type=int, choices=sorted(SOURCES), default=10, help="제출 차수 (10시 또는 17시)")
    ap.add_argument("--refresh-label", action="store_true", help="KPX 제주계통운영정보 최근 2개월을 다시 받는다")
    ap.add_argument("--no-download", action="store_true", help="예보를 새로 받지 않는다 (과거 날짜 재현용)")
    args = ap.parse_args()

    day = pd.Timestamp(args.date) if args.date else pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    deadline = day - pd.Timedelta(days=1) + pd.Timedelta(hours=args.submission)
    if args.refresh_label:
        now = pd.Timestamp.now()
        K.download(f"{now - pd.DateOffset(months=1):%Y-%m}", f"{now:%Y-%m}", K.RAW_DIR)
        K.build_hourly(K.RAW_DIR).to_csv(K.HOURLY_CSV)
    preds = []
    for source, module, scenario in SOURCES[args.submission]:
        if not args.no_download:
            fetch_run(module, scenario, day)
        preds.append(forecast_one(source, day, deadline))
    out = pd.concat(preds, axis=1)
    out.insert(0, "pred_mw", out.mean(axis=1, skipna=False))
    if out["pred_mw"].isna().any():
        raise RuntimeError("예측에 결측이 있습니다 (예보원 중 하나가 비어 있음)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"forecast_{day:%Y-%m-%d}_s{args.submission}.csv"
    out.round(3).to_csv(path, index_label="interval_end")
    print(f"거래일 {day:%Y-%m-%d} ({args.submission}시 제출용, 학습 라벨 ~{deadline}) → {path}")
    print(f"  예보원 {list(out.columns[1:])} | 일 합계 {out['pred_mw'].sum():.0f} MWh, 최대 {out['pred_mw'].max():.0f} MW "
          f"({out['pred_mw'].idxmax():%H}시)")


if __name__ == "__main__":
    main()
