#!/usr/bin/env python3
"""
NOAA GFS 0.25° 풍속 예보 내려받기 (풍력 하루 전 평가용) — AWS Open Data 원본 GRIB2
===============================================================================

`download_nwp_gfs_aws.py`(태양광용)와 같은 방식으로, 제주 격자의 10 m·100 m 바람(U, V)을 받는다.
파일마다 업로드 시각(S3 Last-Modified)을 기록해 제출 마감 전 공개 여부를 확인한다.

시나리오: D-2 18 UTC 초기화, 예보시간 f022 ~ f045 → 거래일 D의 01:00 ~ 24:00 KST (24시간 전부).
          업로드 ≈ D-1 07시 전후(KST) → D-1 10:00 마감 전.
변수: UGRD·VGRD (10 m above ground, 100 m above ground), 순간값, m/s.

출력
  data/external/nwp/raw/gfs_wind/t18z/{YYYYMMDD}.csv   초기화 1회분의 제주 격자값 (위도 33.0~33.75, 경도 126.0~127.0)
  data/external/nwp/gfs_wind_d1_10kst.csv               유효 시각(KST)별 wide: 격자별 100 m 풍속 20개,
                                                        격자 평균 10 m·100 m 풍속, 평균 벡터의 풍향, posted_utc, deadline_ok

사용법
  python scripts/download_nwp_gfs_wind.py --start 2024-01-01 --end 2026-10-31
  python scripts/download_nwp_gfs_wind.py --build-only
"""

import argparse
import email.utils
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.download_nwp_gfs_aws import (  # noqa: E402
    BOX_LATS,
    BOX_LONS,
    BUCKET_URL,
    KST_OFFSET,
    OUT_DIR,
    THREADS_PER_RUN,
    _decode_box,
    _get_message,
    _http_get,
    _parse_idx,
)

CYCLE = 18
INIT_DAY_OFFSET = -2
LEAD_FIRST, LEAD_LAST = 22, 45
DEADLINE_KST_HOUR = 10
MESSAGES: Dict[Tuple[str, str], str] = {
    ("UGRD", "10 m above ground"): "u10", ("VGRD", "10 m above ground"): "v10",
    ("UGRD", "100 m above ground"): "u100", ("VGRD", "100 m above ground"): "v100",
}
GRID_COLS = [f"g_{la:.2f}_{lo:.2f}" for la in BOX_LATS for lo in BOX_LONS]
RAW_DIR = OUT_DIR / "raw" / "gfs_wind" / f"t{CYCLE:02d}z"
TIDY_CSV = OUT_DIR / "gfs_wind_d1_10kst.csv"


def _fetch_lead(init: pd.Timestamp, lead: int) -> List[dict]:
    url = f"{BUCKET_URL}/gfs.{init:%Y%m%d}/{CYCLE:02d}/atmos/gfs.t{CYCLE:02d}z.pgrb2.0p25.f{lead:03d}"
    idx_text, _ = _http_get(url + ".idx")
    rows = []
    for var, level, trange, start, end in _parse_idx(idx_text.decode("utf-8")):
        name = MESSAGES.get((var, level))
        if name is None or end is None or "ave" in trange:
            continue
        buf, last_modified = _get_message(url, start, end)
        box, step_type, step_range = _decode_box(buf)
        if step_type != "instant" or step_range != str(lead):
            raise ValueError(f"{url} {var}: 순간값이 아님 (stepType={step_type}, stepRange={step_range})")
        posted = email.utils.parsedate_to_datetime(last_modified) if last_modified else None
        row = {"init_utc": init, "lead_h": lead, "var": name,
               "posted_utc": None if posted is None else posted.strftime("%Y-%m-%d %H:%M:%S")}
        row.update(dict(zip(GRID_COLS, box.ravel().astype(float))))
        rows.append(row)
    if {r["var"] for r in rows} != set(MESSAGES.values()):
        raise ValueError(f"{url}: 바람 메시지가 빠져 있습니다")
    return rows


def download_run(init_date: str, raw_dir: str) -> Tuple[str, str]:
    """초기화 1회분 → raw CSV. 이미 있으면 건너뛴다."""
    out = Path(raw_dir) / f"{init_date}.csv"
    if out.exists():
        return init_date, "cached"
    init = pd.Timestamp(init_date) + pd.Timedelta(hours=CYCLE)
    try:
        with ThreadPoolExecutor(THREADS_PER_RUN) as pool:
            parts = list(pool.map(lambda ld: _fetch_lead(init, ld), range(LEAD_FIRST, LEAD_LAST + 1)))
    except FileNotFoundError as e:
        return init_date, f"missing: {e}"
    except Exception as e:  # noqa: BLE001 — 이 초기화만 건너뛰고 상태에 남긴다 (다시 실행하면 빈 날짜만 받는다)
        return init_date, f"error: {type(e).__name__}: {e}"
    df = pd.DataFrame([r for part in parts for r in part]).sort_values(["lead_h", "var"])
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(out)
    return init_date, "ok"


def build_tidy(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """raw 격자 U·V → 유효 시각(KST)별 풍속 피처 원재료"""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}에 내려받은 파일이 없습니다")
    raw = pd.concat([pd.read_csv(f, parse_dates=["init_utc", "posted_utc"]) for f in files], ignore_index=True)
    key = ["init_utc", "lead_h"]
    comp = {v: raw[raw["var"] == v].set_index(key)[GRID_COLS] for v in MESSAGES.values()}
    ws100 = np.sqrt(comp["u100"] ** 2 + comp["v100"] ** 2)
    ws10 = np.sqrt(comp["u10"] ** 2 + comp["v10"] ** 2)
    out = ws100.rename(columns=lambda c: c.replace("g_", "ws100_"))
    out["ws100_mean"], out["ws100_max"], out["ws100_min"] = ws100.mean(axis=1), ws100.max(axis=1), ws100.min(axis=1)
    out["ws10_mean"] = ws10.mean(axis=1)
    u, v = comp["u100"].mean(axis=1), comp["v100"].mean(axis=1)
    out["wd100_deg"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0     # 바람이 불어오는 방향 (기상 관례)
    out["posted_utc"] = raw.groupby(key)["posted_utc"].max()
    out = out.reset_index()
    out["datetime"] = out["init_utc"] + pd.to_timedelta(out["lead_h"], unit="h") + KST_OFFSET
    trade_day = (out["datetime"] - pd.Timedelta(hours=1)).dt.normalize()
    deadline = trade_day - pd.Timedelta(days=1) + pd.Timedelta(hours=DEADLINE_KST_HOUR) - KST_OFFSET
    out["deadline_ok"] = out["posted_utc"].notna() & (out["posted_utc"] <= deadline)
    if out["datetime"].duplicated().any():
        raise ValueError("같은 유효 시각에 초기화가 둘 이상 겹칩니다")
    return out.sort_values("datetime").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-01-01", help="첫 거래일 D (KST)")
    ap.add_argument("--end", default="2026-10-31", help="마지막 거래일 D (KST)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.build_only:
        days = pd.date_range(args.start, args.end, freq="D")
        init_dates = [f"{d + pd.Timedelta(days=INIT_DAY_OFFSET):%Y%m%d}" for d in days]
        t0, bad = time.time(), {}
        with ProcessPoolExecutor(args.workers) as pool:
            futs = [pool.submit(download_run, d, str(RAW_DIR)) for d in init_dates]
            for k, fut in enumerate(as_completed(futs), 1):
                d, st = fut.result()
                if st not in ("ok", "cached"):
                    bad[d] = st
                if k % 50 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} ({time.time() - t0:.0f}s) 최근 {d}: {st[:60]}", flush=True)
        print(f"완료: {len(init_dates) - len(bad)}회분, 받지 못함 {len(bad)}회분 {sorted(bad)[:10]}")
    tidy = build_tidy()
    tidy.to_csv(TIDY_CSV, index=False)
    late = tidy[~tidy["deadline_ok"]]
    print(f"saved {TIDY_CSV} rows={len(tidy)} {tidy['datetime'].min()} ~ {tidy['datetime'].max()} | "
          f"마감 후 업로드 {late['init_utc'].nunique()}회분")


if __name__ == "__main__":
    main()
