#!/usr/bin/env python3
"""
NOAA GFS 0.25° 운영 예보 내려받기 — AWS Open Data(noaa-gfs-bdp-pds) 원본 GRIB2
=============================================================================

하루 전 예측을 운영 조건으로 재현하려면 '제출 마감 전에 실제로 공개돼 있던 예보'가 필요하다.
이 스크립트는 NOAA가 실시간으로 올린 GFS 원본 파일에서 제주 격자만 잘라 오고,
파일마다 S3 `Last-Modified`(업로드 시각)를 함께 기록해 마감 전 공개 여부를 파일 단위로 검증한다.

시나리오 (거래일 D, KST)
  d1_10kst : D-2 18 UTC 초기화 (= D-1 03:00 KST). 예보시간 f026~f041 → D 05:00~20:00 KST
             실제 업로드 ≈ 초기화 + 3시간 40~55분 (D-1 07시 전후 KST) → D-1 10:00 마감 전
  d1_17kst : D-1 00 UTC 초기화 (= D-1 09:00 KST). 예보시간 f020~f035 → D 05:00~20:00 KST
             실제 업로드 ≈ D-1 12:40~13:00 KST → 10시 마감에는 못 쓰고 17:00 제출에만 쓸 수 있다

변수 (pgrb2.0p25, 1시간 간격)
  DSWRF:surface              W/m², 6시간 구간 시작부터 f까지의 '평균' (예: 24-27 hour ave)
                             → 구간 (f-1, f] 평균 = n·A(f) − (n−1)·A(f−1), n = f − 구간 시작
  TCDC:entire atmosphere     %, 순간값(f hour fcst)
  TMP:2 m above ground       K, 순간값
  RH:2 m above ground        %, 순간값

출력
  data/external/nwp/raw/gfs_aws/t{CC}z/{YYYYMMDD}.csv  초기화 1회분의 제주 격자값(위도 33.0~33.75, 경도 126.0~127.0)
  data/external/nwp/gfs_aws_{시나리오}.csv              관측소별 tidy (구간 끝 시각 T KST, ASOS와 같은 단위)
      solar_radiation MJ/m² (T-1h, T] 누적 / temperature °C / humidity % / cloud_cover 0~10
      posted_utc = 그 값에 쓰인 파일들의 가장 늦은 업로드 시각, deadline_ok = 마감 전 업로드 여부

격자 → 관측소: ASOS 좌표로 쌍선형 보간 (주변 0.25° 격자 4점).
의존성: pip install eccodes (GRIB2 디코더, 바이너리 포함). 라이선스: NOAA 공공 데이터(제한 없음).

사용법
  python scripts/download_nwp_gfs_aws.py --scenario d1_10kst --start 2024-01-01 --end 2025-12-31
  python scripts/download_nwp_gfs_aws.py --scenario d1_17kst --start 2024-01-01 --end 2025-12-31
  python scripts/download_nwp_gfs_aws.py --scenario d1_10kst --build-only     # 내려받은 원본으로 tidy만 다시 생성
"""

import argparse
import email.utils
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUCKET_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT_DIR = ROOT / "data" / "external" / "nwp"

# ASOS 관측소 좌표 (scripts/download_nwp_openmeteo.py와 같은 값)
STATION_COORDS: Dict[str, Tuple[float, float]] = {
    "jeju": (33.5141, 126.5297),
    "gosan": (33.2938, 126.1628),
    "seongsan": (33.3868, 126.8802),
    "seogwipo": (33.2462, 126.5653),
}
GRID_STEP = 0.25
BOX_LATS = [33.75, 33.5, 33.25, 33.0]            # GFS 격자는 북→남
BOX_LONS = [126.0, 126.25, 126.5, 126.75, 127.0]
GRID_NI, GRID_NJ = 1440, 721

# 내려받을 GRIB 메시지: (변수, 층) → (열 이름, 시간 구간 종류)
MESSAGES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("DSWRF", "surface"): ("dswrf_avg", "ave"),
    ("TCDC", "entire atmosphere"): ("tcdc", "inst"),
    ("TMP", "2 m above ground"): ("tmp2m", "inst"),
    ("RH", "2 m above ground"): ("rh2m", "inst"),
}
DSWRF_BUCKET_HOURS = 6       # GFS 평균 플럭스는 6시간마다 다시 시작한다

# 시나리오: 초기화 시각(UTC), 거래일 D 대비 초기화 날짜, 예보시간 범위, 제출 마감(D-1 KST 시)
SCENARIOS: Dict[str, Dict[str, int]] = {
    "d1_10kst": {"cycle": 18, "init_day_offset": -2, "lead_first": 26, "lead_last": 41, "deadline_kst_hour": 10},
    "d1_17kst": {"cycle": 0, "init_day_offset": -1, "lead_first": 20, "lead_last": 35, "deadline_kst_hour": 17},
}
KST_OFFSET = pd.Timedelta(hours=9)
W_TO_MJ_PER_HOUR = 0.0036
KELVIN = 273.15

TIMEOUT_SEC = 60
MAX_RETRIES = 5
RETRY_PAUSE_SEC = 2.0
THREADS_PER_RUN = 8


def _http_get(url: str, byte_range: Optional[Tuple[int, int]] = None) -> Tuple[bytes, Optional[str]]:
    """GET (선택적 Range). (본문, Last-Modified 헤더)를 돌려준다. 404는 FileNotFoundError."""
    headers = {} if byte_range is None else {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_SEC) as resp:
                return resp.read(), resp.headers.get("Last-Modified")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
        time.sleep(RETRY_PAUSE_SEC * (attempt + 1))
    raise RuntimeError(f"내려받기 실패 ({MAX_RETRIES}회 재시도): {url} — {last_err}")


def _get_message(url: str, start: int, end: int) -> Tuple[bytes, Optional[str]]:
    """GRIB 메시지 1개를 받아 길이와 앞뒤 표식을 확인한다. 잘려서 온 응답은 다시 받는다."""
    for _ in range(MAX_RETRIES):
        buf, last_modified = _http_get(url, (start, end))
        if len(buf) == end - start + 1 and buf[:4] == b"GRIB" and buf[-4:] == b"7777":
            return buf, last_modified
        time.sleep(RETRY_PAUSE_SEC)
    raise ValueError(f"GRIB 메시지가 온전하지 않습니다 (idx와 파일이 어긋났을 수 있음): {url} bytes={start}-{end}")


def _parse_idx(text: str) -> List[Tuple[str, str, str, int, Optional[int]]]:
    """wgrib2 idx → [(변수, 층, 시간 구간 문자열, 시작 바이트, 끝 바이트)]"""
    lines = [ln for ln in text.strip().split("\n") if ln]
    out = []
    for i, ln in enumerate(lines):
        p = ln.split(":")
        end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
        out.append((p[3], p[4], p[5], int(p[1]), end))
    return out


def _decode_box(buf: bytes) -> Tuple[np.ndarray, str, str]:
    """GRIB2 메시지 1개 → (제주 격자 4×5 값, stepType, stepRange). 격자 정의가 다르면 실패시킨다."""
    import eccodes  # 지연 import: --build-only는 디코더 없이도 돈다

    gid = eccodes.codes_new_from_message(buf)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        lat0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        if (ni, nj) != (GRID_NI, GRID_NJ) or abs(lat0 - 90.0) > 1e-6 or abs(lon0) > 1e-6:
            raise ValueError(f"예상과 다른 격자: Ni={ni} Nj={nj} lat0={lat0} lon0={lon0}")
        step_type = str(eccodes.codes_get(gid, "stepType"))
        step_range = str(eccodes.codes_get(gid, "stepRange"))
        vals = eccodes.codes_get_values(gid).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    jj = [int(round((90.0 - la) / GRID_STEP)) for la in BOX_LATS]
    ii = [int(round(lo / GRID_STEP)) for lo in BOX_LONS]
    return vals[np.ix_(jj, ii)], step_type, step_range


def _fetch_lead(init: pd.Timestamp, cycle: int, lead: int) -> List[dict]:
    """예보시간 1개 파일에서 필요한 메시지 4개를 내려받아 제주 격자값 행으로 만든다"""
    url = f"{BUCKET_URL}/gfs.{init:%Y%m%d}/{cycle:02d}/atmos/gfs.t{cycle:02d}z.pgrb2.0p25.f{lead:03d}"
    idx_text, _ = _http_get(url + ".idx")
    rows = []
    found = set()
    for var, level, trange, start, end in _parse_idx(idx_text.decode("utf-8")):
        key = (var, level)
        if key not in MESSAGES or end is None:
            continue
        name, kind = MESSAGES[key]
        if (kind == "ave") != ("ave" in trange):
            continue
        buf, last_modified = _get_message(url, start, end)
        box, step_type, step_range = _decode_box(buf)
        if kind == "ave" and step_type != "avg":
            raise ValueError(f"{url} {var}: 평균값이 아님 (stepType={step_type})")
        if kind == "inst" and (step_type != "instant" or step_range != str(lead)):
            raise ValueError(f"{url} {var}: 순간값이 아님 (stepType={step_type}, stepRange={step_range})")
        posted = email.utils.parsedate_to_datetime(last_modified) if last_modified else None
        row = {"init_utc": init, "lead_h": lead, "var": name, "step_range": step_range,
               "posted_utc": None if posted is None else posted.strftime("%Y-%m-%d %H:%M:%S")}
        for a, la in enumerate(BOX_LATS):
            for b, lo in enumerate(BOX_LONS):
                row[f"g_{la:.2f}_{lo:.2f}"] = float(box[a, b])
        rows.append(row)
        found.add(name)
    missing = {v[0] for v in MESSAGES.values()} - found
    if missing:
        raise ValueError(f"{url}: 메시지 없음 {sorted(missing)}")
    return rows


def download_run(init_date: str, cycle: int, lead_first: int, lead_last: int, raw_dir: str) -> Tuple[str, str]:
    """초기화 1회분 → raw CSV. 반환 (init_date, 상태). 이미 있으면 건너뛴다."""
    out = Path(raw_dir) / f"{init_date}.csv"
    if out.exists():
        return init_date, "cached"
    init = pd.Timestamp(init_date) + pd.Timedelta(hours=cycle)
    leads = list(range(lead_first, lead_last + 1))
    try:
        with ThreadPoolExecutor(THREADS_PER_RUN) as pool:
            parts = list(pool.map(lambda ld: _fetch_lead(init, cycle, ld), leads))
    except FileNotFoundError as e:
        return init_date, f"missing: {e}"
    except Exception as e:  # noqa: BLE001 — 디코더 오류 종류가 많다. 이 초기화만 건너뛰고 상태에 남긴다 (다시 실행하면 빈 날짜만 받는다)
        return init_date, f"error: {type(e).__name__}: {e}"
    df = pd.DataFrame([r for part in parts for r in part]).sort_values(["lead_h", "var"])
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(out)                      # 중간에 끊긴 파일을 완성본으로 오인하지 않게 원자적으로 교체
    return init_date, "ok"


def _bilinear_weights(lat: float, lon: float) -> Dict[str, float]:
    """관측소 좌표를 둘러싼 0.25° 격자 4점의 쌍선형 가중치 (열 이름 → 가중치)"""
    la0 = np.floor(lat / GRID_STEP) * GRID_STEP
    lo0 = np.floor(lon / GRID_STEP) * GRID_STEP
    fy, fx = (lat - la0) / GRID_STEP, (lon - lo0) / GRID_STEP
    w = {(la0, lo0): (1 - fy) * (1 - fx), (la0, lo0 + GRID_STEP): (1 - fy) * fx,
         (la0 + GRID_STEP, lo0): fy * (1 - fx), (la0 + GRID_STEP, lo0 + GRID_STEP): fy * fx}
    out = {}
    for (la, lo), wt in w.items():
        if la not in BOX_LATS or lo not in BOX_LONS:
            raise ValueError(f"관측소 ({lat}, {lon})가 내려받은 격자 범위 밖입니다")
        out[f"g_{la:.2f}_{lo:.2f}"] = wt
    return out


def build_tidy(scenario: str, raw_dir: Path) -> pd.DataFrame:
    """raw 격자값 → 관측소별 tidy (구간 끝 시각 T KST). DSWRF는 1시간 평균으로 푼다."""
    cfg = SCENARIOS[scenario]
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}에 내려받은 파일이 없습니다")
    raw = pd.concat([pd.read_csv(f, parse_dates=["init_utc", "posted_utc"]) for f in files], ignore_index=True)
    grid_cols = [c for c in raw.columns if c.startswith("g_")]
    weights = {st: _bilinear_weights(*xy) for st, xy in STATION_COORDS.items()}
    for st, w in weights.items():
        raw[st] = sum(raw[c] * wt for c, wt in w.items())
    stations = list(STATION_COORDS)

    wide = raw.pivot(index=["init_utc", "lead_h"], columns="var", values=stations)
    posted = raw.groupby(["init_utc", "lead_h"])["posted_utc"].max()
    step = raw[raw["var"] == "dswrf_avg"].set_index(["init_utc", "lead_h"])["step_range"]

    rows = []
    for st in stations:
        df = pd.DataFrame({v: wide[(st, v)] for v in ("dswrf_avg", "tcdc", "tmp2m", "rh2m")})
        df["posted_utc"] = posted
        df["step_range"] = step
        df = df.reset_index().sort_values(["init_utc", "lead_h"])
        bucket_start = df["step_range"].str.split("-").str[0].astype(int)
        n = df["lead_h"] - bucket_start
        if not ((n >= 1) & (n <= DSWRF_BUCKET_HOURS)).all():
            raise ValueError("DSWRF 평균 구간이 예상(6시간 누적 재시작)과 다릅니다")
        prev = df.groupby("init_utc")[["dswrf_avg", "posted_utc", "lead_h"]].shift(1)
        same_bucket = (n > 1)
        if (same_bucket & (prev["lead_h"] != df["lead_h"] - 1)).iloc[1:].any():
            # 각 초기화의 첫 예보시간은 직전 값이 없어도 된다(아래에서 따로 검사)
            bad = df[same_bucket & (prev["lead_h"] != df["lead_h"] - 1) & (df["lead_h"] != cfg["lead_first"])]
            if len(bad):
                raise ValueError(f"DSWRF를 풀 직전 예보시간이 없습니다: {bad[['init_utc', 'lead_h']].head()}")
        hourly = np.where(same_bucket, n * df["dswrf_avg"] - (n - 1) * prev["dswrf_avg"], df["dswrf_avg"])
        # 첫 예보시간이 구간 중간이면 풀 수 없다 → 결측 (KST 05시 이전이라 야간이며 모델이 쓰지 않는다)
        first_mid_bucket = (df["lead_h"] == cfg["lead_first"]) & same_bucket
        hourly = np.where(first_mid_bucket, np.nan, hourly)
        posted_used = np.where(same_bucket & prev["posted_utc"].notna(),
                               np.maximum(df["posted_utc"].values, prev["posted_utc"].values), df["posted_utc"].values)
        valid_kst = df["init_utc"] + pd.to_timedelta(df["lead_h"], unit="h") + KST_OFFSET
        rows.append(pd.DataFrame({
            "datetime": valid_kst.values, "station": st, "scenario": scenario,
            "init_utc": df["init_utc"].values, "lead_h": df["lead_h"].values,
            "posted_utc": pd.to_datetime(posted_used),
            "temperature": df["tmp2m"].values - KELVIN,
            "humidity": df["rh2m"].values,
            "solar_radiation": np.clip(hourly, 0, None) * W_TO_MJ_PER_HOUR,
            "cloud_cover": df["tcdc"].values / 10.0,
        }))
    tidy = pd.concat(rows, ignore_index=True)
    # 마감: 거래일 D의 전날 deadline_kst_hour시 (KST) → UTC
    trade_day = (tidy["datetime"] - pd.Timedelta(hours=1)).dt.normalize()
    deadline_utc = trade_day - pd.Timedelta(days=1) + pd.Timedelta(hours=cfg["deadline_kst_hour"]) - KST_OFFSET
    tidy["deadline_utc"] = deadline_utc
    tidy["deadline_ok"] = tidy["posted_utc"].notna() & (tidy["posted_utc"] <= deadline_utc)
    if tidy.duplicated(["datetime", "station"]).any():
        raise ValueError("같은 유효 시각에 초기화가 둘 이상 겹칩니다")
    return tidy.sort_values(["station", "datetime"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--start", default="2024-01-01", help="첫 거래일 D (KST)")
    ap.add_argument("--end", default="2025-12-31", help="마지막 거래일 D (KST)")
    ap.add_argument("--workers", type=int, default=6, help="동시에 받는 초기화 수 (프로세스)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--build-only", action="store_true", help="내려받지 않고 tidy만 다시 만든다")
    args = ap.parse_args()

    cfg = SCENARIOS[args.scenario]
    raw_dir = args.out_dir / "raw" / "gfs_aws" / f"t{cfg['cycle']:02d}z"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.build_only:
        days = pd.date_range(args.start, args.end, freq="D")
        init_dates = [f"{d + pd.Timedelta(days=cfg['init_day_offset']):%Y%m%d}" for d in days]
        t0 = time.time()
        status: Dict[str, str] = {}
        with ProcessPoolExecutor(args.workers) as pool:
            futs = [pool.submit(download_run, d, cfg["cycle"], cfg["lead_first"], cfg["lead_last"], str(raw_dir))
                    for d in init_dates]
            for k, fut in enumerate(as_completed(futs), 1):
                d, st = fut.result()
                status[d] = st
                if k % 20 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} ({time.time() - t0:.0f}s) 최근 {d}: {st}", flush=True)
        bad = {d: s for d, s in status.items() if s not in ("ok", "cached")}
        print(f"완료: ok/cached {len(status) - len(bad)}, 없음 {len(bad)}")
        for d, s in sorted(bad.items()):
            print(f"  {d}: {s}", file=sys.stderr)

    tidy = build_tidy(args.scenario, raw_dir)
    out = args.out_dir / f"gfs_aws_{args.scenario}.csv"
    tidy.to_csv(out, index=False)
    late = tidy[~tidy["deadline_ok"]]
    print(f"saved {out} rows={len(tidy)} 거래일 {tidy['datetime'].min()} ~ {tidy['datetime'].max()} | "
          f"마감 후 업로드 {late['init_utc'].nunique()}회분 {len(late)}행")


if __name__ == "__main__":
    main()
