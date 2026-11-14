#!/usr/bin/env python3
"""
ECMWF IFS 공개 자료(0.25°) 내려받기 — 공개 클라우드 미러의 실시간 파일
==============================================================================

ECMWF가 실시간으로 공개한 운영 예보 파일에서 제주 격자만 잘라 오고, 파일마다 S3 `Last-Modified`를 기록한다.
재예측(hindcast)이 아니라 그때 실제로 올라온 파일이다. 라이선스 CC BY 4.0 (출처 표기).

시나리오 (거래일 D, KST)
  d1_10kst : D-2 12 UTC 초기화, 예보시간 30~48시간(3시간 간격) → D 03:00~21:00 KST
             업로드 ≈ 초기화 + 7~8시간 반 (D-1 04~06시 KST) → D-1 10:00 마감 전
  d1_17kst : D-2 18 UTC 초기화, 예보시간 24~42시간 → D 03:00~21:00 KST
             업로드 ≈ D-1 09:30~10:30 KST → 10시 마감에는 빠듯하거나 늦고 17시 제출에는 쓸 수 있다

변수
  ssrd  J/m², 초기시각부터의 누적 하향 단파복사 → 3시간 구간 에너지 = 차분
  2t    K, 2 m 기온 (순간값)
  2d    K, 2 m 이슬점 (순간값) → 습도는 Magnus 식
  전운량(tcc)은 2024~2025년 공개 자료에 없다 → cloud_cover는 결측으로 둔다

3시간 → 1시간
  일사: 3시간 구간 에너지를 그 안의 맑은하늘 일사(Haurwitz, 구간 적분) 비율대로 나눈다 (구간 안에서 맑음지수 일정 가정)
  기온·이슬점: 시각에 대해 선형 보간

미러: Google Cloud(ecmwf-open-data)에서 먼저 찾고, 없으면 AWS(ecmwf-forecasts)에서 받는다 (MIRRORS 주석 참고).
경로: 2024-02-28까지 {날짜}/{HH}z/0p25/…, 그 뒤 {날짜}/{HH}z/ifs/0p25/…. 18 UTC 스트림 이름은 scda → oper로 바뀌었다.
0.25° 자료는 2024년 1월 말부터 있다 (그 전은 0.4°).

출력
  data/external/nwp/raw/ecmwf_aws/t{CC}z/{YYYYMMDD}.csv
  data/external/nwp/ecmwf_aws_{시나리오}.csv   GFS tidy와 같은 열 (cloud_cover는 NaN)

사용법
  python scripts/download_nwp_ecmwf_aws.py --scenario d1_10kst --start 2024-02-03 --end 2026-10-31
  python scripts/download_nwp_ecmwf_aws.py --scenario d1_10kst --build-only
"""

import argparse
import email.utils
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.download_nwp_gfs_aws import (  # noqa: E402
    BOX_LATS,
    BOX_LONS,
    GRID_NI,
    GRID_NJ,
    GRID_STEP,
    KELVIN,
    KST_OFFSET,
    OUT_DIR,
    STATION_COORDS,
    _bilinear_weights,
    _http_get,
)
from src.features.solar_features_v2 import interval_irradiance  # noqa: E402

# 같은 실시간 파일의 공개 미러. AWS는 요청이 몰리면 503을 자주 돌려주고, Google Cloud에는 2024년 중반 18 UTC GRIB이 없다.
# Google Cloud의 업로드 시각은 AWS보다 2~3분 늦다 (마감 판정에는 더 보수적).
MIRRORS: Dict[str, str] = {
    "gcs": "https://storage.googleapis.com/ecmwf-open-data",
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
}
PARAMS: Dict[str, str] = {"ssrd": "ssrd", "2t": "t2m", "2d": "d2m"}   # index의 param → 열 이름
STEP_HOURS = 3
J_TO_MJ = 1e-6
THREADS_PER_RUN = 3            # 이 버킷은 요청이 몰리면 503(SlowDown)을 돌려준다 → GFS보다 천천히 받는다
THROTTLE_RETRIES = 6
THROTTLE_PAUSE_SEC = 8.0
MAGNUS_A, MAGNUS_B = 17.625, 243.04   # Alduchov & Eskridge (1996)

SCENARIOS: Dict[str, Dict] = {
    "d1_10kst": {"cycle": 12, "init_day_offset": -2, "steps": list(range(30, 49, STEP_HOURS)),
                 "streams": ["oper"], "deadline_kst_hour": 10},
    "d1_17kst": {"cycle": 18, "init_day_offset": -2, "steps": list(range(24, 43, STEP_HOURS)),
                 "streams": ["scda", "oper"], "deadline_kst_hour": 17},
}


def _candidate_urls(init_date: str, cycle: int, step: int, streams: List[str], mirrors: List[str]) -> List[str]:
    """미러·경로 규칙·스트림 이름의 가능한 조합을 순서대로 돌려준다 (확장자 없는 기본 경로)"""
    out = []
    for mirror in mirrors:
        for stream in streams:
            name = f"{init_date}{cycle:02d}0000-{step}h-{stream}-fc"
            out.append(f"{MIRRORS[mirror]}/{init_date}/{cycle:02d}z/ifs/0p25/{stream}/{name}")
            out.append(f"{MIRRORS[mirror]}/{init_date}/{cycle:02d}z/0p25/{stream}/{name}")
    return out


def _decode_box(buf: bytes) -> np.ndarray:
    """GRIB2 메시지 1개 → 제주 격자 4×5 값. ECMWF 격자는 경도 180°에서 시작한다."""
    import eccodes

    gid = eccodes.codes_new_from_message(buf)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        lat0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        if (ni, nj) != (GRID_NI, GRID_NJ) or abs(lat0 - 90.0) > 1e-6:
            raise ValueError(f"예상과 다른 격자: Ni={ni} Nj={nj} lat0={lat0} lon0={lon0}")
        vals = eccodes.codes_get_values(gid).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    jj = [int(round((90.0 - la) / GRID_STEP)) for la in BOX_LATS]
    ii = [int(round(((lo - lon0) % 360.0) / GRID_STEP)) % ni for lo in BOX_LONS]
    return vals[np.ix_(jj, ii)]


def _get_patiently(url: str, byte_range: Optional[Tuple[int, int]] = None) -> Tuple[bytes, Optional[str]]:
    """_http_get에 긴 대기를 한 겹 더 씌운다 (503이 이어질 때). 404는 그대로 올린다."""
    for attempt in range(THROTTLE_RETRIES):
        try:
            return _http_get(url, byte_range)
        except RuntimeError:
            if attempt == THROTTLE_RETRIES - 1:
                raise
            time.sleep(THROTTLE_PAUSE_SEC * (attempt + 1))
    raise RuntimeError(f"도달할 수 없는 분기: {url}")


def _fetch_step(init: pd.Timestamp, cycle: int, step: int, streams: List[str], mirrors: List[str]) -> List[dict]:
    """예보시간 1개 파일에서 ssrd·2t·2d를 내려받는다. 어느 경로에도 없으면 FileNotFoundError."""
    base = None
    for cand in _candidate_urls(f"{init:%Y%m%d}", cycle, step, streams, mirrors):
        try:
            idx_text, _ = _get_patiently(cand + ".index")
            base = cand
            break
        except FileNotFoundError:
            continue
    if base is None:
        raise FileNotFoundError(f"{init:%Y%m%d} {cycle:02d}z step {step}")
    rows, found = [], set()
    for line in idx_text.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("param") not in PARAMS or rec.get("levtype") != "sfc" or int(rec["step"]) != step:
            continue
        buf, last_modified = _get_patiently(base + ".grib2", (rec["_offset"], rec["_offset"] + rec["_length"] - 1))
        if len(buf) != rec["_length"] or buf[:4] != b"GRIB" or buf[-4:] != b"7777":
            raise RuntimeError(f"GRIB 메시지가 온전하지 않습니다: {base} {rec['param']}")
        box = _decode_box(buf)
        posted = email.utils.parsedate_to_datetime(last_modified) if last_modified else None
        row = {"init_utc": init, "step_h": step, "var": PARAMS[rec["param"]],
               "posted_utc": None if posted is None else posted.strftime("%Y-%m-%d %H:%M:%S")}
        for a, la in enumerate(BOX_LATS):
            for b, lo in enumerate(BOX_LONS):
                row[f"g_{la:.2f}_{lo:.2f}"] = float(box[a, b])
        rows.append(row)
        found.add(PARAMS[rec["param"]])
    missing = set(PARAMS.values()) - found
    if missing:
        raise ValueError(f"{base}: 메시지 없음 {sorted(missing)}")
    return rows


def download_run(init_date: str, scenario: str, raw_dir: str, mirrors: List[str]) -> Tuple[str, str]:
    """초기화 1회분 → raw CSV. 반환 (init_date, 상태). 이미 있으면 건너뛴다."""
    cfg = SCENARIOS[scenario]
    out = Path(raw_dir) / f"{init_date}.csv"
    if out.exists():
        return init_date, "cached"
    init = pd.Timestamp(init_date) + pd.Timedelta(hours=cfg["cycle"])
    try:
        with ThreadPoolExecutor(THREADS_PER_RUN) as pool:
            parts = list(pool.map(lambda st: _fetch_step(init, cfg["cycle"], st, cfg["streams"], mirrors),
                                  cfg["steps"]))
    except FileNotFoundError as e:
        return init_date, f"missing: {e}"
    except (RuntimeError, ValueError) as e:   # 막히거나 변수가 빠진 초기화만 건너뛴다. 다시 실행하면 빈 날짜만 받는다
        return init_date, f"error: {e}"
    df = pd.DataFrame([r for part in parts for r in part]).sort_values(["step_h", "var"])
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(out)
    return init_date, "ok"


def relative_humidity(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """기온·이슬점(°C) → 상대습도(%). Magnus 식, 100%로 자른다."""
    e = np.exp(MAGNUS_A * td_c / (MAGNUS_B + td_c))
    es = np.exp(MAGNUS_A * t_c / (MAGNUS_B + t_c))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def build_tidy(scenario: str, raw_dir: Path) -> pd.DataFrame:
    """raw 격자값 → 관측소별 시간값 tidy (구간 끝 시각 T KST)"""
    cfg = SCENARIOS[scenario]
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}에 내려받은 파일이 없습니다")
    raw = pd.concat([pd.read_csv(f, parse_dates=["init_utc", "posted_utc"]) for f in files], ignore_index=True)
    for st, xy in STATION_COORDS.items():
        raw[st] = sum(raw[c] * wt for c, wt in _bilinear_weights(*xy).items())
    stations = list(STATION_COORDS)
    wide = raw.pivot(index=["init_utc", "step_h"], columns="var", values=stations)
    posted = raw.groupby(["init_utc", "step_h"])["posted_utc"].max()

    rows = []
    for init, grp in wide.groupby(level="init_utc"):
        steps = grp.index.get_level_values("step_h").to_numpy()
        if list(steps) != cfg["steps"]:
            continue                                   # 예보시간이 빠진 초기화는 쓰지 않는다
        t_inst = init + pd.to_timedelta(steps, unit="h") + KST_OFFSET        # 순간값 시각 (KST)
        t_hour = pd.date_range(t_inst[0] + pd.Timedelta(hours=1), t_inst[-1], freq="h")   # 구간 끝 시각
        block = (np.arange(len(t_hour)) // STEP_HOURS)                       # 각 시간이 속한 3시간 구간
        hcs = interval_irradiance(t_hour)["Hcs"].to_numpy()
        hcs_block = np.bincount(block, weights=hcs, minlength=len(steps) - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(hcs_block[block] > 0, hcs / hcs_block[block], 1.0 / STEP_HOURS)
        post = posted.loc[init].reindex(steps)
        post_block = np.maximum(post.values[:-1], post.values[1:])[block]     # 구간 양 끝 파일 중 늦은 업로드
        frac = (np.arange(len(t_hour)) % STEP_HOURS + 1) / STEP_HOURS         # 구간 안 위치 (1/3, 2/3, 1)
        for st in stations:
            g = grp[st]
            energy = np.diff(g["ssrd"].to_numpy()) * J_TO_MJ                  # 3시간 구간 에너지 (MJ/m²)
            t2 = g["t2m"].to_numpy() - KELVIN
            d2 = g["d2m"].to_numpy() - KELVIN
            temp = t2[:-1][block] * (1 - frac) + t2[1:][block] * frac
            dew = d2[:-1][block] * (1 - frac) + d2[1:][block] * frac
            rows.append(pd.DataFrame({
                "datetime": t_hour, "station": st, "scenario": scenario, "init_utc": init,
                "lead_h": steps[0] + np.arange(1, len(t_hour) + 1), "posted_utc": pd.to_datetime(post_block),
                "temperature": temp, "humidity": relative_humidity(temp, dew),
                "solar_radiation": np.clip(energy[block], 0, None) * share, "cloud_cover": np.nan,
            }))
    if not rows:
        raise ValueError("완전한 초기화가 하나도 없습니다")
    tidy = pd.concat(rows, ignore_index=True)
    trade_day = (tidy["datetime"] - pd.Timedelta(hours=1)).dt.normalize()
    tidy["deadline_utc"] = trade_day - pd.Timedelta(days=1) + pd.Timedelta(hours=cfg["deadline_kst_hour"]) - KST_OFFSET
    tidy["deadline_ok"] = tidy["posted_utc"].notna() & (tidy["posted_utc"] <= tidy["deadline_utc"])
    if tidy.duplicated(["datetime", "station"]).any():
        raise ValueError("같은 유효 시각에 초기화가 둘 이상 겹칩니다")
    return tidy.sort_values(["station", "datetime"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--start", default="2024-02-03", help="첫 거래일 D (KST)")
    ap.add_argument("--end", default="2026-10-31", help="마지막 거래일 D (KST)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--mirrors", default="gcs,aws", help="찾아볼 미러 순서 (쉼표 구분). 앞 미러에 없으면 다음 미러에서 받는다")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    cfg = SCENARIOS[args.scenario]
    mirrors = [m.strip() for m in args.mirrors.split(",") if m.strip()]
    unknown = [m for m in mirrors if m not in MIRRORS]
    if unknown or not mirrors:
        ap.error(f"알 수 없는 미러: {unknown} (가능: {list(MIRRORS)})")
    raw_dir = args.out_dir / "raw" / "ecmwf_aws" / f"t{cfg['cycle']:02d}z"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.build_only:
        days = pd.date_range(args.start, args.end, freq="D")
        init_dates = [f"{d + pd.Timedelta(days=cfg['init_day_offset']):%Y%m%d}" for d in days]
        t0 = time.time()
        status: Dict[str, str] = {}
        with ProcessPoolExecutor(args.workers) as pool:
            futs = [pool.submit(download_run, d, args.scenario, str(raw_dir), mirrors) for d in init_dates]
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
    out = args.out_dir / f"ecmwf_aws_{args.scenario}.csv"
    tidy.to_csv(out, index=False)
    late = tidy[~tidy["deadline_ok"]]
    print(f"saved {out} rows={len(tidy)} {tidy['datetime'].min()} ~ {tidy['datetime'].max()} | "
          f"마감 후 업로드 {late['init_utc'].nunique()}회분 {len(late)}행")


if __name__ == "__main__":
    main()
