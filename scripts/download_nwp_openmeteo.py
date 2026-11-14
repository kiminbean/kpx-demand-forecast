#!/usr/bin/env python3
"""
과거 수치예보(NWP) 내려받기 — Open-Meteo Previous Runs API
=========================================================

하루 전 예측을 정직하게 평가하려면 관측 기상 대신 '그때 발표된 예보'가 필요하다.
Previous Runs API는 유효 시각보다 N일 앞서 발표된 예보 값을 `{변수}_previous_dayN`으로 돌려준다.
- previous_day2: 48시간 전 발표분 → KPX 하루 전 10시 제출 시점에 확실히 존재 (보수적, 주 지표)
- previous_day1: 24시간 전 발표분 → 일부 시간은 제출 시점 이후 발표분이 섞일 수 있음 (낙관적, 보조)

모델: JMA MSM (5 km, 제주 포함) — 2024년 1월부터 일사·기온·습도·운량이 모두 있는 유일한 모델
(ECMWF IFS 0.25는 2024년 3월 이후만 확인됨, KMA·GFS는 2024년 보관분 없음).

단위 변환: 일사 W/m²(직전 1시간 평균) × 0.0036 → MJ/m²(ASOS와 같은 직전 1시간 누적), 운량 % ÷ 10 → 0~10.
라이선스: Open-Meteo 무료 API는 비상업 용도, 데이터는 CC BY 4.0 (출처 표기 필요).

사용법
  python scripts/download_nwp_openmeteo.py                         # JMA MSM, 2024 전체 → data/external/nwp/
  python scripts/download_nwp_openmeteo.py --model ecmwf_ifs025 --start 2024-03-01
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# ASOS 관측소 좌표 (기상청 지점정보)
STATION_COORDS = {
    "jeju": (33.5141, 126.5297),
    "gosan": (33.2938, 126.1628),
    "seongsan": (33.3868, 126.8802),
    "seogwipo": (33.2462, 126.5653),
}
NWP_VARS = {  # Open-Meteo 변수 → ASOS 열 이름, 단위 변환 계수
    "shortwave_radiation": ("solar_radiation", 0.0036),
    "temperature_2m": ("temperature", 1.0),
    "relative_humidity_2m": ("humidity", 1.0),
    "cloud_cover": ("cloud_cover", 0.1),
}
LEAD_DAYS = (1, 2)
REQUEST_PAUSE_SEC = 1.0
TIMEOUT_SEC = 60


def fetch_chunk(model: str, start: str, end: str) -> list:
    """4개 관측소 좌표를 한 번에 요청 (응답은 좌표별 객체 리스트)"""
    lats = ",".join(f"{c[0]:.4f}" for c in STATION_COORDS.values())
    lons = ",".join(f"{c[1]:.4f}" for c in STATION_COORDS.values())
    hourly = ",".join(f"{v}_previous_day{d}" for v in NWP_VARS for d in LEAD_DAYS)
    query = urllib.parse.urlencode({
        "latitude": lats, "longitude": lons, "hourly": hourly, "models": model,
        "start_date": start, "end_date": end, "timezone": "Asia/Seoul",
    })
    with urllib.request.urlopen(f"{API_URL}?{query}", timeout=TIMEOUT_SEC) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if isinstance(payload, dict):
        if payload.get("error"):
            raise RuntimeError(f"Open-Meteo 오류: {payload.get('reason')}")
        payload = [payload]
    if len(payload) != len(STATION_COORDS):
        raise RuntimeError(f"좌표 {len(STATION_COORDS)}개를 요청했는데 응답이 {len(payload)}개입니다")
    return payload


def to_tidy(payload: list) -> pd.DataFrame:
    """응답 → (datetime, station, lead_days, temperature, humidity, solar_radiation, cloud_cover)"""
    rows = []
    for st, loc in zip(STATION_COORDS, payload):
        h = loc["hourly"]
        times = pd.to_datetime(h["time"])
        for d in LEAD_DAYS:
            df = pd.DataFrame({"datetime": times, "station": st, "lead_days": d,
                               "grid_lat": loc.get("latitude"), "grid_lon": loc.get("longitude")})
            for var, (col, factor) in NWP_VARS.items():
                df[col] = pd.to_numeric(pd.Series(h[f"{var}_previous_day{d}"]), errors="coerce") * factor
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="jma_msm")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01", help="마지막 구간 끝(2025-01-01 00시)까지 포함")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "external" / "nwp")
    args = ap.parse_args()

    raw_dir = args.out_dir / "raw" / args.model
    raw_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.date_range(args.start, args.end, freq="MS").union([pd.Timestamp(args.start), pd.Timestamp(args.end)])
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        end = b - pd.Timedelta(days=1) if b != pd.Timestamp(args.end) else b
        start_s, end_s = f"{a:%Y-%m-%d}", f"{end:%Y-%m-%d}"
        payload = fetch_chunk(args.model, start_s, end_s)
        (raw_dir / f"{start_s}_{end_s}.json").write_text(json.dumps(payload), encoding="utf-8")
        parts.append(to_tidy(payload))
        print(f"  {args.model} {start_s} ~ {end_s}: {sum(len(p['hourly']['time']) for p in payload)} rows")
        time.sleep(REQUEST_PAUSE_SEC)

    tidy = pd.concat(parts, ignore_index=True).drop_duplicates(["datetime", "station", "lead_days"]).sort_values(
        ["lead_days", "station", "datetime"])
    missing = tidy[["solar_radiation", "temperature", "humidity", "cloud_cover"]].isna().mean()
    out = args.out_dir / f"openmeteo_previous_runs_{args.model}.csv"
    tidy.to_csv(out, index=False)
    print(f"saved {out} rows={len(tidy)} | 결측 비율 {missing.round(4).to_dict()}")
    if (missing > 0.01).any():
        print("경고: 결측이 1%를 넘는 변수가 있습니다 — 해당 기간 보관분이 없을 수 있습니다", file=sys.stderr)


if __name__ == "__main__":
    main()
