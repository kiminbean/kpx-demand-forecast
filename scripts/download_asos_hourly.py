#!/usr/bin/env python3
"""
기상청 ASOS 시간자료 내려받기 (제주 4개 관측소) — 공공데이터포털 API
==================================================================

`data/raw/weather/jeju_weather_{연도}_all_stations.csv`를 2025년 파일과 같은 열 이름으로 만든다.
API는 전일 자료까지만 준다. 이미 있는 파일은 덮어쓰지 않는다 (원본 보호).

사용법
  python scripts/download_asos_hourly.py --year 2026                    # 1월 1일 ~ 어제
  python scripts/download_asos_hourly.py --year 2026 --end 2026-10-31

환경변수: KMA_API_KEY (.env)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
OUT_DIR = ROOT / "data" / "raw" / "weather"
STATIONS: Dict[str, str] = {"184": "제주", "185": "고산", "188": "성산", "189": "서귀포"}

# API 열 → 이 프로젝트의 열 이름 (2025년 파일과 같다). 나머지 열은 API 이름 그대로 둔다
COLMAP: Dict[str, str] = {
    "tm": "datetime", "stnId": "station_id", "stnNm": "station_name",
    "ta": "temperature", "rn": "precipitation", "ws": "wind_speed", "wd": "wind_direction",
    "hm": "humidity", "pa": "pressure", "ps": "sea_pressure", "ss": "sunshine_hours",
    "icsr": "solar_radiation", "dc10Tca": "cloud_cover", "vs": "visibility",
}
NUMERIC = ["temperature", "precipitation", "wind_speed", "wind_direction", "humidity", "pressure",
           "sea_pressure", "sunshine_hours", "solar_radiation", "cloud_cover", "visibility"]
ROWS_PER_PAGE = 999
TIMEOUT_SEC = 60
MAX_RETRIES = 4
REQUEST_PAUSE_SEC = 0.5


def load_api_key() -> str:
    """환경변수 → .env 순서로 찾는다. 줄바꿈·따옴표가 섞여 있을 수 있어 정리한다."""
    key = os.environ.get("KMA_API_KEY", "")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("KMA_API_KEY="):
                key = line.split("=", 1)[1]
    key = key.strip().strip('"').strip("'")
    if not key:
        raise ValueError("KMA_API_KEY가 없습니다 (.env 또는 환경변수)")
    return key


def fetch(key: str, station: str, start: pd.Timestamp, end: pd.Timestamp) -> List[dict]:
    """관측소 1곳의 [start, end] 시간자료 (페이지를 끝까지 넘긴다)"""
    items: List[dict] = []
    page = 1
    while True:
        params = {"serviceKey": key, "numOfRows": ROWS_PER_PAGE, "pageNo": page, "dataType": "JSON",
                  "dataCd": "ASOS", "dateCd": "HR", "startDt": f"{start:%Y%m%d}", "endDt": f"{end:%Y%m%d}",
                  "startHh": "00", "endHh": "23", "stnIds": station}
        body = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(API_URL, params=params, timeout=TIMEOUT_SEC)
                resp.raise_for_status()
                data = resp.json()["response"]
                if data["header"]["resultCode"] != "00":
                    raise RuntimeError(f"API 오류 [{data['header']['resultCode']}] {data['header']['resultMsg']}")
                body = data["body"]
                break
            except (requests.RequestException, ValueError, KeyError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"ASOS {station} {start:%Y-%m-%d}~{end:%Y-%m-%d} 실패: {e}") from e
                time.sleep(REQUEST_PAUSE_SEC * (attempt + 2))
        got = body["items"]["item"]
        items.extend(got)
        if page * ROWS_PER_PAGE >= int(body["totalCount"]) or not got:
            return items
        page += 1
        time.sleep(REQUEST_PAUSE_SEC)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--end", default=None, help="마지막 날 (기본: 어제 또는 그 해 12-31)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    out = args.out_dir / f"jeju_weather_{args.year}_all_stations.csv"
    if out.exists():
        raise SystemExit(f"{out} 가 이미 있습니다. 원본은 덮어쓰지 않습니다.")
    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    last = min(pd.Timestamp(args.end) if args.end else pd.Timestamp(f"{args.year}-12-31"), yesterday)
    first = pd.Timestamp(f"{args.year}-01-01")
    if last < first:
        raise SystemExit(f"{args.year}년 자료는 아직 없습니다")

    key = load_api_key()
    rows: List[dict] = []
    for station in STATIONS:
        for m_start in pd.date_range(first, last, freq="MS"):
            m_end = min(m_start + pd.offsets.MonthEnd(0), last)
            got = fetch(key, station, m_start, m_end)
            rows.extend(got)
            print(f"  {STATIONS[station]} {m_start:%Y-%m}: {len(got)} rows", flush=True)
            time.sleep(REQUEST_PAUSE_SEC)
    df = pd.DataFrame(rows).rename(columns=COLMAP)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["station_id"] = df["station_id"].astype(int)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")   # 빈 문자열(미관측) → NaN
    df = df.sort_values(["station_id", "datetime"]).drop_duplicates(["station_id", "datetime"])
    expected = len(STATIONS) * (int((last - first).days) + 1) * 24
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out} rows={len(df)} (기대 {expected}) {df['datetime'].min()} ~ {df['datetime'].max()}")
    if len(df) < 0.99 * expected:
        print("경고: 받은 행이 기대보다 1% 넘게 적습니다", file=sys.stderr)


if __name__ == "__main__":
    main()
