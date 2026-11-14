#!/usr/bin/env python3
"""
KPX 제주계통운영정보(5분) 내려받기 — 대국민 전력수급현황 공유 시스템
====================================================================

https://openapi.kpx.or.kr/chejusukub.do 의 "조회 → 저장(CSV)" 기능을 그대로 쓴다 (로그인·인증키 없음).
열: 기준일시, 공급능력(MW), 현재수요(MW), 신재생총합(MW), 신재생태양광(MW), 신재생풍력(MW). 2019-12-01부터 있다.

왜 필요한가
  공개 거래량(data.go.kr 15065269 / 15161308)의 `태양광`·`풍력`에는 재생에너지 입찰제도에 등록한 자원이 빠진다.
  이 자원은 `급전가능 재생에너지`(연료코드 53)로 따로 집계되고, 거래기간 경계(3/1, 9/1)마다 넘어가는 설비가 늘어
  공개 시계열의 포괄 범위가 계단식으로 줄었다 (2024-06-01, 2024-09-01, 2025-03-01, 2025-09-01).
  계통운영정보의 태양광·풍력은 입찰자원을 포함하므로 이 단절이 없고, 2026년 자료도 있다.

주의
  - 값은 5분 간격 순시 출력(MW)이고 출력제어가 반영된 실적이다. 정산 계량값(MWh)과 정의가 다르다.
  - 1회 조회는 최대 1년. 이 스크립트는 월 단위로 나눠 받는다.

출력
  data/raw/kpx/jeju_system_5min/{YYYY-MM}.csv      원본 (EUC-KR → UTF-8로만 바꿔 저장)
  data/processed/kpx_jeju_system_hourly.csv        구간 끝 시각 T(KST) 기준 1시간 평균
      solar_mw      표시 지연 10분을 당긴 태양광 (구간 (T-1h+10분, T+10분]의 5분값 평균) — 모델 라벨로 쓰는 값
      solar_raw_mw, wind_mw, renew_mw, demand_mw, supply_mw, n_samples   (구간 (T-1h, T]의 5분값 평균)

사용법
  python scripts/download_kpx_jeju_system.py --start 2019-12 --end 2026-11
  python scripts/download_kpx_jeju_system.py --build-only
"""

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://openapi.kpx.or.kr"
RAW_DIR = ROOT / "data" / "raw" / "kpx" / "jeju_system_5min"
HOURLY_CSV = ROOT / "data" / "processed" / "kpx_jeju_system_hourly.csv"

COLUMNS = {"기준일시": "time", "공급능력(MW)": "supply_mw", "현재수요(MW)": "demand_mw",
           "신재생총합(MW)": "renew_mw", "신재생태양광(MW)": "solar_mw", "신재생풍력(MW)": "wind_mw"}
SAMPLES_PER_HOUR = 12
MIN_SAMPLES_PER_HOUR = 9         # 5분값이 9개 미만인 시간은 결측으로 둔다
# 계통 태양광 값은 시장 계량값보다 늦게 찍힌다. 시간 평균과 공개 거래량(코드 13)의 상관이 가장 높은 지연:
# 2020~2023년 상반기 35~45분, 2023년 하반기~2024년 상반기 15분, 2024년 하반기~2025년 10분.
# 이 프로젝트가 쓰는 2024년 이후 구간에 맞춰 10분으로 고정한다 (그 이전 구간에는 맞지 않는다).
SOLAR_LAG_MIN = 10
REQUEST_PAUSE_SEC = 1.0
TIMEOUT_SEC = 120
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (J-REF research; data download via the site's own CSV export)"


def fetch_month(session: requests.Session, month: pd.Period) -> str:
    """한 달치 CSV 본문(UTF-8 문자열). 조회를 먼저 해야 저장이 동작한다."""
    start, end = f"{month.start_time:%Y-%m-%d}", f"{month.end_time:%Y-%m-%d}"
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            session.post(f"{BASE_URL}/chejusukub.do", timeout=TIMEOUT_SEC,
                         data={"pageIndex": 1, "startDate": start, "endDate": end, "searchUseYn": "Y"}).raise_for_status()
            resp = session.post(f"{BASE_URL}/downloadChejuSukubCSV.do", timeout=TIMEOUT_SEC,
                                data={"startDate": start, "endDate": end})
            resp.raise_for_status()
            text = resp.content.decode("euc-kr")
            if not text.startswith("기준일시"):
                raise ValueError(f"CSV가 아닌 응답: {text[:80]!r}")
            return text
        except (requests.RequestException, UnicodeDecodeError, ValueError) as e:
            last_err = e
            time.sleep(REQUEST_PAUSE_SEC * (attempt + 2))
    raise RuntimeError(f"{month} 내려받기 실패: {last_err}")


def download(start: str, end: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Referer": f"{BASE_URL}/chejusukub.do"})
    session.get(f"{BASE_URL}/chejusukub.do", timeout=TIMEOUT_SEC).raise_for_status()  # 세션 쿠키
    this_month = pd.Timestamp.now().to_period("M")
    for month in pd.period_range(start, end, freq="M"):
        out = raw_dir / f"{month}.csv"
        if out.exists() and month < this_month:   # 지난 달은 다시 받지 않는다 (이번 달은 자료가 계속 늘어난다)
            continue
        text = fetch_month(session, month)
        out.write_text(text, encoding="utf-8")
        print(f"  {month}: {text.count(chr(10)) - 1} rows", flush=True)
        time.sleep(REQUEST_PAUSE_SEC)


def build_hourly(raw_dir: Path) -> pd.DataFrame:
    """5분 순시값 → 구간 끝 시각 T 기준 1시간 평균. 표본이 모자란 시간은 결측."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}에 원본이 없습니다 — 먼저 내려받으세요")
    parts = []
    for f in files:
        df = pd.read_csv(io.StringIO(f.read_text(encoding="utf-8")), dtype={"기준일시": str})
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name}: 열이 없습니다 {missing}")
        parts.append(df.rename(columns=COLUMNS)[list(COLUMNS.values())])
    raw = pd.concat(parts, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"], format="%Y%m%d%H%M%S")
    raw = raw.drop_duplicates("time").set_index("time").sort_index()
    # 구간 (T-1h, T]: 00:05~01:00 표본이 T=01:00에 속한다
    t_end = raw.index.ceil("h")
    grouped = raw.groupby(t_end)
    hourly = grouped.mean().rename(columns={"solar_mw": "solar_raw_mw"})
    hourly["n_samples"] = grouped["solar_mw"].count()
    value_cols = [c for c in hourly.columns if c != "n_samples"]
    hourly.loc[hourly["n_samples"] < MIN_SAMPLES_PER_HOUR, value_cols] = float("nan")
    # 태양광은 표시 시각이 실제보다 늦다 → 그만큼 당겨서 평균한다 (풍력은 지연이 없어 그대로 둔다)
    lagged = raw["solar_mw"].copy()
    lagged.index = lagged.index - pd.Timedelta(minutes=SOLAR_LAG_MIN)
    g_solar = lagged.groupby(lagged.index.ceil("h"))
    solar = g_solar.mean().where(g_solar.count() >= MIN_SAMPLES_PER_HOUR)
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full)
    hourly.insert(0, "solar_mw", solar.reindex(full))
    hourly["n_samples"] = hourly["n_samples"].fillna(0).astype(int)
    hourly.index.name = "datetime"
    return hourly


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2019-12", help="첫 달 (YYYY-MM)")
    ap.add_argument("--end", default=f"{pd.Timestamp.now():%Y-%m}", help="마지막 달 (YYYY-MM)")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=HOURLY_CSV)
    ap.add_argument("--build-only", action="store_true", help="내려받지 않고 시간 평균만 다시 만든다")
    args = ap.parse_args()

    if not args.build_only:
        download(args.start, args.end, args.raw_dir)
    hourly = build_hourly(args.raw_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.out)
    gaps = int((hourly["n_samples"] < MIN_SAMPLES_PER_HOUR).sum())
    print(f"saved {args.out} rows={len(hourly)} {hourly.index.min()} ~ {hourly.index.max()} | 결측 시간 {gaps}")
    if gaps > 0.02 * len(hourly):
        print("경고: 결측 시간이 2%를 넘습니다", file=sys.stderr)


if __name__ == "__main__":
    main()
