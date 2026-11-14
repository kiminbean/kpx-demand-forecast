#!/usr/bin/env python3
"""
제주 태양광·풍력 공개 시계열의 포괄 범위 점검
============================================

공개 거래량(data.go.kr 15065269)의 제주 `풍력`이 2024-06과 2025-03에, `태양광`이 2024-06과 2025-09에
계단식으로 줄어든 이유를 KPX 공개 자료끼리 대조해 확인한다. 모델 학습·평가는 하지 않는다.

입력 (data/raw/kpx/, 모두 data.go.kr에서 로그인 없이 받는 파일)
  15161308  제주발전기 연료원 시간대별 전력거래량 2010~2025 (ZIP) — 연료코드 53 `급전가능 재생에너지` 포함
  15157963  비중앙발전기 연료원별 일별 시간대별 전력거래량 2020~2025 — 설비용량(MW) 열 포함 (전국)
  제주계통운영정보 5분값 (scripts/download_kpx_jeju_system.py) — 입찰자원을 포함한 태양광·풍력

출력: 표준 출력 + models/coverage_analysis/coverage_summary.json

사용법
  python scripts/analyze_jeju_coverage.py
"""

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FUEL_ZIP = ROOT / "data/raw/kpx/한국전력거래소_제주발전기 연료원 시간대별 전력거래량_2010_2025_20261114.zip"
NONCENTRAL_CSV = ROOT / "data/raw/kpx/한국전력거래소_비중앙발전기 연료원별 일별 시간대별 전력거래량_15157963_20261114.csv"
SYSTEM_RAW_DIR = ROOT / "data/raw/kpx/jeju_system_5min"
SYSTEM_HOURLY = ROOT / "data/processed/kpx_jeju_system_hourly.csv"
OUT_DIR = ROOT / "models" / "coverage_analysis"

FUELS = {"태양광": "pv", "풍력": "wind", "급전가능 재생에너지": "disp"}
YEARS = range(2020, 2026)
NIGHT_HOURS = (22, 23, 0, 1, 2, 3, 4)       # 구간 끝 시각 기준. 해가 없는 시간 → 코드 53은 사실상 입찰 풍력
CAPACITY_STEP_MW = 15.0                      # 이보다 크게 바뀐 날만 '계단'으로 본다
LAG_GRID_MIN = range(0, 65, 5)


def _member_name(info: zipfile.ZipInfo) -> str:
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def load_fuel_hourly(years=YEARS) -> pd.DataFrame:
    """제주 연료원별 시간 거래량 (구간 끝 시각 인덱스, 열 pv·wind·disp, MWh)"""
    parts = []
    with zipfile.ZipFile(FUEL_ZIP) as z:
        members = {_member_name(i): i for i in z.infolist()}
        for year in years:
            info = next(i for n, i in members.items() if f"{year}년도" in n)
            df = pd.read_csv(io.StringIO(z.read(info).decode("utf-8-sig")))
            df = df[df["FUEL_NM"].isin(FUELS)]
            df["t"] = pd.to_datetime(df["TRAD_YMD"]) + pd.to_timedelta(df["TRAD_HH"].astype(int), unit="h")
            parts.append(df.pivot_table(index="t", columns="FUEL_NM", values="AMGO", aggfunc="sum"))
    wide = pd.concat(parts).rename(columns=FUELS)
    return wide.reindex(columns=list(FUELS.values())).fillna(0.0).sort_index()


def capacity_steps() -> Dict[str, List[dict]]:
    """전국 비중앙 발전기 설비용량이 크게 바뀐 날. `급전가능 재생에너지`는 제주 입찰자원뿐이다."""
    df = pd.read_csv(NONCENTRAL_CSV, encoding="cp949")
    df["거래일"] = pd.to_datetime(df["거래일"].astype(str))
    out: Dict[str, List[dict]] = {}
    daily = {f: df[df["연료원"] == f].groupby("거래일")["설비용량(MW)"].max() for f in FUELS}
    disp = daily["급전가능 재생에너지"]
    step_days = [disp.index.min()] + list(disp.index[disp.diff().abs() > CAPACITY_STEP_MW])
    for fuel, cap in daily.items():
        diff = cap.diff()
        out[FUELS[fuel]] = [{"date": f"{d:%Y-%m-%d}", "capacity_mw": round(float(cap.get(d, np.nan)), 2),
                             "change_mw": round(float(diff.get(d, np.nan)), 2)} for d in step_days]
    return out


def best_lag(system_5min: pd.Series, market_hourly: pd.Series, min_mw: float = 1.0) -> Dict[str, float]:
    """5분값을 몇 분 당겨 시간 평균했을 때 시장 거래량과 상관이 가장 높은가"""
    best = {"lag_min": 0, "corr": -1.0}
    for lag in LAG_GRID_MIN:
        s = system_5min.copy()
        s.index = s.index - pd.Timedelta(minutes=lag)
        hourly = s.groupby(s.index.ceil("h")).mean()
        j = pd.concat([market_hourly, hourly], axis=1, keys=["m", "s"]).dropna()
        j = j[j["m"] > min_mw]
        c = float(j.corr().iloc[0, 1])
        if c > best["corr"]:
            best = {"lag_min": lag, "corr": round(c, 4)}
    return best


def load_system_5min(year: int) -> pd.DataFrame:
    files = sorted(SYSTEM_RAW_DIR.glob(f"{year}-*.csv"))
    raw = pd.concat([pd.read_csv(f, dtype={"기준일시": str}) for f in files])
    raw["t"] = pd.to_datetime(raw["기준일시"], format="%Y%m%d%H%M%S")
    return raw.drop_duplicates("t").set_index("t").sort_index()


def main() -> None:
    pd.set_option("display.width", 200)
    fuel = load_fuel_hourly()
    day = (fuel.index - pd.Timedelta(hours=1)).normalize()
    summary: dict = {}

    annual = fuel.groupby(day.year).sum().round(0)
    print("== 제주 연간 거래량 (MWh)\n", annual.loc[2023:].to_string())
    summary["annual_mwh"] = {int(y): r.to_dict() for y, r in annual.loc[2023:].iterrows()}
    first = fuel.index[fuel["disp"] > 0].min()
    print(f"== 연료코드 53 첫 시각: {first} (구간 끝)")
    summary["dispatchable_first_hour"] = str(first)

    night = fuel[np.isin(fuel.index.hour, NIGHT_HOURS)]
    nm = night.groupby((night.index - pd.Timedelta(hours=1)).to_period("M")).sum()
    share = (nm["wind"] / (nm["wind"] + nm["disp"])).round(3)
    print("== 야간 공개 `풍력` ÷ (풍력 + 코드 53), 월별\n", share.loc["2024-01":].to_string())
    summary["night_public_wind_share"] = {str(k): float(v) for k, v in share.loc["2024-01":].items()}

    steps = capacity_steps()
    print("== 설비용량 계단 (전국 비중앙 발전기 자료, MW)")
    for k, rows in steps.items():
        print(f"  {k}: " + ", ".join(f"{r['date']} {r['capacity_mw']} ({r['change_mw']:+.2f})" for r in rows))
    summary["capacity_steps"] = steps

    sysh = pd.read_csv(SYSTEM_HOURLY, parse_dates=["datetime"], index_col="datetime")
    j = fuel.join(sysh[["solar_raw_mw", "wind_mw"]], how="inner").dropna()
    q = (j.index - pd.Timedelta(hours=1)).to_period("Q")
    g = j.groupby(q).sum()
    jn = j[np.isin(j.index.hour, NIGHT_HOURS)]
    gn = jn.groupby((jn.index - pd.Timedelta(hours=1)).to_period("Q")).sum()
    ratios = pd.DataFrame({
        "시장 합계/계통 합계": (g["pv"] + g["wind"] + g["disp"]) / (g["solar_raw_mw"] + g["wind_mw"]),
        "공개 태양광/계통 태양광": g["pv"] / g["solar_raw_mw"],
        "공개 풍력/계통 풍력": g["wind"] / g["wind_mw"],
        "야간 (풍력+53)/계통 풍력": (gn["wind"] + gn["disp"]) / gn["wind_mw"],
    }).round(3)
    print("== 공개 거래량 ÷ 제주계통운영정보, 분기별\n", ratios.to_string())
    summary["market_over_system"] = {str(k): r.to_dict() for k, r in ratios.iterrows()}

    lags = {}
    for year in YEARS:
        raw = load_system_5min(year)
        mk = fuel[(fuel.index - pd.Timedelta(hours=1)).year == year]
        lags[year] = {"solar": best_lag(raw["신재생태양광(MW)"], mk["pv"]),
                      "wind": best_lag(raw["신재생풍력(MW)"], mk["wind"] + mk["disp"], min_mw=5.0)}
        print(f"== {year} 계통 표시 지연(분): 태양광 {lags[year]['solar']}, 풍력 {lags[year]['wind']}")
    summary["system_display_lag"] = lags

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")
    print(f"[저장] {OUT_DIR / 'coverage_summary.json'}")


if __name__ == "__main__":
    main()
