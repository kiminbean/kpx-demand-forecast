#!/usr/bin/env python3
"""
태양광 하루 전 예측 재검증 — 마감 전 공개가 확인된 예보 입력 (사전등록 2026-11-14)
==================================================================================

사전등록 문서: docs/prereg_solar_dayahead_verified_2026-11-14.md (결과를 보기 전에 작성).
이 스크립트는 그 문서의 비교를 그대로 실행한다. 운영 모델·API는 건드리지 않는다.

입력 예보
  gfs10 : NOAA GFS D-2 18 UTC 초기화 — AWS 업로드 시각이 D-1 10:00 KST 이전임을 파일 단위로 확인
  gfs17 : NOAA GFS D-1 00 UTC 초기화 — 업로드 시각이 D-1 17:00 KST 이전 (10시 마감에는 못 씀)
  ifs10 : ECMWF IFS D-2 12 UTC 초기화 (공개 자료 0.25°, 3시간 간격) — 업로드 시각 확인, 10시 마감 전
  ifs17 : ECMWF IFS D-2 18 UTC 초기화 — 17시 제출용
  jma2  : Open-Meteo JMA MSM previous_day2 — 48시간 전 예보. 파일 단위 공개 시각 기록은 없다
  obs   : ASOS 관측 — 하루 전 성능의 상한

라벨
  public : KPX 공개 거래량 `태양광` (15065269 / 15161308). 입찰제도 등록 자원이 빠져 2024-06-01,
           2024-09-01, 2025-03-01, 2025-09-01에 포괄 범위가 줄었다
  system : KPX 제주계통운영정보 태양광 5분값의 시간 평균. 입찰자원을 포함해 단절이 없고 2026년 자료가 있다

단계 (선택 → 확인)
  A   선택  public  2024-05-01 ~ 2024-11-07 06:00
  B   확인  public  2025년            (A의 판정을 따른다)
  C   확인  public  2023년, 학습 2022-01-02~   (A의 판정을 따른다. 이 프로젝트에서 처음 쓰는 구간)
  As  선택  system  A와 같은 구간
  D   확인  system  2026-01-01 ~ 10-31 (As의 판정을 따른다. 처음 쓰는 구간, 라벨 단절 없음)
  2024-11-07 ~ 12-31 (소진된 테스트 구간)은 어느 단계에서도 평가하지 않는다 (확인 단계의 학습 자료로만 쓴다).
  확인 단계는 1회만 실행된다.

사용법
  python scripts/experiment_solar_dayahead_verified.py --stage A
  python scripts/experiment_solar_dayahead_verified.py --stage B --dry-run   # 일정·자료만 점검
  python scripts/experiment_solar_dayahead_verified.py --stage B
"""

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_solar_v2 as T  # noqa: E402
from scripts.download_asos_hourly import COLMAP as ASOS_COLMAP  # noqa: E402
from src.evaluation import evaluate_generation  # noqa: E402
from src.evaluation.metrics import KPX_MIN_CAPACITY_FACTOR  # noqa: E402
from src.features.solar_features_v2 import (  # noqa: E402
    ALL_STATIONS,
    RAD_STATIONS,
    STATIONS,
    WEATHER_VARS,
    build_solar_frame,
    create_island_mean_features,
    create_solar_features_v2,
    generation_level,
    load_kpx_hourly,
    physics_features,
    to_island_mean,
)
from src.models.solar_v2 import LEVEL_COL, SolarV2Model  # noqa: E402

KPX_2024 = T.KPX_RAW
KPX_2025 = ROOT / "data/raw/kpx/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20251231_20261114_agent.csv"
KPX_FUEL_ZIP = ROOT / "data/raw/kpx/한국전력거래소_제주발전기 연료원 시간대별 전력거래량_2010_2025_20261114.zip"
KPX_SYSTEM_HOURLY = ROOT / "data/processed/kpx_jeju_system_hourly.csv"
WEATHER_DIR = ROOT / "data" / "raw" / "weather"
OBS_YEARS = range(2022, 2027)
NWP_DIR = ROOT / "data" / "external" / "nwp"
# 파일 단위로 업로드 시각을 확인한 예보 (tidy CSV, deadline_ok 열 포함)
FORECAST_CSV = {"gfs10": NWP_DIR / "gfs_aws_d1_10kst.csv", "gfs17": NWP_DIR / "gfs_aws_d1_17kst.csv",
                "ifs10": NWP_DIR / "ecmwf_aws_d1_10kst.csv", "ifs17": NWP_DIR / "ecmwf_aws_d1_17kst.csv"}
JMA_CSVS = [NWP_DIR / "jma_msm_2022_2023" / "openmeteo_previous_runs_jma_msm.csv", T.NWP_CSV,
            NWP_DIR / "jma_msm_2025" / "openmeteo_previous_runs_jma_msm.csv",
            NWP_DIR / "jma_msm_2026" / "openmeteo_previous_runs_jma_msm.csv"]
OUT_DIR = ROOT / "models" / "dayahead_verified"
FUEL_ZIP_YEARS = (2022, 2023)                   # public 라벨의 2024년 이전 구간 (연료코드 13 태양광)
FUEL_NAME_SOLAR = "태양광"

TEST_START = T.TEST_START                       # 2024-11-07 06:00 — 선택 단계는 이 뒤 자료를 읽지 않는다
DEADLINE_HOUR_KST = 10                          # 학습 라벨 컷오프는 모든 시나리오에서 D-1 10:00 (예보만 달라지게)
OBS_LAG_DAYS = 1                                # MOS 적합에 쓰는 ASOS 관측은 D-2 24:00까지 (data.go.kr API는 전일까지 제공)

WEEKLY_STEP_DAYS = 7
MOS_MIN_PAIRS = 100
MOS_QUANTILES = np.linspace(0.02, 0.98, 25)
CLOUD_MAX = 10.0
BLOCK_DAYS = 7
N_BOOT = 2000
BOOT_SEED = 42
DONE_MARKER = "DONE.json"

# 사전등록 채택 기준 (선택 단계): 풀링 FER 0.25%p 이상 개선 + 6개 폴드 중 4개 이상 개선
ADOPT_MIN_GAIN_PP = 0.25
ADOPT_MIN_FOLDS = 4


@dataclass(frozen=True)
class Stage:
    """평가 단계. 구간은 구간 시작 시각 기준 [eval_start, eval_end)."""
    key: str
    label: str                     # public | system
    train_start: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    selection: bool                # True면 eval_end 뒤 자료를 싣지 않는다
    decided_by: Optional[str] = None
    has_ifs: bool = True           # ECMWF 공개 자료(0.25°)는 2024-02부터 있다
    # public 라벨 포괄 범위가 바뀐 날 (구간별 보고용). 입찰제도 거래기간 경계
    breaks: Tuple[str, ...] = ()


STAGES: Dict[str, Stage] = {
    "A": Stage("A", "public", T.TRAIN_START, pd.Timestamp(T.CV_FOLDS[0][0]), TEST_START, True),
    "As": Stage("As", "system", T.TRAIN_START, pd.Timestamp(T.CV_FOLDS[0][0]), TEST_START, True),
    "B": Stage("B", "public", T.TRAIN_START, pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), False, "A",
               breaks=("2025-03-01", "2025-09-01")),
    "C": Stage("C", "public", pd.Timestamp("2022-01-02"), pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"),
               False, "A", has_ifs=False),
    "D": Stage("D", "system", T.TRAIN_START, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-01"), False, "As"),
}
# 분모(MW). A·B는 이전 라운드와 같은 값. 나머지는 같은 규칙으로 자료에서 정한다 (capacities 참고)
CAP_A = {"selection_384.87": T.CAPACITY_SELECTION}
CAP_B = {"proxy_max_2025": 320.6, "eff_359_2024": 359.4, "market_449.5": 449.5}


@dataclass(frozen=True)
class Config:
    """실험 1건. name은 출력 파일 이름으로도 쓴다."""
    name: str
    variant: str                 # island_mean | island_level | stations_level
    weather: str                 # gfs10 | gfs17 | ifs10 | ifs17 | jma2 | obs
    cadence: str                 # monthly | weekly | frozen
    label_lag_days: int = 0      # 학습 라벨 공개 지연 (일)
    level_lag_days: int = 2      # lvl_p에 쓰는 발전량 지연 (일)
    mos: str = ""                # '+'로 이은 보정 (rad_linear, rad_linear_cloud, rad_quantile, cloud_linear)


def _cfg(variant: str, weather: str, cadence: str, **kw) -> Config:
    short = {"island_mean": "im", "island_level": "il", "stations_level": "sl"}[variant]
    name = f"{short}_{weather}_{cadence}"
    if kw.get("label_lag_days"):
        name += f"_lab{kw['label_lag_days']}"
    if kw.get("level_lag_days", 2) != 2:
        name += f"_lvl{kw['level_lag_days']}"
    if kw.get("mos"):
        name += f"_mos-{kw['mos']}"
    return Config(name=name, variant=variant, weather=weather, cadence=cadence, **kw)


MOS_CANDIDATES = ["rad_linear", "rad_linear_cloud", "rad_quantile", "cloud_linear", "rad_linear_cloud+cloud_linear"]
BASELINE = "im_gfs10_monthly"

# 예측 평균(다중 예보). 구성원이 모두 있을 때만 계산한다
ENSEMBLES: Dict[str, List[str]] = {
    "ens_gfs10_jma2": ["im_gfs10_monthly", "im_jma2_monthly"],
    "ens_gfs10_ifs10": ["im_gfs10_monthly", "im_ifs10_monthly"],
    "ens_gfs10_ifs10_jma2": ["im_gfs10_monthly", "im_ifs10_monthly", "im_jma2_monthly"],
    "ens_gfs17_ifs17": ["im_gfs17_monthly", "im_ifs17_monthly"],
}

SELECTION_CONFIGS: List[Config] = (
    [_cfg("island_mean", w, "monthly") for w in ("gfs10", "gfs17", "ifs10", "ifs17", "jma2", "obs")]
    + [_cfg(v, "gfs10", "monthly") for v in ("island_level", "stations_level")]
    + [_cfg(v, "gfs10", c) for v in ("island_mean", "island_level", "stations_level") for c in ("frozen", "weekly")]
    + [_cfg("island_mean", "gfs10", "monthly", mos=m) for m in MOS_CANDIDATES]
    + [_cfg("island_mean", "gfs10", "monthly", label_lag_days=d) for d in (2, 7, 35)]
    + [_cfg("island_level", "gfs10", "monthly", level_lag_days=d) for d in (3, 7)]
)

CONFIRM_CONFIGS: List[Config] = (
    [_cfg("island_mean", "gfs10", c) for c in ("monthly", "frozen", "weekly")]
    + [_cfg(v, "gfs10", c) for v in ("island_level", "stations_level") for c in ("monthly", "frozen")]
    + [_cfg("island_mean", w, "monthly") for w in ("gfs17", "ifs10", "ifs17", "jma2")]
    + [_cfg(v, "obs", c) for v in ("island_mean", "island_level", "stations_level") for c in ("monthly", "frozen")]
    + [_cfg("island_mean", "gfs10", "monthly", label_lag_days=d) for d in (7, 35)]
    + [_cfg("island_level", "gfs10", "monthly", level_lag_days=d) for d in (3, 7)]
)

# (이름, a, b) — 차이 = a − b (음수면 a가 낫다). 확인 단계 비교 목록
CONFIRM_COMPARISONS: List[Tuple[str, str, str]] = [
    ("재학습: 월별 − 고정", "im_gfs10_monthly", "im_gfs10_frozen"),
    ("재학습: 주별 − 월별", "im_gfs10_weekly", "im_gfs10_monthly"),
    ("발전 수준(섬 평균): 월별", "il_gfs10_monthly", "im_gfs10_monthly"),
    ("발전 수준(관측소별): 월별", "sl_gfs10_monthly", "im_gfs10_monthly"),
    ("발전 수준(섬 평균): 고정", "il_gfs10_frozen", "im_gfs10_frozen"),
    ("발전 수준(섬 평균), 관측 기상: 월별", "il_obs_monthly", "im_obs_monthly"),
    ("발전 수준(섬 평균), 관측 기상: 고정", "il_obs_frozen", "im_obs_frozen"),
    ("17시 제출 − 10시 제출", "im_gfs17_monthly", "im_gfs10_monthly"),
    ("GFS − JMA", "im_gfs10_monthly", "im_jma2_monthly"),
    ("GFS+JMA 평균 − GFS", "ens_gfs10_jma2", "im_gfs10_monthly"),
    ("GFS+JMA 평균 − JMA", "ens_gfs10_jma2", "im_jma2_monthly"),
    ("IFS − GFS", "im_ifs10_monthly", "im_gfs10_monthly"),
    ("GFS+IFS 평균 − GFS", "ens_gfs10_ifs10", "im_gfs10_monthly"),
    ("GFS+IFS+JMA 평균 − GFS", "ens_gfs10_ifs10_jma2", "im_gfs10_monthly"),
    ("17시 다중 예보 − 10시 다중 예보", "ens_gfs17_ifs17", "ens_gfs10_ifs10"),
    ("예보 오차 비용: GFS − 관측", "im_gfs10_monthly", "im_obs_monthly"),
    ("관측 기상 재학습: 월별 − 고정", "im_obs_monthly", "im_obs_frozen"),
]


# ---------------------------------------------------------------------------
# 자료
# ---------------------------------------------------------------------------

def _zip_member_name(info: zipfile.ZipInfo) -> str:
    try:
        return info.filename.encode("cp437").decode("cp949")   # 한글 파일명이 cp949로 들어 있다
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def load_fuel_zip_solar(years: Tuple[int, ...] = FUEL_ZIP_YEARS, path: Path = KPX_FUEL_ZIP) -> pd.Series:
    """제주 연료원별 시간대별 전력거래량(15161308)에서 `태양광`만 — 구간 끝 라벨"""
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — data.go.kr 15161308에서 내려받으세요")
    parts = []
    with zipfile.ZipFile(path) as z:
        members = {_zip_member_name(i): i for i in z.infolist()}
        for year in years:
            hit = [i for n, i in members.items() if f"{year}년도" in n]
            if len(hit) != 1:
                raise ValueError(f"{path.name}에 {year}년 파일이 없습니다")
            df = pd.read_csv(io.StringIO(z.read(hit[0]).decode("utf-8-sig")))
            df = df[df["FUEL_NM"].astype(str).str.strip() == FUEL_NAME_SOLAR]
            t_end = pd.to_datetime(df["TRAD_YMD"]) + pd.to_timedelta(df["TRAD_HH"].astype(int), unit="h")
            parts.append(pd.Series(df["AMGO"].astype(float).values, index=pd.DatetimeIndex(t_end)))
    return pd.concat(parts).sort_index().rename("generation_mw")


def load_generation(label: str) -> pd.Series:
    """구간 끝 시각 T 인덱스의 시간 발전량. 결측 시간은 뺀다."""
    if label == "public":
        gen = pd.concat([load_fuel_zip_solar(), load_kpx_hourly(KPX_2024), load_kpx_hourly(KPX_2025)]).sort_index()
    elif label == "system":
        if not KPX_SYSTEM_HOURLY.exists():
            raise FileNotFoundError(f"{KPX_SYSTEM_HOURLY} 없음 — 먼저 python scripts/download_kpx_jeju_system.py 실행")
        df = pd.read_csv(KPX_SYSTEM_HOURLY, parse_dates=["datetime"], index_col="datetime")
        gen = df["solar_mw"].dropna()
    else:
        raise ValueError(f"알 수 없는 라벨: {label}")
    if not gen.index.is_unique:
        raise ValueError(f"{label} 발전량 시계열에 중복 시각이 있습니다")
    return gen.rename("generation_mw")


def _load_asos_year(path: Path) -> pd.DataFrame:
    """연도별 ASOS 파일 → wide. 2020~2023년 파일은 API 원래 열 이름이라 바꿔 읽는다."""
    w = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "datetime" not in w.columns:
        w = w.drop(columns=[c for c in ("station_id", "station_name") if c in w.columns]).rename(columns=ASOS_COLMAP)
    w["datetime"] = pd.to_datetime(w["datetime"])
    missing = [c for c in ["datetime", "station_id"] + WEATHER_VARS if c not in w.columns]
    if missing:
        raise ValueError(f"{path.name}에 열이 없습니다: {missing}")
    w["st"] = w["station_id"].astype(int).map(STATIONS)
    if w["st"].isna().any():
        raise ValueError(f"{path.name}: 알 수 없는 관측소")
    for v in WEATHER_VARS:
        w[v] = pd.to_numeric(w[v], errors="coerce")
    wide = w.pivot(index="datetime", columns="st", values=WEATHER_VARS)
    wide.columns = [f"{v}_{st}" for v, st in wide.columns]
    return wide


def load_observed() -> pd.DataFrame:
    parts = [_load_asos_year(WEATHER_DIR / f"jeju_weather_{y}_all_stations.csv") for y in OBS_YEARS
             if (WEATHER_DIR / f"jeju_weather_{y}_all_stations.csv").exists()]
    wx = pd.concat(parts)
    wx = wx[~wx.index.duplicated()].sort_index()
    return wx.reindex(pd.date_range(wx.index.min(), wx.index.max(), freq="h"))


def _pad_weather(w: pd.DataFrame) -> pd.DataFrame:
    """예보에 없는 변수 열을 NaN으로 채우고 1시간 간격으로 재색인"""
    for v in WEATHER_VARS:
        for st in ALL_STATIONS:
            if f"{v}_{st}" not in w:
                w[f"{v}_{st}"] = np.nan
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))


def load_verified_forecast(source: str) -> pd.DataFrame:
    """GFS·IFS tidy → wide. 마감 뒤에 올라온 값은 결측으로 둔다 (그 날은 평가에서 빠진다)."""
    path = FORECAST_CSV[source]
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 scripts/download_nwp_gfs_aws.py / download_nwp_ecmwf_aws.py 실행")
    df = pd.read_csv(path, parse_dates=["datetime"])
    cols = ["temperature", "humidity", "solar_radiation", "cloud_cover"]
    df.loc[~df["deadline_ok"].astype(bool), cols] = np.nan
    w = df.pivot(index="datetime", columns="station", values=cols)
    w.columns = [f"{v}_{st}" for v, st in w.columns]
    return _pad_weather(w)


def load_jma(lead_days: int = 2) -> pd.DataFrame:
    parts = [T.load_nwp_weather(lead_days, p) for p in JMA_CSVS if p.exists()]
    if not parts:
        raise FileNotFoundError("JMA 예보 파일이 없습니다 — python scripts/download_nwp_openmeteo.py")
    w = pd.concat(parts)
    w = w[~w.index.duplicated(keep="first")].sort_index()
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))


def load_weather(source: str) -> pd.DataFrame:
    if source in FORECAST_CSV:
        return load_verified_forecast(source)
    if source == "jma2":
        return load_jma(2)
    if source == "obs":
        return load_observed()
    raise ValueError(f"알 수 없는 기상 입력: {source}")


def build_features(weather: pd.DataFrame, gen: pd.Series, variant: str,
                   level_lag_days: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(frame, X, y). 인덱스는 구간 끝 T."""
    frame = build_solar_frame(weather, generation=gen)
    if variant == "island_mean":
        X = create_island_mean_features(frame)
    elif variant == "island_level":
        # API와 같은 섬 평균 기상 1세트 + 발전 수준. 수준의 분모도 섬 평균 물리 출력대용값으로 만든다
        X = create_island_mean_features(frame)
        p_mean = physics_features(to_island_mean(frame))["p_mean"]
        X = pd.concat([X, generation_level(frame, frame["generation_mw"], p_mean, level_lag_days)], axis=1)
    elif variant == "stations_level":
        X = create_solar_features_v2(frame)
        lvl = generation_level(frame, frame["generation_mw"], physics_features(frame)["p_mean"], level_lag_days)
        X = pd.concat([X, lvl], axis=1)
    else:
        raise ValueError(f"알 수 없는 변형: {variant}")
    return frame, X, frame["generation_mw"]


# ---------------------------------------------------------------------------
# 일정 · 마스크
# ---------------------------------------------------------------------------

def origins_for(stage: Stage, cadence: str) -> List[pd.Timestamp]:
    """재학습 시점(거래일 00:00) 목록. 각 시점의 모델이 다음 시점 직전까지 예측한다."""
    start, end = stage.eval_start, stage.eval_end
    if cadence == "frozen":
        return [start]
    if cadence == "monthly":
        if stage.selection:  # 기존 CV 폴드와 똑같이: 마지막 폴드(10-01 학습)가 11월 첫 주까지 예측한다
            return [pd.Timestamp(fs) for fs, _ in T.CV_FOLDS]
        return [d for d in pd.date_range(start, end, freq="MS") if d < end]
    if cadence == "weekly":
        return list(pd.date_range(start, end - pd.Timedelta(seconds=1), freq=f"{WEEKLY_STEP_DAYS}D"))
    raise ValueError(f"알 수 없는 재학습 주기: {cadence}")


def training_mask(index: pd.DatetimeIndex, origin: pd.Timestamp, train_start: pd.Timestamp,
                  label_lag_days: int = 0) -> np.ndarray:
    """origin(첫 거래일) 전날 10시에서 공개 지연만큼 더 이전에 끝난 구간만 학습에 쓴다"""
    deadline = (origin.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=DEADLINE_HOUR_KST)
                - pd.Timedelta(days=label_lag_days))
    return np.asarray((index - pd.Timedelta(hours=1) >= train_start) & (index <= deadline))


# ---------------------------------------------------------------------------
# 예보 보정 (MOS) — 시점마다 그때까지의 관측으로만 적합
# ---------------------------------------------------------------------------

def fit_mos(fc: pd.DataFrame, obs: pd.DataFrame, parts: List[str], fit_start: pd.Timestamp,
            fit_end: pd.Timestamp) -> Dict[str, dict]:
    """예보→관측 매핑 계수. 표본이 모자란 관측소는 보정하지 않는다."""
    idx = fc.index[(fc.index - pd.Timedelta(hours=1) >= fit_start) & (fc.index <= fit_end)]
    o = obs.reindex(idx)
    coeffs: Dict[str, dict] = {}
    for part in parts:
        if part.startswith("rad_"):
            for st in RAD_STATIONS:
                f, y = fc.loc[idx, f"solar_radiation_{st}"], o[f"solar_radiation_{st}"]
                c = fc.loc[idx, f"cloud_cover_{st}"]
                ok = (np.isfinite(f) & np.isfinite(y) & (np.isfinite(c) if part == "rad_linear_cloud" else True)).values
                if ok.sum() < MOS_MIN_PAIRS:
                    continue
                fv, yv = f.values[ok], y.values[ok]
                if part == "rad_linear":
                    a, b = np.polyfit(fv, yv, 1)
                    coeffs[f"solar_radiation_{st}"] = {"type": "linear", "a": float(a), "b": float(b)}
                elif part == "rad_linear_cloud":
                    A = np.column_stack([fv, c.values[ok], np.ones(ok.sum())])
                    (a, cc, b), *_ = np.linalg.lstsq(A, yv, rcond=None)
                    coeffs[f"solar_radiation_{st}"] = {"type": "linear_cloud", "a": float(a), "c": float(cc),
                                                       "b": float(b), "cloud": f"cloud_cover_{st}"}
                elif part == "rad_quantile":
                    fq, keep = np.unique(np.quantile(fv, MOS_QUANTILES), return_index=True)
                    coeffs[f"solar_radiation_{st}"] = {"type": "quantile", "fq": fq.tolist(),
                                                       "oq": np.quantile(yv, MOS_QUANTILES)[keep].tolist(),
                                                       "max": float(yv.max())}
                else:
                    raise ValueError(f"알 수 없는 보정: {part}")
        elif part == "cloud_linear":
            for st in ALL_STATIONS:
                f, y = fc.loc[idx, f"cloud_cover_{st}"], o[f"cloud_cover_{st}"]
                ok = (np.isfinite(f) & np.isfinite(y)).values
                if ok.sum() < MOS_MIN_PAIRS:
                    continue
                a, b = np.polyfit(f.values[ok], y.values[ok], 1)
                coeffs[f"cloud_cover_{st}"] = {"type": "cloud_linear", "a": float(a), "b": float(b)}
        else:
            raise ValueError(f"알 수 없는 보정: {part}")
    return coeffs


def apply_mos(fc: pd.DataFrame, coeffs: Dict[str, dict]) -> pd.DataFrame:
    """보정된 사본을 돌려준다. 일사 보정에 쓰는 운량은 보정 전 예보값이다."""
    out = fc.copy()
    for col, e in coeffs.items():
        v = fc[col].to_numpy(dtype=float)
        if e["type"] == "linear":
            cv = np.maximum(v * e["a"] + e["b"], 0.0)
        elif e["type"] == "linear_cloud":
            cv = np.maximum(v * e["a"] + fc[e["cloud"]].to_numpy(dtype=float) * e["c"] + e["b"], 0.0)
        elif e["type"] == "quantile":
            cv = np.clip(np.interp(v, e["fq"], e["oq"]), 0.0, e["max"])
        elif e["type"] == "cloud_linear":
            cv = np.clip(v * e["a"] + e["b"], 0.0, CLOUD_MAX)
        else:
            raise ValueError(f"알 수 없는 보정 유형: {e['type']}")
        out[col] = np.where(np.isnan(v), np.nan, cv)
    return out


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def _predict_rows(model: SolarV2Model, X: pd.DataFrame, frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """rows 위치의 예측. 낮인데 입력이 빠진 행(예보 결측일)은 NaN으로 남긴다."""
    Xe, dl = X[rows], frame["daylight_frac"][rows]
    day = dl.values > 0
    need = [c for c in model.feature_names if not model.nan_allowed.get(c, True) and c != LEVEL_COL]
    bad = day & Xe[need].isna().any(axis=1).values
    if "p_mean" in Xe.columns:
        bad |= day & ~np.isfinite(Xe["p_mean"].values)
    if LEVEL_COL in model.feature_names:
        lvl = Xe[LEVEL_COL].values
        bad |= day & ~(np.isfinite(lvl) & (lvl > 0))
    out = np.full(len(Xe), np.nan)
    if (~bad).any():
        out[~bad] = model.predict(Xe[~bad], frame["daylight_frac"])
    return out


def load_stage_inputs(stage: Stage, weather_source: str, need_obs: bool):
    """단계에 맞게 잘라 낸 (발전량, 예보, 관측). 선택 단계는 평가 구간 끝 뒤 자료를 싣지 않는다."""
    gen = load_generation(stage.label)
    weather = load_weather(weather_source)
    obs = load_observed() if need_obs else None
    cut = stage.eval_end
    gen = gen[(gen.index - pd.Timedelta(hours=1) >= stage.train_start - pd.Timedelta(days=1)) & (gen.index <= cut)]
    if stage.selection:
        weather = weather[weather.index <= cut]
        obs = None if obs is None else obs[obs.index <= cut]
    return gen, weather, obs


def run_config(cfg: Config, stage_key: str) -> pd.Series:
    """설정 1건의 walk-forward 예측 (인덱스 = 구간 끝 T, 평가 구간 전체)"""
    stage = STAGES[stage_key]
    gen, weather, obs = load_stage_inputs(stage, cfg.weather, bool(cfg.mos))
    origins = origins_for(stage, cfg.cadence)
    bounds = origins[1:] + [stage.eval_end]
    starts = gen.index - pd.Timedelta(hours=1)
    pred = pd.Series(np.nan, index=gen.index[(starts >= stage.eval_start) & (starts < stage.eval_end)])

    frame = X = y = model = None
    if not cfg.mos:
        frame, X, y = build_features(weather, gen, cfg.variant, cfg.level_lag_days)
    for origin, nxt in zip(origins, bounds):
        if cfg.mos:
            fit_end = origin.normalize() - pd.Timedelta(days=OBS_LAG_DAYS)
            coeffs = fit_mos(weather, obs, cfg.mos.split("+"), stage.train_start, fit_end)
            frame, X, y = build_features(apply_mos(weather, coeffs), gen, cfg.variant, cfg.level_lag_days)
        s = frame.index - pd.Timedelta(hours=1)
        tr = training_mask(frame.index, origin, stage.train_start, cfg.label_lag_days) & T.usable_rows(frame).values
        ev = np.asarray((s >= origin) & (s < nxt))
        if frame.index[tr].max() > origin - pd.Timedelta(days=1) + pd.Timedelta(hours=DEADLINE_HOUR_KST):
            raise RuntimeError(f"{cfg.name}: 학습 자료가 제출 마감을 넘습니다")
        if cfg.cadence != "frozen" or model is None:
            model = SolarV2Model().fit(X[tr], y[tr], frame["daylight_frac"][tr])
        pred[frame.index[ev]] = _predict_rows(model, X, frame, ev)
    if stage.selection and pred.index.max() > TEST_START:
        raise RuntimeError("선택 단계 예측이 테스트 구간에 닿았습니다")
    return pred


def _run_and_save(args: Tuple[Config, str]) -> Tuple[str, str]:
    cfg, stage_key = args
    out = OUT_DIR / stage_key / f"{cfg.name}.csv"
    if out.exists():
        return cfg.name, "cached"
    pred = run_config(cfg, stage_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.rename("pred_mw").to_csv(out, index_label="datetime")
    return cfg.name, "ok"


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def capacities(stage: Stage, y_eval: pd.Series) -> Tuple[Dict[str, float], str]:
    """(분모 목록, 주 분모 이름). A·B는 이전 라운드 값, 그 밖은 같은 규칙을 자료에 적용한다."""
    if stage.key == "A":
        return CAP_A, "selection_384.87"
    if stage.key == "B":
        return CAP_B, "proxy_max_2025"
    if stage.selection:
        # 선택용 대용값 = 테스트 전 구간의 최대 시간 발전량 (A의 384.87 MW와 같은 규칙)
        full = load_generation(stage.label)
        s = full.index - pd.Timedelta(hours=1)
        return {"proxy_max_pretest": float(full[(s >= stage.train_start) & (s < stage.eval_end)].max())}, "proxy_max_pretest"
    # 확인용 대용값 = 평가 구간의 최대 시간 발전량 (B의 320.6 MW와 같은 규칙)
    return {"proxy_max_eval": float(y_eval.max())}, "proxy_max_eval"


def paired_fer_diff(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, index: pd.DatetimeIndex, cap: float,
                    n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Dict[str, float]:
    """FER(a) − FER(b)와 7일 블록 부트스트랩 95% 신뢰구간 (평가 대상 시간 공통)"""
    m = y >= KPX_MIN_CAPACITY_FACTOR * cap
    d = (np.abs(np.clip(pa, 0, None) - y) - np.abs(np.clip(pb, 0, None) - y))[m] / cap * 100.0
    day = (index[m] - pd.Timedelta(hours=1)).normalize()
    block = ((day - day.min()).days // BLOCK_DAYS).values
    sums = np.bincount(block, weights=d)
    counts = np.bincount(block).astype(float)
    keep = counts > 0
    sums, counts = sums[keep], counts[keep]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff_pp": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_eval_hours": int(m.sum())}


def load_predictions(stage_key: str, names: List[str]) -> pd.DataFrame:
    cols = {}
    for n in names:
        p = OUT_DIR / stage_key / f"{n}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음")
        cols[n] = pd.read_csv(p, parse_dates=["datetime"], index_col="datetime")["pred_mw"]
    return pd.DataFrame(cols)


def summarize(stage_key: str, configs: List[Config], comparisons: List[Tuple[str, str, str]]) -> dict:
    """공통 평가 시간에서 설정별 FER·폴드별 FER, 비교, (선택 단계) 채택 판정"""
    stage = STAGES[stage_key]
    preds = load_predictions(stage_key, [c.name for c in configs])
    gen = load_generation(stage.label)
    y = gen.reindex(preds.index)
    if stage.selection and preds.index.max() > TEST_START:
        raise RuntimeError("선택 단계 요약에 테스트 구간이 들어 있습니다")
    for ens, members in ENSEMBLES.items():
        if all(m in preds for m in members):
            preds[ens] = preds[members].mean(axis=1, skipna=False)
    preds["persistence48"] = gen.shift(48, freq="h").reindex(preds.index)
    common = preds.notna().all(axis=1) & y.notna()
    P, y_s = preds[common], y[common]
    yv = y_s.values
    caps, head = capacities(stage, y_s)
    # 폴드 = 월별 재학습 시점 단위 (선택 단계 마지막 폴드는 기존 CV처럼 11월 첫 주를 포함한다)
    monthly = pd.DatetimeIndex(origins_for(stage, "monthly"))
    period = monthly[np.searchsorted(monthly, P.index - pd.Timedelta(hours=1), side="right") - 1].strftime("%Y-%m")
    out: dict = {"stage": stage_key, "label": stage.label, "window": [str(stage.eval_start), str(stage.eval_end)],
                 "train_start": str(stage.train_start),
                 "n_hours_total": int(len(preds)), "n_hours_common": int(common.sum()),
                 "dropped_days": sorted({str(d.date()) for d in (preds.index[~common] - pd.Timedelta(hours=1))}),
                 "capacities": caps, "headline_capacity": head, "configs": {}}
    for name in P.columns:
        res = {k: evaluate_generation(yv, P[name].values, c) for k, c in caps.items()}
        folds = {per: evaluate_generation(yv[period == per], P[name].values[period == per],
                                          caps[head])["fer_mean_pct"] for per in sorted(set(period))}
        out["configs"][name] = {"by_capacity": res, "folds_fer": folds}

    def cmp(a: str, b: str) -> dict:
        r = paired_fer_diff(yv, P[a].values, P[b].values, P.index, caps[head])
        fa, fb = out["configs"][a]["folds_fer"], out["configs"][b]["folds_fer"]
        r["folds_better"] = int(sum(fa[k] < fb[k] for k in fa))
        r["n_folds"] = len(fa)
        return r

    if stage.selection:
        pairs = {n: (n, BASELINE) for n in P.columns if n not in (BASELINE, "persistence48")}
        pairs["im_gfs10_weekly_vs_monthly"] = ("im_gfs10_weekly", BASELINE)
        pairs["il_gfs10_weekly_vs_il_monthly"] = ("il_gfs10_weekly", "il_gfs10_monthly")
        out["comparisons"] = {k: dict(cmp(a, b), a=a, b=b) for k, (a, b) in pairs.items()}

        def passes(key: str) -> bool:
            c = out["comparisons"][key]
            return bool(-c["diff_pp"] >= ADOPT_MIN_GAIN_PP and c["folds_better"] >= ADOPT_MIN_FOLDS)

        mos_names = [_cfg("island_mean", "gfs10", "monthly", mos=m).name for m in MOS_CANDIDATES]
        best_mos = min(mos_names, key=lambda n: out["configs"][n]["by_capacity"][head]["fer_mean_pct"])
        out["decisions"] = {
            "rule": f"풀링 FER {ADOPT_MIN_GAIN_PP}%p 이상 개선 + 폴드 {ADOPT_MIN_FOLDS}개 이상 개선",
            "level_island": passes("il_gfs10_monthly"),
            "level_stations": passes("sl_gfs10_monthly"),
            "weekly_over_monthly": passes("im_gfs10_weekly_vs_monthly"),
            "ensembles": {e: passes(e) for e in ENSEMBLES if e in P.columns},
            "mos_best": best_mos,
            "mos_adopt": passes(best_mos),
        }
    else:
        out["comparisons"] = {label: dict(cmp(a, b), a=a, b=b) for label, a, b in comparisons
                              if a in P.columns and b in P.columns}
        cap = caps[head]
        m = yv >= KPX_MIN_CAPACITY_FACTOR * cap
        if BASELINE in P and "im_gfs17_monthly" in P:
            # KPX는 10시·17시 두 번 제출한 예측의 오차율을 평균한다
            e10 = np.abs(np.clip(P[BASELINE].values, 0, None) - yv)[m]
            e17 = np.abs(np.clip(P["im_gfs17_monthly"].values, 0, None) - yv)[m]
            out["two_submission_mean_fer_pct"] = float(((e10 + e17) / 2).mean() / cap * 100.0)
        if stage.breaks:
            # 라벨 포괄 범위가 바뀐 날로 나눠, 구간마다 그 구간 최대 발전량을 분모로 다시 본다
            edges = [stage.eval_start] + [pd.Timestamp(b) for b in stage.breaks] + [stage.eval_end]
            s = P.index - pd.Timedelta(hours=1)
            out["segments"] = {}
            for lo, hi in zip(edges[:-1], edges[1:]):
                seg = np.asarray((s >= lo) & (s < hi))
                c_seg = float(yv[seg].max())
                out["segments"][f"{lo:%Y-%m-%d}~{hi:%Y-%m-%d}"] = {
                    "capacity_mw": c_seg,
                    "fer": {n: evaluate_generation(yv[seg], P[n].values[seg], c_seg)["fer_mean_pct"] for n in P.columns}}
    return out


def print_summary(s: dict) -> None:
    head = s["headline_capacity"]
    print(f"\n=== 단계 {s['stage']} ({s['label']}) {s['window'][0]} ~ {s['window'][1]} | 공통 평가 "
          f"{s['n_hours_common']}/{s['n_hours_total']}시간 | 제외 {len(s['dropped_days'])}일 | "
          f"분모 {head} = {s['capacities'][head]:.1f} MW ===")
    rows = sorted(s["configs"].items(), key=lambda kv: kv[1]["by_capacity"][head]["fer_mean_pct"])
    for name, r in rows:
        e = r["by_capacity"][head]
        print(f"  {name:48s} FER {e['fer_mean_pct']:6.3f}%  MAE {e['mae_mw']:6.2f}  MBE {e['mbe_mw']:+6.2f}  "
              f"n={e['n_eval_hours']} | " + " ".join(f"{v:.2f}" for v in r["folds_fer"].values()))
    print("  --- 비교 (a − b, %p, 95% CI) ---")
    for k, c in s["comparisons"].items():
        print(f"  {k:44s} {c['diff_pp']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] 폴드 {c['folds_better']}/{c['n_folds']}")
    if "decisions" in s:
        print("  --- 채택 판정 ---", json.dumps(s["decisions"], ensure_ascii=False))
    if "two_submission_mean_fer_pct" in s:
        print(f"  10시·17시 2회 제출 평균 FER {s['two_submission_mean_fer_pct']:.3f}%")
    for seg, r in s.get("segments", {}).items():
        top = " ".join(f"{n}:{v:.2f}" for n, v in sorted(r["fer"].items(), key=lambda kv: kv[1])[:6])
        print(f"  구간 {seg} (C={r['capacity_mw']:.1f}): {top}")


def stage_plan(stage_key: str) -> Tuple[List[Config], List[Tuple[str, str, str]]]:
    """단계에서 돌릴 설정과 비교 목록. 확인 단계는 선택 단계 판정에 따라 보정 후보 1개가 추가될 수 있다."""
    stage = STAGES[stage_key]

    def available(c: Config) -> bool:
        if c.weather.startswith("ifs") and not stage.has_ifs:
            return False
        path = FORECAST_CSV.get(c.weather)
        if path is not None and not path.exists():   # 아직 받지 못한 예보는 건너뛴다 (결과 문서에 적는다)
            print(f"  건너뜀: {c.name} — {path.name} 없음", flush=True)
            return False
        return True

    if stage.selection:
        return [c for c in SELECTION_CONFIGS if available(c)], []
    configs = [c for c in CONFIRM_CONFIGS if available(c)]
    comparisons = list(CONFIRM_COMPARISONS)
    dec_path = OUT_DIR / stage.decided_by / "summary.json"
    if not dec_path.exists():
        raise SystemExit(f"선택 단계 {stage.decided_by} 요약이 없습니다 — 먼저 --stage {stage.decided_by}를 실행하세요")
    dec = json.loads(dec_path.read_text(encoding="utf-8"))["decisions"]
    if dec["mos_adopt"]:  # 선택 단계에서 기준을 넘은 보정만 확인 단계에 올린다
        mos = dec["mos_best"].split("_mos-")[1]
        configs.append(_cfg("island_mean", "gfs10", "monthly", mos=mos))
        comparisons.append(("예보 보정(선택 단계 통과)", configs[-1].name, BASELINE))
    return configs, comparisons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="자료·일정만 점검하고 학습·평가는 하지 않는다")
    args = ap.parse_args()

    stage = STAGES[args.stage]
    out_dir = OUT_DIR / args.stage
    marker = out_dir / DONE_MARKER
    if not stage.selection and marker.exists():
        raise SystemExit(f"확인 단계 {args.stage}는 이미 실행됐습니다 ({marker}). 다시 돌리지 않습니다.")
    configs, comparisons = stage_plan(args.stage)

    if args.dry_run:
        print(f"단계 {args.stage} ({stage.label}): 학습 {stage.train_start} ~, 평가 {stage.eval_start} ~ {stage.eval_end}, "
              f"설정 {len(configs)}건")
        for cad in ("frozen", "monthly", "weekly"):
            o = origins_for(stage, cad)
            print(f"  {cad}: 재학습 {len(o)}회, 첫 {o[0]}, 끝 {o[-1]}")
        for src in sorted({c.weather for c in configs}):
            gen, w, _ = load_stage_inputs(stage, src, False)
            w = w[(w.index >= stage.train_start) & (w.index <= stage.eval_end)]
            rad = w[[f"solar_radiation_{st}" for st in ("jeju", "gosan")]].notna().all(axis=1)
            days = rad.groupby(w.index.normalize()).any()
            print(f"  {src}: {w.index.min()} ~ {w.index.max()}, 일사 있는 날 {int(days.sum())}/{len(days)}")
        print(f"  발전량({stage.label}): {gen.index.min()} ~ {gen.index.max()}, {len(gen)}시간")
        for c in configs:
            print("  ", asdict(c))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        for name, status in pool.map(_run_and_save, [(c, args.stage) for c in configs]):
            print(f"  {name}: {status}", flush=True)
    s = summarize(args.stage, configs, comparisons)
    (out_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(s)
    if not stage.selection:
        marker.write_text(json.dumps({"done_at": str(pd.Timestamp.now()), "configs": [c.name for c in configs]},
                                     ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
