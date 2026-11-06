"""
Solar Feature Engineering V2 - 구간 정렬 + 경사면 일사 물리 피처
==================================================================

V1(create_solar_features)에서 바뀐 점

1. 시간 정렬
   KPX 거래시간 h는 (h-1)~h시 발전량이고, ASOS 일사(t)는 (t-1)~t시 누적값이다.
   기존 병합 파일은 발전 구간 (t, t+1]을 일사 구간 (t-1, t]와 짝지어 1시간 어긋나 있었다
   (발전량-일사 상관 0.894 → 정렬 시 0.948). 표본을 발전 구간 끝 시각 T로 라벨링해 통일한다.
2. 태양 위치
   구간 중앙(T-30분), 균시차 포함 (NOAA/Spencer 근사). 구간 적분 일사(대기권 밖, 맑은하늘)는 5분 간격 적분.
3. 경사면 일사 물리 체인 (관측소별)
   Erbs 산란비 → Hay–Davies 경사면 일사(경사 25°, 남/동남/남서향) → NOCT 셀온도 보정 출력대용값.
   수평 일사 1 MJ/m²당 발전량이 1월 약 145 MW에서 7월 약 64 MW로 계절 따라 절반이 되는 기하 효과를
   피처가 직접 설명해, 학습 범위 밖 계절로의 외삽 오류(6월 과대예측 등)를 줄인다.

입력 단위: 일사 MJ/m² (구간 누적), 기온 °C, 운량 0~10, 습도 %.
"""

import warnings
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# 관측소: ASOS 지점번호 → 이름 (일사: jeju·gosan 연중, seogwipo 2024-05 이후, seongsan 미관측)
STATIONS: Dict[int, str] = {184: "jeju", 185: "gosan", 188: "seongsan", 189: "seogwipo"}
ALL_STATIONS: List[str] = ["jeju", "gosan", "seongsan", "seogwipo"]
RAD_STATIONS: List[str] = ["jeju", "gosan", "seogwipo"]
WEATHER_VARS: List[str] = ["temperature", "humidity", "wind_speed", "wind_direction", "pressure",
                           "solar_radiation", "cloud_cover", "precipitation", "visibility"]

SITE_LAT, SITE_LON = 33.38, 126.55  # 제주도 중심 근사
TZ_HOURS = 9  # KST

SOLAR_CONSTANT = 1361.0          # W/m²
MJ_TO_W = 1e6 / 3600.0           # 시간 누적 MJ/m² → 평균 W/m²
INTEGRATION_STEPS = 12           # 구간 적분 간격 5분

# 물리 체인 상수 (문헌값, 데이터로 맞추지 않음)
PANEL_TILT_DEG = 25.0            # 제주 태양광 일반 경사 20~30°의 중간
PLANE_AZIMUTHS = {"": 180.0, "_e": 135.0, "_w": 225.0}  # 남향 + 동남/남서향 (fleet 방위 혼재 → 오전·오후 비대칭)
ALBEDO = 0.2
NOCT_C = 45.0
TEMP_COEF = -0.004               # 결정질 실리콘 출력 온도계수 (/°C)
MIN_DEN_MJ = 0.1                 # kt·kc 분모 하한 (저고도 폭주 방지)
COSZ_MIN = 0.087                 # 천정각 85° 제한
RB_MAX = 5.0
KT_MAX, KC_MAX = 1.2, 1.5        # 구름 가장자리 증폭 허용 상한
P_CS_MIN = 30.0                  # 맑은하늘 출력 대비 비율을 정의할 최소 맑은하늘 출력 (W/m²)

COMPACT_FEATURES: List[str] = [
    "p_mean", "p_jeju", "p_gosan", "p_seogwipo", "poa_mean", "p_cs", "p_ratio_cs",
    "kt_mean", "kc_mean", "fd_mean", "tc_mean", "p_mean_e", "p_mean_w",
    "H0", "cosz_mid", "zenith_mid", "azimuth_mid",
    "cloud_mean", "humidity_mean", "temp_mean", "dayofyear", "doy_sin", "doy_cos",
]


# ---------------------------------------------------------------------------
# 데이터 로드 (구간 끝 라벨)
# ---------------------------------------------------------------------------

def load_kpx_hourly(path, region: str = "제주도", fuel: str = "태양광") -> pd.Series:
    """KPX '지역별 시간별 태양광 및 풍력 발전량' CSV → 구간 끝 시각 T 인덱스의 시간 거래량(MWh)

    거래시간 h(1~24)는 (h-1)~h시 구간이므로 T = 거래일 + h시간.
    """
    raw = pd.read_csv(path, encoding="cp949")
    date_col = "거래일자" if "거래일자" in raw.columns else "거래일"  # 2025년 파일부터 헤더 변경
    for col in (date_col, "거래시간", "지역", "연료원", "전력거래량(MWh)"):
        if col not in raw.columns:
            raise ValueError(f"KPX 파일에 '{col}' 열이 없습니다: {list(raw.columns)}")
    raw = raw[(raw["지역"].astype(str).str.strip() == region) & (raw["연료원"].astype(str).str.strip() == fuel)]
    if raw.empty:
        raise ValueError(f"KPX 파일에 {region}/{fuel} 행이 없습니다")
    hours = raw["거래시간"].astype(int)
    if not hours.between(1, 24).all():
        raise ValueError("거래시간은 1~24여야 합니다")
    t_end = pd.to_datetime(raw[date_col].astype(str).str.strip()) + pd.to_timedelta(hours, unit="h")
    s = pd.Series(raw["전력거래량(MWh)"].astype(float).values, index=pd.DatetimeIndex(t_end), name="generation_mw")
    s = s.sort_index()
    if not s.index.is_unique:
        raise ValueError("KPX 시계열에 중복 시각이 있습니다")
    return s


def load_asos_stations(path) -> pd.DataFrame:
    """ASOS 관측소별 시간자료(datetime, station_id, 변수…) → wide (열: {변수}_{관측소}), 1시간 간격 재색인"""
    w = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["datetime"])
    missing = [c for c in ["datetime", "station_id"] + WEATHER_VARS if c not in w.columns]
    if missing:
        raise ValueError(f"ASOS 파일에 열이 없습니다: {missing}")
    w["st"] = w["station_id"].map(STATIONS)
    if w["st"].isna().any():
        raise ValueError(f"알 수 없는 관측소: {sorted(w.loc[w['st'].isna(), 'station_id'].unique())}")
    wide = w.pivot(index="datetime", columns="st", values=WEATHER_VARS)
    wide.columns = [f"{v}_{st}" for v, st in wide.columns]
    full = pd.date_range(wide.index.min(), wide.index.max() + pd.Timedelta(hours=1), freq="h")
    return wide.reindex(full)


# ---------------------------------------------------------------------------
# 태양 기하
# ---------------------------------------------------------------------------

def to_kst_naive(times) -> pd.DatetimeIndex:
    """시간대가 붙은 시각은 KST로 바꾼 뒤 시간대를 뗀다 (UTC 'Z' 입력이 9시간 밀리는 것을 막음)"""
    times = pd.DatetimeIndex(times)
    if times.tz is not None:
        times = times.tz_convert("Asia/Seoul").tz_localize(None)
    return times


def solar_geometry(times: pd.DatetimeIndex, lat: float = SITE_LAT, lon: float = SITE_LON) -> pd.DataFrame:
    """KST 시각의 태양 천정각·방위각 (NOAA/Spencer 근사, 균시차 포함)"""
    times = to_kst_naive(times)
    hh = times.hour.values + times.minute.values / 60 + times.second.values / 3600
    doy = times.dayofyear.values
    g = 2 * np.pi / 365 * (doy - 1 + (hh - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    true_solar_min = hh * 60 + eot + 4 * lon - 60 * TZ_HOURS
    ha = np.radians(true_solar_min / 4 - 180)
    lat_r = np.radians(lat)
    cosz = np.clip(np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha), -1, 1)
    sinz = np.sqrt(np.clip(1 - cosz ** 2, 1e-12, None))
    cos_az = np.clip((np.sin(decl) - np.sin(lat_r) * cosz) / (np.cos(lat_r) * sinz), -1, 1)
    az = np.degrees(np.arccos(cos_az))
    az = np.where(ha > 0, 360 - az, az)
    g0 = 2 * np.pi * (doy - 1) / 365
    e0 = SOLAR_CONSTANT * (1.00011 + 0.034221 * np.cos(g0) + 0.00128 * np.sin(g0)
                           + 0.000719 * np.cos(2 * g0) + 0.000077 * np.sin(2 * g0))
    return pd.DataFrame({"cosz": cosz, "zenith": np.degrees(np.arccos(cosz)), "azimuth": az, "e0": e0},
                        index=times)


def interval_irradiance(t_end: pd.DatetimeIndex) -> pd.DataFrame:
    """구간 (T-1h, T] 적분: 대기권 밖 수평 일사 H0, Haurwitz 맑은하늘 일사 Hcs (MJ/m²), 낮 비율, 중앙 기하"""
    t_end = to_kst_naive(t_end)
    h0 = np.zeros(len(t_end))
    hcs = np.zeros(len(t_end))
    day = np.zeros(len(t_end))
    for k in range(INTEGRATION_STEPS):
        tt = t_end - pd.Timedelta(hours=1) + pd.to_timedelta((k + 0.5) * 60 / INTEGRATION_STEPS, unit="min")
        geo = solar_geometry(tt)
        cz = np.clip(geo["cosz"].values, 0, None)
        h0 += geo["e0"].values * cz
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            hcs += np.where(cz > 0, 1098 * cz * np.exp(-0.057 / np.where(cz > 0, cz, 1)), 0)
        day += cz > 0
    mid = solar_geometry(t_end - pd.Timedelta(minutes=30))
    return pd.DataFrame({
        "H0": h0 / INTEGRATION_STEPS / MJ_TO_W,
        "Hcs": hcs / INTEGRATION_STEPS / MJ_TO_W,
        "daylight_frac": day / INTEGRATION_STEPS,
        "zenith_mid": mid["zenith"].values,
        "azimuth_mid": mid["azimuth"].values,
        "cosz_mid": mid["cosz"].values,
    }, index=t_end)


def build_solar_frame(weather_wide: pd.DataFrame, index: Optional[pd.DatetimeIndex] = None,
                      generation: Optional[pd.Series] = None) -> pd.DataFrame:
    """구간 끝 시각 T 인덱스의 기본 프레임.

    열: {변수}_{관측소}(T 관측, 일사는 구간 누적), prev_{변수}_{관측소}(T-1h 관측), 구간 태양 기하,
        generation_mw(주어진 경우)
    야간(구간 내내 해가 지평선 아래) 일사 결측은 0으로 확정한다 (ASOS는 야간 일사를 기록하지 않음).
    """
    if index is None:
        if generation is None:
            raise ValueError("index 또는 generation 중 하나는 필요합니다")
        index = generation.index
    index = to_kst_naive(index)
    if not index.is_unique:
        raise ValueError("구간 끝 시각에 중복이 있습니다")
    if pd.DatetimeIndex(weather_wide.index).tz is not None:
        raise ValueError("기상 자료 시각은 KST naive여야 합니다")
    frame = pd.DataFrame(index=index)
    if generation is not None:
        frame["generation_mw"] = generation.reindex(index)
    cur = weather_wide.reindex(index)
    prev = weather_wide.reindex(index - pd.Timedelta(hours=1)).set_axis(index).add_prefix("prev_")
    geo = interval_irradiance(index)
    frame = pd.concat([frame, cur, prev, geo], axis=1)
    night = frame["daylight_frac"].values == 0
    prev_night = interval_irradiance(index - pd.Timedelta(hours=1))["daylight_frac"].values == 0
    for st in RAD_STATIONS:
        for col, mask in ((f"solar_radiation_{st}", night), (f"prev_solar_radiation_{st}", prev_night)):
            if col in frame:
                frame.loc[mask & frame[col].isna().values, col] = 0.0
    return frame


def to_island_mean(frame: pd.DataFrame) -> pd.DataFrame:
    """관측소별 값이 없고 섬 평균 기상 1세트만 있는 운영 환경(API)과 같은 입력으로 바꾼다.

    일사는 연중 관측되는 제주·고산 평균, 기온·습도·운량은 4개 관측소 평균을 모든 관측소 열에 넣는다.
    """
    b = frame.copy()
    for pre in ("", "prev_"):
        rad = b[[f"{pre}solar_radiation_jeju", f"{pre}solar_radiation_gosan"]].mean(axis=1)
        b[f"{pre}solar_radiation_jeju"] = rad
        b[f"{pre}solar_radiation_gosan"] = rad
        b[f"{pre}solar_radiation_seogwipo"] = np.nan
        for v in ("temperature", "humidity", "cloud_cover"):
            m = b[[f"{pre}{v}_{st}" for st in ALL_STATIONS]].mean(axis=1)
            for st in ALL_STATIONS:
                b[f"{pre}{v}_{st}"] = m
    return b


# ---------------------------------------------------------------------------
# 경사면 일사 물리 체인
# ---------------------------------------------------------------------------

def erbs_diffuse_fraction(kt: np.ndarray) -> np.ndarray:
    """Erbs(1982) 산란비. kt 결측(저고도)은 전량 산란으로 둔다 — 저고도에서는 직달 분리가 무의미"""
    kt = np.where(np.isnan(kt), 0.0, kt)
    fd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.80,
                           0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                           0.165))
    return np.clip(fd, 0.0, 1.0)


def _plane_factors(frame: pd.DataFrame, tilt: float, az_plane: float):
    """직달 경사 계수 Rb, 하늘·지면 시야계수"""
    z = np.radians(frame["zenith_mid"].values)
    azs = np.radians(frame["azimuth_mid"].values)
    b = np.radians(tilt)
    cosz = np.maximum(frame["cosz_mid"].values, COSZ_MIN)
    cos_t = np.cos(z) * np.cos(b) + np.sin(z) * np.sin(b) * np.cos(azs - np.radians(az_plane))
    rb = np.clip(np.maximum(cos_t, 0) / cosz, 0, RB_MAX)
    rb = np.where(frame["cosz_mid"].values > 0, rb, 0.0)
    return rb, (1 + np.cos(b)) / 2, (1 - np.cos(b)) / 2, cosz


def _poa_chain(ghi_w: np.ndarray, g0h_w: np.ndarray, rb, vf_sky, vf_gnd, cosz):
    """수평 일사(W/m²) → (kt, 산란비, 경사면 일사). Hay–Davies 이방성 하늘 모델"""
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(g0h_w >= MIN_DEN_MJ * MJ_TO_W, ghi_w / g0h_w, np.nan)
        kt = np.clip(kt, 0, KT_MAX)
        fd = erbs_diffuse_fraction(kt)
        dhi = fd * ghi_w
        bh = ghi_w - dhi
        ai = np.where(g0h_w > 1.0, np.clip(bh / g0h_w, 0, 1), 0.0)
    sky = dhi * (ai * rb + (1 - ai) * vf_sky)
    poa = bh * rb + sky + ALBEDO * ghi_w * vf_gnd
    return kt, fd, poa


def _power_proxy(poa: np.ndarray, ta: np.ndarray):
    tc = ta + (NOCT_C - 20.0) / 800.0 * poa
    return tc, poa * (1 + TEMP_COEF * (tc - 25.0))


def _station_power(frame: pd.DataFrame, st: str, az_plane: float, tilt: float = PANEL_TILT_DEG) -> Dict[str, np.ndarray]:
    rb, vs, vg, cosz = _plane_factors(frame, tilt, az_plane)
    rad = frame[f"solar_radiation_{st}"].values
    ghi = rad * MJ_TO_W
    g0h = frame["H0"].values * MJ_TO_W
    ta = np.nanmean(np.c_[frame[f"temperature_{st}"].values, frame[f"prev_temperature_{st}"].values], axis=1)
    kt, fd, poa = _poa_chain(ghi, g0h, rb, vs, vg, cosz)
    tc, p = _power_proxy(poa, ta)
    hcs = frame["Hcs"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        kc = np.clip(np.where(hcs >= MIN_DEN_MJ, rad / hcs, np.nan), 0, KC_MAX)
    missing = np.isnan(ghi)  # 결측 관측소(2024년 1~4월 서귀포)는 파생값도 결측 유지
    out = {"kt": kt, "kc": kc, "fd": fd, "poa": poa, "tc": tc, "p": p}
    return {k: np.where(missing, np.nan, v) for k, v in out.items()}


def physics_features(frame: pd.DataFrame, tilt: float = PANEL_TILT_DEG) -> pd.DataFrame:
    """관측소별 물리량, 관측소 평균, 맑은하늘 출력과 그 비율"""
    out = pd.DataFrame(index=frame.index)
    south = {st: _station_power(frame, st, PLANE_AZIMUTHS[""], tilt) for st in RAD_STATIONS}
    # 저고도에서 모든 관측소 kt가 결측인 행은 평균도 결측이 정상 → 경고 억제
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in ("kt", "kc", "fd", "poa", "tc", "p"):
            out[f"{k}_mean"] = np.nanmean(np.c_[tuple(south[st][k] for st in RAD_STATIONS)], axis=1)
        for st in RAD_STATIONS:
            out[f"p_{st}"] = south[st]["p"]
        for suffix in ("_e", "_w"):
            per = [_station_power(frame, st, PLANE_AZIMUTHS[suffix], tilt)["p"] for st in RAD_STATIONS]
            out[f"p_mean{suffix}"] = np.nanmean(np.c_[tuple(per)], axis=1)
    # 맑은하늘(Haurwitz) 수평 일사로 같은 체인 → 기상 무관 최대 출력 곡선
    rb, vs, vg, cosz = _plane_factors(frame, tilt, PLANE_AZIMUTHS[""])
    ta_all = frame[[f"temperature_{st}" for st in ALL_STATIONS]
                   + [f"prev_temperature_{st}" for st in ALL_STATIONS]].mean(axis=1).values
    _, _, poa_cs = _poa_chain(frame["Hcs"].values * MJ_TO_W, frame["H0"].values * MJ_TO_W, rb, vs, vg, cosz)
    _, out["p_cs"] = _power_proxy(poa_cs, ta_all)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["p_ratio_cs"] = np.where(out["p_cs"] > P_CS_MIN, np.clip(out["p_mean"] / out["p_cs"], 0, KC_MAX), np.nan)
    return out


def create_solar_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    """물리 주도 피처 23개 (COMPACT_FEATURES 순서)

    Args:
        frame: build_solar_frame 결과 (관측소별 기상 + 구간 태양 기하)
    """
    phys = physics_features(frame)
    ctx = pd.DataFrame(index=frame.index)
    ctx["cloud_mean"] = frame[[f"cloud_cover_{st}" for st in ALL_STATIONS]].mean(axis=1)
    ctx["humidity_mean"] = frame[[f"humidity_{st}" for st in ALL_STATIONS]].mean(axis=1)
    ctx["temp_mean"] = frame[[f"temperature_{st}" for st in ALL_STATIONS]].mean(axis=1)
    doy = frame.index.dayofyear.values
    ctx["dayofyear"] = doy
    ctx["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    ctx["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    X = pd.concat([phys, frame[["H0", "cosz_mid", "zenith_mid", "azimuth_mid"]], ctx], axis=1)
    return X[COMPACT_FEATURES]


def cloud_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    """구간 양 끝(T-1h, T)의 4개 관측소 평균 운량과 변화량 — 구간 중 구름 변화의 단서"""
    cc0 = frame[[f"cloud_cover_{st}" for st in ALL_STATIONS]].mean(axis=1)
    cc_m1 = frame[[f"prev_cloud_cover_{st}" for st in ALL_STATIONS]].mean(axis=1)
    return pd.DataFrame({"cc_m1": cc_m1, "cc0": cc0, "cc_d_lag": cc0 - cc_m1}, index=frame.index)


ISLAND_MEAN_DUPLICATES = ["p_gosan", "p_seogwipo"]  # 섬 평균 입력에서는 p_jeju와 같은 값


def create_island_mean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """섬 평균 기상 1세트만 쓰는 운영형 피처 (발전 이력 불필요). 학습 스크립트와 API가 같은 함수를 쓴다."""
    X = create_solar_features_v2(to_island_mean(frame)).drop(columns=ISLAND_MEAN_DUPLICATES)
    return pd.concat([X, cloud_trend_features(frame)], axis=1)


def frame_from_point_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """관측소 구분 없는 시간별 기상(API 입력·수치예보 지점값) → build_solar_frame과 같은 열 구성

    Args:
        weather: 인덱스 = 시각 T (KST, 1시간 간격 연속), 열 = WEATHER_VARS.
            solar_radiation은 구간 (T-1h, T] 누적 MJ/m² (제주·고산 ASOS 평균에 해당), 모르면 NaN.

    Returns:
        구간 끝 T 인덱스 프레임. 첫 행은 직전 관측이 없어 prev_* 열이 결측이다.
    """
    required = ["temperature", "humidity", "solar_radiation", "cloud_cover"]
    missing = [v for v in required if v not in weather.columns]
    if missing:
        raise ValueError(f"기상 입력에 열이 없습니다: {missing}")
    index = to_kst_naive(weather.index)
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("기상 입력 시각은 중복 없이 오름차순이어야 합니다")
    wide = pd.DataFrame(index=index)
    for v in WEATHER_VARS:
        values = weather[v].to_numpy(dtype=float) if v in weather.columns else np.full(len(index), np.nan)
        for st in ALL_STATIONS:
            wide[f"{v}_{st}"] = values
    # 학습 때 섬 평균 일사는 연중 관측 관측소(제주·고산)만으로 만들었다
    wide["solar_radiation_seongsan"] = np.nan
    wide["solar_radiation_seogwipo"] = np.nan
    return build_solar_frame(wide, index=index)


# ---------------------------------------------------------------------------
# 발전 수준 (과거 발전량, D-2 컷오프)
# ---------------------------------------------------------------------------

LEVEL_WINDOW_DAYS = 14
LEVEL_LAG_DAYS = 2      # 하루 전 10시 제출 시점에 확정된 발전량은 D-2까지
LEVEL_MIN_P = 100.0     # 집계에 쓰는 시간: 물리 출력대용값 ≥ 100 W/m²


def generation_level(frame: pd.DataFrame, generation: pd.Series, p_mean: pd.Series) -> pd.Series:
    """최근 14일(대상일 D 기준 D-2까지) Σ발전량 / Σ물리 출력대용값 → lvl_p

    물리 피처가 계절 기하를 설명하고 남는 '발전 수준' 변화를 따라간다:
    2024-06-01 제주 재생에너지 입찰제도 시행 후 KPX 공개 거래량이 포괄하는 설비가 약 79%로 줄었고,
    그 뒤로도 수준이 계속 내려갔다 (1~5월 ≈0.38 → 9~11월 ≈0.26 MW per W/m²).
    발전량이 없는 날(미공개·결측)은 창에서 빠지고, 14일 창에 실제 자료가 있는 날이 4일 미만이면 NaN을 돌려준다.
    오래된 값을 이어 붙이지 않는다 (공급 지연이 길면 조용히 낡은 수준을 쓰는 대신 예측 단계에서 실패시키기 위함).

    Args:
        frame: build_solar_frame 결과 (구간 끝 T 인덱스)
        generation: 실측 발전량 (frame.index 기준, 미래 구간은 NaN이어도 됨)
        p_mean: physics_features의 p_mean
    """
    day = (frame.index - pd.Timedelta(hours=1)).normalize()  # 구간 시작 시각의 날짜 = 거래일
    y = generation.reindex(frame.index)
    use = (p_mean.reindex(frame.index) >= LEVEL_MIN_P) & y.notna()
    g = pd.DataFrame({"Y": y.where(use, 0.0), "R": p_mean.reindex(frame.index).where(use, 0.0),
                      "n": use.astype(int)}).groupby(day.values).sum()
    observed = g["n"] > 0
    Y, R = g["Y"].where(observed), g["R"].where(observed)
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    Y, R = Y.reindex(full), R.reindex(full)
    w = LEVEL_WINDOW_DAYS
    ys = Y.rolling(w, min_periods=w // 3).sum()
    rs = R.rolling(w, min_periods=w // 3).sum()
    lvl = (ys / rs.where(rs > 0)).shift(LEVEL_LAG_DAYS)
    return lvl.reindex(day).set_axis(frame.index).rename("lvl_p")


# ---------------------------------------------------------------------------
# 야간 후처리 (태양광 연계 ESS 저녁 방전)
# ---------------------------------------------------------------------------

def night_profile(generation: pd.Series, daylight_frac: pd.Series, window_days: int = 28) -> Dict[int, float]:
    """해가 진 구간(daylight_frac == 0)의 시각별 평균 발전량 (구간 끝 시각 T의 시 기준).

    KPX 태양광 거래량에는 태양광 연계 ESS의 저녁 방전이 포함돼 일몰 후에도 2~6 MW가 나온다.
    기상 모델은 이를 설명하지 못하므로 학습 기간 마지막 window_days일의 평균 프로파일로 대체한다.
    """
    night = daylight_frac.reindex(generation.index).values == 0
    idx = generation.index
    recent = night & (idx > idx.max() - pd.Timedelta(days=window_days))
    prof = generation[recent].groupby(idx[recent].hour).mean()
    prof_all = generation[night].groupby(idx[night].hour).mean()
    return {h: float(prof.get(h, prof_all.get(h, 0.0))) for h in range(24)}


def apply_night_profile(pred: np.ndarray, index: pd.DatetimeIndex, daylight_frac: Iterable[float],
                        profile: Dict[int, float]) -> np.ndarray:
    """야간 구간 예측을 프로파일 값으로 바꾸고 음수를 0으로 자른다"""
    p = np.clip(np.asarray(pred, dtype=float), 0, None).copy()
    night = np.asarray(list(daylight_frac), dtype=float) == 0
    hours = pd.DatetimeIndex(index)[night].hour
    p[night] = [profile.get(int(h), 0.0) for h in hours]
    return p
