"""
J-REF Dashboard - Streamlit Cloud Version

제주도 신재생에너지 발전량 예측 대시보드
Optimized for Streamlit Cloud deployment

Usage:
    streamlit run streamlit_app.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="J-REF Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)

SOLAR_COLOR = "#d68420"
WIND_COLOR = "#147e88"
INK_COLOR = "#183a45"


def apply_dashboard_style():
    """Apply a consistent, responsive presentation to Streamlit components."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gowun+Dodum&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, .stApp, p, h1, h2, h3, label { font-family: 'Manrope', 'Gowun Dodum', sans-serif; }
    .stApp, [data-testid="stHeader"] { background: #f4f8f8; color: #183a45; }
    [data-testid="stSidebar"] { background: #e8f1f0; border-right: 1px solid #d4e4e1; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #4d6b70; }
    [data-testid="stMainBlockContainer"] { max-width: 1450px; padding-top: 2.5rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: #183a45 !important; letter-spacing: -.035em; }
    h1 { font-weight: 800 !important; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #deebe9; border-radius: 14px; padding: 1.1rem 1.25rem; min-height: 125px; }
    [data-testid="stMetricLabel"] { color: #55747a; }
    [data-testid="stMetricValue"] { color: #183a45; letter-spacing: -.04em; }
    [data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #deebe9; border-radius: 14px; padding: .4rem; }
    [data-testid="stTabs"] [role="tab"] { padding: .8rem 1rem; }
    .jref-kicker { color: #147e88; font-size: .8rem; font-weight: 800; letter-spacing: .08em; margin-bottom: .5rem; }
    .jref-rule { height: 4px; width: 56px; background: #147e88; border-radius: 8px; margin: .4rem 0 1.3rem; }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding: 1.4rem 1rem 3rem; }
        [data-testid="stMetric"] { min-height: 0; }
        h1 { font-size: 1.85rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)


def style_chart(fig, height=360):
    """Use the dashboard palette and legible chart defaults."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK_COLOR, size=12),
        margin=dict(l=24, r=20, t=55, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#edf3f2", zeroline=False)
    return fig


@st.cache_data
def load_data():
    """데이터 로드 (캐싱)"""
    # 실측 KPX 데이터만 사용한다 (merged_dataset.csv는 물리식으로 만든 합성 데이터라 제외)
    data_paths = [
        Path("data/processed/kpx_real_merged.csv"),  # Local full data
        Path("data/sample/sample_data.csv"),  # Same real data, committed for cloud
    ]

    for data_path in data_paths:
        if data_path.exists():
            df = pd.read_csv(data_path, encoding="utf-8-sig")
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df, str(data_path)

    return None, None


def plot_generation_overview(df: pd.DataFrame):
    """발전량 개요 플롯"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("시간별 발전량", "월별 평균 발전량", "발전량 분포", "발전량 비율"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "pie"}]],
    )

    # 최근 7일 시계열
    recent = df.tail(168)  # 7일 * 24시간

    fig.add_trace(
        go.Scatter(x=recent["datetime"], y=recent["solar_mw"],
                   name="태양광", line=dict(color="#FFA500")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=recent["datetime"], y=recent["wind_mw"],
                   name="풍력", line=dict(color="#00BFFF")),
        row=1, col=1
    )

    # 월별 평균
    monthly = df.groupby(df["datetime"].dt.month).agg({
        "solar_mw": "mean",
        "wind_mw": "mean"
    }).reset_index()

    fig.add_trace(
        go.Bar(x=monthly["datetime"], y=monthly["solar_mw"],
               name="태양광 (월평균)", marker_color="#FFA500"),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=monthly["datetime"], y=monthly["wind_mw"],
               name="풍력 (월평균)", marker_color="#00BFFF"),
        row=1, col=2
    )

    # 히스토그램
    fig.add_trace(
        go.Histogram(x=df["solar_mw"], name="태양광", marker_color="#FFA500", opacity=0.7),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=df["wind_mw"], name="풍력", marker_color="#00BFFF", opacity=0.7),
        row=2, col=1
    )

    # 파이 차트
    total_solar = df["solar_mw"].sum()
    total_wind = df["wind_mw"].sum()
    fig.add_trace(
        go.Pie(labels=["태양광", "풍력"], values=[total_solar, total_wind],
               marker_colors=["#FFA500", "#00BFFF"]),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=True)
    return fig


def plot_hourly_pattern(df: pd.DataFrame):
    """시간대별 패턴 플롯"""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour

    hourly = df.groupby("hour").agg({
        "solar_mw": ["mean", "std"],
        "wind_mw": ["mean", "std"]
    }).reset_index()
    hourly.columns = ["hour", "solar_mean", "solar_std", "wind_mean", "wind_std"]

    fig = go.Figure()

    # 태양광
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["solar_mean"],
        name="태양광",
        line=dict(color="#FFA500", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 165, 0, 0.3)"
    ))

    # 풍력
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["wind_mean"],
        name="풍력",
        line=dict(color="#00BFFF", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 191, 255, 0.3)"
    ))

    fig.update_layout(
        title="시간대별 평균 발전량",
        xaxis_title="시간 (Hour)",
        yaxis_title="발전량 (MW)",
        height=400,
    )

    return fig


def plot_weather_correlation(df: pd.DataFrame):
    """기상-발전량 상관관계 플롯"""
    weather_cols = ["temperature", "humidity", "wind_speed", "pressure", "solar_radiation", "cloud_cover"]
    available_cols = [col for col in weather_cols if col in df.columns and df[col].notna().any()]

    if not available_cols:
        return None

    # 상관계수 계산
    corr_data = []
    for col in available_cols:
        solar_corr = df["solar_mw"].corr(df[col])
        wind_corr = df["wind_mw"].corr(df[col])
        corr_data.append({
            "variable": col,
            "solar_corr": solar_corr,
            "wind_corr": wind_corr,
        })

    corr_df = pd.DataFrame(corr_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="태양광",
        x=corr_df["variable"],
        y=corr_df["solar_corr"],
        marker_color="#FFA500"
    ))
    fig.add_trace(go.Bar(
        name="풍력",
        x=corr_df["variable"],
        y=corr_df["wind_corr"],
        marker_color="#00BFFF"
    ))

    fig.update_layout(
        title="기상 변수와 발전량 상관관계",
        xaxis_title="기상 변수",
        yaxis_title="상관계수",
        barmode="group",
        height=400,
    )

    return fig


def main():
    apply_dashboard_style()
    # 사이드바
    st.sidebar.markdown("<div class='jref-kicker'>JEJU RENEWABLE ENERGY</div>", unsafe_allow_html=True)
    st.sidebar.title("J-REF")
    st.sidebar.markdown("제주 재생에너지 데이터 대시보드")

    # 데이터 로드
    df, data_source = load_data()

    if df is None:
        st.error("데이터를 찾을 수 없습니다.")
        st.info("GitHub에서 프로젝트를 클론하고 데이터를 생성하세요.")
        st.code("""
git clone https://github.com/kiminbean/kpx-demand-forecast.git
cd kpx-demand-forecast
python scripts/download_data.py --year 2024
        """)
        return

    # 페이지 선택
    page = st.sidebar.radio(
        "메뉴",
        ["📊 개요", "📈 발전량 분석", "🔮 모델 성능"],
    )

    # 날짜 필터
    st.sidebar.markdown("---")
    st.sidebar.subheader("조회 설정")

    date_range = st.sidebar.date_input(
        "날짜 범위",
        value=(df["datetime"].min().date(), df["datetime"].max().date()),
        min_value=df["datetime"].min().date(),
        max_value=df["datetime"].max().date(),
    )

    if len(date_range) == 2:
        mask = (df["datetime"].dt.date >= date_range[0]) & (df["datetime"].dt.date <= date_range[1])
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        st.warning("선택한 기간에 데이터가 없습니다. 날짜 범위를 변경해 주세요.")
        return

    # 메인 컨텐츠
    if page == "📊 개요":
        st.markdown("<div class='jref-kicker'>제주의 에너지 흐름</div>", unsafe_allow_html=True)
        st.title("재생에너지 발전 현황")
        st.markdown("<div class='jref-rule'></div>", unsafe_allow_html=True)
        st.caption(
            f"KPX 실측 시간별 거래량 (제주, "
            f"{df['datetime'].min():%Y-%m-%d} ~ {df['datetime'].max():%Y-%m-%d}) "
            f"· 기상청 ASOS 관측"
        )
        st.info("공개 거래량 기준으로, 입찰자원·자가소비 등은 포함되지 않습니다. 2024년 6월부터 집계 범위가 달라져 기간 간 직접 비교에 주의하세요.", icon="ℹ️")
        st.subheader("선택 기간 요약")

        # KPI 카드
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="관측 데이터",
                value=f"{len(filtered_df):,}건",
                help="선택한 기간의 시간별 데이터 건수입니다."
            )

        with col2:
            st.metric(
                label="태양광 평균",
                value=f"{filtered_df['solar_mw'].mean():.1f} MW",
                help=f"선택 기간 최대 {filtered_df['solar_mw'].max():.1f} MW"
            )

        with col3:
            st.metric(
                label="풍력 평균",
                value=f"{filtered_df['wind_mw'].mean():.1f} MW",
                help=f"선택 기간 최대 {filtered_df['wind_mw'].max():.1f} MW"
            )

        with col4:
            total_col = "total_mw" if "total_mw" in filtered_df.columns else None
            if total_col:
                st.metric(
                    label="태양광 + 풍력 평균",
                    value=f"{filtered_df[total_col].mean():.1f} MW",
                    help=f"선택 기간 최대 {filtered_df[total_col].max():.1f} MW"
                )
            else:
                total_avg = filtered_df['solar_mw'].mean() + filtered_df['wind_mw'].mean()
                st.metric(
                    label="태양광 + 풍력 평균",
                    value=f"{total_avg:.1f} MW",
                )

        st.subheader("발전량 추이")
        st.caption("선택 기간의 마지막 7일 · 시간별 평균 전력 (MW)")
        recent = filtered_df.sort_values("datetime").tail(168)
        fig = go.Figure()
        for column, label, color in (("solar_mw", "태양광", SOLAR_COLOR), ("wind_mw", "풍력", WIND_COLOR)):
            fig.add_trace(go.Scatter(x=recent["datetime"], y=recent[column], name=label,
                                     line=dict(color=color, width=2.5),
                                     hovertemplate="%{x|%m/%d %H:%M}<br>%{y:.1f} MW<extra>%{fullData.name}</extra>"))
        fig.update_yaxes(title_text="MW", rangemode="tozero")
        st.plotly_chart(style_chart(fig), use_container_width=True)
        st.subheader("월별 평균")
        st.caption("선택 기간에 포함된 관측값 기준 · MW")
        monthly = filtered_df.groupby(filtered_df["datetime"].dt.to_period("M"))[["solar_mw", "wind_mw"]].mean()
        fig_monthly = go.Figure()
        for column, label, color in (("solar_mw", "태양광", SOLAR_COLOR), ("wind_mw", "풍력", WIND_COLOR)):
            fig_monthly.add_bar(x=monthly.index.astype(str), y=monthly[column], name=label, marker_color=color)
        fig_monthly.update_layout(barmode="group")
        st.plotly_chart(style_chart(fig_monthly, 320), use_container_width=True)

    elif page == "📈 발전량 분석":
        st.title("발전량 분석")
        st.caption("선택한 기간의 발전 패턴과 기상 관측값을 살펴봅니다.")
        st.warning("이 데이터의 발전량은 시간 구간 시작, 기상 관측은 구간 종료 기준입니다. 기상 상관관계는 시간 정렬 차이를 감안해 해석하세요.")

        tab1, tab2, tab3 = st.tabs(["⏰ 시간대별 패턴", "🌤️ 기상 상관관계", "📋 원본 데이터"])

        with tab1:
            st.plotly_chart(style_chart(plot_hourly_pattern(filtered_df), 400), use_container_width=True)

            # 요일별 히트맵
            temp_df = filtered_df.copy()
            temp_df["dayofweek"] = temp_df["datetime"].dt.dayofweek
            temp_df["hour"] = temp_df["datetime"].dt.hour

            pivot = temp_df.pivot_table(
                values="solar_mw", index="hour", columns="dayofweek", aggfunc="mean"
            )

            fig_heatmap = px.imshow(
                pivot,
                labels=dict(x="요일 (0=월)", y="시간", color="발전량 (MW)"),
                title="태양광 발전량 히트맵 (시간 x 요일)",
                color_continuous_scale="YlOrRd",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with tab2:
            fig_corr = plot_weather_correlation(filtered_df)
            if fig_corr:
                st.plotly_chart(style_chart(fig_corr, 400), use_container_width=True)
            else:
                st.info("기상 데이터가 없습니다.")

            # 산점도
            if "solar_radiation" in filtered_df.columns:
                sample_size = min(1000, len(filtered_df))
                sample = filtered_df.sample(sample_size, random_state=42)
                fig_scatter = px.scatter(
                    sample,
                    x="solar_radiation",
                    y="solar_mw",
                    color=sample["datetime"].dt.hour,
                    title="일사량 vs 태양광 발전량",
                    labels={"solar_radiation": "일사량", "solar_mw": "발전량 (MW)", "color": "시간"},
                )
                st.plotly_chart(style_chart(fig_scatter, 400), use_container_width=True)

        with tab3:
            st.dataframe(filtered_df.tail(100), use_container_width=True)

            # 다운로드 버튼
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name="jref_data.csv",
                mime="text/csv",
            )

    elif page == "🔮 모델 성능":
        st.title("🔮 모델 성능")
        st.caption("과거 평가 결과 · 수치가 낮을수록 예측 오차가 작습니다")

        st.subheader("운영 모델과 평가 결과")
        st.markdown(
            "기본 API는 입찰자원을 포함한 **제주 계통 발전량**을 예측합니다. "
            "태양광과 풍력 모두 LightGBM을 사용하며, 태양광은 가용한 발전 이력에 따라 "
            "기상 전용 또는 발전 수준 모델을 사용합니다. `SERVING_LABEL=public` 설정 시 "
            "이전 공개 라벨 모델을 사용합니다."
        )
        st.info(
            "아래 수치는 서로 다른 기간과 조건에서 수행한 과거 실험입니다. "
            "현재 API 모델의 실시간 성능이나 향후 성능을 보장하지 않습니다."
        )

        st.subheader("하루 전 예보 입력 · FER")
        st.caption("제출 마감 전 공개가 확인된 수치예보를 사용한 과거 평가")
        st.markdown("""
        | 대상 | 입력 · 평가 조건 | 2025년 | 2026년 1~10월 |
        |---|---|---:|---:|
        | 태양광 | GFS, 10시 제출 · 월별 재학습 | **9.55%** | **9.83%** |
        | 태양광 | GFS + IFS 평균, 10시 제출 · 월별 재학습 | **8.95%** | **9.04%** |
        | 풍력 | GFS 풍속, 월별 재학습 | **11.14%** | **9.24%** |
        """)
        st.caption(
            "태양광 2025년은 공개 태양광 라벨(C=320.6 MW), 2026년은 계통 라벨 "
            "(C=490.1 MW)입니다. 풍력은 두 기간 모두 계통 라벨이며 "
            "분모는 각 기간 최대 발전량(360.2 / 348.1 MW)입니다. "
            "라벨과 분모가 달라 기간·대상 간 단순 비교에 주의하세요."
        )

        st.subheader("관측 기상 입력 · 참고 평가")
        st.markdown("""
        | 대상 | 평가 조건 | 2025년 | 2026년 1~10월 |
        |---|---|---:|---:|
        | 태양광 | 관측 기상 · 월별 재학습 | **6.47%** | **7.34%** |
        | 풍력 | 관측 기상 · 월별 재학습 | **11.80%** | **9.52%** |
        """)
        st.caption(
            "같은 기간·라벨의 예보 입력 평가와 비교한 과거 실험입니다. "
            "관측 기상은 하루 전 실제로 사용할 수 있는 입력이 아니며, "
            "특히 풍력에서는 예보 입력보다 반드시 정확한 것은 아닙니다."
        )

        with st.expander("FER 기준과 출처"):
            st.markdown("""
            **FER**은 시간별 절대 예측오차를 기준 용량으로 나눈 비율의 평균이며,
            실측 발전량이 기준 용량의 10% 이상인 시간만 평가합니다. 위 실험의
            기준 용량은 설비용량이 아닌 기간 최대 발전량 등 대용값일 수 있습니다.

            - 태양광: `docs/solar_dayahead_verified_2026-11-14.md`
            - 풍력: `docs/wind_dayahead_gfs_2026-11-15.md`
            - 이전 공개 라벨의 관측 기상 태양광 FER 4.18%는 과거 테스트 결과로,
              현재 계통 라벨 API 모델이나 하루 전 예보의 성능이 아닙니다.
            """)

    # 푸터
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        **J-REF**

        제주도 신재생에너지 발전량 예측 시스템

        KPX 공개 거래량 · {len(df):,}건

        ---

        [GitHub](https://github.com/kiminbean/kpx-demand-forecast) |
        [License](https://github.com/kiminbean/kpx-demand-forecast/blob/main/LICENSE)
        """
    )


if __name__ == "__main__":
    main()
