"""
J-REF Dashboard - Streamlit Cloud Version

제주도 신재생에너지 발전량 예측 대시보드
Optimized for Streamlit Cloud deployment

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="J-REF Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    # 사이드바
    st.sidebar.title("⚡ J-REF Dashboard")
    st.sidebar.markdown("제주도 신재생에너지 발전량 예측")

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
        "페이지 선택",
        ["📊 개요", "📈 발전량 분석", "🔮 모델 성능"],
    )

    # 날짜 필터
    st.sidebar.markdown("---")
    st.sidebar.subheader("필터")

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

    # 메인 컨텐츠
    if page == "📊 개요":
        st.title("🌴 제주도 신재생에너지 발전량 현황")
        st.caption(
            f"데이터: KPX 실측 시간별 발전량 (제주, "
            f"{df['datetime'].min():%Y-%m-%d} ~ {df['datetime'].max():%Y-%m-%d}) "
            f"+ 기상청 ASOS 관측"
        )

        # KPI 카드
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📁 총 데이터",
                value=f"{len(filtered_df):,}건",
                delta=f"{len(df):,}건 전체"
            )

        with col2:
            st.metric(
                label="☀️ 태양광 평균",
                value=f"{filtered_df['solar_mw'].mean():.1f} MW",
                delta=f"최대 {filtered_df['solar_mw'].max():.1f} MW"
            )

        with col3:
            st.metric(
                label="💨 풍력 평균",
                value=f"{filtered_df['wind_mw'].mean():.1f} MW",
                delta=f"최대 {filtered_df['wind_mw'].max():.1f} MW"
            )

        with col4:
            total_col = "total_mw" if "total_mw" in filtered_df.columns else None
            if total_col:
                st.metric(
                    label="⚡ 총 발전량",
                    value=f"{filtered_df[total_col].mean():.1f} MW",
                    delta=f"최대 {filtered_df[total_col].max():.1f} MW"
                )
            else:
                total_avg = filtered_df['solar_mw'].mean() + filtered_df['wind_mw'].mean()
                st.metric(
                    label="⚡ 총 발전량",
                    value=f"{total_avg:.1f} MW",
                )

        # 개요 플롯
        st.plotly_chart(plot_generation_overview(filtered_df), use_container_width=True)

    elif page == "📈 발전량 분석":
        st.title("📈 발전량 상세 분석")

        tab1, tab2, tab3 = st.tabs(["⏰ 시간대별 패턴", "🌤️ 기상 상관관계", "📋 원본 데이터"])

        with tab1:
            st.plotly_chart(plot_hourly_pattern(filtered_df), use_container_width=True)

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
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("기상 데이터가 없습니다.")

            # 산점도
            if "solar_radiation" in filtered_df.columns:
                sample_size = min(1000, len(filtered_df))
                fig_scatter = px.scatter(
                    filtered_df.sample(sample_size),
                    x="solar_radiation",
                    y="solar_mw",
                    color=filtered_df.sample(sample_size)["datetime"].dt.hour,
                    title="일사량 vs 태양광 발전량",
                    labels={"solar_radiation": "일사량", "solar_mw": "발전량 (MW)", "color": "시간"},
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

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
        st.title("🔮 모델 성능 분석")

        st.markdown("""
        ### 📊 현재 모델 성능 (2024-12-17 기준)

        | 모델 | R² | RMSE | MAE | 상태 |
        |------|-----|------|-----|------|
        | **Solar (LightGBM)** | 0.9134 | 17.63 MW | 9.77 MW | ✅ Production |
        | **Wind (LightGBM)** | 0.7576 | 14.96 MW | 11.57 MW | ✅ Production |

        ---

        ### 🚀 Wind 모델 개선 과정
        """)

        # 성능 개선 차트
        phases = ["Phase 0\n(Baseline)", "Phase 1\n(Physics)", "Phase 2\n(5yr Pattern)", "Phase 5\n(Final)"]
        r2_values = [0.2530, 0.5153, 0.6357, 0.7576]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=phases,
            y=r2_values,
            marker_color=["#FF6B6B", "#FFA500", "#4ECDC4", "#45B7D1"],
            text=[f"R² {v:.2f}" for v in r2_values],
            textposition="outside"
        ))

        fig.update_layout(
            title="Wind 모델 R² 개선 과정 (+199.4%)",
            yaxis_title="R² Score",
            height=400,
            yaxis_range=[0, 1]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### 🔑 핵심 피처 (Wind 모델)

        1. `wind_speed_anomaly` - 5년 평균 대비 풍속 이상치
        2. `dayofyear` - 계절성
        3. `temperature_min` - 월-시간별 최저기온
        4. `wind_speed_rolling_mean_3h` - 3시간 롤링 평균
        5. `wind_power_density` - 풍력 에너지 밀도 (½ρv³)
        """)

    # 푸터
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        **J-REF v0.1**

        제주도 신재생에너지 발전량 예측 시스템

        - 📊 Data: {len(df):,}건
        - 📁 Source: `{data_source}`

        ---

        [GitHub](https://github.com/kiminbean/kpx-demand-forecast) |
        [License](https://github.com/kiminbean/kpx-demand-forecast/blob/main/LICENSE)
        """
    )


if __name__ == "__main__":
    main()
