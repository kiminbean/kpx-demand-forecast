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

        # 운영 모델 성능: Solar V2는 재학습 산출물(metrics_test.json)에서 직접 로드
        solar_metrics_path = Path("models/solar_v2/island_mean/metrics_test.json")
        solar_date, solar_r2, solar_rmse, solar_mae, solar_fer = (
            "2026-11-06", 0.9685, 10.63, 5.25, 4.18,
        )
        if solar_metrics_path.exists():
            try:
                _m = json.loads(solar_metrics_path.read_text())
                _r = _m["by_capacity"]["eff_359"]["jeju_2024_rule"]
                solar_r2 = _r["r2"]
                solar_rmse = _r["rmse_mw"]
                solar_mae = _r["mae_mw"]
                solar_fer = _r["fer_mean_pct"]
                solar_date = datetime.fromtimestamp(
                    solar_metrics_path.stat().st_mtime
                ).strftime("%Y-%m-%d")
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

        # MAPE(주간): 0에 가까운 실측을 분모로 쓰면 폭주하므로 KPX 평가 대상 시간
        # (실측 ≥ 설비용량의 10%)만 계산하고 제외 시간 비율을 함께 표시한다.
        solar_capacity_mw = 359.4
        solar_mape, solar_mape_n, solar_mape_excl_pct = 16.54, 370, 71.8
        solar_predictions_path = Path("models/solar_v2/island_mean/predictions_test.csv")
        if solar_predictions_path.exists():
            try:
                _p = pd.read_csv(solar_predictions_path, index_col=0, parse_dates=True)
                _y = _p["actual_mw"].to_numpy()
                _pred = np.clip(_p["pred_mw"].to_numpy(), 0.0, None)
                _mask = _y >= 0.10 * solar_capacity_mw
                solar_mape = float(
                    np.mean(np.abs(_pred[_mask] - _y[_mask]) / _y[_mask]) * 100.0
                )
                solar_mape_n = int(_mask.sum())
                solar_mape_excl_pct = float(np.mean(~_mask) * 100.0)
            except (KeyError, ValueError):
                pass

        # Wind Phase-5 Fair: 모델 미변경(2024-12-17 학습) — 문서 기준 고정값
        wind_r2, wind_rmse, wind_mae = 0.7576, 14.96, 11.57

        st.markdown(f"""
        ### 📊 현재 모델 성능 (Solar {solar_date} 재학습 기준)

        | 모델 | R² | RMSE | MAE | FER | MAPE(주간) | 상태 |
        |------|-----|------|-----|-----|-----------|------|
        | **Solar V2 (LightGBM, island_mean)** | {solar_r2:.4f} | {solar_rmse:.2f} MW | {solar_mae:.2f} MW | {solar_fer:.2f}% | {solar_mape:.2f}% | ✅ Production |
        | **Wind (LightGBM, Phase-5)** | {wind_r2:.4f} | {wind_rmse:.2f} MW | {wind_mae:.2f} MW | — | — | ✅ Production |

        ※ Solar V2는 테스트셋(2024-11-07~12-31, FER 평가 370시간, C=359.4 MW) 실측 산출물을
        `models/solar_v2/island_mean/metrics_test.json`에서 읽어 표시하므로 재학습 시 자동 갱신됩니다.
        ※ MAPE(주간) {solar_mape:.2f}%는 야간·일출·일몰처럼 실측이 0에 가까워 MAPE가 왜곡되지 않도록
        실측 ≥ 설비용량의 10%(≥ {0.10 * solar_capacity_mw:.1f} MW)인 평가 대상 {solar_mape_n}시간만 계산한 값입니다.
        제외 시간은 전체의 {solar_mape_excl_pct:.1f}%이며, FER({solar_fer:.2f}%)과 수치가 다른 것은
        분모 차이(실측 발전량 vs 설비용량) 때문입니다.
        ※ 관측 기상 입력 기준 수치로, 하루 전 예보(48시간 전 NWP) 조건의 상한선입니다.

        ### 🏁 타사 발전량 예측 오차율 비교 (FER 기준)

        발전량 예측 "오차율"은 업계 표준인 **KPX FER**(예측오차율 = |예측−실측| / 설비용량 × 100,
        전력시장운영규칙 제14.1.1조)으로만 비교합니다. 타사 홈페이지의 "발전량 예측 평균 오차율
        8% 이내" 등 표기는 모두 이 FER 기준이며, MAPE(주간) {solar_mape:.2f}%는 분모(실측 발전량)가
        달라 타사 오차율과 직접 비교할 수 없습니다.

        | 항목 | 기준 | 오차율 |
        |------|------|--------|
        | **Solar V2 (본 프로젝트)** | KPX FER, 관측기상(성능 상한) | **{solar_fer:.2f}%** |
        | Solar V2 하루 전 예보 | KPX FER, 48h 전 NWP(실제 제출 조건) | 8.53% |
        | Wind (본 프로젝트) | KPX FER, 관측기상, 실효용량 104.9 MW | 17.61% (17.4~19.7%) |
        | 타사 주장(풍력, 전국 포트폴리오) | KPX FER | 8% 이내 |
        | 정산 1티어 경계 | 전력시장운영규칙 별표2 | 6% (Jeju ×0.5, 2024-06~) |

        ※ 풍력 FER은 interval-END 라벨 재평가 결과입니다(테스트 2024-11-07~12-31, 평가 1,027h,
        분모는 커버리지 브레이크 후 실효 용량 92.8~106.7 MW, 48h 지속성 34.13% 대비 양호).
        관측 기상 입력의 상한 조건이므로 하루 전 조건에서는 더 나쁩니다 — 풍력은 타사 8%와
        비교 가능한 수준이 아니며 개선이 선행됩니다. 상세: docs/wind_fer_evaluation.md

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
