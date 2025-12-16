"""
J-REF Dashboard

제주도 신재생에너지 발전량 예측 대시보드

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, DEVICE


# 페이지 설정
st.set_page_config(
    page_title="J-REF Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    """데이터 로드 (캐싱)"""
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"

    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_resource
def load_models():
    """모델 로드 (캐싱)"""
    try:
        from src.inference import load_predictor
        solar_predictor = load_predictor("solar")
        wind_predictor = load_predictor("wind")
        return {"solar": solar_predictor, "wind": wind_predictor}
    except Exception as e:
        st.warning(f"모델 로드 실패: {e}")
        return None


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


def make_prediction(models, df, model_type, n_rows=100):
    """예측 수행"""
    if models is None:
        return None, "모델이 로드되지 않았습니다."

    try:
        predictor = models.get(model_type)
        if predictor is None:
            return None, f"{model_type} 모델이 없습니다."

        test_df = df.tail(n_rows).copy()
        predictions = predictor.predict(test_df)

        return predictions, None
    except Exception as e:
        return None, str(e)


def main():
    # 사이드바
    st.sidebar.title("J-REF Dashboard")
    st.sidebar.markdown("제주도 신재생에너지 발전량 예측")

    # 데이터 로드
    df = load_data()
    models = load_models()

    if df is None:
        st.error("데이터를 찾을 수 없습니다. 먼저 샘플 데이터를 생성하세요.")
        st.code("python scripts/download_data.py --sample --year 2024")
        return

    # 페이지 선택
    page = st.sidebar.radio(
        "페이지 선택",
        ["개요", "발전량 분석", "예측", "모델 성능"],
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
    if page == "개요":
        st.title("제주도 신재생에너지 발전량 현황")

        # KPI 카드
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="총 데이터",
                value=f"{len(filtered_df):,}건",
                delta=f"{len(df):,}건 전체"
            )

        with col2:
            st.metric(
                label="태양광 평균",
                value=f"{filtered_df['solar_mw'].mean():.1f} MW",
                delta=f"최대 {filtered_df['solar_mw'].max():.1f} MW"
            )

        with col3:
            st.metric(
                label="풍력 평균",
                value=f"{filtered_df['wind_mw'].mean():.1f} MW",
                delta=f"최대 {filtered_df['wind_mw'].max():.1f} MW"
            )

        with col4:
            st.metric(
                label="총 발전량",
                value=f"{filtered_df['total_mw'].mean():.1f} MW",
                delta=f"최대 {filtered_df['total_mw'].max():.1f} MW"
            )

        # 개요 플롯
        st.plotly_chart(plot_generation_overview(filtered_df), use_container_width=True)

    elif page == "발전량 분석":
        st.title("발전량 상세 분석")

        tab1, tab2, tab3 = st.tabs(["시간대별 패턴", "기상 상관관계", "원본 데이터"])

        with tab1:
            st.plotly_chart(plot_hourly_pattern(filtered_df), use_container_width=True)

            # 요일별 히트맵
            filtered_df["dayofweek"] = filtered_df["datetime"].dt.dayofweek
            filtered_df["hour"] = filtered_df["datetime"].dt.hour

            pivot = filtered_df.pivot_table(
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
                fig_scatter = px.scatter(
                    filtered_df.sample(min(1000, len(filtered_df))),
                    x="solar_radiation",
                    y="solar_mw",
                    color="hour",
                    title="일사량 vs 태양광 발전량",
                    labels={"solar_radiation": "일사량", "solar_mw": "발전량 (MW)"},
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.dataframe(filtered_df.tail(100), use_container_width=True)

            # 다운로드 버튼
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="CSV 다운로드",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv",
            )

    elif page == "예측":
        st.title("발전량 예측")

        if models is None:
            st.warning("모델이 로드되지 않았습니다. 먼저 모델을 학습하세요.")
            st.code("python scripts/train.py --model all --quick")
            return

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "모델 선택",
                ["solar", "wind"],
                format_func=lambda x: "태양광" if x == "solar" else "풍력"
            )

        with col2:
            n_rows = st.slider("입력 데이터 크기", 50, 500, 100)

        if st.button("예측 실행"):
            with st.spinner("예측 중..."):
                predictions, error = make_prediction(models, filtered_df, model_type, n_rows)

                if error:
                    st.error(f"예측 실패: {error}")
                else:
                    st.success(f"예측 완료: {len(predictions)}개 시점")

                    # 예측 결과 표시
                    target_col = "solar_mw" if model_type == "solar" else "wind_mw"
                    seq_length = 24 if model_type == "solar" else 48

                    test_df = filtered_df.tail(n_rows).copy()
                    pred_times = test_df["datetime"].values[seq_length - 1:][:len(predictions)]
                    actuals = test_df[target_col].values[seq_length - 1:][:len(predictions)]

                    result_df = pd.DataFrame({
                        "datetime": pred_times,
                        "actual": actuals,
                        "predicted": predictions,
                    })

                    # 시계열 플롯
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result_df["datetime"],
                        y=result_df["actual"],
                        name="실제값",
                        line=dict(color="blue", width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=result_df["datetime"],
                        y=result_df["predicted"],
                        name="예측값",
                        line=dict(color="red", width=2, dash="dash")
                    ))

                    fig.update_layout(
                        title=f"{'태양광' if model_type == 'solar' else '풍력'} 발전량 예측 결과",
                        xaxis_title="시간",
                        yaxis_title="발전량 (MW)",
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # 메트릭
                    st.subheader("예측 통계")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("예측 평균", f"{predictions.mean():.2f}")
                    with col2:
                        st.metric("예측 표준편차", f"{predictions.std():.2f}")
                    with col3:
                        st.metric("예측 범위", f"{predictions.min():.2f} ~ {predictions.max():.2f}")

    elif page == "모델 성능":
        st.title("모델 성능 분석")

        # 학습 히스토리 로드
        for model_type in ["solar", "wind"]:
            history_path = MODELS_DIR / f"{model_type}_model" / "training_history.json"

            if history_path.exists():
                import json
                with open(history_path) as f:
                    history = json.load(f)

                st.subheader(f"{'태양광' if model_type == 'solar' else '풍력'} 모델")

                fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "MAPE"))

                # Loss
                fig.add_trace(go.Scatter(
                    y=history["train_loss"],
                    name="Train Loss",
                    line=dict(color="blue")
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    y=history["val_loss"],
                    name="Val Loss",
                    line=dict(color="orange")
                ), row=1, col=1)

                # MAPE
                if "mape" in history:
                    fig.add_trace(go.Scatter(
                        y=history["mape"],
                        name="MAPE",
                        line=dict(color="green")
                    ), row=1, col=2)

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # 최종 성능
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
                with col2:
                    st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
                with col3:
                    if "mape" in history:
                        st.metric("Final MAPE", f"{history['mape'][-1]:.2f}%")

            else:
                st.info(f"{model_type} 모델의 학습 히스토리가 없습니다.")

    # 푸터
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **J-REF v0.1**

        제주도 신재생에너지 발전량 예측 시스템

        - Device: """ + str(DEVICE) + """
        - Data: """ + (f"{len(df):,}건" if df is not None else "없음") + """
        """
    )


if __name__ == "__main__":
    main()
