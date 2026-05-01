#!/usr/bin/env python3
"""
Quant Investing Model - 전문 대시보드

참고용이지만 충분히 완성도 있는 퀀트 리서치 웹앱.
단일 종목의 주가 예측 모델 성과를 분석하고 해석하는 대시보드.

실행: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD"]
MODEL_OPTIONS = [
    "Random Forest",
    "LSTM",
    "GRU",
    "Transformer",
    "Autoformer",
    "QuantFormer",
    "ConvLSTM",
    "THGNN (Temporal Heterogeneous Graph Neural Network)",
    "Time Series Prediction",
]

# ============================================================================
# 1. Mock Data Generator
# ============================================================================

def generate_mock_prediction_data(ticker: str = "ticker_name", n_days: int = 60):
    """백테스트 및 예측 결과 mock 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # 실제값 시계열 (수익률)
    actual_returns = np.random.normal(0.0005, 0.015, n_days)
    
    # 예측값 (약간의 노이즈를 섞은 실제값과 유사한 패턴)
    predicted_returns = actual_returns + np.random.normal(0, 0.008, n_days)
    
    # 누적 수익률 (백테스트 성과)
    cumulative_actual = np.cumprod(1 + actual_returns) - 1
    cumulative_predicted = np.cumprod(1 + predicted_returns) - 1
    
    return {
        "dates": dates,
        "actual_returns": actual_returns,
        "predicted_returns": predicted_returns,
        "cumulative_actual": cumulative_actual,
        "cumulative_predicted": cumulative_predicted,
    }


def generate_mock_metrics(actual: np.ndarray, predicted: np.ndarray):
    """모델 성능 지표 mock 데이터"""
    return {
        "mae": calculate_mae(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "r2": calculate_r2(actual, predicted),
        "directional_accuracy": calculate_directional_accuracy(actual, predicted),
        "hit_ratio": calculate_hit_ratio(actual, predicted),
        "sharpe_ratio": calculate_sharpe_ratio(predicted),
        "max_drawdown": calculate_max_drawdown(predicted),
        "train_r2": max(0.0, min(0.99, calculate_r2(actual, predicted) + 0.08)),
        "test_r2": calculate_r2(actual, predicted),
    }


def generate_mock_feature_importance():
    """Feature Importance mock 데이터"""
    features = [
        "Recent Return (20d)",
        "Volume Rate",
        "Volatility",
        "RSI (14)",
        "MACD Signal",
        "SMA Ratio",
        "ATR",
        "Price Momentum",
    ]
    importance = np.array([0.28, 0.18, 0.15, 0.12, 0.11, 0.08, 0.05, 0.03])
    return pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=False)


def generate_mock_latest_predictions(ticker: str = "ticker_name", n_stocks: int = 5):
    """최신 예측 결과 mock 데이터"""
    tickers = DEFAULT_TICKERS[:n_stocks]
    predictions = np.random.normal(0.001, 0.008, n_stocks)
    
    return pd.DataFrame({
        "Ticker": tickers,
        "Pred_Return": predictions,
        "Direction": ["↓" if p < -0.002 else "↑" if p > 0.002 else "→" for p in predictions],
    })


def get_tickers_from_provider(query: str) -> list[str]:
    """입력값과 일치하는 mock 티커 목록 반환"""
    normalized = query.strip().upper()
    if not normalized:
        return DEFAULT_TICKERS

    matches = [ticker for ticker in DEFAULT_TICKERS if normalized in ticker]
    return matches or DEFAULT_TICKERS


# ============================================================================
# 2. 지표 계산 함수
# ============================================================================

def calculate_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """방향성 정확도: 상승/하락 방향을 맞춘 비율"""
    actual_dir = np.sign(actual)
    pred_dir = np.sign(predicted)
    accuracy = np.mean(actual_dir == pred_dir)
    return accuracy


def calculate_hit_ratio(actual: np.ndarray, predicted: np.ndarray, threshold: float = 0.001) -> float:
    """Hit Ratio: |예측값 - 실제값| < threshold 비율"""
    errors = np.abs(predicted - actual)
    hit_ratio = np.mean(errors < threshold)
    return hit_ratio


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE 계산"""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAE 계산"""
    return np.mean(np.abs(actual - predicted))


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R² 계산"""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """Sharpe Ratio 계산"""
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """최대 낙폭 계산"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


# ============================================================================
# 3. 예측 해석 함수
# ============================================================================

def interpret_prediction(pred_return: float, confidence: float = 0.5) -> dict:
    """예측값을 투자 신호로 해석"""
    abs_pred = abs(pred_return)
    
    # 신호 강도 판정
    if abs_pred < 0.002:
        signal_strength = "약함"
        signal_power = "Low"
    elif abs_pred < 0.005:
        signal_strength = "중간"
        signal_power = "Medium"
    else:
        signal_strength = "강함"
        signal_power = "High"
    
    # 방향성 판정
    if pred_return > 0.002:
        direction = "Bullish"
        direction_kr = "상승"
        color = "green"
    elif pred_return < -0.002:
        direction = "Bearish"
        direction_kr = "하락"
        color = "red"
    else:
        direction = "Neutral"
        direction_kr = "중립"
        color = "gray"
    
    # 해석 문장
    if direction == "Bullish":
        interpretation = f"다음 거래일 약 +{abs_pred*100:.2f}%의 상승을 예측합니다. ({signal_strength} 신호)"
    elif direction == "Bearish":
        interpretation = f"다음 거래일 약 -{abs_pred*100:.2f}%의 하락을 예측합니다. ({signal_strength} 신호)"
    else:
        interpretation = f"다음 거래일 거의 변화가 없을 것으로 예측합니다. (신호 강도: {signal_strength})"
    
    return {
        "direction": direction,
        "direction_kr": direction_kr,
        "signal_strength": signal_strength,
        "signal_power": signal_power,
        "interpretation": interpretation,
        "color": color,
        "pred_return_pct": pred_return * 100,
    }


def generate_summary_text(ticker: str, model_name: str, pred_return: float, 
                          directional_acc: float, rmse: float, test_r2: float) -> str:
    """결과 요약 문장 생성"""
    interp = interpret_prediction(pred_return)
    
    summary = (
        f"### 📊 분석 요약\n\n"
        f"**{ticker}** 종목에 대해 **{model_name}** 모델은 다음 거래일 수익률을 "
        f"**{interp['pred_return_pct']:.2f}%**로 예측하고 있습니다.\n\n"
        f"방향성은 **{interp['direction_kr']} 신호({interp['signal_strength']})**이며, "
        f"최근 테스트 기준 방향성 정확도는 **{directional_acc*100:.1f}%**, "
        f"R²는 **{test_r2:.3f}**입니다.\n\n"
        f"**💡 해석**: {interp['interpretation']}"
    )
    
    # 신뢰도 평가
    if test_r2 < 0.1:
        summary += "\n\n⚠️ **주의**: 모델 설명력이 매우 낮습니다. 신뢰도는 낮게 평가됩니다."
    elif directional_acc < 0.5:
        summary += "\n\n⚠️ **주의**: 방향성 정확도가 50% 미만입니다. 신중한 해석이 필요합니다."
    
    return summary


# ============================================================================
# 4. 차트 생성 함수
# ============================================================================

def plot_actual_vs_predicted(data: dict, title: str = "실제값 vs 예측값 (수익률)"):
    """실제값 vs 예측값 선 그래프"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["dates"],
        y=data["actual_returns"] * 100,
        name="실제 수익률",
        mode="lines",
        line=dict(color="blue", width=2),
        opacity=0.7,
    ))
    
    fig.add_trace(go.Scatter(
        x=data["dates"],
        y=data["predicted_returns"] * 100,
        name="예측 수익률",
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        opacity=0.7,
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="수익률 (%)",
        hovermode="x unified",
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig


def plot_cumulative_returns(data: dict):
    """누적 수익률 곡선"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["dates"],
        y=data["cumulative_actual"] * 100,
        name="실제 누적 수익률",
        mode="lines",
        line=dict(color="blue", width=3),
        fill="tozeroy",
        fillcolor="rgba(0, 0, 255, 0.1)",
    ))
    
    fig.add_trace(go.Scatter(
        x=data["dates"],
        y=data["cumulative_predicted"] * 100,
        name="예측 기반 누적 수익률",
        mode="lines",
        line=dict(color="green", width=3),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 0, 0.1)",
    ))
    
    fig.update_layout(
        title="누적 수익률 비교 (백테스트)",
        xaxis_title="날짜",
        yaxis_title="누적 수익률 (%)",
        hovermode="x unified",
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig


def plot_feature_importance(feature_imp_df: pd.DataFrame):
    """Feature Importance 막대 차트"""
    fig = px.bar(
        feature_imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        labels={"Importance": "중요도", "Feature": "피처"},
        color="Importance",
        color_continuous_scale="Blues",
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=200, r=50, t=50, b=50),
    )
    
    return fig


def plot_prediction_residuals(actual: np.ndarray, predicted: np.ndarray):
    """예측 오차 분포"""
    residuals = actual - predicted
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals * 100,
        name="예측 오차",
        nbinsx=30,
        marker_color="rgba(100, 150, 255, 0.7)",
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="정확 예측")
    
    fig.update_layout(
        title="예측 오차 분포",
        xaxis_title="오차 (%)",
        yaxis_title="빈도",
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig


# ============================================================================
# 5. UI 컴포넌트
# ============================================================================

def render_kpi_card(label: str, value: str, sublabel: str = "", color: str = "blue"):
    """KPI 카드 렌더링"""
    color_map = {
        "green": "🟢",
        "red": "🔴",
        "gray": "⚫",
        "blue": "🔵",
    }
    
    st.metric(
        label=f"{color_map.get(color, '●')} {label}",
        value=value,
        delta=sublabel if sublabel else None,
    )


def render_model_info_card(model_name: str, target: str, features_used: int, 
                          train_period: str, test_period: str):
    """모델 정보 카드"""
    with st.expander("📋 모델 상세 정보", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**모델명**")
            st.write(model_name)
            st.write("**예측 목표**")
            st.write(target)
            st.write("**사용 피처**")
            st.write(f"{features_used}개")
        
        with col2:
            st.write("**학습 기간**")
            st.write(train_period)
            st.write("**테스트 기간**")
            st.write(test_period)
            st.write("**데이터 소스**")
            st.write("Yahoo Finance")


def render_metrics_grid(metrics: dict):
    """성능 지표 그리드"""
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_list = [
        ("MAE", f"{metrics['mae']:.4f}", "평균 절대 오차 (낮을수록 좋음)", col1),
        ("RMSE", f"{metrics['rmse']:.4f}", "평균 제곱근 오차 (낮을수록 좋음)", col2),
        ("R²", f"{metrics['r2']:.3f}", "설명력 (1에 가까울수록 좋음)", col3),
        ("정확도", f"{metrics['directional_accuracy']*100:.1f}%", "상승/하락 방향 맞춘 비율", col4),
    ]
    
    for label, value, tooltip, col in metrics_list:
        with col:
            st.metric(label, value)
            st.caption(f"💡 {tooltip}")


def render_latest_predictions_table(pred_df: pd.DataFrame):
    """최신 예측 테이블"""
    # 예측값 해석
    pred_df_display = pred_df.copy()
    pred_df_display["Pred_Return_Pct"] = (pred_df_display["Pred_Return"] * 100).apply(
        lambda x: f"{x:+.2f}%"
    )
    pred_df_display["Signal"] = pred_df_display["Pred_Return"].apply(
        lambda x: "↑ 상승" if x > 0.002 else "↓ 하락" if x < -0.002 else "→ 중립"
    )
    pred_df_display["Interpretation"] = pred_df_display["Pred_Return"].apply(
        lambda x: "매수 신호" if x > 0.005 else "약한 매수" if x > 0.002 
                  else "매도 신호" if x < -0.005 else "약한 매도" if x < -0.002
                  else "중립"
    )
    pred_df_display["예측 시각"] = datetime.now().strftime("%H:%M:%S")
    
    display_df = pred_df_display[["Ticker", "Pred_Return_Pct", "Signal", "Interpretation", "예측 시각"]]
    display_df.columns = ["종목", "예측 수익률", "신호", "해석", "예측 시각"]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================================
# 6. Main 실행 함수
# ============================================================================

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="Quant Research Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # 커스텀 스타일
    st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .summary-block { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5rem; 
                     border-left: 4px solid #1976d2; border-radius: 0.5rem; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)
    
    # ============ 헤더 ============
    st.title("📈 Quant Research Dashboard")
    st.caption("머신러닝 기반 주가 예측 모델 성과 분석 및 해석 대시보드")
    
    # ============ 사이드바 ============
    with st.sidebar:
        st.header("⚙️ 설정")
        ticker = DEFAULT_TICKERS[0]
        # 1단계: 텍스트 입력
        query = st.text_input("종목 검색", placeholder="분석할 종목의 티커를 입력하세요.")
        # 2단계: 입력값 기반으로 프로바이더에서 검색 후 선택
        if query:
            results = get_tickers_from_provider(query)
            ticker = st.selectbox("종목 선택", options=results)
        else:
            ticker = st.selectbox("종목 선택", options=DEFAULT_TICKERS)
        model_name = st.selectbox("모델", MODEL_OPTIONS)
        lookback_days = st.slider("분석 기간 (일)", 30, 365, 60)
        
        st.divider()
        st.info("💡 이 대시보드는 참고용 모델의 성과를 분석하는 도구입니다. 실제 투자 판단에는 추가 검증이 필요합니다.")
    
    # ============ Mock 데이터 로드 ============
    pred_data = generate_mock_prediction_data(ticker, lookback_days)
    metrics = generate_mock_metrics(
        pred_data["actual_returns"],
        pred_data["predicted_returns"],
    )
    feature_imp = generate_mock_feature_importance()
    latest_preds = generate_mock_latest_predictions(ticker, 5)
    
    # 추가 지표 계산
    directional_acc = metrics["directional_accuracy"]
    
    latest_pred = np.mean(pred_data["predicted_returns"][-5:])
    interp = interpret_prediction(latest_pred)
    
    # ============ 섹션 1: 분석 요약 ============
    st.markdown(generate_summary_text(
        ticker, model_name, latest_pred,
        directional_acc, metrics["rmse"], metrics["test_r2"]
    ))
    
    # ============ 섹션 2: KPI 카드 ============
    st.subheader("🎯 핵심 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "현재 신호",
            interp["direction_kr"],
            interp["signal_strength"],
        )
    
    with col2:
        st.metric(
            "예측 수익률",
            f"{interp['pred_return_pct']:.2f}%",
        )
    
    with col3:
        st.metric(
            "방향성 정확도",
            f"{directional_acc*100:.1f}%",
        )
    
    with col4:
        st.metric(
            "Test R²",
            f"{metrics['test_r2']:.3f}",
        )
    
    # ============ 섹션 3: 성능 지표 상세 ============
    st.subheader("📊 모델 성능 상세 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**오차 지표**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("MAE", f"{metrics['mae']:.4f}")
            st.caption("프낸 평균 절대 오차")
        with col_b:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
            st.caption("평균 제곱근 오차")
        with col_c:
            st.metric("R²", f"{metrics['r2']:.3f}")
            st.caption("설명력 지수")
    
    with col2:
        st.markdown("**정확도 지표**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("정확도", f"{directional_acc*100:.1f}%")
            st.caption("방향성 정확도")
        with col_b:
            st.metric("Hit Ratio", f"{metrics['hit_ratio']*100:.1f}%")
            st.caption("오차 내 비율")
        with col_c:
            st.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
            st.caption("위험조정 수익률")
    
    # ============ 섹션 4: 차트 ============
    st.subheader("📈 예측 성과 시각화")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "실제 vs 예측",
        "누적 수익률",
        "오차 분포",
        "Feature 중요도"
    ])
    
    with tab1:
        st.plotly_chart(
            plot_actual_vs_predicted(pred_data),
            use_container_width=True
        )
    
    with tab2:
        st.plotly_chart(
            plot_cumulative_returns(pred_data),
            use_container_width=True
        )
    
    with tab3:
        st.plotly_chart(
            plot_prediction_residuals(
                pred_data["actual_returns"],
                pred_data["predicted_returns"]
            ),
            use_container_width=True
        )
    
    with tab4:
        st.plotly_chart(
            plot_feature_importance(feature_imp),
            use_container_width=True
        )
    
    # ============ 섹션 5: 최신 예측 ============
    st.subheader("🎪 최신 예측 결과")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**예측 테이블**")
        render_latest_predictions_table(latest_preds)
    
    with col2:
        st.markdown("**예측값 분포**")
        fig_dist = px.histogram(
            latest_preds,
            x="Pred_Return",
            nbinsx=10,
            title="예측 수익률 분포",
            labels={"Pred_Return": "예측 수익률 (%)"},
        )
        fig_dist.update_xaxes(tickformat=".2%")
        fig_dist.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # ============ 섹션 6: 모델 정보 ============
    st.divider()
    st.subheader("ℹ️ 모델 및 데이터 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_model_info_card(
            model_name=model_name,
            target="다음 거래일 수익률 예측",
            features_used=len(feature_imp),
            train_period="2023-01-01 ~ 2025-01-01",
            test_period="2024-01-01 ~ 현재"
        )
    
    with col2:
        st.markdown("**주의사항**")
        st.warning(
            "⚠️ 본 대시보드는 참고용입니다.\n\n"
            "실제 매매는 다음을 고려하세요:\n"
            "• 추가 기본분석\n"
            "• 리스크 관리\n"
            "• 포트폴리오 다양화\n"
            "• 시장 상황 변화\n"
        )
    
    with col3:
        st.markdown("**해석 가이드**")
        st.info(
            "**신호 강도:**\n"
            "• 강함: |수익률| > 0.5%\n"
            "• 중간: 0.2% ~ 0.5%\n"
            "• 약함: < 0.2%\n\n"
            "**신호 방향:**\n"
            "• 상승(Bullish): > +0.2%\n"
            "• 하락(Bearish): < -0.2%\n"
            "• 중립(Neutral): ±0.2% 이내"
        )
    
    # ============ 푸터 ============
    st.divider()
    st.caption(
        "📊 AI Based Quantitative Investing | "
        "학습용 머신러닝 모델 분석 도구 | "
        f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
