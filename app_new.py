#!/usr/bin/env python3
"""
Quantitative Investment Model - 초보자 친화 웹 앱

Streamlit 기반 웹 인터페이스.
단일 티커의 주가 예측 및 정량 투자 분석을 브라우저에서 실행합니다.

실행: streamlit run app_new.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

try:
    from quant_investing_model import (
        download_data_with_volume,
        calculate_metrics,
        calculate_var,
        download_fama_french_factors,
        run_factor_model,
        THGNN_DEPS_OK,
    )
    from prediction_hub import (
        run_stock_prediction,
        prediction_model_ids,
        prediction_model_descriptions,
        PredictionHubError,
    )

    DEPS_OK = True
    try:
        import sklearn  # noqa: F401
        SKLEARN_OK = True
    except ImportError:
        SKLEARN_OK = False
except ImportError as e:
    DEPS_OK = False
    SKLEARN_OK = False
    IMPORT_ERROR = str(e)


# ============ 페이지 설정 ============
st.set_page_config(
    page_title="Quant Investing Model - 초보자용",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stAlert { margin-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .pipeline-step {
        background-color: #e8f4f8;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0099cc;
        border-radius: 0.25rem;
    }
    .pipeline-success {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .pipeline-error {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# ============ 초기화 ============
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []


def log_pipeline(stage: str, message: str, status: str = "info"):
    """파이프라인 로그 기록"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "stage": stage,
        "message": message,
        "status": status,
    }
    st.session_state.pipeline_logs.append(log_entry)


def display_pipeline():
    """파이프라인 현황 표시"""
    if st.session_state.pipeline_logs:
        st.subheader("🔄 분석 진행 상황")
        for log in st.session_state.pipeline_logs:
            css_class = "pipeline-step"
            if log["status"] == "success":
                css_class += " pipeline-success"
            elif log["status"] == "error":
                css_class += " pipeline-error"
            
            icon = "✅" if log["status"] == "success" else "❌" if log["status"] == "error" else "⏳"
            st.markdown(
                f'<div class="{css_class}">'
                f'{icon} <strong>[{log["timestamp"]}] {log["stage"]}</strong>: {log["message"]}'
                f'</div>',
                unsafe_allow_html=True
            )


# ============ 차트 함수들 ============
def render_price_chart(prices: pd.DataFrame, ticker: str):
    """가격 시계열 차트"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices.index, prices[ticker], linewidth=2, color="#0099cc")
    ax.set_title(f"{ticker} 가격 시계열", fontsize=12, fontweight="bold")
    ax.set_ylabel("가격 ($)", fontsize=10)
    ax.set_xlabel("날짜", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_cumulative_return_chart(prices: pd.DataFrame, ticker: str):
    """누적 수익률 차트"""
    returns = prices.pct_change().dropna()
    cum_returns = (1 + returns).cumprod() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum_returns.index, cum_returns[ticker] * 100, linewidth=2, color="#00aa44")
    ax.fill_between(cum_returns.index, cum_returns[ticker] * 100, alpha=0.2, color="#00aa44")
    ax.set_title(f"{ticker} 누적 수익률", fontsize=12, fontweight="bold")
    ax.set_ylabel("수익률 (%)", fontsize=10)
    ax.set_xlabel("날짜", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_factor_betas_chart(factor_results: dict, ticker: str):
    """팩터 β 막대 차트"""
    if ticker not in factor_results:
        return None
    
    res = factor_results[ticker]
    params = res.get("params", {})
    factors = ["Mkt-RF", "SMB", "HML"]
    betas = [params.get(f, 0) for f in factors]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if b >= 0 else "#e74c3c" for b in betas]
    ax.bar(factors, betas, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("β 계수", fontsize=10)
    ax.set_title(f"{ticker} - 팩터 노출도 (Fama-French)", fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def render_metrics_chart(metrics: pd.DataFrame, ticker: str):
    """주요 지표 시각화"""
    if ticker not in metrics.index:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # metrics는 행=티커, 열=지표명
    ticker_metrics = metrics.loc[ticker]
    
    metric_names = ["누적 수익률", "연율화 수익률", "연율화 변동성", "샤프 비율"]
    values = []
    for name in metric_names:
        val = ticker_metrics.get(name, 0)
        if isinstance(val, str):
            val = float(val.replace("%", ""))
        values.append(val)
    
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    bars = ax.bar(metric_names, values, color=colors, alpha=0.8, edgecolor="black")
    
    # 값 라벨 추가
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{val:.2f}" if abs(val) < 100 else f"{val/100:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight="bold")
    
    ax.set_ylabel("값", fontsize=10)
    ax.set_title(f"{ticker} - 주요 지표", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    return fig


# ============ 메인 함수 ============
def main():
    if not DEPS_OK:
        st.error(
            f"필수 패키지가 설치되지 않았습니다.\n\n"
            f"```bash\npip install -r requirements.txt\n```\n\n"
            f"오류: {IMPORT_ERROR}"
        )
        return

    # ============ 헤더 ============
    st.title("📈 Quant Investing Model")
    st.caption("초보자도 쉽게 시작하는 정량 투자 분석 & 주가 예측")

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>면책사항</strong>: 이 도구는 투자 자문이 아니며, 모든 의사결정과 손익은 본인 책임입니다.
        과거 데이터 기반 분석이므로 미래 수익을 보장하지 않습니다.
    </div>
    """, unsafe_allow_html=True)

    # ============ 사이드바 입력 ============
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        
        ticker = st.text_input(
            "📍 종목 티커 (예: AAPL)",
            value="AAPL",
            max_chars=10,
            placeholder="AAPL, MSFT, TSLA 등..."
        ).strip().upper()
        
        if not ticker:
            st.error("티커를 입력해주세요.")
            return
        
        st.divider()
        
        lookback_days = st.slider(
            "📅 분석 기간 (일)",
            min_value=60,
            max_value=730,
            value=365,
            step=30,
            help="과거 몇 일의 데이터를 사용해 분석할지 선택합니다."
        )
        
        risk_free_rate = st.number_input(
            "📊 무위험 수익률 (연율, %)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help="일반적으로 10년물 미국 국채 수익률 사용"
        ) / 100.0
        
        st.divider()
        
        # 분석 모델 선택
        st.subheader("🎯 분석 모델 선택 (중복 가능)")
        
        analysis_models = []
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("📊 위험도 분석 (VaR)", value=True):
                analysis_models.append("var")
            if st.checkbox("🧮 팩터 분석 (FF3)", value=True):
                analysis_models.append("fama_french")
            if st.checkbox("🤖 LSTM 예측", value=False, help="시간이 소요될 수 있습니다"):
                analysis_models.append("lstm")
        
        with col2:
            if st.checkbox("⚡ QuantFormer", value=False, help="최신 양자화 모델"):
                analysis_models.append("quantformer")
            if st.checkbox("🌲 Random Forest", value=False, help="앙상블 모델"):
                analysis_models.append("random_forest")
            if st.checkbox("📈 Ridge 회귀", value=False, help="선형 모델"):
                analysis_models.append("ridge")
        
        if not analysis_models:
            st.warning("최소 1개 이상의 분석 모델을 선택해주세요.")
        
        st.divider()
        
        analyze_btn = st.button(
            "🚀 분석 시작",
            type="primary",
            use_container_width=True,
            disabled=not analysis_models or not ticker
        )

    if not analyze_btn:
        st.info("👈 **좌측 사이드바**에서 티커를 입력하고 분석 모델을 선택한 후 **분석 시작** 버튼을 누르세요.")
        return

    # ============ 파이프라인 시작 ============
    st.session_state.pipeline_logs = []  # 초기화
    
    # 전체 컨테이너
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.subheader("🔄 분석 진행 상황")
        progress_placeholder = st.empty()
    
    # ============ 1단계: 데이터 다운로드 ============
    # ============ 1단계: 데이터 다운로드 ============
    log_pipeline("데이터 수집", "yfinance에서 주가 데이터 추출 중...", "info")
    
    try:
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # yfinance만 사용 (API 키 불필요)
        prices = None
        volumes = None
        
        try:
            prices, volumes = download_data_with_volume(
                [ticker],
                start_date,
                end_date,
                provider="yfinance",
                use_local_store=False,
                force_refresh=True,
            )
        except:
            from quant_investing_model import download_data
            prices = download_data(
                [ticker],
                start_date,
                end_date,
                provider="yfinance",
                use_local_store=False,
                force_refresh=True,
            )
        
        if prices is None or prices.empty or ticker not in prices.columns:
            raise ValueError(f"yfinance에서 {ticker} 데이터를 가져올 수 없습니다.")
        
        log_pipeline("데이터 수집", "✅ yfinance에서 데이터 수집 성공", "success")
        
        if prices is None or prices.empty:
            raise ValueError("yfinance에서 데이터를 가져올 수 없습니다.")
        
        prices = prices[[ticker]]  # 단일 컬럼만
        
        with progress_placeholder.container():
            display_pipeline()
        
    except Exception as e:
        log_pipeline("데이터 수집", f"❌ 실패: {str(e)}", "error")
        with progress_placeholder.container():
            display_pipeline()
        st.error(f"데이터 수집 실패: {e}")
        return

    # ============ 결과 표시 섹션 ============
    with results_container:
        st.divider()
        st.success(f"✅ 데이터 로드 완료 ({len(prices)} 거래일, yfinance 제공)")
        
        # 2단계: 메트릭 계산
        log_pipeline("메트릭 계산", "수익률, 변동성, 샤프 비율 계산 중...", "info")
        with progress_placeholder.container():
            display_pipeline()
        
        try:
            metrics = calculate_metrics(prices, risk_free_rate)
            log_pipeline("메트릭 계산", "✅ 완료", "success")
        except Exception as e:
            log_pipeline("메트릭 계산", f"❌ 실패: {str(e)[:50]}", "error")
            metrics = None
        
        with progress_placeholder.container():
            display_pipeline()
        
        # ============섹션 1: 주요 지표 ============
        st.header("📊 주요 투자 지표")
        
        if metrics is not None and ticker in metrics.index:
            col1, col2, col3, col4 = st.columns(4)
            
            # metrics는 행=티커, 열=지표명
            ticker_metrics = metrics.loc[ticker]
            
            with col1:
                val = ticker_metrics.get("누적 수익률", 0)
                if isinstance(val, str):
                    val = float(val.replace("%", ""))
                st.metric("누적 수익률", f"{val:.2%}")
            
            with col2:
                val = ticker_metrics.get("연율화 수익률", 0)
                if isinstance(val, str):
                    val = float(val.replace("%", ""))
                st.metric("연율 수익률", f"{val:.2%}")
            
            with col3:
                val = ticker_metrics.get("연율화 변동성", 0)
                if isinstance(val, str):
                    val = float(val.replace("%", ""))
                st.metric("연율 변동성", f"{val:.2%}")
            
            with col4:
                val = ticker_metrics.get("샤프 비율", 0)
                if isinstance(val, str):
                    val = float(val.replace("%", ""))
                st.metric("샤프 비율", f"{val:.3f}")
            
            # 상세 보기 (DataFrame 전치 - 티커별 지표 보기)
            st.dataframe(pd.DataFrame(ticker_metrics).T, use_container_width=True)
        elif metrics is not None:
            st.warning(f"'{ticker}' 데이터에 대한 메트릭이 없습니다.")
        
        # ============ 섹션 2: 차트 ============
        st.header("📈 시각 분석")
        
        tab1, tab2, tab3 = st.tabs(["가격 추이", "수익률 추이", "지표 대시보드"])
        
        with tab1:
            st.pyplot(render_price_chart(prices, ticker))
        
        with tab2:
            st.pyplot(render_cumulative_return_chart(prices, ticker))
        
        with tab3:
            if metrics is not None:
                st.pyplot(render_metrics_chart(metrics, ticker))
        
        # ============ 섹션 3: VaR 분석 ============
        if "var" in analysis_models:
            log_pipeline("VaR 분석", "Value-at-Risk 계산 중...", "info")
            with progress_placeholder.container():
                display_pipeline()
            
            try:
                returns = prices.pct_change().dropna()
                var_95 = calculate_var(returns[ticker], confidence=0.95)
                var_99 = calculate_var(returns[ticker], confidence=0.99)
                
                log_pipeline("VaR 분석", "✅ 완료", "success")
                
                st.header("⚠️ 위험도 분석 (VaR)")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("일일 VaR (95% 신뢰도)", f"{var_95:.2%}", 
                             help="하루에 95% 확률로 이만큼 이상 손실할 수 있음")
                with col2:
                    st.metric("일일 VaR (99% 신뢰도)", f"{var_99:.2%}",
                             help="하루에 99% 확률로 이만큼 이상 손실할 수 있음")
            except Exception as e:
                log_pipeline("VaR 분석", f"❌ 실패: {str(e)[:50]}", "error")
                st.warning(f"VaR 계산 실패: {e}")
            
            with progress_placeholder.container():
                display_pipeline()
        
        # ============ 섹션 4: Fama-French 분석 ============
        if "fama_french" in analysis_models:
            log_pipeline("팩터 분석", "Fama-French 3-팩터 데이터 다운로드 중...", "info")
            with progress_placeholder.container():
                display_pipeline()
            
            try:
                returns = prices.pct_change().dropna()
                factors = download_fama_french_factors(start_date, end_date)
                factor_result = run_factor_model(returns, factors, risk_free_rate)
                
                log_pipeline("팩터 분석", "✅ 완료", "success")
                
                st.header("📐 Fama-French 3-팩터 분석")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = render_factor_betas_chart(factor_result, ticker)
                    if fig:
                        st.pyplot(fig)
                
                with col2:
                    if ticker in factor_result:
                        res = factor_result[ticker]
                        params = res.get("params", {})
                        
                        st.write("**팩터 민감도 (β)**")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Market (β_Mkt)", f"{params.get('Mkt-RF', 0):.3f}")
                        with col_b:
                            st.metric("Size (β_SMB)", f"{params.get('SMB', 0):.3f}")
                        with col_c:
                            st.metric("Value (β_HML)", f"{params.get('HML', 0):.3f}")
                        
                        if "R-squared" in res:
                            st.metric("R² (설명력)", f"{res['R-squared']:.3f}")
                
                # 상세 회귀 요약
                with st.expander("📋 상세 회귀 결과"):
                    if ticker in factor_result and "summary" in factor_result[ticker]:
                        st.text(factor_result[ticker]["summary"])
            
            except Exception as e:
                log_pipeline("팩터 분석", f"❌ 실패: {str(e)[:50]}", "error")
                st.warning(f"팩터 분석 실패: {e}")
            
            with progress_placeholder.container():
                display_pipeline()
        
        # ============ 섹션 5: 머신러닝 예측 ============
        ml_models = [m for m in analysis_models if m in ["lstm", "quantformer", "random_forest", "ridge"]]
        
        if ml_models:
            st.header("🤖 머신러닝 활용 주가 예측")
            st.caption(f"선택된 모델: {', '.join(ml_models).upper()}")
            
            results_tabs = st.tabs([m.upper() for m in ml_models])
            
            for model_name, tab in zip(ml_models, results_tabs):
                with tab:
                    log_pipeline("ML 예측", f"{model_name} 모델 학습 중...", "info")
                    with progress_placeholder.container():
                        display_pipeline()
                    
                    try:
                        # 데이터 크기에 맞게 시퀀스 길이 자동 조정
                        n_trading_days = len(prices)
                        # seq_len은 데이터의 10~30% (최소 5, 최대 30)
                        adaptive_seq_len = max(5, min(30, int(n_trading_days * 0.15)))
                        
                        # 기본 하이퍼파라미터
                        pred_kw = {
                            "lookback": adaptive_seq_len,
                            "seq_len": adaptive_seq_len,
                            "train_ratio": 0.7,
                            "epochs": 8 if model_name in ("lstm", "quantformer") else 12,
                            "lr": 1e-3,
                            "hidden": 32,
                            "num_layers": 1,
                            "rho": 3,
                            "phi": 0.2,
                            "corr_window": min(20, adaptive_seq_len),
                            "corr_threshold": 0.6,
                            "top_k": max(1, min(5, n_trading_days // 50)),
                        }
                        
                        st.info(f"💡 데이터 {n_trading_days}일 기반 자동 조정: seq_len={adaptive_seq_len}")
                        
                        pred_result = run_stock_prediction(
                            model_name,
                            prices,
                            volumes,
                            **pred_kw
                        )
                        
                        log_pipeline("ML 예측", f"✅ {model_name} 완료", "success")
                        
                        # 결과 표시
                        st.success(f"✅ **{model_name.upper()}** 학습 완료")
                        
                        if "train_metrics" in pred_result:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if "loss" in pred_result["train_metrics"]:
                                    st.metric("Train Loss", 
                                             f"{pred_result['train_metrics']['loss']:.4f}")
                            with col_b:
                                if "test_metrics" in pred_result and "loss" in pred_result["test_metrics"]:
                                    st.metric("Test Loss",
                                             f"{pred_result['test_metrics']['loss']:.4f}")
                        
                        # 최신 예측 화면
                        if "pred" in pred_result:
                            st.write("**최신 주가 예측**")
                            pred_df = pd.DataFrame({
                                "기간": pred_result["pred"].index,
                                "예측값": pred_result["pred"].values
                            })
                            st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    except PredictionHubError as e:
                        log_pipeline("ML 예측", f"❌ 예측 실패: {str(e)[:50]}", "error")
                        st.error(f"예측 실패: {e}")
                    except Exception as e:
                        log_pipeline("ML 예측", f"❌ 예측 중 오류: {str(e)[:50]}", "error")
                        st.error(f"예측 중 오류 발생: {e}")
                    
                    with progress_placeholder.container():
                        display_pipeline()
        
        # ============ 최종 파이프라인 완료 ============
        log_pipeline("분석", "✅ 모든 분석 완료!", "success")
        with progress_placeholder.container():
            display_pipeline()


if __name__ == "__main__":
    main()
