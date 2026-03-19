#!/usr/bin/env python3
"""
Quantitative Investment Model - 웹 앱

Streamlit 기반 웹 인터페이스.
정량 투자 분석을 브라우저에서 실행합니다. 개인 실전 활용용.

실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 백엔드
import matplotlib.pyplot as plt

try:
    from data_providers import get_provider_names
    from quant_investing_model import (
        download_data,
        calculate_metrics,
        calculate_var,
        download_fama_french_factors,
        run_factor_model,
    )
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    IMPORT_ERROR = str(e)

# 페이지 설정
st.set_page_config(
    page_title="Quantitative Investment Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 커스텀 스타일
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
</style>
""", unsafe_allow_html=True)


def render_price_chart(prices: pd.DataFrame):
    """가격 시계열 차트 (기준일=100)."""
    norm_prices = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in norm_prices.columns:
        ax.plot(norm_prices.index, norm_prices[c], label=c)
    ax.set_title("가격 시계열 (기준일=100)", fontsize=12)
    ax.set_ylabel("지수")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_cumulative_return_chart(prices: pd.DataFrame):
    """누적 수익률 차트."""
    returns = prices.pct_change().dropna()
    cum_returns = (1 + returns).cumprod() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in cum_returns.columns:
        ax.plot(cum_returns.index, cum_returns[c], label=c)
    ax.set_title("누적 수익률", fontsize=12)
    ax.set_ylabel("누적 수익률")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    return fig


def render_factor_betas_chart(factor_results: dict):
    """팩터 β 막대 차트."""
    fig, ax = plt.subplots(figsize=(8, 4))
    tickers = list(factor_results.keys())
    x = np.arange(len(tickers))
    width = 0.25
    for i, f in enumerate(["Mkt-RF", "SMB", "HML"]):
        vals = [
            factor_results[t].get("params", pd.Series()).get(f, 0)
            for t in tickers
        ]
        ax.bar(x + i * width, vals, width, label=f)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("β")
    ax.set_title("팩터 노출도 (β)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_sharpe_chart(metrics: pd.DataFrame):
    """샤프 비율 막대 차트."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sharpe = metrics["샤프 비율"].dropna()
    if len(sharpe) > 0:
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sharpe.values]
        ax.bar(sharpe.index, sharpe.values, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("샤프 비율")
    ax.set_ylabel("샤프 비율")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    if not DEPS_OK:
        st.error(
            f"필수 패키지가 설치되지 않았습니다. 아래 명령으로 설치 후 다시 실행하세요.\n\n"
            f"```bash\npip install -r requirements.txt\n```\n\n"
            f"오류: {IMPORT_ERROR}"
        )
        return

    st.title("📈 Quantitative Investment Model")
    st.caption("정량 투자 분석 · 개인 실전 활용용")

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>면책</strong>: 투자 자문·매매 권유가 아니며, 결과에 따른 의사결정과 손익은 전적으로 본인 책임입니다.
    </div>
    """, unsafe_allow_html=True)

    # 사이드바: 입력 파라미터
    with st.sidebar:
        st.header("⚙️ 분석 설정")

        provider = st.selectbox(
            "데이터 소스",
            options=get_provider_names(),
            index=0,
            help="yfinance는 키 불필요. 나머지는 API 키가 필요할 수 있습니다.",
        )
        api_keys = None
        if provider != "yfinance":
            api_key_val = st.text_input(
                f"API 키 ({provider})",
                type="password",
                help="비워두면 환경 변수 사용 (예: ALPHAVANTAGE_API_KEY)",
            )
            if api_key_val.strip():
                api_keys = {provider: api_key_val.strip()}

        ticker_input = st.text_input(
            "자산 티커 (쉼표 구분)",
            value="AAPL, MSFT, SPY",
            help="예: AAPL, MSFT, SPY, QQQ",
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작일", value=pd.to_datetime("2020-01-01"))
        with col2:
            end_date = st.date_input("종료일", value=pd.to_datetime("2024-12-31"))

        risk_free_rate = st.number_input(
            "연간 무위험 수익률",
            min_value=0.0,
            max_value=0.2,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="예: 0.03 = 3%",
        )

        with st.expander("💰 비용·세금·미시구조 반영 (선택)"):
            st.caption("편도 비용(%)·세율(%)을 소수으로 입력. 0이면 미적용.")
            transaction_cost_pct = st.number_input(
                "거래비용 (편도, %)",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="예: 0.1 = 0.1%",
            ) / 100.0
            slippage_pct = st.number_input(
                "슬리피지 (편도, %)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.2f",
            ) / 100.0
            spread_pct = st.number_input(
                "스프레드/미시구조 (편도, %)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="예: bid-ask 절반",
            ) / 100.0
            tax_rate = st.number_input(
                "세율 (양의 수익에 대한 %, %)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                format="%.1f",
                help="예: 25 = 25%",
            ) / 100.0

        st.divider()
        calc_var = st.checkbox("VaR 계산", value=False)
        var_confidence = 0.95
        if calc_var:
            var_confidence = st.select_slider(
                "VaR 신뢰수준",
                options=[0.90, 0.95, 0.99],
                value=0.95,
                format_func=lambda x: f"{x:.0%}",
            )

        run_factor = st.checkbox("Fama-French 3-팩터 분석", value=True)

        st.divider()
        analyze_btn = st.button("🔍 분석 실행", type="primary", use_container_width=True)

    if not analyze_btn:
        st.info("👈 사이드바에서 파라미터를 설정한 뒤 **분석 실행** 버튼을 눌러주세요.")
        return

    if not tickers:
        st.error("최소 1개 이상의 티커를 입력해주세요.")
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if start_date >= end_date:
        st.error("시작일은 종료일보다 이전이어야 합니다.")
        return

    # 분석 실행
    with st.spinner("데이터를 불러오는 중..."):
        try:
            prices = download_data(
                tickers, start_str, end_str, provider=provider, api_keys=api_keys
            )
        except (ValueError, ConnectionError) as e:
            st.error(f"데이터 다운로드 실패: {e}")
            return

    st.success(f"데이터 로드 완료: {list(prices.columns)}")

    returns = prices.pct_change().dropna()

    # 수익률·리스크 지표 (비용·세금·미시구조 선택 반영)
    metrics = calculate_metrics(
        prices,
        risk_free_rate,
        transaction_cost_pct=transaction_cost_pct,
        slippage_pct=slippage_pct,
        spread_pct=spread_pct,
        tax_rate=tax_rate,
    )

    # VaR
    var_results = None
    if calc_var:
        var_results = {t: calculate_var(returns[t], confidence=var_confidence) for t in returns.columns}

    # Fama-French
    factor_results = None
    if run_factor:
        with st.spinner("Fama-French 팩터 데이터 다운로드 중..."):
            try:
                factors = download_fama_french_factors(start_str, end_str)
                factor_results = run_factor_model(returns, factors, risk_free_rate)
            except (ValueError, ConnectionError) as e:
                st.warning(f"팩터 분석 건너뜀: {e}")
                factor_results = None

    # === 결과 표시 ===

    st.header("📊 수익률·리스크 지표")
    if any(c for c in metrics.columns if "비용" in c or "세금" in c):
        st.caption("아래 '(비용·세금 반영)' 컬럼은 거래비용·슬리피지·스프레드·세금을 반영한 값입니다.")
    display_metrics = metrics.copy()
    for c in display_metrics.columns:
        if display_metrics[c].dtype in (np.float64, float):
            display_metrics[c] = display_metrics[c].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "-"
            )
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    if var_results:
        st.subheader("Value-at-Risk (일별, 파라메트릭)")
        var_df = pd.DataFrame([
            {"티커": t, "VaR": f"{v:.2%}"} for t, v in var_results.items()
        ])
        st.dataframe(var_df, use_container_width=True, hide_index=True)

    # 차트
    st.header("📈 시각화")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(render_price_chart(prices))
    with col2:
        st.pyplot(render_cumulative_return_chart(prices))

    col3, col4 = st.columns(2)
    with col3:
        if factor_results:
            st.pyplot(render_factor_betas_chart(factor_results))
        else:
            st.info("팩터 분석을 실행하지 않았습니다.")
    with col4:
        st.pyplot(render_sharpe_chart(metrics))

    # Fama-French 회귀 요약
    if factor_results:
        st.header("📐 Fama-French 3-팩터 회귀 결과")
        for ticker, res in factor_results.items():
            with st.expander(f"**{ticker}** 회귀 요약"):
                st.text(res["summary"])

    st.divider()
    st.caption("Quantitative Investment Model · 개인 실전 활용")


if __name__ == "__main__":
    main()
