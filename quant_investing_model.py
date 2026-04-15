#!/usr/bin/env python3
"""
Quantitative Investment Model / Calculator

교육용 Python 기반 투자 모델입니다.
역사적 가격 데이터를 조회하고, 수익률·리스크 지표를 계산하며,
Fama-French 3-팩터 모델로 팩터 노출도를 추정합니다.

면책 조항: 이 도구는 교육 목적으로만 제공됩니다.
실제 투자 조언을 제공하지 않으며, 실제 거래에 사용해서는 안 됩니다.
"""

import io
import zipfile
import urllib.request
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from data_providers import (
    fetch_prices,
    get_provider_names,
    normalize_provider_name,
    validate_provider_api_key,
)
from market_data_store import (
    MarketDataStore,
)

try:
    import torch

    from thgnn import (
        THGNN,
        THGNNBatch,
        build_features_from_close,
        make_daily_batches,
        train_thgnn,
        evaluate_batches,
    )

    THGNN_DEPS_OK = True
except Exception:
    THGNN_DEPS_OK = False


# =============================================================================
# 1. 데이터 다운로드
# =============================================================================


def download_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    provider: str = "yfinance",
    api_keys: Optional[dict] = None,
    *,
    use_local_store: bool = True,
    store_path: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    지정 기간의 조정 종가(또는 종가)를 선택한 데이터 프로바이더로 다운로드합니다.

    프로바이더: yfinance, alpha_vantage, financial_modeling_prep, eodhd,
    polygon, finnhub, barchart, quodd, databento.
    API 키는 api_keys dict 또는 환경 변수(ALPHAVANTAGE_API_KEY 등)로 전달.

    Parameters
    ----------
    tickers : list[str]
        자산 티커 심볼 목록 (예: ['AAPL', 'MSFT', 'SPY'])
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)
    provider : str
        데이터 소스 (기본: yfinance)
    api_keys : dict, optional
        { "alpha_vantage": "KEY", "fmp": "KEY", ... }

    Returns
    -------
    pd.DataFrame
        날짜 인덱스, 티커별 조정 종가(또는 종가) 컬럼
    """
    if not tickers:
        raise ValueError("티커 목록이 비어 있습니다.")
    provider = normalize_provider_name(provider)
    if provider not in get_provider_names():
        raise ValueError(f"지원하지 않는 프로바이더: {provider}. 사용 가능: {get_provider_names()}")
    validate_provider_api_key(provider, api_keys)

    store = MarketDataStore(store_path) if use_local_store else None
    dfs = []
    for t in tickers:
        series = _load_or_fetch_close_series(
            ticker=t,
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            api_keys=api_keys,
            store=store,
            force_refresh=force_refresh,
        )
        if series is not None and not series.empty:
            series.index = pd.to_datetime(series.index).tz_localize(None)
            dfs.append(series)
    if not dfs:
        raise ValueError(
            f"다운로드된 데이터가 없습니다. 티커({tickers}), 날짜({start_date}~{end_date}), "
            f"프로바이더({provider}) 및 API 키를 확인하세요."
        )
    result = pd.concat(dfs, axis=1)
    result = result.dropna(how="all")

    if result.empty or result.shape[1] == 0:
        raise ValueError(
            f"다운로드된 데이터가 없습니다. 티커({tickers})와 날짜({start_date}~{end_date})를 확인하세요."
        )

    result = result.dropna(axis=1, thresh=int(len(result) * 0.5))
    return result


def download_data_with_volume(
    tickers: list[str],
    start_date: str,
    end_date: str,
    provider: str = "yfinance",
    api_keys: Optional[dict] = None,
    *,
    use_local_store: bool = True,
    store_path: Optional[str] = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    종가와 거래량을 같은 캘린더로 정렬해 다운로드합니다.

    QuantFormer 논문의 turnover(회전율) 입력에 거래량을 쓰기 위한 용도입니다.
    현재는 **yfinance만** 지원합니다. 다른 프로바이더는 ``download_data``만 사용하세요.
    """
    if not tickers:
        raise ValueError("티커 목록이 비어 있습니다.")
    provider = normalize_provider_name(provider)
    if provider != "yfinance":
        raise ValueError(
            "종가+거래량 동시 수집은 현재 yfinance만 지원합니다. "
            f"(요청 프로바이더: {provider})"
        )

    store = MarketDataStore(store_path) if use_local_store else None
    close_list = []
    vol_list = []
    for t in tickers:
        close_series, volume_series = _load_or_fetch_close_and_volume(
            ticker=t,
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            store=store,
            force_refresh=force_refresh,
        )
        if close_series is None or close_series.empty:
            continue
        close_list.append(close_series)
        vol_list.append(volume_series)

    if not close_list:
        raise ValueError(
            f"다운로드된 데이터가 없습니다. 티커({tickers}), 날짜({start_date}~{end_date})를 확인하세요."
        )

    prices = pd.concat(close_list, axis=1)
    volumes = pd.concat(vol_list, axis=1)
    volumes = volumes.reindex(columns=prices.columns)
    prices = prices.dropna(how="all")
    volumes = volumes.reindex(prices.index)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    volumes = volumes.reindex(columns=prices.columns)
    return prices, volumes


def _load_or_fetch_close_series(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    provider: str,
    api_keys: Optional[dict],
    store: Optional[MarketDataStore],
    force_refresh: bool,
) -> Optional[pd.Series]:
    if store and not force_refresh and store.covers_range(provider, ticker, start_date, end_date):
        cached = store.load_series(provider, ticker, start_date, end_date, field="close")
        if not cached.empty:
            return cached

    series = fetch_prices(provider, ticker, start_date, end_date, api_keys=api_keys)
    if series is None or series.empty:
        if store:
            return store.load_series(provider, ticker, start_date, end_date, field="close")
        return series

    series.index = pd.to_datetime(series.index).tz_localize(None)
    if store:
        store.upsert_daily_bars(
            provider,
            ticker,
            pd.DataFrame({"close": series}),
            source="remote",
            requested_start=start_date,
            requested_end=end_date,
        )
        return store.load_series(provider, ticker, start_date, end_date, field="close")
    return series


def _load_or_fetch_close_and_volume(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    provider: str,
    store: Optional[MarketDataStore],
    force_refresh: bool,
) -> tuple[Optional[pd.Series], pd.Series]:
    if store and not force_refresh and store.covers_range(
        provider, ticker, start_date, end_date, require_volume=True
    ):
        cached_close = store.load_series(provider, ticker, start_date, end_date, field="close")
        cached_volume = store.load_series(provider, ticker, start_date, end_date, field="volume")
        if not cached_close.empty:
            if cached_volume.empty:
                cached_volume = pd.Series(index=cached_close.index, dtype=float, name=ticker)
            return cached_close, cached_volume.reindex(cached_close.index)

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        data = pd.DataFrame()

    if data.empty:
        if store:
            cached_close = store.load_series(provider, ticker, start_date, end_date, field="close")
            cached_volume = store.load_series(provider, ticker, start_date, end_date, field="volume")
            if cached_volume.empty and not cached_close.empty:
                cached_volume = pd.Series(index=cached_close.index, dtype=float, name=ticker)
            return cached_close, cached_volume
        return None, pd.Series(dtype=float, name=ticker)

    data.index = pd.to_datetime(data.index).tz_localize(None)
    if "Close" in data.columns:
        close = data["Close"].copy()
    else:
        close = data.iloc[:, 3].copy()
    close.name = ticker

    if "Volume" in data.columns:
        volume = data["Volume"].copy().astype(float)
    else:
        volume = pd.Series(index=close.index, dtype=float)
    volume.name = ticker

    if store:
        frame = pd.DataFrame({"close": close, "volume": volume})
        store.upsert_daily_bars(
            provider,
            ticker,
            frame,
            source="remote",
            requested_start=start_date,
            requested_end=end_date,
        )
        close = store.load_series(provider, ticker, start_date, end_date, field="close")
        volume = store.load_series(provider, ticker, start_date, end_date, field="volume")
        if volume.empty:
            volume = pd.Series(index=close.index, dtype=float, name=ticker)
        else:
            volume = volume.reindex(close.index)
    return close, volume


# =============================================================================
# 2. 수익률·리스크 지표 계산
# =============================================================================


def calculate_metrics(
    prices: pd.DataFrame,
    risk_free_rate: float,
    *,
    transaction_cost_pct: float = 0.0,
    slippage_pct: float = 0.0,
    spread_pct: float = 0.0,
    tax_rate: float = 0.0,
) -> pd.DataFrame:
    """
    일별 수익률로부터 수익률·리스크 지표를 계산합니다.

    - 누적 수익률: (최종가/최초가) - 1
    - 연율화 수익률: (1 + 누적수익률)^(252/거래일수) - 1
    - 연율화 변동성: 일별 수익률 표준편차 × √252
    - 샤프 비율: (연율화 수익률 - 무위험 수익률) / 연율화 변동성
    - 최대 낙폭: 피크 대비 최대 골(peak-to-trough) 하락률

    비용·세금 반영(선택):
    - 거래비용·슬리피지·스프레드(편도)를 합산해 매매 시점(진입/청산)에 반영.
    - 세금: 양의 수익에 대해 지정 세율을 적용한 세후 수익률을 추가 계산.
    - 반영 시 '누적 수익률(비용·세금 반영)', '연율화 수익률(비용·세금 반영)',
      '샤프 비율(비용·세금 반영)', '최대 낙폭(비용 반영)' 컬럼이 추가됩니다.

    Parameters
    ----------
    prices : pd.DataFrame
        조정 종가 (날짜 인덱스)
    risk_free_rate : float
        연간 무위험 수익률 (예: 0.03 = 3%)
    transaction_cost_pct : float, optional
        편도 거래비용 (소수, 예: 0.001 = 0.1%)
    slippage_pct : float, optional
        편도 슬리피지 (소수)
    spread_pct : float, optional
        편도 스프레드·미시구조 비용 (예: bid-ask 절반, 소수)
    tax_rate : float, optional
        양의 수익에 대한 세율 (소수, 예: 0.25 = 25%)

    Returns
    -------
    pd.DataFrame
        티커별 지표 요약 (비용·세금 사용 시 추가 컬럼 포함)
    """
    # 일별 수익률: r_t = (P_t / P_{t-1}) - 1
    returns = prices.pct_change().dropna()

    if returns.empty:
        raise ValueError("수익률 계산에 충분한 데이터가 없습니다.")

    n_days = len(returns)
    trading_days_per_year = 252

    # 편도 비용 합계 (진입/청산 각 1회 적용 → 왕복 2회)
    one_way_cost = transaction_cost_pct + slippage_pct + spread_pct
    use_costs = one_way_cost > 0 or tax_rate > 0

    metrics_list = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 2:
            continue

        # 누적 수익률 (Gross)
        cum_return = (1 + r).prod() - 1

        # 연율화 수익률
        ann_return = (1 + cum_return) ** (trading_days_per_year / n_days) - 1

        # 연율화 변동성
        ann_vol = r.std() * np.sqrt(trading_days_per_year)

        # 샤프 비율
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

        # 최대 낙폭 (Gross)
        cum_wealth = (1 + r).cumprod()
        rolling_max = cum_wealth.expanding().max()
        drawdown = (cum_wealth - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        row = {
            "티커": ticker,
            "누적 수익률": cum_return,
            "연율화 수익률": ann_return,
            "연율화 변동성": ann_vol,
            "샤프 비율": sharpe,
            "최대 낙폭": max_drawdown,
        }

        if use_costs:
            # 비용 반영: 진입 시 (1-c), 청산 시 (1-c) → 순수익 (1-c)^2 * (1 + cum_gross) - 1
            cum_net_before_tax = (1 - one_way_cost) ** 2 * (1 + cum_return) - 1
            # 세금: 양의 수익만 세율 적용
            cum_net_after_tax = (
                cum_net_before_tax * (1 - tax_rate) if cum_net_before_tax > 0 else cum_net_before_tax
            )
            ann_return_net = (1 + cum_net_after_tax) ** (trading_days_per_year / n_days) - 1
            sharpe_net = (ann_return_net - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
            # 비용 반영 누적 자산 곡선으로 최대 낙폭
            net_wealth = (1 - one_way_cost) * (1 + r).cumprod()
            rolling_max_net = net_wealth.expanding().max()
            drawdown_net = (net_wealth - rolling_max_net) / rolling_max_net
            max_drawdown_net = drawdown_net.min()

            row["누적 수익률(비용·세금 반영)"] = cum_net_after_tax
            row["연율화 수익률(비용·세금 반영)"] = ann_return_net
            row["샤프 비율(비용·세금 반영)"] = sharpe_net
            row["최대 낙폭(비용 반영)"] = max_drawdown_net

        metrics_list.append(row)

    return pd.DataFrame(metrics_list).set_index("티커")


# =============================================================================
# 3. 파라메트릭 VaR
# =============================================================================


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    정규분포 가정 하의 파라메트릭 Value-at-Risk (VaR)를 계산합니다.

    VaR: 주어진 신뢰수준에서 일별 최대 손실(음수)의 예상값.
    정규분포 가정: VaR = μ + σ × z_α
    여기서 z_α는 표준정규분포의 α 분위수 (예: 95% → z ≈ -1.645)

    Parameters
    ----------
    returns : pd.Series
        일별 수익률
    confidence : float
        신뢰수준 (예: 0.95 = 95%)

    Returns
    -------
    float
        일별 VaR (음수 = 손실)
    """
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    # α = 1 - confidence (왼쪽 꼬리)
    alpha = 1 - confidence
    z = _norm_ppf(alpha)
    var = mu + z * sigma
    return var


# 표준정규분포 분위수 (scipy 의존성 없이 일반적인 신뢰수준 지원)
Z_SCORES = {
    0.90: -1.2816,
    0.95: -1.6449,
    0.99: -2.3263,
    0.975: -1.9600,
    0.995: -2.5758,
}


def _norm_ppf(p: float) -> float:
    """표준정규분포 분위수 근사 (p < 0.5, VaR용)."""
    if p in Z_SCORES:
        return Z_SCORES[p]
    if p <= 0 or p >= 1:
        return np.nan
    # 선형 보간
    sorted_ps = sorted(Z_SCORES.keys())
    for i, cp in enumerate(sorted_ps):
        if p >= cp:
            if i == 0:
                return Z_SCORES[cp]
            p0, p1 = sorted_ps[i - 1], cp
            z0, z1 = Z_SCORES[p0], Z_SCORES[p1]
            return z0 + (z1 - z0) * (p - p0) / (p1 - p0)
    return Z_SCORES[sorted_ps[-1]]


# =============================================================================
# 4. Fama-French 3-팩터 모델
# =============================================================================

FF3_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def download_fama_french_factors(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Ken French 데이터 라이브러리에서 Fama-French 3-팩터 일별 데이터를 다운로드합니다.

    팩터: Mkt-RF(시장 초과수익률), SMB(소형-대형), HML(가치-성장), RF(무위험 수익률)
    값은 % 단위이므로 100으로 나누어 소수로 변환합니다.

    Parameters
    ----------
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        날짜 인덱스, Mkt-RF, SMB, HML, RF 컬럼
    """
    try:
        with urllib.request.urlopen(FF3_DAILY_URL, timeout=30) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
                name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
                # FF 포맷: 처음 4행은 헤더/설명, 그 다음 YYYYMMDD, Mkt-RF, SMB, HML, RF
                df = pd.read_csv(zf.open(name), skiprows=4)
    except Exception as e:
        raise ConnectionError(
            f"Fama-French 팩터 데이터 다운로드 실패: {e}. "
            "인터넷 연결을 확인하세요."
        ) from e

    # 컬럼 정리 (공백 제거)
    df.columns = df.columns.str.strip()
    # 첫 컬럼이 날짜 (YYYYMMDD)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_numeric(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")
    df = df.set_index("Date")

    # 값이 % 단위이므로 100으로 나눔
    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    df = df.dropna(how="all")
    # 기간 필터
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df.loc[(df.index >= start) & (df.index <= end)]
    return df


def run_factor_model(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    risk_free_rate: float,
) -> dict:
    """
    Fama-French 3-팩터 모델 OLS 회귀를 실행합니다.

    모델: R_i - R_f = α + β_Mkt*(Mkt-RF) + β_SMB*SMB + β_HML*HML + ε
    - α(알파): 팩터로 설명되지 않는 초과 수익
    - β: 각 팩터에 대한 노출도(감도)

    Parameters
    ----------
    returns : pd.DataFrame
        자산 일별 수익률 (날짜 인덱스)
    factors : pd.DataFrame
        Mkt-RF, SMB, HML, RF 컬럼 포함
    risk_free_rate : float
        연간 무위험 수익률 (일별로 변환하여 사용 가능, 또는 RF 사용)

    Returns
    -------
    dict
        ticker -> {'model': OLS 결과, 'params': Series, 'summary': str}
    """
    # 일별 무위험 수익률 (FF 데이터는 %→소수, 이미 일별)
    rf_daily = factors["RF"] if "RF" in factors.columns else pd.Series(
        risk_free_rate / 252, index=factors.index
    )

    # 날짜 정렬 및 공통 기간
    common_idx = returns.index.intersection(factors.index).sort_values().drop_duplicates()
    if len(common_idx) < 30:
        raise ValueError("팩터와 수익률 데이터의 공통 기간이 너무 짧습니다 (최소 30일).")

    R = returns.reindex(common_idx).ffill().bfill()
    F = factors.reindex(common_idx).ffill().bfill()
    rf_aligned = rf_daily.reindex(common_idx).ffill().bfill()

    results = {}
    X = add_constant(F[["Mkt-RF", "SMB", "HML"]])

    for ticker in R.columns:
        y = R[ticker] - rf_aligned
        valid = y.notna() & X.notna().all(axis=1)
        y_clean = y[valid].dropna()
        X_clean = X.loc[y_clean.index]
        if len(y_clean) < 20:
            continue
        model = OLS(y_clean, X_clean).fit()
        results[ticker] = {
            "model": model,
            "params": model.params,
            "summary": model.summary().as_text(),
        }
    return results


# =============================================================================
# 5. 결과 출력 및 시각화
# =============================================================================


def display_results(
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    var_results: Optional[dict] = None,
    factor_results: Optional[dict] = None,
    save_plots: bool = True,
) -> None:
    """
    테이블과 그래프로 결과를 출력합니다.

    - 수익률·리스크 지표 테이블
    - VaR 테이블 (선택)
    - 팩터 회귀 요약 및 β 막대 그래프 (선택)
    - 가격 시계열, 누적 수익률 그래프
    """
    print("\n" + "=" * 60)
    print("수익률·리스크 지표 요약")
    print("=" * 60)
    display_df = metrics.copy()
    for c in display_df.columns:
        if display_df[c].dtype in (np.float64, float):
            display_df[c] = display_df[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    print(display_df.to_string())

    if var_results:
        print("\n" + "=" * 60)
        print("Value-at-Risk (일별, 파라메트릭)")
        print("=" * 60)
        for ticker, var in var_results.items():
            print(f"  {ticker}: {var:.2%}")

    if factor_results:
        print("\n" + "=" * 60)
        print("Fama-French 3-팩터 모델 회귀 결과")
        print("=" * 60)
        for ticker, res in factor_results.items():
            print(f"\n--- {ticker} ---")
            print(res["summary"])

    # 그래프
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) 가격 시계열 (정규화: 100 기준)
    ax1 = axes[0, 0]
    norm_prices = prices / prices.iloc[0] * 100
    for c in norm_prices.columns:
        ax1.plot(norm_prices.index, norm_prices[c], label=c)
    ax1.set_title("가격 시계열 (기준일=100)")
    ax1.set_ylabel("지수")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2) 누적 수익률
    ax2 = axes[0, 1]
    returns = prices.pct_change().dropna()
    cum_returns = (1 + returns).cumprod() - 1
    for c in cum_returns.columns:
        ax2.plot(cum_returns.index, cum_returns[c], label=c)
    ax2.set_title("누적 수익률")
    ax2.set_ylabel("누적 수익률")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3) 팩터 β 막대 그래프
    ax3 = axes[1, 0]
    if factor_results:
        tickers = list(factor_results.keys())
        factors_ff = ["const", "Mkt-RF", "SMB", "HML"]
        x = np.arange(len(tickers))
        width = 0.2
        for i, f in enumerate(["Mkt-RF", "SMB", "HML"]):
            vals = [
                factor_results[t].get("params", pd.Series()).get(f, 0)
                for t in tickers
            ]
            ax3.bar(x + i * width, vals, width, label=f)
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(tickers)
        ax3.set_ylabel("β")
        ax3.set_title("팩터 노출도 (β)")
        ax3.legend()
        ax3.axhline(0, color="black", linewidth=0.5)
    else:
        ax3.text(0.5, 0.5, "팩터 분석 없음", ha="center", va="center", transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)

    # 4) 샤프 비율 막대
    ax4 = axes[1, 1]
    sharpe = metrics["샤프 비율"].dropna()
    if len(sharpe) > 0:
        ax4.bar(sharpe.index, sharpe.values, color="steelblue", alpha=0.8)
        ax4.set_title("샤프 비율")
        ax4.set_ylabel("샤프 비율")
        ax4.axhline(0, color="black", linewidth=0.5)
    else:
        ax4.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig("quant_investing_results.png", dpi=150, bbox_inches="tight")
        print("\n그래프가 'quant_investing_results.png'로 저장되었습니다.")
    plt.show()


# =============================================================================
# 6. CLI 메인
# =============================================================================


def parse_date(s: str) -> str:
    """날짜 문자열 검증 및 정규화."""
    try:
        d = datetime.strptime(s.strip(), "%Y-%m-%d")
        return d.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 사용: '{s}'")


def main() -> None:
    """
    커맨드라인 인터페이스.
    나중에 GUI나 API로 확장할 수 있도록 입력/로직을 분리했습니다.
    """
    print("=" * 60)
    print("Quantitative Investment Model (교육용)")
    print("=" * 60)
    print("면책: 이 도구는 교육 목적이며, 투자 조언이 아닙니다.\n")

    # 1) 티커 입력
    ticker_input = input("자산 티커 (쉼표 구분, 예: AAPL, MSFT, SPY): ").strip()
    if not ticker_input:
        ticker_input = "AAPL, MSFT, SPY"
        print(f"기본값 사용: {ticker_input}")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    # 2) 기간 입력
    start_input = input("시작일 (YYYY-MM-DD, 예: 2020-01-01): ").strip() or "2020-01-01"
    end_input = input("종료일 (YYYY-MM-DD, 예: 2024-12-31): ").strip() or "2024-12-31"
    try:
        start_date = parse_date(start_input)
        end_date = parse_date(end_input)
    except ValueError as e:
        print(f"오류: {e}")
        return

    # 3) 데이터 프로바이더
    providers = get_provider_names()
    print(f"데이터 소스: {', '.join(providers)}")
    provider_input = input("데이터 소스 (기본 yfinance): ").strip().lower() or "yfinance"
    provider = provider_input if provider_input in providers else "yfinance"
    api_keys = None
    if provider != "yfinance":
        key_input = input(f"  API 키 ({provider}, 비워두면 환경변수): ").strip()
        if key_input:
            api_keys = {provider: key_input}

    # 4) 무위험 수익률
    rf_input = input("연간 무위험 수익률 (예: 0.03 = 3%): ").strip() or "0.03"
    try:
        risk_free_rate = float(rf_input)
    except ValueError:
        print("오류: 숫자를 입력하세요 (예: 0.03)")
        return

    # 5) VaR 신뢰수준 (선택)
    var_choice = input("VaR 계산 여부 (y/n, 기본 n): ").strip().lower() or "n"
    var_confidence = 0.95
    if var_choice == "y":
        conf_input = input("VaR 신뢰수준 (예: 0.95 = 95%): ").strip() or "0.95"
        try:
            var_confidence = float(conf_input)
        except ValueError:
            var_confidence = 0.95

    # 6) 팩터 모델 실행 여부
    factor_choice = input("Fama-French 3-팩터 분석 실행 (y/n, 기본 y): ").strip().lower() or "y"

    # 7) 비용·세금·미시구조 반영 (선택)
    cost_choice = input("거래비용·슬리피지·스프레드·세금 반영 (y/n, 기본 n): ").strip().lower() or "n"
    transaction_cost_pct = slippage_pct = spread_pct = tax_rate = 0.0
    if cost_choice == "y":
        try:
            transaction_cost_pct = float(input("  거래비용 편도 (%%, 예: 0.1): ").strip() or "0") / 100.0
            slippage_pct = float(input("  슬리피지 편도 (%%, 예: 0.05): ").strip() or "0") / 100.0
            spread_pct = float(input("  스프레드/미시구조 편도 (%%, 예: 0.05): ").strip() or "0") / 100.0
            tax_rate = float(input("  세율 양의수익 (%%, 예: 25): ").strip() or "0") / 100.0
        except ValueError:
            print("  숫자로 입력하지 않아 비용·세금 미적용.")

    # --- 실행 ---
    try:
        print(f"\n데이터 다운로드 중 ({provider})...")
        prices = download_data(
            tickers, start_date, end_date, provider=provider, api_keys=api_keys
        )
        print(f"다운로드 완료: {list(prices.columns)}")

        returns = prices.pct_change().dropna()
        metrics = calculate_metrics(
            prices,
            risk_free_rate,
            transaction_cost_pct=transaction_cost_pct,
            slippage_pct=slippage_pct,
            spread_pct=spread_pct,
            tax_rate=tax_rate,
        )

        var_results = None
        if var_choice == "y":
            var_results = {}
            for t in returns.columns:
                var_results[t] = calculate_var(returns[t], confidence=var_confidence)

        factor_results = None
        if factor_choice == "y":
            print("Fama-French 팩터 데이터 다운로드 중...")
            factors = download_fama_french_factors(start_date, end_date)
            factor_results = run_factor_model(returns, factors, risk_free_rate)

        display_results(
            prices=prices,
            metrics=metrics,
            var_results=var_results,
            factor_results=factor_results,
            save_plots=True,
        )
    except (ValueError, ConnectionError) as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        raise


if __name__ == "__main__":
    main()


# =============================================================================
# 7. THGNN 기반 주가 움직임 예측 (논문 로직 반영)
# =============================================================================


def thgnn_prepare(
    prices: pd.DataFrame,
    *,
    lookback: int = 20,
    corr_window: int = 20,
    corr_threshold: float = 0.6,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
) -> dict:
    """
    THGNN 학습용 일자별 배치 생성.

    - 입력: 종가 DataFrame (index=날짜, columns=티커)
    - 출력: torch 배치 리스트 및 메타데이터
    """
    if not THGNN_DEPS_OK:
        raise ImportError("THGNN 실행을 위한 의존성(torch)이 준비되지 않았습니다. requirements.txt를 설치하세요.")

    if prices.shape[1] < 4:
        raise ValueError("THGNN은 종목 간 관계 학습이 핵심이라 최소 4개 이상 티커를 권장합니다.")

    close = prices.sort_index().astype(float).to_numpy()  # (T, L)
    feats = build_features_from_close(close)  # (T, L, 6)

    L = prices.shape[1]
    if top_k is None:
        top_k = max(1, min(10, L // 5))

    raw_batches = make_daily_batches(
        feats,
        close,
        lookback=lookback,
        corr_window=corr_window,
        corr_threshold=corr_threshold,
        top_k=int(top_k),
    )
    if not raw_batches:
        raise ValueError("THGNN 배치를 만들 수 없습니다. 기간이 너무 짧거나 파라미터가 과도합니다.")

    dev = None
    if device:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batches: list[THGNNBatch] = []
    for x, A_pos, A_neg, y, mask in raw_batches:
        batches.append(
            THGNNBatch(
                x=torch.tensor(x, dtype=torch.float32),
                A_pos=torch.tensor(A_pos, dtype=torch.float32),
                A_neg=torch.tensor(A_neg, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.float32),
                mask=torch.tensor(mask, dtype=torch.float32),
            )
        )

    return {
        "batches": batches,
        "tickers": list(prices.columns),
        "dates": list(prices.sort_index().index),
        "device": dev,
        "top_k": int(top_k),
    }


def thgnn_train_and_predict(
    prices: pd.DataFrame,
    *,
    lookback: int = 20,
    corr_window: int = 20,
    corr_threshold: float = 0.6,
    top_k: Optional[int] = None,
    train_ratio: float = 0.7,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """
    THGNN을 학습하고(간단 train/test split) 마지막 배치 기준 예측 확률을 반환.
    """
    if not THGNN_DEPS_OK:
        raise ImportError("THGNN 실행을 위한 의존성(torch)이 준비되지 않았습니다. requirements.txt를 설치하세요.")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    prep = thgnn_prepare(
        prices,
        lookback=lookback,
        corr_window=corr_window,
        corr_threshold=corr_threshold,
        top_k=top_k,
        device=device,
    )
    batches: list[THGNNBatch] = prep["batches"]
    dev = prep["device"]
    tickers = prep["tickers"]

    n = len(batches)
    split = int(max(1, min(n - 1, round(n * float(train_ratio)))))
    train_batches = batches[:split]
    test_batches = batches[split:]

    model = THGNN(feature_dim=6)
    hist = train_thgnn(
        model,
        train_batches=train_batches,
        valid_batches=test_batches if test_batches else None,
        device=dev,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
    )
    train_metrics = evaluate_batches(model, train_batches, dev)
    test_metrics = evaluate_batches(model, test_batches, dev) if test_batches else {"loss": np.nan, "acc": np.nan, "n_days": 0}

    # 마지막 day 배치로 확률 예측
    last = batches[-1]
    model.eval()
    with torch.no_grad():
        p = model(last.x.to(dev), last.A_pos.to(dev), last.A_neg.to(dev)).detach().cpu().numpy()

    pred = pd.Series(p, index=tickers, name="p(up)")
    pred = pred.sort_values(ascending=False)
    return {
        "model": model,
        "history": hist,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "pred": pred,
        "top_k": prep["top_k"],
        "device": str(dev),
    }
