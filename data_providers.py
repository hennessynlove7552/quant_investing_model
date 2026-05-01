"""
다중 데이터 프로바이더: 일별 조정 종가(또는 종가) 수집.

지원: yfinance, Alpha Vantage, Financial Modeling Prep, EODHD, Polygon.io, Finnhub, Barchart, Databento.
API 키는 환경 변수 또는 download_data(..., api_keys=...) 로 전달.
"""

import os
import time
from typing import Optional

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

import yfinance as yf


# 프로바이더별 환경 변수 키 이름
PROVIDER_ENV_KEYS = {
    "yfinance": None,
    "finnhub": "FINNHUB_API_KEY",
}

PROVIDER_ALIASES = {
    "alpha": "alpha_vantage",
    "alphavantage": "alpha_vantage",
    "fmp": "financial_modeling_prep",
    "financialmodelingprep": "financial_modeling_prep",
}


def get_provider_names() -> list[str]:
    """지원하는 프로바이더 이름 목록."""
    return list(PROVIDER_ENV_KEYS.keys())


def normalize_provider_name(provider: str) -> str:
    """프로바이더 별칭을 정규 이름으로 변환합니다."""
    return PROVIDER_ALIASES.get(provider, provider)


def get_provider_env_key(provider: str) -> Optional[str]:
    """프로바이더에 대응하는 환경 변수명을 반환합니다."""
    provider = normalize_provider_name(provider)
    return PROVIDER_ENV_KEYS.get(provider)


def _get_api_key(provider: str, api_keys: Optional[dict] = None) -> Optional[str]:
    provider = normalize_provider_name(provider)
    if api_keys:
        if provider in api_keys:
            return api_keys[provider]
        for alias, normalized in PROVIDER_ALIASES.items():
            if normalized == provider and alias in api_keys:
                return api_keys[alias]
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return os.environ.get(env_key) if env_key else None


def validate_provider_api_key(provider: str, api_keys: Optional[dict] = None) -> None:
    """
    API 키가 필요한 프로바이더의 설정을 확인합니다.

    키가 없으면 어떤 환경 변수를 설정해야 하는지까지 포함한 메시지를 제공합니다.
    """
    provider = normalize_provider_name(provider)
    env_key = get_provider_env_key(provider)
    if env_key is None:
        return
    if _get_api_key(provider, api_keys):
        return
    raise ValueError(
        f"`{provider}` 사용에는 API 키가 필요합니다. "
        f"사이드바 입력창에 키를 넣거나 환경 변수 `{env_key}` 를 설정하세요."
    )


def _fetch_yfinance(ticker: str, start_date: str, end_date: str, **kwargs) -> Optional[pd.Series]:
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None
        close = data["Close"] if "Close" in data.columns else data.iloc[:, 3]
        close = close.rename(ticker)
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    except Exception:
        return None


def _fetch_finnhub(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    if not api_key or not requests:
        return None
    try:
        from_ts = int(pd.Timestamp(start_date).timestamp())
        to_ts = int(pd.Timestamp(end_date).timestamp())
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {"symbol": ticker, "resolution": "D", "from": from_ts, "to": to_ts, "token": api_key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "c" not in j or not j["c"]:
            return None
        df = pd.DataFrame({"t": j["t"], "c": j["c"]})
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df = df.set_index("t").sort_index()
        close = df["c"].astype(float)
        close.name = ticker
        return close
    except Exception:
        return None


_FETCHERS = {
    "yfinance": _fetch_yfinance,
    "finnhub": _fetch_finnhub,
}


def fetch_prices(
    provider: str,
    ticker: str,
    start_date: str,
    end_date: str,
    api_keys: Optional[dict] = None,
) -> Optional[pd.Series]:
    """
    지정 프로바이더로 티커의 일별 종가(또는 조정 종가) 시계열을 가져옵니다.

    Parameters
    ----------
    provider : str
        프로바이더 이름 (get_provider_names() 참고)
    ticker : str
        종목 심볼
    start_date : str
        YYYY-MM-DD
    end_date : str
        YYYY-MM-DD
    api_keys : dict, optional
        provider -> api_key 매핑. 없으면 환경 변수 사용.

    Returns
    -------
    pd.Series
        인덱스=날짜, 값=종가(또는 조정종가). 실패 시 None.
    """
    provider = normalize_provider_name(provider)
    if provider not in _FETCHERS:
        return None
    key = _get_api_key(provider, api_keys)
    fn = _FETCHERS[provider]
    kwargs = {}
    if provider != "yfinance":
        kwargs["api_key"] = key
    result = fn(ticker, start_date, end_date, **kwargs)
    if provider == "alpha_vantage":
        time.sleep(12.1)  # 무료 한도 5회/분
    return result
