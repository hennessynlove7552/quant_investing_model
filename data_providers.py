"""
다중 데이터 프로바이더: 일별 조정 종가(또는 종가) 수집.

지원: yfinance, Alpha Vantage, Financial Modeling Prep, EODHD, Polygon.io,
     Finnhub, Barchart, Quodd, Databento.
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
    "alpha_vantage": "ALPHAVANTAGE_API_KEY",
    "financial_modeling_prep": "FMP_API_KEY",
    "eodhd": "EODHD_API_KEY",
    "polygon": "POLYGON_API_KEY",
    "finnhub": "FINNHUB_API_KEY",
    "barchart": "BARCHART_API_KEY",
    "quodd": "QUODD_API_KEY",
    "databento": "DATABENTO_API_KEY",
}


def get_provider_names() -> list[str]:
    """지원하는 프로바이더 이름 목록."""
    return list(PROVIDER_ENV_KEYS.keys())


def _get_api_key(provider: str, api_keys: Optional[dict] = None) -> Optional[str]:
    if api_keys and provider in api_keys:
        return api_keys[provider]
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return os.environ.get(env_key) if env_key else None


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


def _fetch_alpha_vantage(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    if not api_key or not requests:
        return None
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "datatype": "json",
            "apikey": api_key,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        key = "Time Series (Daily)"
        if key not in j:
            return None
        df = pd.DataFrame.from_dict(j[key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        close = df["5. adjusted close"].astype(float)
        close.name = ticker
        close = close.loc[start_date:end_date]
        return close
    except Exception:
        return None


def _fetch_fmp(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    if not api_key or not requests:
        return None
    try:
        url = "https://financialmodelingprep.com/api/v3/historical-price-full/" + ticker
        params = {"from": start_date, "to": end_date, "apikey": api_key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "historical" not in j or not j["historical"]:
            return None
        df = pd.DataFrame(j["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        close = df["adjClose"] if "adjClose" in df.columns else df["close"]
        close = close.astype(float)
        close.name = ticker
        return close
    except Exception:
        return None


def _fetch_eodhd(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    if not api_key or not requests:
        return None
    try:
        # EODHD US stocks: SYMBOL.US
        symbol = f"{ticker}.US" if "." not in ticker else ticker
        url = f"https://eodhistoricaldata.com/api/eod/{symbol}"
        params = {
            "from": start_date,
            "to": end_date,
            "api_token": api_key,
            "fmt": "json",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        close = df["adjusted_close"] if "adjusted_close" in df.columns else df["close"]
        close = close.astype(float)
        close.name = ticker
        return close
    except Exception:
        return None


def _fetch_polygon(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    if not api_key or not requests:
        return None
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "results" not in j or not j["results"]:
            return None
        df = pd.DataFrame(j["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("t").sort_index()
        close = df["c"].astype(float)
        close.name = ticker
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


def _fetch_barchart(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    if not api_key or not requests:
        return None
    try:
        url = "https://ondemand.websol.barchart.com/getHistory.json"
        params = {
            "symbol": ticker,
            "type": "daily",
            "startDate": start_date,
            "endDate": end_date,
            "apiKey": api_key,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "results" not in j or not j["results"]:
            return None
        df = pd.DataFrame(j["results"])
        if "tradingDay" in df.columns:
            df["date"] = pd.to_datetime(df["tradingDay"])
        else:
            df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        close = df["close"].astype(float)
        close.name = ticker
        return close
    except Exception:
        return None


def _fetch_quodd(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    """Quodd: 엔터프라이즈 API. REST 엔드포인트는 문서 확인 필요. 스텁."""
    if not api_key or not requests:
        return None
    # 공개 문서상 REST 예시가 제한적. 나중에 엔드포인트 확정 시 구현.
    return None


def _fetch_databento(
    ticker: str, start_date: str, end_date: str, api_key: Optional[str], **kwargs
) -> Optional[pd.Series]:
    """Databento: Historical API는 클라이언트 라이브러리 권장. REST로 EOD 요청 시도."""
    if not api_key or not requests:
        return None
    try:
        # Databento Historical API - timeseries range (EOD)
        url = "https://hist.databento.com/v0/timeseries.get_range"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "dataset": "XNAS.ITP",
            "symbols": [ticker],
            "schema": "ohlcv-1d",
            "start": start_date,
            "end": end_date,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not data.get("data"):
            return None
        rows = data["data"]
        df = pd.DataFrame(rows)
        if "close" in df.columns and "ts_event" in df.columns:
            df["date"] = pd.to_datetime(df["ts_event"], unit="ns").dt.normalize()
        else:
            return None
        df = df.set_index("date").sort_index()
        close = df["close"].astype(float)
        close.name = ticker
        return close
    except Exception:
        return None


_FETCHERS = {
    "yfinance": _fetch_yfinance,
    "alpha_vantage": _fetch_alpha_vantage,
    "financial_modeling_prep": _fetch_fmp,
    "eodhd": _fetch_eodhd,
    "polygon": _fetch_polygon,
    "finnhub": _fetch_finnhub,
    "barchart": _fetch_barchart,
    "quodd": _fetch_quodd,
    "databento": _fetch_databento,
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
