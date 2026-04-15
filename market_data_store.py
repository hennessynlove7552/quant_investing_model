"""
로컬 마켓 데이터 저장소.

실전형 데이터 파이프라인의 첫 단계로, 외부 프로바이더에서 받은 일별 시계열을
SQLite에 적재하고 재사용할 수 있게 합니다.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "market_data.db"


def get_default_market_data_store_path() -> str:
    """기본 로컬 저장소 경로를 반환합니다."""
    return str(DEFAULT_DB_PATH)


class MarketDataStore:
    """일별 가격/거래량 데이터를 SQLite에 저장하고 조회합니다."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_bars (
                    provider TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (provider, ticker, date)
                );

                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    requested_start TEXT NOT NULL,
                    requested_end TEXT NOT NULL,
                    rows_written INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_daily_bars_lookup
                ON daily_bars (provider, ticker, date);
                """
            )
            conn.commit()

    def upsert_daily_bars(
        self,
        provider: str,
        ticker: str,
        frame: pd.DataFrame,
        *,
        source: str = "remote",
        requested_start: Optional[str] = None,
        requested_end: Optional[str] = None,
    ) -> int:
        """
        일별 OHLCV 데이터를 적재합니다.

        입력 DataFrame은 date index 또는 date 컬럼을 가져야 하며,
        없는 컬럼은 NULL로 저장됩니다.
        """
        normalized = self._normalize_bars_frame(frame)
        if normalized.empty:
            return 0

        fetched_at = datetime.utcnow().isoformat(timespec="seconds")
        rows = [
            (
                provider,
                ticker,
                row.date,
                row.open,
                row.high,
                row.low,
                row.close,
                row.adj_close,
                row.volume,
                fetched_at,
            )
            for row in normalized.itertuples(index=False)
        ]

        with closing(self._connect()) as conn:
            conn.executemany(
                """
                INSERT INTO daily_bars (
                    provider, ticker, date, open, high, low, close, adj_close, volume, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, ticker, date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    adj_close=excluded.adj_close,
                    volume=excluded.volume,
                    fetched_at=excluded.fetched_at
                """,
                rows,
            )
            conn.execute(
                """
                INSERT INTO fetch_log (
                    provider, ticker, requested_start, requested_end, rows_written, source, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    ticker,
                    requested_start or normalized["date"].min(),
                    requested_end or normalized["date"].max(),
                    len(rows),
                    source,
                    fetched_at,
                ),
            )
            conn.commit()
        return len(rows)

    def load_series(
        self,
        provider: str,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        field: str = "close",
    ) -> pd.Series:
        """저장소에서 단일 필드 시계열을 조회합니다."""
        if field not in {"open", "high", "low", "close", "adj_close", "volume"}:
            raise ValueError(f"지원하지 않는 필드: {field}")

        query = f"""
            SELECT date, {field}
            FROM daily_bars
            WHERE provider = ?
              AND ticker = ?
              AND date BETWEEN ? AND ?
              AND {field} IS NOT NULL
            ORDER BY date
        """
        with closing(self._connect()) as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(provider, ticker, start_date, end_date),
            )

        if df.empty:
            return pd.Series(dtype=float, name=ticker)

        df["date"] = pd.to_datetime(df["date"])
        series = pd.Series(df[field].astype(float).to_numpy(), index=df["date"], name=ticker)
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return series.sort_index()

    def get_coverage(self, provider: str, ticker: str) -> Optional[dict]:
        """저장된 범위와 기본 건수를 요약합니다."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT
                    MIN(date) AS min_date,
                    MAX(date) AS max_date,
                    COUNT(*) AS rows,
                    SUM(CASE WHEN close IS NOT NULL THEN 1 ELSE 0 END) AS close_rows,
                    SUM(CASE WHEN volume IS NOT NULL THEN 1 ELSE 0 END) AS volume_rows
                FROM daily_bars
                WHERE provider = ? AND ticker = ?
                """,
                (provider, ticker),
            ).fetchone()

        if not row or row[0] is None:
            return None
        return {
            "min_date": row[0],
            "max_date": row[1],
            "rows": int(row[2] or 0),
            "close_rows": int(row[3] or 0),
            "volume_rows": int(row[4] or 0),
        }

    def covers_range(
        self,
        provider: str,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        require_volume: bool = False,
        tolerance_days: int = 5,
    ) -> bool:
        """
        요청 구간이 저장소로 충분히 커버되는지 판단합니다.

        거래일과 요청 종료일이 정확히 일치하지 않을 수 있어 tolerance를 둡니다.
        """
        coverage = self.get_coverage(provider, ticker)
        if coverage is None:
            return False

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        min_ts = pd.Timestamp(coverage["min_date"])
        max_ts = pd.Timestamp(coverage["max_date"])

        if min_ts > start_ts:
            return False
        if max_ts < end_ts - pd.Timedelta(days=tolerance_days):
            return False
        if require_volume and coverage["volume_rows"] <= 0:
            return False
        return True

    def summary(self) -> dict:
        """저장소 전체 요약 정보."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS rows,
                    COUNT(DISTINCT provider || ':' || ticker) AS instruments,
                    MIN(date) AS min_date,
                    MAX(date) AS max_date
                FROM daily_bars
                """
            ).fetchone()
        return {
            "db_path": str(self.db_path),
            "rows": int(row[0] or 0),
            "instruments": int(row[1] or 0),
            "min_date": row[2],
            "max_date": row[3],
        }

    @staticmethod
    def _normalize_bars_frame(frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame(name="close")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        rename_map = {}
        for col in df.columns:
            low = str(col).strip().lower()
            if low in {"open", "high", "low", "close", "adj close", "adj_close", "volume"}:
                rename_map[col] = low.replace(" ", "_")
        df = df.rename(columns=rename_map)

        wanted_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in wanted_cols:
            if col not in df.columns:
                df[col] = pd.NA

        normalized = df[wanted_cols].reset_index().rename(columns={"index": "date"})
        normalized["date"] = normalized["date"].dt.strftime("%Y-%m-%d")
        for col in wanted_cols:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized = normalized.dropna(how="all", subset=wanted_cols)
        return normalized
