"""
db_manager.py
─────────────
SQLite 데이터베이스 관리 모듈.
- 스키마 초기화
- OHLCV / 시총 / DART 공시 / 스캔 결과 CRUD
- 증분 업데이트 (이미 존재하는 날짜는 스킵)
"""
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── DDL ────────────────────────────────────────────────────────────────────

DDL = """
-- OHLCV 일봉
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,   -- YYYYMMDD
    open        INTEGER NOT NULL,
    high        INTEGER NOT NULL,
    low         INTEGER NOT NULL,
    close       INTEGER NOT NULL,
    volume      INTEGER NOT NULL,
    amount      INTEGER NOT NULL,   -- 거래대금 (원)
    PRIMARY KEY (ticker, date)
);

-- 시총 / 상장주식수
CREATE TABLE IF NOT EXISTS market_cap (
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    market_cap      INTEGER NOT NULL,   -- 시가총액 (원)
    shares_listed   INTEGER NOT NULL,   -- 상장주식수
    close           INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- DART 공시 (CB / BW)
CREATE TABLE IF NOT EXISTS dart_disclosure (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    name            TEXT,
    report_nm       TEXT,           -- 공시명
    rcept_no        TEXT UNIQUE,    -- 접수번호
    rcept_dt        TEXT,           -- 접수일 YYYYMMDD
    report_type     TEXT,           -- CB / BW / 기타
    amount_billion  REAL,           -- 금액 (억원)
    created_at      TEXT DEFAULT (datetime('now','localtime'))
);

-- 스캔 결과 (종합 점수)
CREATE TABLE IF NOT EXISTS scan_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    name            TEXT,
    total_score     REAL    NOT NULL,
    sector          TEXT,
    market_cap_억   INTEGER,
    signals_json    TEXT,           -- 개별 신호 점수 JSON
    created_at      TEXT DEFAULT (datetime('now','localtime'))
);

-- 스캔 실행 이력
CREATE TABLE IF NOT EXISTS scan_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date       TEXT    NOT NULL,
    universe_count  INTEGER,
    result_count    INTEGER,
    duration_sec    REAL,
    created_at      TEXT DEFAULT (datetime('now','localtime'))
);

-- 수익률 트래킹
CREATE TABLE IF NOT EXISTS return_tracking (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    entry_price     INTEGER,
    price_5d        INTEGER,
    price_10d       INTEGER,
    price_20d       INTEGER,
    return_5d       REAL,
    return_10d      REAL,
    return_20d      REAL,
    updated_at      TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE (scan_date, ticker)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date   ON ohlcv (ticker, date);
CREATE INDEX IF NOT EXISTS idx_mktcap_date          ON market_cap (date);
CREATE INDEX IF NOT EXISTS idx_scan_date            ON scan_results (scan_date);
CREATE INDEX IF NOT EXISTS idx_dart_ticker          ON dart_disclosure (ticker);
CREATE INDEX IF NOT EXISTS idx_dart_dt              ON dart_disclosure (rcept_dt);
"""


class DBManager:
    """SQLite 데이터베이스 관리자"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._memory_conn: Optional[sqlite3.Connection] = None

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def conn(self):
        """
        컨텍스트 매니저로 커넥션 제공.
        :memory: DB는 단일 커넥션 유지 (테스트용).
        """
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            con = self._memory_conn
            try:
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
        else:
            con = sqlite3.connect(self.db_path, timeout=30)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA foreign_keys=ON")
            try:
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
            finally:
                con.close()

    def _init_db(self) -> None:
        with self.conn() as con:
            con.executescript(DDL)
        logger.info(f"Database initialized: {self.db_path}")

    # ── OHLCV ──────────────────────────────────────────────────────────────

    def get_ohlcv_max_date(self, ticker: str) -> Optional[str]:
        """해당 종목의 DB 내 최신 날짜 조회 (증분 업데이트용)"""
        with self.conn() as con:
            row = con.execute(
                "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
            ).fetchone()
            return row[0] if row else None

    def get_ohlcv_date_range(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """종목의 DB 데이터 날짜 범위"""
        with self.conn() as con:
            row = con.execute(
                "SELECT MIN(date), MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
            ).fetchone()
            return (row[0], row[1]) if row else (None, None)

    def upsert_ohlcv(self, df: pd.DataFrame, ticker: str) -> int:
        """
        OHLCV 데이터 upsert (INSERT OR IGNORE → 증분 업데이트).

        Args:
            df: pykrx get_market_ohlcv 결과 DataFrame (컬럼: 시가 고가 저가 종가 거래량 거래대금)
            ticker: 종목코드

        Returns:
            삽입된 행 수
        """
        if df.empty:
            return 0

        records = []
        for date_val, row in df.iterrows():
            date_str = (
                date_val.strftime("%Y%m%d")
                if hasattr(date_val, "strftime")
                else str(date_val).replace("-", "")[:8]
            )
            records.append((
                ticker,
                date_str,
                int(row.get("시가", row.get("Open", 0))),
                int(row.get("고가", row.get("High", 0))),
                int(row.get("저가", row.get("Low", 0))),
                int(row.get("종가", row.get("Close", 0))),
                int(row.get("거래량", row.get("Volume", 0))),
                int(row.get("거래대금", row.get("Amount", 0))),
            ))

        with self.conn() as con:
            cur = con.executemany(
                """INSERT OR IGNORE INTO ohlcv
                   (ticker, date, open, high, low, close, volume, amount)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                records,
            )
            inserted = cur.rowcount

        logger.debug(f"ohlcv upsert {ticker}: {inserted}/{len(records)} rows inserted")
        return inserted

    def load_ohlcv(
        self, ticker: str, fromdate: str, todate: str
    ) -> pd.DataFrame:
        """DB에서 OHLCV 로드"""
        with self.conn() as con:
            df = pd.read_sql_query(
                """SELECT date, open, high, low, close, volume, amount
                   FROM ohlcv
                   WHERE ticker = ? AND date BETWEEN ? AND ?
                   ORDER BY date""",
                con,
                params=(ticker, fromdate, todate),
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df.set_index("date", inplace=True)
            # 한글 컬럼명으로 통일 (signal_engine 호환)
            df.columns = ["시가", "고가", "저가", "종가", "거래량", "거래대금"]
        return df

    # ── 시총 ───────────────────────────────────────────────────────────────

    def upsert_market_cap(self, records: List[Dict]) -> int:
        """시총 데이터 upsert"""
        if not records:
            return 0
        rows = [
            (r["ticker"], r["date"], r["market_cap"], r["shares_listed"], r["close"])
            for r in records
        ]
        with self.conn() as con:
            cur = con.executemany(
                """INSERT OR IGNORE INTO market_cap
                   (ticker, date, market_cap, shares_listed, close)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
            return cur.rowcount

    def load_market_cap(self, date: str) -> pd.DataFrame:
        """특정 날짜의 전종목 시총 로드.
        pykrx가 시가총액=0을 반환한 경우 OHLCV 종가 × 상장주식수로 자동 보정."""
        with self.conn() as con:
            return pd.read_sql_query(
                """SELECT mc.ticker,
                          mc.date,
                          CASE WHEN mc.market_cap > 0
                               THEN mc.market_cap
                               ELSE CAST(mc.shares_listed AS REAL) * COALESCE(o.close, 0)
                          END AS market_cap,
                          mc.shares_listed,
                          CASE WHEN mc.close > 0
                               THEN mc.close
                               ELSE COALESCE(o.close, 0)
                          END AS close
                   FROM market_cap mc
                   LEFT JOIN ohlcv o
                          ON mc.ticker = o.ticker AND o.date = mc.date
                   WHERE mc.date = ?""",
                con,
                params=(date,),
            )

    # ── DART 공시 ──────────────────────────────────────────────────────────

    def upsert_dart(self, records: List[Dict]) -> int:
        """DART 공시 upsert (rcept_no 기준 중복 방지)"""
        if not records:
            return 0
        rows = [
            (
                r["ticker"],
                r.get("name", ""),
                r.get("report_nm", ""),
                r.get("rcept_no", ""),
                r.get("rcept_dt", ""),
                r.get("report_type", ""),
                r.get("amount_billion"),
            )
            for r in records
        ]
        with self.conn() as con:
            cur = con.executemany(
                """INSERT OR IGNORE INTO dart_disclosure
                   (ticker, name, report_nm, rcept_no, rcept_dt, report_type, amount_billion)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            return cur.rowcount

    def load_dart_recent(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """최근 N일 DART 공시 조회"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        with self.conn() as con:
            return pd.read_sql_query(
                """SELECT * FROM dart_disclosure
                   WHERE ticker = ? AND rcept_dt >= ?
                   ORDER BY rcept_dt DESC""",
                con,
                params=(ticker, cutoff),
            )

    # ── 스캔 결과 ──────────────────────────────────────────────────────────

    def save_scan_results(self, results: List[Dict], scan_date: str) -> None:
        """스캔 결과 저장"""
        rows = [
            (
                scan_date,
                r["ticker"],
                r.get("name", ""),
                r.get("total_score", 0),
                r.get("sector", ""),
                r.get("market_cap_억", 0),
                json.dumps(r.get("signals", {}), ensure_ascii=False),
            )
            for r in results
        ]
        with self.conn() as con:
            con.executemany(
                """INSERT INTO scan_results
                   (scan_date, ticker, name, total_score, sector, market_cap_억, signals_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        logger.info(f"Saved {len(results)} scan results for {scan_date}")

    def save_scan_history(
        self, scan_date: str, universe_count: int,
        result_count: int, duration_sec: float
    ) -> None:
        with self.conn() as con:
            con.execute(
                """INSERT INTO scan_history
                   (scan_date, universe_count, result_count, duration_sec)
                   VALUES (?, ?, ?, ?)""",
                (scan_date, universe_count, result_count, round(duration_sec, 2)),
            )

    def load_scan_results(self, scan_date: str) -> pd.DataFrame:
        """특정 날짜 스캔 결과 로드"""
        with self.conn() as con:
            df = pd.read_sql_query(
                "SELECT * FROM scan_results WHERE scan_date = ? ORDER BY total_score DESC",
                con,
                params=(scan_date,),
            )
        return df

    def load_recent_scan_dates(self, n: int = 30) -> List[str]:
        """최근 N번 스캔 날짜 목록"""
        with self.conn() as con:
            rows = con.execute(
                "SELECT DISTINCT scan_date FROM scan_history ORDER BY scan_date DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [r[0] for r in rows]

    # ── 수익률 트래킹 ──────────────────────────────────────────────────────

    def init_return_tracking(self, scan_date: str, results: List[Dict]) -> None:
        """스캔 직후 entry price 기록"""
        rows = [(scan_date, r["ticker"], r.get("close", 0)) for r in results]
        with self.conn() as con:
            con.executemany(
                """INSERT OR IGNORE INTO return_tracking (scan_date, ticker, entry_price)
                   VALUES (?, ?, ?)""",
                rows,
            )

    def update_return_tracking(
        self, scan_date: str, ticker: str,
        price_5d: int = None, price_10d: int = None, price_20d: int = None
    ) -> None:
        """수익률 업데이트"""
        with self.conn() as con:
            row = con.execute(
                "SELECT entry_price FROM return_tracking WHERE scan_date=? AND ticker=?",
                (scan_date, ticker),
            ).fetchone()
            if not row or not row[0]:
                return
            entry = row[0]

            def ret(p):
                return round((p - entry) / entry * 100, 2) if p else None

            con.execute(
                """UPDATE return_tracking
                   SET price_5d=?, price_10d=?, price_20d=?,
                       return_5d=?, return_10d=?, return_20d=?,
                       updated_at=datetime('now','localtime')
                   WHERE scan_date=? AND ticker=?""",
                (
                    price_5d, price_10d, price_20d,
                    ret(price_5d), ret(price_10d), ret(price_20d),
                    scan_date, ticker,
                ),
            )
