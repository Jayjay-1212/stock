"""
data_collector.py
─────────────────
pykrx + OpenDartReader 를 이용한 데이터 수집 모듈.

기능:
1. 코스닥 전종목 OHLCV 일봉 (최근 120일, 증분 업데이트)
2. 시총 / 상장주식수 데이터
3. DART CB/BW 공시 자동 파싱
4. API 레이트 리밋 대응 (sleep + retry)
5. 실패 종목 별도 로그 기록
"""
import logging
import logging.handlers
import re
import time
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pykrx import stock

from db_manager import DBManager

logger = logging.getLogger(__name__)


# ── 실패 종목 전용 로거 ────────────────────────────────────────────────────

def get_failed_logger(log_path: str) -> logging.Logger:
    fl = logging.getLogger("failed_tickers")
    if not fl.handlers:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(log_path, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
        fl.addHandler(h)
        fl.setLevel(logging.WARNING)
    return fl


# ── 유틸 ──────────────────────────────────────────────────────────────────

def _date_str(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _retry(func, retries: int = 3, delay: float = 2.0):
    """함수 실행 재시도 데코레이터 (함수형)"""
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Retry {attempt+1}/{retries} after error: {e}")
                time.sleep(delay * (attempt + 1))
            else:
                raise


def _latest_trading_date(offset: int = 0) -> str:
    """최근 거래일 탐색 (공휴일/주말 건너뜀).

    단순히 종목 리스트 존재 여부가 아닌, 실제 가격 데이터(OHLCV)가
    있는 날짜를 반환한다. 장 마감 후 KRX 데이터 게시 전이거나 공휴일인
    경우를 올바르게 처리한다.
    """
    target = datetime.now() - timedelta(days=offset)
    for _ in range(15):
        ds = _date_str(target)
        try:
            tickers = stock.get_market_ticker_list(ds, market="KOSDAQ")
            if tickers:
                # 실제 OHLCV 데이터가 있는지 샘플 종목으로 확인
                try:
                    sample = stock.get_market_ohlcv(ds, ds, tickers[0])
                    if not sample.empty:
                        return ds
                except Exception:
                    pass
        except Exception:
            pass
        target -= timedelta(days=1)
    return _date_str(datetime.now())


# ── DataCollector ─────────────────────────────────────────────────────────

class DataCollector:
    """코스닥 데이터 수집기"""

    def __init__(self, config: dict, db: DBManager):
        self.config = config
        self.db = db

        data_cfg = config.get("data", {})
        self.ohlcv_days = data_cfg.get("ohlcv_days", 120)
        self.retry_count = data_cfg.get("retry_count", 3)
        self.retry_delay = data_cfg.get("retry_delay", 2)
        self.rate_delay = data_cfg.get("rate_limit_delay", 0.3)
        self.failed_log = data_cfg.get("failed_log", "logs/failed_tickers.log")

        dart_cfg = config.get("dart", {})
        self.dart_enabled = dart_cfg.get("enabled", False)
        self.dart_api_key = dart_cfg.get("api_key", "")
        self.dart_lookback = dart_cfg.get("lookback_days", 90)
        self.dart_types = dart_cfg.get("report_types", ["CB", "BW"])

        self.failed_logger = get_failed_logger(self.failed_log)

    # ── 종목 리스트 ────────────────────────────────────────────────────────

    def get_kosdaq_tickers(self, date: str = None) -> List[Tuple[str, str]]:
        """코스닥 전종목 (ticker, name) 반환"""
        date = date or _latest_trading_date()

        def _fetch():
            tickers = stock.get_market_ticker_list(date, market="KOSDAQ")
            result = []
            for t in tqdm(tickers, desc="Fetching Ticker Names", ncols=80):
                try:
                    name = stock.get_market_ticker_name(t)
                    result.append((t, name))
                    time.sleep(self.rate_delay * 0.3)
                except Exception as e:
                    logger.debug(f"Name fetch failed {t}: {e}")
                    result.append((t, t))
            return result

        return _retry(_fetch, self.retry_count, self.retry_delay)

    # ── 시총 ───────────────────────────────────────────────────────────────

    def collect_market_cap(self, date: str = None) -> pd.DataFrame:
        """
        코스닥 전종목 시총 수집 및 DB 저장.

        Returns:
            시총 DataFrame (columns: ticker, name, market_cap, shares_listed, close)
        """
        date = date or _latest_trading_date()
        logger.info(f"[MarketCap] Fetching KOSDAQ market cap for {date}")

        def _fetch():
            return stock.get_market_cap_by_ticker(date, market="KOSDAQ")

        try:
            df = _retry(_fetch, self.retry_count, self.retry_delay)
        except Exception as e:
            logger.error(f"[MarketCap] Failed: {e}")
            return pd.DataFrame()

        if df.empty:
            logger.warning(f"[MarketCap] Empty result for {date}")
            return pd.DataFrame()

        df.index.name = "ticker"
        df = df.reset_index()

        # 종목명 추가
        df["name"] = df["ticker"].apply(
            lambda t: self._safe_get_name(t)
        )

        # DB 저장
        records = []
        for _, row in df.iterrows():
            records.append({
                "ticker": row["ticker"],
                "date": date,
                "market_cap": int(row.get("시가총액", 0)),
                "shares_listed": int(row.get("상장주식수", 0)),
                "close": int(row.get("종가", 0)),
            })
        inserted = self.db.upsert_market_cap(records)
        logger.info(f"[MarketCap] Saved {inserted}/{len(records)} records")

        return df

    def _safe_get_name(self, ticker: str) -> str:
        try:
            return stock.get_market_ticker_name(ticker)
        except Exception:
            return ticker

    # ── OHLCV ──────────────────────────────────────────────────────────────

    def collect_ohlcv_single(
        self, ticker: str, fromdate: str, todate: str
    ) -> bool:
        """
        단일 종목 OHLCV 수집 (증분 업데이트).

        Returns:
            True = 성공, False = 실패
        """
        # 이미 있는 데이터 최신 날짜 확인 (증분)
        max_date = self.db.get_ohlcv_max_date(ticker)
        if max_date and max_date >= todate:
            logger.debug(f"[OHLCV] {ticker} already up-to-date ({max_date})")
            return True

        # 이미 있으면 그 다음 날부터만 수집
        if max_date:
            dt = datetime.strptime(max_date, "%Y%m%d") + timedelta(days=1)
            fromdate = _date_str(dt)

        def _fetch():
            df = stock.get_market_ohlcv(fromdate, todate, ticker)
            return df

        try:
            df = _retry(_fetch, self.retry_count, self.retry_delay)
        except Exception as e:
            msg = f"{ticker}\t{fromdate}~{todate}\t{e}"
            self.failed_logger.warning(msg)
            logger.warning(f"[OHLCV] Failed {ticker}: {e}")
            return False

        if df is None or df.empty:
            logger.debug(f"[OHLCV] Empty data for {ticker} {fromdate}~{todate}")
            return True  # 거래 없는 기간은 정상

        inserted = self.db.upsert_ohlcv(df, ticker)
        logger.debug(f"[OHLCV] {ticker}: +{inserted} rows")
        return True

    def collect_ohlcv_universe(
        self, tickers: List[Tuple[str, str]], date: str = None
    ) -> Dict[str, bool]:
        """
        유니버스 전종목 OHLCV 수집.

        Args:
            tickers: [(ticker, name), ...]
            date: 기준 날짜 (None=오늘)

        Returns:
            {ticker: success}
        """
        todate = date or _latest_trading_date()
        from_dt = datetime.strptime(todate, "%Y%m%d") - timedelta(days=self.ohlcv_days)
        fromdate = _date_str(from_dt)

        logger.info(f"[OHLCV] Collecting {len(tickers)} tickers {fromdate}~{todate}")

        results = {}
        for i, (ticker, name) in enumerate(tqdm(tickers, desc="OHLCV Collection", ncols=80)):
            # if i % 50 == 0:
            #     logger.info(f"[OHLCV] Progress {i}/{len(tickers)}")

            success = self.collect_ohlcv_single(ticker, fromdate, todate)
            results[ticker] = success
            time.sleep(self.rate_delay)

        failed = [t for t, ok in results.items() if not ok]
        logger.info(
            f"[OHLCV] Done: {len(tickers)-len(failed)} success, {len(failed)} failed"
        )
        if failed:
            logger.warning(f"[OHLCV] Failed tickers: {failed[:10]}...")

        return results

    # ── DART 공시 ──────────────────────────────────────────────────────────

    def collect_dart_disclosures(
        self, tickers: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        DART에서 CB/BW 공시 수집.
        OpenDartReader API 키 필요.

        Returns:
            공시 레코드 리스트
        """
        if not self.dart_enabled:
            logger.info("[DART] Disabled (set dart.enabled=true in config)")
            return []

        if not self.dart_api_key:
            logger.warning("[DART] API key not set")
            return []

        try:
            import OpenDartReader as odr
        except ImportError:
            logger.error("[DART] OpenDartReader not installed: pip install OpenDartReader")
            return []

        dart = odr.OpenDartReader(self.dart_api_key)
        cutoff = _date_str(datetime.now() - timedelta(days=self.dart_lookback))
        today = _date_str(datetime.now())

        all_records = []
        ticker_map = {t: n for t, n in tickers}

        for report_type in self.dart_types:
            logger.info(f"[DART] Fetching {report_type} disclosures {cutoff}~{today}")
            try:
                # 전체 공시 조회 (코스닥)
                df = _retry(
                    lambda rt=report_type: dart.list(
                        corp_cls="K",        # 코스닥
                        bgn_de=cutoff,
                        end_de=today,
                        pblntf_detail_ty=rt,
                    ),
                    self.retry_count,
                    self.retry_delay,
                )

                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    ticker = self._dart_corp_to_ticker(dart, row.get("corp_code", ""))
                    if not ticker:
                        continue

                    record = {
                        "ticker": ticker,
                        "name": row.get("corp_name", ticker_map.get(ticker, "")),
                        "report_nm": row.get("report_nm", ""),
                        "rcept_no": row.get("rcept_no", ""),
                        "rcept_dt": str(row.get("rcept_dt", "")).replace("-", ""),
                        "report_type": report_type,
                        "amount_billion": self._parse_amount(row.get("report_nm", "")),
                    }
                    all_records.append(record)
                    time.sleep(self.rate_delay * 0.5)

            except Exception as e:
                logger.error(f"[DART] {report_type} fetch failed: {e}")
                continue

        if all_records:
            inserted = self.db.upsert_dart(all_records)
            logger.info(f"[DART] Saved {inserted}/{len(all_records)} disclosures")

        return all_records

    def _dart_corp_to_ticker(self, dart, corp_code: str) -> Optional[str]:
        """DART corp_code → 주식 ticker 변환"""
        try:
            info = dart.company(corp_code)
            if info is not None and "stock_code" in info:
                return str(info["stock_code"]).zfill(6)
        except Exception:
            pass
        return None

    def _parse_amount(self, report_nm: str) -> Optional[float]:
        """공시명에서 금액(억원) 파싱"""
        # 예: "전환사채권 발행결정 (500억원)" → 500.0
        patterns = [
            r"(\d[\d,]*)\s*억",
            r"(\d[\d,]*)\s*억원",
        ]
        for pat in patterns:
            m = re.search(pat, report_nm)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
        return None

    # ── 시총 보정 ──────────────────────────────────────────────────────────

    def _supplement_market_cap_from_ohlcv(
        self, df: pd.DataFrame, date: str
    ) -> pd.DataFrame:
        """pykrx가 시가총액 0 반환 시 OHLCV 종가 × 상장주식수로 보정.

        OHLCV 수집 완료 후 호출해야 DB에 종가 데이터가 있음.
        """
        logger.info("[MarketCap] Supplementing zero 시가총액 from OHLCV close prices")
        try:
            tickers = df["ticker"].tolist()
            with self.db.conn() as con:
                ph = ",".join("?" * len(tickers))
                rows = con.execute(
                    f"SELECT ticker, close FROM ohlcv WHERE date=? AND ticker IN ({ph})",
                    [date] + tickers,
                ).fetchall()
            close_map = {r[0]: r[1] for r in rows}

            df = df.copy()
            for idx, row in df.iterrows():
                ticker = str(row["ticker"])
                close = close_map.get(ticker, 0)
                shares = int(row.get("상장주식수", 0))
                if close > 0 and shares > 0:
                    df.loc[idx, "시가총액"] = close * shares
                    df.loc[idx, "종가"] = close

            # DB의 0 레코드도 UPDATE
            to_update = [
                (int(row["시가총액"]), int(row["종가"]), str(row["ticker"]), date)
                for _, row in df.iterrows()
                if int(row.get("시가총액", 0)) > 0
            ]
            if to_update:
                with self.db.conn() as con:
                    con.executemany(
                        "UPDATE market_cap SET market_cap=?, close=? "
                        "WHERE ticker=? AND date=? AND market_cap=0",
                        to_update,
                    )
                logger.info(f"[MarketCap] Updated {len(to_update)} DB records via OHLCV close")

        except Exception as e:
            logger.warning(f"[MarketCap] Supplement failed: {e}")

        return df

    # ── 전체 수집 파이프라인 ───────────────────────────────────────────────

    def run(self, date: str = None) -> Dict:
        """
        전체 데이터 수집 실행.

        Returns:
            {market_cap_df, ohlcv_results, dart_records}
        """
        date = date or _latest_trading_date()
        logger.info(f"=== Data Collection Start: {date} ===")

        # 1. 시총 수집 → 유니버스 확보
        mktcap_df = self.collect_market_cap(date)

        tickers = []
        if not mktcap_df.empty:
            tickers = [
                (str(row["ticker"]), str(row.get("name", row["ticker"])))
                for _, row in mktcap_df.iterrows()
            ]

        # 2. OHLCV 수집
        ohlcv_results = {}
        if tickers:
            ohlcv_results = self.collect_ohlcv_universe(tickers, date)

        # 3. 시총 0 보정: pykrx 미제공 시 OHLCV 종가 × 상장주식수 활용
        if (
            not mktcap_df.empty
            and "시가총액" in mktcap_df.columns
            and (mktcap_df["시가총액"] == 0).all()
        ):
            mktcap_df = self._supplement_market_cap_from_ohlcv(mktcap_df, date)

        # 4. DART 공시 수집
        dart_records = self.collect_dart_disclosures(tickers)

        logger.info(
            f"=== Data Collection Done: {len(tickers)} tickers, "
            f"{sum(ohlcv_results.values())} ohlcv ok, "
            f"{len(dart_records)} dart records ==="
        )

        return {
            "market_cap_df": mktcap_df,
            "ohlcv_results": ohlcv_results,
            "dart_records": dart_records,
        }

    def collect_history(self, years: int = 1) -> None:
        """
        최초 실행 시 과거 N년 데이터 일괄 수집.
        월말 기준으로 시총 스냅샷, 전체 OHLCV 수집.
        """
        logger.info(f"[History] Collecting {years} year(s) of history data")

        end = datetime.now()
        start = end - timedelta(days=365 * years)

        # 월 단위로 시총 수집
        current = start
        while current <= end:
            ds = _date_str(current)
            logger.info(f"[History] Market cap snapshot: {ds}")
            try:
                self.collect_market_cap(ds)
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"[History] market cap {ds} failed: {e}")

            # 다음 달
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # 현재 유니버스로 OHLCV 전체 수집
        today = _latest_trading_date()
        from_dt = end - timedelta(days=365 * years)
        fromdate = _date_str(from_dt)

        tickers = self.get_kosdaq_tickers(today)
        logger.info(f"[History] Collecting OHLCV for {len(tickers)} tickers")

        for i, (ticker, name) in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"[History] OHLCV progress {i}/{len(tickers)}")
            try:
                self.collect_ohlcv_single(ticker, fromdate, today)
                time.sleep(self.rate_delay)
            except Exception as e:
                logger.warning(f"[History] {ticker} failed: {e}")

        logger.info("[History] Done")
