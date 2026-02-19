"""
main.py
───────
코스닥 세력 매집 감지 시스템 - 메인 진입점.

실행 방법:
  python main.py               → APScheduler 시작 (매일 18:00)
  python main.py --run-now     → 즉시 1회 실행
  python main.py --history     → 과거 1년 데이터 수집
  python main.py --backtest    → 백테스트 실행
  python main.py --date 20240115  → 특정 날짜 스캔
"""
import argparse
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from backtester import Backtester
from data_collector import DataCollector, _latest_trading_date
from db_manager import DBManager
from report_generator import ChartGenerator, HTMLReportGenerator, TelegramNotifier, BacktestReportGenerator
from signal_engine import SignalEngine
from universe_filter import UniverseFilter


# ── 설정 로드 ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    # .env 파일 로드 (있으면)
    env_path = Path(path).parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    import os
                    os.environ.setdefault(key.strip(), val.strip())

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 환경변수 오버라이드 (토큰/키 등 민감 정보)
    import os
    tg = cfg.setdefault("telegram", {})
    if os.environ.get("TELEGRAM_TOKEN"):
        tg["token"] = os.environ["TELEGRAM_TOKEN"]
    if os.environ.get("TELEGRAM_CHAT_ID"):
        tg["chat_id"] = os.environ["TELEGRAM_CHAT_ID"]
    dart = cfg.setdefault("dart", {})
    if os.environ.get("DART_API_KEY"):
        dart["api_key"] = os.environ["DART_API_KEY"]

    return cfg


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    log_dir = Path(log_cfg.get("dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_dir / log_cfg.get("filename", "scanner.log"),
            maxBytes=log_cfg.get("max_bytes", 10_485_760),
            backupCount=log_cfg.get("backup_count", 5),
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO"), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ── 메인 스캐너 ───────────────────────────────────────────────────────────

class StockScanner:
    """전체 파이프라인 조율"""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("StockScanner")

        # 컴포넌트 초기화
        db_path = config.get("database", {}).get("path", "db/stock_history.db")
        self.db          = DBManager(db_path)
        self.collector   = DataCollector(config, self.db)
        self.uf          = UniverseFilter(config, self.db)
        self.engine      = SignalEngine(config, self.db)
        self.chart_gen   = ChartGenerator(config)
        self.html_gen    = HTMLReportGenerator(config)
        self.notifier    = TelegramNotifier(config)
        self.backtester  = Backtester(config, self.db)

    # ── 스캔 실행 ─────────────────────────────────────────────────────────

    def run(self, date: str = None) -> None:
        """1회 전체 스캔 실행"""
        t0 = time.time()
        date = date or _latest_trading_date()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  스캔 시작: {date}")
        self.logger.info(f"{'='*60}")

        try:
            # 1. 데이터 수집 (OHLCV + 시총 + DART)
            self.logger.info("Step 1: 데이터 수집")
            collect_result = self.collector.run(date)
            mktcap_df = collect_result.get("market_cap_df")

            # 2. 유니버스 필터링
            self.logger.info("Step 2: 유니버스 필터링")
            universe = self.uf.filter(date, mktcap_df)
            if universe.empty:
                self.logger.error("유니버스 비어있음 → 스캔 중단")
                return
            self.logger.info(f"  유니버스: {len(universe)}종목")

            # 3. 시그널 스캔
            self.logger.info("Step 3: 시그널 스캔")
            results = self.engine.scan(universe, date)
            self.logger.info(f"  감지 종목: {len(results)}")

            if not results:
                self.logger.info("  감지된 종목 없음")
                self._save_history(date, len(universe), 0, time.time() - t0)
                return

            # 4. 결과 출력 (콘솔)
            df_result = self.engine.to_dataframe(results)
            self.logger.info(
                f"\n{df_result[['ticker','name','total_score','sector']].to_string(index=False)}"
            )

            # 5. DB 저장
            self.logger.info("Step 4: DB 저장")
            result_dicts = [r.to_dict() for r in results]
            self.db.save_scan_results(result_dicts, date)
            self.db.init_return_tracking(date, result_dicts)

            # 6. 차트 생성
            self.logger.info("Step 5: 차트 생성")
            chart_paths = self.chart_gen.generate_all(results, self.db)

            # 7. HTML 리포트
            self.logger.info("Step 6: HTML 리포트")
            report_path = self.html_gen.generate(results, date, chart_paths)
            self.logger.info(f"  리포트: {report_path}")

            # 8. Telegram 알림
            self.logger.info("Step 7: Telegram 전송")
            self.notifier.notify(results, date, chart_paths)

        except Exception as e:
            self.logger.error(f"스캔 오류: {e}", exc_info=True)

        finally:
            duration = time.time() - t0
            self._save_history(date, len(universe) if "universe" in dir() else 0,
                               len(results) if "results" in dir() else 0, duration)
            self.logger.info(f"  완료: {duration:.1f}초")

    def _save_history(
        self, date: str, universe_count: int, result_count: int, duration: float
    ) -> None:
        try:
            self.db.save_scan_history(date, universe_count, result_count, duration)
        except Exception as e:
            self.logger.warning(f"scan_history 저장 실패: {e}")

    # ── 기타 모드 ─────────────────────────────────────────────────────────

    def collect_history(self) -> None:
        """과거 1년 데이터 수집"""
        years = self.config.get("data", {}).get("history_years", 1)
        self.logger.info(f"과거 {years}년 데이터 수집 시작")
        self.collector.collect_history(years)

    def run_backtest(self, months: int = None, limit: int = None) -> None:
        """백테스트 실행"""
        self.logger.info("백테스트 시작")
        date = _latest_trading_date()
        tickers = self.collector.get_kosdaq_tickers(date)

        if limit:
            tickers = tickers[:limit]

        # Run backtest on all tickers
        self.logger.info(f"Target Tickers: {len(tickers)}")
        result = self.backtester.run(tickers)
        
        if not result.empty:
            # 1. Console Summary
            self.backtester.print_summary(result)
            summary_dict = self.backtester.summary(result)

            # 2. Score Band Analysis (Log)
            for fd in self.config.get("backtest", {}).get("forward_days", [10, 20, 30]):
                band_df = self.backtester.accuracy_by_score_band(result, fd)
                if not band_df.empty:
                    self.logger.info(f"\n{fd}일 후 점수구간별 승률:\n{band_df.to_string(index=False)}")

            # 3. CSV Save
            out_csv = Path("reports") / f"backtest_{date}.csv"
            out_csv.parent.mkdir(exist_ok=True)
            result.to_csv(out_csv, index=False, encoding="utf-8-sig")
            self.logger.info(f"백테스트 CSV 저장: {out_csv}")

            # 4. HTML Report
            report_gen = BacktestReportGenerator(self.config)
            report_gen.generate(result, summary_dict)



# ── 스케줄러 ─────────────────────────────────────────────────────────────

def start_scheduler(scanner: StockScanner, config: dict) -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched_cfg = config.get("scheduler", {})
    hour     = sched_cfg.get("hour", 18)
    minute   = sched_cfg.get("minute", 0)
    timezone = sched_cfg.get("timezone", "Asia/Seoul")

    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(
        func=scanner.run,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=timezone),
        id="daily_scan",
        name="코스닥 세력 매집 스캔",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    logger = logging.getLogger("scheduler")
    logger.info(f"스케줄러 시작: 매일 {hour:02d}:{minute:02d} ({timezone})")
    logger.info("중지: Ctrl+C")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료")


# ── 진입점 ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="코스닥 세력 매집 감지 시스템")
    parser.add_argument("--run-now",  action="store_true", help="즉시 1회 스캔")
    parser.add_argument("--history",  action="store_true", help="과거 1년 데이터 수집")
    parser.add_argument("--backtest", action="store_true", help="백테스트 실행")
    parser.add_argument("--date",     type=str,            help="특정 날짜 스캔 (YYYYMMDD)")
    parser.add_argument("--limit",    type=int,            help="백테스트/스캔 종목 수 제한 (테스트용)")
    parser.add_argument("--config",   type=str, default="config.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    scanner = StockScanner(config)

    if args.run_now:
        scanner.run()
    elif args.date:
        scanner.run(args.date)
    elif args.history:
        scanner.collect_history()
    elif args.backtest:
        scanner.run_backtest(limit=args.limit)
    else:
        start_scheduler(scanner, config)


if __name__ == "__main__":
    main()
