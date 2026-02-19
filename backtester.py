"""
backtester.py
─────────────
과거 6개월 데이터로 시그널 정확도 검증.

검증 방법:
  1. 과거 날짜별로 시그널 계산
  2. 시그널 발생 후 N일(5/10/20) 수익률 측정
  3. Accuracy / 평균수익률 / 승률 리포트 출력
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from db_manager import DBManager
from signal_engine import SignalCalculator, Normalizer

logger = logging.getLogger(__name__)


class Backtester:
    """시그널 백테스터"""

    def __init__(self, config: dict, db: DBManager):
        self.config = config
        self.db = db
        self.bt_cfg = config.get("backtest", {})

        self.forward_days: List[int] = self.bt_cfg.get("forward_days", [5, 10, 20])
        self.min_score: float = self.bt_cfg.get("min_score_threshold", 60)
        self.history_months: int = self.bt_cfg.get("history_months", 6)

        self.calculator = SignalCalculator()
        self.sig_cfg = config.get("signals", {})

    # ── 공개 API ──────────────────────────────────────────────────────────

    def run(
        self,
        tickers: List[Tuple[str, str]],
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        백테스트 실행.

        Args:
            tickers: [(ticker, name), ...]
            end_date: 백테스트 종료 날짜 (None=오늘)

        Returns:
            종목별 백테스트 결과 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(
            days=self.history_months * 30
        )
        start_date = start_dt.strftime("%Y%m%d")

        logger.info(
            f"[Backtest] {start_date}~{end_date}, "
            f"{len(tickers)} tickers, forward={self.forward_days}"
        )

        all_records = []

        for ticker, name in tqdm(tickers, desc="Backtesting", ncols=80):
            try:
                records = self._backtest_ticker(ticker, name, start_date, end_date)
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"[Backtest] {ticker} {name} failed: {e}")

        if not all_records:
            logger.warning("[Backtest] No records generated")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info(f"[Backtest] Total {len(df)} signal events")
        return df

    def summary(self, result_df: pd.DataFrame) -> Dict:
        """백테스트 요약 통계"""
        if result_df.empty:
            return {}

        summary = {}
        for fd in self.forward_days:
            col = f"return_{fd}d"
            if col not in result_df.columns:
                continue
            valid = result_df[col].dropna()
            if valid.empty:
                continue
            summary[f"{fd}d"] = {
                "count": len(valid),
                "win_rate": round((valid > 0).mean() * 100, 1),
                "avg_return": round(valid.mean(), 2),
                "median_return": round(valid.median(), 2),
                "best": round(valid.max(), 2),
                "worst": round(valid.min(), 2),
            }
        return summary

    def print_summary(self, result_df: pd.DataFrame) -> None:
        """요약 통계 출력"""
        s = self.summary(result_df)
        print("\n" + "=" * 60)
        print("  백테스트 결과 요약")
        print("=" * 60)
        for period, stats in s.items():
            print(f"\n▶ {period} 후 수익률")
            print(f"  샘플 수  : {stats['count']}")
            print(f"  승률     : {stats['win_rate']}%")
            print(f"  평균수익률: {stats['avg_return']:+.2f}%")
            print(f"  중앙값   : {stats['median_return']:+.2f}%")
            print(f"  최고     : {stats['best']:+.2f}%")
            print(f"  최저     : {stats['worst']:+.2f}%")
        print("=" * 60)

    # ── 내부 메서드 ───────────────────────────────────────────────────────

    def _backtest_ticker(
        self, ticker: str, name: str, start_date: str, end_date: str
    ) -> List[Dict]:
        """단일 종목 백테스트"""

        # 전체 OHLCV 로드 (forward 기간 포함)
        max_forward = max(self.forward_days)
        ext_end = (
            datetime.strptime(end_date, "%Y%m%d") + timedelta(days=max_forward + 30)
        ).strftime("%Y%m%d")

        df = self.db.load_ohlcv(ticker, start_date, ext_end)
        if df.empty or len(df) < 80:
            return []

        # 날짜 목록 (start~end 사이, 주 1회 샘플링)
        dates = df.index[(df.index >= pd.Timestamp(start_date)) &
                         (df.index <= pd.Timestamp(end_date))]
        dates = dates[::5]  # 5영업일 간격 샘플링

        records = []
        for signal_date in dates:
            date_str = signal_date.strftime("%Y%m%d")
            window_df = df[df.index <= signal_date].tail(100)

            if len(window_df) < 70:
                continue

            # 시그널 계산
            score = self._calc_composite_score(window_df, ticker)
            if score < self.min_score:
                continue

            # 진입 가격
            entry_close = int(window_df["종가"].iloc[-1])

            record = {
                "ticker": ticker,
                "name": name,
                "signal_date": date_str,
                "score": round(score, 2),
                "entry_price": entry_close,
            }

            # N일 후 수익률 계산
            for fd in self.forward_days:
                target_dt = signal_date + timedelta(days=fd)
                future = df[df.index > target_dt]

                if not future.empty:
                    future_close = int(future.iloc[0]["종가"])
                    ret = (future_close - entry_close) / entry_close * 100
                    record[f"price_{fd}d"] = future_close
                    record[f"return_{fd}d"] = round(ret, 2)
                else:
                    record[f"price_{fd}d"] = None
                    record[f"return_{fd}d"] = None

            records.append(record)

        return records

    def _calc_composite_score(self, df: pd.DataFrame, ticker: str) -> float:
        """단순화된 종합 점수 계산 (백테스트용)"""
        weights = self.config.get("signals", {}).get("weights", {})
        cfg = self.sig_cfg

        signals = [
            self.calculator.obv_divergence(
                df, window=min(60, len(df) - 5)
            ),
            self.calculator.volume_asymmetry(
                df,
                window=cfg.get("volume_window", 20),
            ),
            self.calculator.box_range(
                df,
                window=min(60, len(df) - 5),
                bb_window=cfg.get("bb_window", 20),
            ),
            self.calculator.ma_alignment(
                df,
                short=cfg.get("ma_short", 5),
                mid=cfg.get("ma_mid", 20),
                long=min(cfg.get("ma_long", 60), len(df) - 5),
            ),
        ]

        total_w = 0.0
        weighted_sum = 0.0
        for sig in signals:
            w = weights.get(sig.name, 0)
            sig.weight = w
            total_w += w
            weighted_sum += sig.score * w

        if total_w == 0:
            return 0.0

        return (weighted_sum / total_w) * 10  # 0~100

    def accuracy_by_score_band(self, result_df: pd.DataFrame, forward: int = 10) -> pd.DataFrame:
        """
        점수 구간별 정확도 분석.

        Returns:
            score_band | count | win_rate | avg_return
        """
        col = f"return_{forward}d"
        if col not in result_df.columns:
            return pd.DataFrame()

        bins = [0, 55, 60, 65, 70, 75, 80, 85, 100]
        labels = ["<55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85+"]

        df = result_df.copy()
        df["score_band"] = pd.cut(
            df["score"], bins=bins, labels=labels, right=False
        )

        result = df.groupby("score_band").agg(
            count=(col, "count"),
            win_rate=(col, lambda x: round((x > 0).mean() * 100, 1)),
            avg_return=(col, lambda x: round(x.mean(), 2)),
        ).reset_index()

        return result
