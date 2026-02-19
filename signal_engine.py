"""
signal_engine.py
────────────────
세력 매집 시그널 감지 엔진.

신호 목록 (각 0~10점, 가중 합산):
  1. OBV Divergence       – 주가 기울기 vs OBV 기울기 비교 (60일 선형회귀)
  2. Volume Asymmetry     – 상승/하락일 거래량 비대칭 + 윗꼬리 비율
  3. Box Range            – 박스권 감지 (Rolling 고저폭 + Bollinger Band Width)
  4. MA Alignment         – 이동평균 정배열 / 골든크로스 준비
  5. Institutional Buying – 기관+외국인 순매수 (pykrx)
  6. DART Signal          – CB/BW 공시 감지

종합 점수 = Σ(신호점수 × 가중치) / Σ(가중치) × 섹터가중치
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from db_manager import DBManager

try:
    from pykrx import stock as _pykrx_stock
except ImportError:
    _pykrx_stock = None  # 테스트 환경 또는 미설치 시

logger = logging.getLogger(__name__)

# ── 선형회귀 기울기 (벡터화) ────────────────────────────────────────────────

def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling 선형회귀 기울기 계산 (정규화된 기울기 반환).
    반환값: 각 시점의 기울기 / 해당 구간 평균 (% 단위 비슷한 척도)
    """
    x = np.arange(window, dtype=float)
    x -= x.mean()
    ss = (x ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any() or y.mean() == 0:
            return np.nan
        y_c = y - y.mean()
        raw = np.dot(x, y_c) / ss
        return raw / abs(y.mean())  # 정규화

    return series.rolling(window).apply(_slope, raw=True)


# ── 데이터 클래스 ─────────────────────────────────────────────────────────

@dataclass
class Signal:
    name: str
    score: float      # 0 ~ 10
    weight: float     # 가중치
    detail: str = ""

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class StockScore:
    ticker: str
    name: str
    total_score: float      # 0 ~ 100 (가중 정규화 후 × 섹터가중치)
    signals: List[Signal] = field(default_factory=list)
    sector: str = "기타"
    sector_weight: float = 1.0
    market_cap: int = 0
    date: str = ""
    close: int = 0

    def to_dict(self) -> Dict:
        d = {
            "ticker": self.ticker,
            "name": self.name,
            "total_score": round(self.total_score, 2),
            "sector": self.sector,
            "market_cap_억": round(self.market_cap / 1e8),
            "date": self.date,
            "close": self.close,
            "signals": {s.name: round(s.score, 2) for s in self.signals},
        }
        for s in self.signals:
            d[s.name] = round(s.score, 2)
        return d


# ── 개별 시그널 계산기 ────────────────────────────────────────────────────

class SignalCalculator:
    """
    각 신호 계산 메서드 모음.
    입력 df 컬럼: 시가, 고가, 저가, 종가, 거래량  (index: datetime)
    """

    # 1. OBV Divergence ────────────────────────────────────────────────────

    def obv_divergence(
        self, df: pd.DataFrame, window: int = 60
    ) -> Signal:
        """
        OBV Divergence – 매집의 핵심 신호.

        판정 로직:
          - price_slope < 0 이고 obv_slope > 0  → 강한 매집 (Bullish Divergence)  10점
          - price_slope < 0 이고 obv_slope ≈ 0  → 중간 매집                        7점
          - price_slope ≥ 0 이고 obv_slope > 0  → 상승 확인 (정상 추세)             5점
          - 기타                                                                     2점
        """
        try:
            if len(df) < window:
                return Signal("obv_divergence", 0, 0, "데이터 부족")

            close = df["종가"].astype(float)
            volume = df["거래량"].astype(float)

            # OBV 계산
            direction = np.sign(close.diff().fillna(0))
            obv = (direction * volume).cumsum()

            # 기울기 (정규화)
            price_slope = rolling_slope(close, window).iloc[-1]
            obv_slope = rolling_slope(obv, window).iloc[-1]

            if np.isnan(price_slope) or np.isnan(obv_slope):
                return Signal("obv_divergence", 5, 0, "기울기 계산 불가")

            # 판정
            if price_slope < -0.001 and obv_slope > 0.001:
                score = 10.0
                detail = f"강한매집(price↓OBV↑) price={price_slope:.4f} obv={obv_slope:.4f}"
            elif price_slope < 0 and obv_slope > -0.0005:
                score = 7.0
                detail = f"매집중(price↓OBV→) price={price_slope:.4f} obv={obv_slope:.4f}"
            elif price_slope >= 0 and obv_slope > 0.001:
                score = 5.0
                detail = f"상승확인(price↑OBV↑) price={price_slope:.4f} obv={obv_slope:.4f}"
            elif price_slope > 0.001 and obv_slope < -0.001:
                score = 1.0  # 허수 상승 (가격↑ OBV↓) → 위험 신호
                detail = f"허수상승(price↑OBV↓) price={price_slope:.4f} obv={obv_slope:.4f}"
            else:
                score = 3.0
                detail = f"중립 price={price_slope:.4f} obv={obv_slope:.4f}"

            return Signal("obv_divergence", score, 0, detail)

        except Exception as e:
            return Signal("obv_divergence", 0, 0, f"오류: {e}")

    # 2. Volume Asymmetry ──────────────────────────────────────────────────

    def volume_asymmetry(
        self,
        df: pd.DataFrame,
        window: int = 20,
        wick_threshold: float = 0.6,
        up_vol_threshold: float = 1.5,
    ) -> Signal:
        """
        거래량 비대칭 신호.

          - up_vol_ratio: 상승일 평균 거래량 / 하락일 평균 거래량
          - upper_wick_ratio: (고가 - 종가) / (고가 - 저가)
            → 0.6 이상이면 윗꼬리 (세력이 쏟아내는 신호 → 감점)

        점수 구성 (0~10):
          up_vol_score (7점 만점) + wick_penalty (최대 -3점)
        """
        try:
            if len(df) < window:
                return Signal("volume_asymmetry", 0, 0, "데이터 부족")

            recent = df.tail(window).copy()

            # 상승일 / 하락일 구분
            recent["up"] = recent["종가"] >= recent["시가"]
            up_vol = recent.loc[recent["up"], "거래량"].mean()
            dn_vol = recent.loc[~recent["up"], "거래량"].mean()

            # 하락일이 없으면 매우 강한 매집 신호 → 상한값 부여
            up_vol_ratio = up_vol / dn_vol if dn_vol > 0 else 3.0

            # 윗꼬리 비율 (최근 5일 평균)
            last5 = recent.tail(5)
            hl = last5["고가"] - last5["저가"]
            wick = (last5["고가"] - last5["종가"]) / hl.replace(0, np.nan)
            avg_wick = wick.mean()

            # 점수
            # up_vol_ratio 1.5 이상 → 7점, 1.0 이하 → 0점
            up_score = min(7.0, max(0.0, (up_vol_ratio - 1.0) / (up_vol_threshold - 1.0) * 7))

            # 윗꼬리 페널티
            wick_penalty = min(3.0, max(0.0, (avg_wick - wick_threshold) * 10))

            score = max(0.0, up_score - wick_penalty)

            detail = (
                f"상승/하락거래량비율={up_vol_ratio:.2f}, "
                f"윗꼬리={avg_wick:.2f}"
            )
            return Signal("volume_asymmetry", score, 0, detail)

        except Exception as e:
            return Signal("volume_asymmetry", 0, 0, f"오류: {e}")

    # 3. Box Range ─────────────────────────────────────────────────────────

    def box_range(
        self,
        df: pd.DataFrame,
        window: int = 60,
        bb_window: int = 20,
        bb_std: float = 2.0,
        threshold: float = 0.12,
    ) -> Signal:
        """
        박스권 감지 신호.

          1) Rolling 60일 고점~저점 범위 / 중간가격
          2) Bollinger Band Width (BBW = (상단-하단) / 중간선)
             → BBW가 낮을수록 횡보 → 매집 가능성

        점수 (0~10):
          range_score (5점 만점) + bbw_score (5점 만점)
        """
        try:
            if len(df) < window:
                return Signal("box_range", 0, 0, "데이터 부족")

            close = df["종가"].astype(float)

            # ─ Rolling 가격 범위
            roll_high = df["고가"].rolling(window).max()
            roll_low  = df["저가"].rolling(window).min()
            roll_mid  = (roll_high + roll_low) / 2
            price_range = ((roll_high - roll_low) / roll_mid.replace(0, np.nan)).iloc[-1]

            # ─ Bollinger Band Width
            ma = close.rolling(bb_window).mean()
            std = close.rolling(bb_window).std()
            bbw = (2 * bb_std * std / ma.replace(0, np.nan)).iloc[-1]

            if np.isnan(price_range) or np.isnan(bbw):
                return Signal("box_range", 5, 0, "계산 불가")

            # range_score: threshold 이하 → 만점
            range_score = min(5.0, max(0.0, (threshold - price_range) / threshold * 5 + 5))
            # bbw_score: 0.1 이하 → 만점, 0.4 이상 → 0점
            bbw_score   = min(5.0, max(0.0, (0.4 - bbw) / (0.4 - 0.05) * 5))

            score = range_score + bbw_score

            # 박스권 영역 경계 (차트용)
            box_high = roll_high.iloc[-1]
            box_low  = roll_low.iloc[-1]

            detail = (
                f"범위={price_range:.1%}, BBW={bbw:.3f}, "
                f"박스({box_low:,.0f}~{box_high:,.0f})"
            )
            return Signal("box_range", score, 0, detail)

        except Exception as e:
            return Signal("box_range", 0, 0, f"오류: {e}")

    # 4. MA Alignment ──────────────────────────────────────────────────────

    def ma_alignment(
        self, df: pd.DataFrame,
        short: int = 5, mid: int = 20, long: int = 60
    ) -> Signal:
        """
        이동평균 정배열 / 골든크로스 준비 신호.

        점수 기준:
          정배열 (s>m>l) + 상승 중    → 8~10
          단기가 중기 돌파 직전 (5% 이내) → 6~7
          중기가 장기 위                   → 4~5
          역배열                            → 0~2
        """
        try:
            if len(df) < long + 5:
                return Signal("ma_alignment", 0, 0, "데이터 부족")

            close = df["종가"].astype(float)
            s = close.rolling(short).mean().iloc[-1]
            m = close.rolling(mid).mean().iloc[-1]
            l = close.rolling(long).mean().iloc[-1]

            # 추세 강도 (단기MA 기울기)
            s_slope = (close.rolling(short).mean().diff(3) / close.rolling(short).mean()).iloc[-1]

            if s > m > l:
                base = 8.0
                bonus = min(2.0, s_slope * 50) if s_slope > 0 else 0
                score = base + bonus
            elif s > l and abs(s - m) / (m + 1e-9) < 0.05:
                score = 6.5  # 골든크로스 직전
            elif m > l and s > l:
                score = 5.0
            elif s > l:
                score = 3.5
            elif m > l:
                score = 2.5
            else:
                score = max(0.0, 1.0 - (l - s) / (l + 1e-9) * 5)

            detail = f"MA{short}={s:,.0f} MA{mid}={m:,.0f} MA{long}={l:,.0f}"
            return Signal("ma_alignment", min(10.0, score), 0, detail)

        except Exception as e:
            return Signal("ma_alignment", 0, 0, f"오류: {e}")

    # 5. Institutional Buying ──────────────────────────────────────────────

    def institutional_buying(
        self,
        ticker: str,
        fromdate: str,
        todate: str,
        days: int = 5,
    ) -> Signal:
        """
        기관 + 외국인 순매수 신호 (pykrx).

        점수 기준:
          순매수 양 + 비율 → 0~10
        """
        try:
            stock = _pykrx_stock
            if stock is None:
                return Signal("institutional_buying", 5, 0, "pykrx 미설치")
            df = stock.get_market_trading_volume_by_investor(
                fromdate, todate, ticker
            )
            if df is None or df.empty:
                return Signal("institutional_buying", 5, 0, "데이터 없음")

            # 컬럼 탐색 (pykrx 버전에 따라 다를 수 있음)
            inst_cols = [c for c in df.columns if "기관" in str(c)]
            frgn_cols = [c for c in df.columns if "외국인" in str(c)]

            if not inst_cols or not frgn_cols:
                return Signal("institutional_buying", 5, 0, "컬럼 없음")

            recent = df.tail(days)
            inst_net  = recent[inst_cols[0]].sum()
            frgn_net  = recent[frgn_cols[0]].sum()
            total_net = inst_net + frgn_net

            # 전체 거래량 대비 비율
            vol_col = [c for c in df.columns if "전체" in str(c)]
            avg_vol = abs(recent[vol_col[0]]).mean() if vol_col else 1

            ratio = total_net / (avg_vol + 1e-9)

            # 점수 (순매수 양수일수록 높은 점수)
            if total_net > 0:
                score = min(10.0, 5.0 + ratio * 50)
            elif total_net < 0:
                score = max(0.0, 5.0 + ratio * 50)
            else:
                score = 5.0

            detail = (
                f"기관={inst_net:+,.0f} 외국인={frgn_net:+,.0f} "
                f"합계={total_net:+,.0f}(비율={ratio:.3f})"
            )
            return Signal("institutional_buying", score, 0, detail)

        except Exception as e:
            return Signal("institutional_buying", 5, 0, f"오류: {e}")

    # 6. DART Signal ───────────────────────────────────────────────────────

    def dart_signal(self, ticker: str, db: DBManager, days: int = 90) -> Signal:
        """
        DART CB/BW 공시 신호.

        세력은 CB/BW 발행 직후 주가를 띄우는 경우가 많음.
        공시 있음 → 7점, 없음 → 3점 (중립)
        """
        try:
            dart_df = db.load_dart_recent(ticker, days)
            if dart_df.empty:
                return Signal("dart_signal", 3, 0, "공시 없음")

            # CB/BW 금액 합계
            total_amount = dart_df["amount_billion"].sum() if "amount_billion" in dart_df.columns else 0
            count = len(dart_df)

            # 최근 30일 이내 → 가산점
            recent_mask = dart_df["rcept_dt"] >= (
                datetime.now() - timedelta(days=30)
            ).strftime("%Y%m%d")
            recent_count = recent_mask.sum() if not dart_df.empty else 0

            score = min(10.0, 6.0 + recent_count * 1.5)
            detail = (
                f"CB/BW공시 {count}건, 최근30일 {recent_count}건, "
                f"규모 {total_amount:.0f}억원"
            )
            return Signal("dart_signal", score, 0, detail)

        except Exception as e:
            return Signal("dart_signal", 3, 0, f"오류: {e}")


# ── SignalEngine ──────────────────────────────────────────────────────────

class SignalEngine:
    """세력 매집 종합 시그널 엔진"""

    def __init__(self, config: dict, db: DBManager):
        self.config = config
        self.db = db
        self.sig_cfg = config.get("signals", {})
        self.data_cfg = config.get("data", {})

        self.weights: Dict[str, float] = self.sig_cfg.get("weights", {})
        self.min_score: float = self.sig_cfg.get("min_score", 50)
        self.top_n: int = self.sig_cfg.get("top_n", 20)

        self.rate_delay = self.data_cfg.get("rate_limit_delay", 0.3)
        self.calculator = SignalCalculator()

    # ── 단일 종목 분석 ────────────────────────────────────────────────────

    def analyze(
        self,
        ticker: str,
        name: str,
        date: str,
        sector: str = "기타",
        sector_weight: float = 1.0,
        market_cap: int = 0,
    ) -> Optional[StockScore]:
        """단일 종목 시그널 분석"""
        try:
            # 데이터 기간 설정
            to_dt = datetime.strptime(date, "%Y%m%d")
            from_dt = to_dt - timedelta(days=self.sig_cfg.get("lookback_days", 60) + 30)
            fromdate = from_dt.strftime("%Y%m%d")

            # OHLCV 로드 (DB 우선)
            df = self.db.load_ohlcv(ticker, fromdate, date)
            if df.empty or len(df) < 30:
                logger.debug(f"{ticker} {name}: OHLCV 부족 ({len(df)}행)")
                return None

            close = int(df["종가"].iloc[-1])

            cfg = self.sig_cfg

            # ─ 신호 계산
            signals = [
                self.calculator.obv_divergence(
                    df, window=cfg.get("obv_slope_window", 60)
                ),
                self.calculator.volume_asymmetry(
                    df,
                    window=cfg.get("volume_window", 20),
                    wick_threshold=cfg.get("upper_wick_threshold", 0.6),
                    up_vol_threshold=cfg.get("volume_up_threshold", 1.5),
                ),
                self.calculator.box_range(
                    df,
                    window=cfg.get("box_window", 60),
                    bb_window=cfg.get("bb_window", 20),
                    bb_std=cfg.get("bb_std", 2.0),
                    threshold=cfg.get("box_threshold", 0.12),
                ),
                self.calculator.ma_alignment(
                    df,
                    short=cfg.get("ma_short", 5),
                    mid=cfg.get("ma_mid", 20),
                    long=cfg.get("ma_long", 60),
                ),
                self.calculator.institutional_buying(
                    ticker, fromdate, date,
                    days=cfg.get("institutional_days", 5),
                ),
                self.calculator.dart_signal(ticker, self.db),
            ]

            # ─ 가중치 적용
            for sig in signals:
                sig.weight = self.weights.get(sig.name, 0)

            total_w = sum(s.weight for s in signals)
            if total_w == 0:
                return None

            # ─ 종합 점수 (0~10 → 0~100 환산 후 섹터 가중치 적용)
            raw_score = sum(s.weighted for s in signals) / total_w  # 0~10
            total_score = min(100.0, raw_score * 10 * sector_weight)

            return StockScore(
                ticker=ticker,
                name=name,
                total_score=round(total_score, 2),
                signals=signals,
                sector=sector,
                sector_weight=sector_weight,
                market_cap=market_cap,
                date=date,
                close=close,
            )

        except Exception as e:
            logger.error(f"[Signal] {ticker} {name} 분석 실패: {e}")
            return None

    # ── 유니버스 스캔 ─────────────────────────────────────────────────────

    def scan(self, universe: pd.DataFrame, date: str) -> List[StockScore]:
        """
        유니버스 전종목 스캔.

        Args:
            universe: UniverseFilter.filter() 결과
            date: 기준 날짜 YYYYMMDD

        Returns:
            점수 상위 종목 리스트 (min_score 이상, top_n 개)
        """
        if universe.empty:
            logger.error("[Scan] Empty universe")
            return []

        total = len(universe)
        logger.info(f"[Scan] Scanning {total} stocks for {date}")

        results: List[StockScore] = []

        for idx, row in universe.iterrows():
            ticker = str(row["ticker"])
            name   = str(row.get("name", ticker))
            sector = str(row.get("sector", "기타"))
            sw     = float(row.get("sector_weight", 1.0))
            mktcap = int(row.get("시가총액", 0))

            if idx % 50 == 0:
                logger.info(f"[Scan] {idx}/{total} ...")

            result = self.analyze(ticker, name, date, sector, sw, mktcap)
            if result and result.total_score >= self.min_score:
                results.append(result)

            time.sleep(self.rate_delay * 0.5)

        results.sort(key=lambda x: x.total_score, reverse=True)
        top = results[: self.top_n]

        logger.info(
            f"[Scan] Done: {len(results)}/{total} passed (min={self.min_score}), "
            f"returning top {len(top)}"
        )
        return top

    def to_dataframe(self, results: List[StockScore]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in results])


# ── Normalizer (0~10 정규화 유틸) ────────────────────────────────────────

class Normalizer:
    """지표값을 0~10점으로 정규화하는 유틸 클래스"""

    @staticmethod
    def minmax(value: float, lo: float, hi: float, invert: bool = False) -> float:
        """
        선형 정규화: [lo, hi] → [0, 10].
        invert=True면 값이 낮을수록 높은 점수.
        """
        if hi == lo:
            return 5.0
        score = (value - lo) / (hi - lo) * 10
        score = max(0.0, min(10.0, score))
        return 10.0 - score if invert else score

    @staticmethod
    def sigmoid(value: float, center: float = 0, scale: float = 1.0) -> float:
        """
        Sigmoid 정규화: 연속 값을 0~10 범위로 부드럽게 매핑.
        center: 5점에 해당하는 값
        scale: 기울기 조정
        """
        import math
        z = (value - center) / (scale + 1e-9)
        return 10 / (1 + math.exp(-z))

    @staticmethod
    def threshold_score(
        value: float,
        thresholds: List[Tuple[float, float]],
    ) -> float:
        """
        구간별 점수 매핑.
        thresholds: [(최소값, 점수), ...] 오름차순 정렬
        예: [(0, 0), (0.5, 3), (1.0, 5), (1.5, 7), (2.0, 10)]
        """
        score = 0.0
        for threshold, s in thresholds:
            if value >= threshold:
                score = s
            else:
                break
        return score
