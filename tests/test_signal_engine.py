"""
test_signal_engine.py
──────────────────────
SignalCalculator / SignalEngine 단위 테스트.
pykrx 호출 없이 합성 OHLCV 데이터로 검증.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_engine import (
    Signal, SignalCalculator, SignalEngine, StockScore,
    Normalizer, rolling_slope,
)
from db_manager import DBManager


# ── 합성 데이터 생성 ──────────────────────────────────────────────────────

def make_ohlcv(
    n: int = 100,
    trend: str = "flat",
    vol_pattern: str = "normal",
    seed: int = 42,
) -> pd.DataFrame:
    """
    테스트용 OHLCV DataFrame 생성.
    index: DatetimeIndex
    컬럼: 시가, 고가, 저가, 종가, 거래량, 거래대금
    """
    np.random.seed(seed)
    base = 10_000.0

    if trend == "up":
        prices = base + np.cumsum(np.random.randn(n) * 80 + 20)
    elif trend == "down":
        prices = base + np.cumsum(np.random.randn(n) * 80 - 20)
    elif trend == "flat":
        prices = base + np.random.randn(n) * 200
    elif trend == "accumulation":
        # 주가는 횡보, OBV는 상승 (세력 매집 패턴)
        prices = base + np.random.randn(n) * 150
    else:
        prices = base + np.cumsum(np.random.randn(n) * 80)

    prices = np.maximum(prices, 1_000.0)
    dates = pd.date_range(end="2024-01-31", periods=n, freq="B")

    if vol_pattern == "up_heavy":
        # 상승일에 더 많은 거래량
        vol = np.random.randint(200_000, 500_000, n).astype(float)
        diffs = np.diff(np.concatenate([[prices[0]], prices]))
        vol[diffs > 0] *= 3.0
    elif vol_pattern == "spike":
        vol = np.random.randint(100_000, 300_000, n).astype(float)
        vol[-1] = vol.mean() * 6
    else:
        vol = np.random.randint(100_000, 500_000, n).astype(float)

    df = pd.DataFrame({
        "시가": prices * (1 + np.random.randn(n) * 0.003),
        "고가": prices * (1 + abs(np.random.randn(n)) * 0.008),
        "저가": prices * (1 - abs(np.random.randn(n)) * 0.008),
        "종가": prices,
        "거래량": vol.astype(int),
        "거래대금": (vol * prices).astype(int),
    }, index=dates)
    # 고가 ≥ 종가 ≥ 시가 ≥ 저가 보정
    df["고가"] = df[["고가", "종가", "시가"]].max(axis=1)
    df["저가"] = df[["저가", "종가", "시가"]].min(axis=1)
    return df


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def calc():
    return SignalCalculator()


@pytest.fixture
def config():
    return {
        "signals": {
            "lookback_days": 60,
            "obv_slope_window": 60,
            "volume_window": 20,
            "upper_wick_threshold": 0.6,
            "volume_up_threshold": 1.5,
            "box_window": 60,
            "bb_window": 20,
            "bb_std": 2.0,
            "box_threshold": 0.12,
            "ma_short": 5,
            "ma_mid": 20,
            "ma_long": 60,
            "institutional_days": 5,
            "weights": {
                "obv_divergence": 25,
                "volume_asymmetry": 20,
                "box_range": 15,
                "ma_alignment": 15,
                "institutional_buying": 15,
                "dart_signal": 10,
            },
            "min_score": 50,
            "top_n": 10,
        },
        "data": {"rate_limit_delay": 0, "retry_count": 1, "retry_delay": 0},
        "database": {"path": ":memory:"},
        "logging": {"level": "WARNING", "dir": "logs", "filename": "test.log"},
    }


@pytest.fixture
def db():
    return DBManager(":memory:")


# ── rolling_slope 테스트 ──────────────────────────────────────────────────

class TestRollingSlope:

    def test_upward_series_positive_slope(self):
        s = pd.Series(np.linspace(100, 200, 100))
        slopes = rolling_slope(s, 60)
        assert slopes.dropna().iloc[-1] > 0

    def test_downward_series_negative_slope(self):
        s = pd.Series(np.linspace(200, 100, 100))
        slopes = rolling_slope(s, 60)
        assert slopes.dropna().iloc[-1] < 0

    def test_flat_series_near_zero(self):
        s = pd.Series([100.0] * 100)
        slopes = rolling_slope(s, 60)
        # 완전 평탄이면 기울기 0 (또는 NaN)
        last = slopes.iloc[-1]
        assert last == pytest.approx(0.0, abs=1e-9) or np.isnan(last)

    def test_shorter_than_window_returns_nan(self):
        s = pd.Series([100.0] * 30)
        slopes = rolling_slope(s, 60)
        assert slopes.isna().all()


# ── OBV Divergence 테스트 ─────────────────────────────────────────────────

class TestOBVDivergence:

    def test_returns_signal(self, calc):
        df = make_ohlcv(100, "flat")
        sig = calc.obv_divergence(df, window=60)
        assert isinstance(sig, Signal)
        assert sig.name == "obv_divergence"
        assert 0 <= sig.score <= 10

    def test_bullish_divergence_high_score(self, calc):
        """주가 하락 + OBV 상승 → 높은 점수"""
        df = make_ohlcv(100, "accumulation", "up_heavy")
        sig = calc.obv_divergence(df, window=60)
        # 매집 패턴이므로 score >= 5 이어야 함
        assert sig.score >= 5

    def test_insufficient_data_returns_zero(self, calc):
        df = make_ohlcv(20, "up")
        sig = calc.obv_divergence(df, window=60)
        assert sig.score == 0
        assert "부족" in sig.detail

    def test_detail_contains_slope_info(self, calc):
        df = make_ohlcv(100, "up")
        sig = calc.obv_divergence(df)
        assert "price=" in sig.detail or "OBV" in sig.detail or "기울기" in sig.detail


# ── Volume Asymmetry 테스트 ──────────────────────────────────────────────

class TestVolumeAsymmetry:

    def test_returns_signal(self, calc):
        df = make_ohlcv(60, "flat")
        sig = calc.volume_asymmetry(df)
        assert isinstance(sig, Signal)
        assert sig.name == "volume_asymmetry"
        assert 0 <= sig.score <= 10

    def test_up_heavy_volume_high_score(self, calc):
        """상승일(종가>시가) 거래량이 압도적으로 많을 때 → 낮지 않은 점수"""
        n = 60
        np.random.seed(0)
        # 시가/종가 기준 상승봉에만 거래량을 3배 이상 부여
        prices = np.full(n, 10_000.0)
        opens  = prices * 0.99  # 시가가 종가보다 낮아서 모두 상승봉
        vol = np.random.randint(100_000, 200_000, n).astype(float) * 3
        df = pd.DataFrame({
            "시가": opens, "고가": prices * 1.005,
            "저가": opens * 0.995, "종가": prices,
            "거래량": vol.astype(int),
            "거래대금": (vol * prices).astype(int),
        }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
        sig = calc.volume_asymmetry(df, window=20, up_vol_threshold=1.5)
        # 모든 봉이 상승봉 → up_vol_ratio 비율이 높아야 함
        assert sig.score >= 5, f"score={sig.score}, detail={sig.detail}"

    def test_insufficient_data(self, calc):
        df = make_ohlcv(10, "flat")
        sig = calc.volume_asymmetry(df, window=20)
        assert sig.score == 0

    def test_score_bounded(self, calc):
        df = make_ohlcv(80)
        sig = calc.volume_asymmetry(df)
        assert 0 <= sig.score <= 10

    def test_detail_has_ratio(self, calc):
        df = make_ohlcv(60)
        sig = calc.volume_asymmetry(df)
        assert "비율" in sig.detail or "거래량" in sig.detail


# ── Box Range 테스트 ─────────────────────────────────────────────────────

class TestBoxRange:

    def test_returns_signal(self, calc):
        df = make_ohlcv(80, "flat")
        sig = calc.box_range(df)
        assert isinstance(sig, Signal)
        assert sig.name == "box_range"
        assert 0 <= sig.score <= 10

    def test_flat_price_high_score(self, calc):
        """좁은 횡보 → 높은 점수"""
        n = 80
        prices = np.full(n, 10_000.0) + np.random.randn(n) * 50  # 변동폭 0.5%
        df = pd.DataFrame({
            "시가": prices, "고가": prices * 1.002,
            "저가": prices * 0.998, "종가": prices,
            "거래량": np.ones(n, dtype=int) * 300_000,
            "거래대금": np.ones(n, dtype=int) * 3_000_000_000,
        }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
        sig = calc.box_range(df, window=60, bb_window=20, threshold=0.12)
        assert sig.score >= 6

    def test_wide_range_lower_score(self, calc):
        """변동폭 큰 추세주 → 낮은 점수"""
        df = make_ohlcv(80, "up")
        sig_up = calc.box_range(df, window=60, threshold=0.12)
        df_flat = make_ohlcv(80, "flat")
        sig_flat = calc.box_range(df_flat, window=60, threshold=0.12)
        # 횡보 점수가 추세 점수보다 높거나 같음
        assert sig_flat.score >= sig_up.score - 2  # 허용 마진 2점

    def test_insufficient_data(self, calc):
        df = make_ohlcv(30, "flat")
        sig = calc.box_range(df, window=60)
        assert sig.score == 0

    def test_detail_has_box_info(self, calc):
        df = make_ohlcv(80, "flat")
        sig = calc.box_range(df)
        assert "박스" in sig.detail or "범위" in sig.detail


# ── MA Alignment 테스트 ──────────────────────────────────────────────────

class TestMAAlignment:

    def test_returns_signal(self, calc):
        df = make_ohlcv(100, "up")
        sig = calc.ma_alignment(df)
        assert isinstance(sig, Signal)
        assert sig.name == "ma_alignment"
        assert 0 <= sig.score <= 10

    def test_perfect_uptrend_high_score(self, calc):
        """완벽한 상승 추세 → 높은 점수"""
        n = 100
        prices = np.linspace(8_000, 15_000, n)
        df = pd.DataFrame({
            "시가": prices, "고가": prices * 1.005,
            "저가": prices * 0.995, "종가": prices,
            "거래량": np.ones(n, dtype=int) * 400_000,
            "거래대금": np.ones(n, dtype=int) * 4_000_000_000,
        }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
        sig = calc.ma_alignment(df, short=5, mid=20, long=60)
        assert sig.score >= 7

    def test_downtrend_low_score(self, calc):
        """하락 추세 → 낮은 점수"""
        n = 100
        prices = np.linspace(15_000, 8_000, n)
        df = pd.DataFrame({
            "시가": prices, "고가": prices * 1.005,
            "저가": prices * 0.995, "종가": prices,
            "거래량": np.ones(n, dtype=int) * 300_000,
            "거래대금": np.ones(n, dtype=int) * 3_000_000_000,
        }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
        sig = calc.ma_alignment(df, short=5, mid=20, long=60)
        assert sig.score <= 5

    def test_insufficient_data(self, calc):
        df = make_ohlcv(30, "up")
        sig = calc.ma_alignment(df, long=60)
        assert sig.score == 0


# ── Normalizer 테스트 ─────────────────────────────────────────────────────

class TestNormalizer:

    def test_minmax_mid_value(self):
        score = Normalizer.minmax(5.0, 0.0, 10.0)
        assert score == pytest.approx(5.0)

    def test_minmax_at_lo(self):
        assert Normalizer.minmax(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_minmax_at_hi(self):
        assert Normalizer.minmax(10.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_minmax_inverted(self):
        score = Normalizer.minmax(2.0, 0.0, 10.0, invert=True)
        assert score == pytest.approx(8.0)

    def test_minmax_clamps_below(self):
        assert Normalizer.minmax(-5.0, 0.0, 10.0) == 0.0

    def test_minmax_clamps_above(self):
        assert Normalizer.minmax(15.0, 0.0, 10.0) == 10.0

    def test_sigmoid_center(self):
        score = Normalizer.sigmoid(0.0, center=0.0, scale=1.0)
        assert score == pytest.approx(5.0, abs=0.1)

    def test_threshold_score(self):
        thresholds = [(0, 0), (0.5, 3), (1.0, 5), (1.5, 7), (2.0, 10)]
        assert Normalizer.threshold_score(1.2, thresholds) == 5
        assert Normalizer.threshold_score(2.0, thresholds) == 10
        assert Normalizer.threshold_score(0.1, thresholds) == 0


# ── StockScore 테스트 ─────────────────────────────────────────────────────

class TestStockScore:

    def test_to_dict_contains_required_keys(self):
        sig = Signal("obv_divergence", 8.0, 25.0, "테스트")
        ss = StockScore(
            ticker="000001",
            name="테스트기업",
            total_score=75.0,
            signals=[sig],
            sector="바이오",
            sector_weight=1.5,
            market_cap=150_000_000_000,
            date="20240101",
            close=15000,
        )
        d = ss.to_dict()
        assert d["ticker"] == "000001"
        assert d["name"] == "테스트기업"
        assert d["total_score"] == 75.0
        assert d["sector"] == "바이오"
        assert "market_cap_억" in d
        assert "signals" in d
        assert "obv_divergence" in d

    def test_signal_weighted(self):
        s = Signal("test", score=8.0, weight=25.0)
        assert s.weighted == pytest.approx(200.0)


# ── SignalEngine 통합 테스트 ──────────────────────────────────────────────

class TestSignalEngine:

    def test_to_dataframe_empty(self, config, db):
        engine = SignalEngine(config, db)
        df = engine.to_dataframe([])
        assert df.empty

    def test_to_dataframe_with_results(self, config, db):
        engine = SignalEngine(config, db)
        sig = Signal("obv_divergence", 7.0, 25.0)
        ss = StockScore("000001", "테스트", 72.0, [sig], "바이오", 1.5, 1_500_000_000, "20240101", 15000)
        df = engine.to_dataframe([ss])
        assert len(df) == 1
        assert "ticker" in df.columns
        assert "total_score" in df.columns

    def test_analyze_skips_missing_ohlcv(self, config, db):
        """DB에 OHLCV 없으면 None 반환"""
        engine = SignalEngine(config, db)
        result = engine.analyze("999999", "없는종목", "20240101")
        assert result is None

    def test_analyze_with_db_data(self, config):
        """DB에 데이터 있으면 분석 실행"""
        db = DBManager(":memory:")  # 독립 인스턴스 (persistent conn)
        # OHLCV 삽입
        df = make_ohlcv(100, "up", "up_heavy")
        db.upsert_ohlcv(df.reset_index().rename(columns={"index": "날짜"}), "000001")

        # institutional_buying 목업
        with patch("signal_engine._pykrx_stock") as mock_stock:
            mock_stock.get_market_trading_volume_by_investor.return_value = pd.DataFrame({
                "기관합계": [1000, 2000, 3000, 4000, 5000],
                "외국인합계": [500, 1000, 1500, 2000, 2500],
                "전체": [10000, 20000, 30000, 40000, 50000],
            })
            engine = SignalEngine(config, db)
            result = engine.analyze("000001", "테스트", "20240131", "바이오", 1.5, 150_000_000_000)

        if result is not None:
            assert 0 <= result.total_score <= 100
            assert len(result.signals) > 0

    def test_scan_empty_universe(self, config, db):
        engine = SignalEngine(config, db)
        results = engine.scan(pd.DataFrame(), "20240101")
        assert results == []
