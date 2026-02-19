"""
test_universe_filter.py
────────────────────────
UniverseFilter 단위 테스트.
pykrx 실제 호출 없이 목 데이터로 검증.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from db_manager import DBManager
from universe_filter import UniverseFilter, SECTOR_KEYWORDS


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "universe": {
            "market_cap_min": 100_000_000_000,  # 1000억
            "market_cap_max": 300_000_000_000,  # 3000억
            "float_ratio_min": 0.40,
            "float_ratio_max": 0.60,
            "sector_weights": {
                "바이오": 1.5, "로봇": 1.5, "AI": 1.5, "2차전지": 1.5,
                "반도체": 1.3, "방위": 1.3, "기타": 1.0,
            },
        },
        "data": {"cache_dir": "cache_test", "cache_ttl_hours": 0},
        "database": {"path": ":memory:"},
        "logging": {"level": "WARNING", "dir": "logs_test", "filename": "test.log"},
    }


@pytest.fixture
def db(config):
    return DBManager(":memory:")


@pytest.fixture
def uf(config, db):
    return UniverseFilter(config, db)


@pytest.fixture
def sample_mktcap():
    """테스트용 시총 데이터"""
    return pd.DataFrame({
        "ticker": ["000001", "000002", "000003", "000004", "000005", "000006"],
        "name":   ["한국바이오", "로봇테크", "AI소프트", "2차전지제조", "일반제조", "대형기업"],
        "시가총액": [
            150_000_000_000,  # 1500억 ✓
            200_000_000_000,  # 2000억 ✓
            80_000_000_000,   # 800억  ✗ (하한 미만)
            250_000_000_000,  # 2500억 ✓
            120_000_000_000,  # 1200억 ✓
            500_000_000_000,  # 5000억 ✗ (상한 초과)
        ],
        "상장주식수": [10_000_000] * 6,
        "종가": [15000, 10000, 4000, 6250, 6000, 5000],
    })


# ── 시총 필터 테스트 ─────────────────────────────────────────────────────

class TestMarketCapFilter:

    def test_passes_within_range(self, uf, sample_mktcap):
        result = uf._filter_market_cap(sample_mktcap)
        assert len(result) == 4
        assert set(result["ticker"]) == {"000001", "000002", "000004", "000005"}

    def test_excludes_below_min(self, uf, sample_mktcap):
        result = uf._filter_market_cap(sample_mktcap)
        assert "000003" not in result["ticker"].values

    def test_excludes_above_max(self, uf, sample_mktcap):
        result = uf._filter_market_cap(sample_mktcap)
        assert "000006" not in result["ticker"].values

    def test_empty_input(self, uf):
        assert uf._filter_market_cap(pd.DataFrame()).empty

    def test_all_filtered_out(self, uf):
        df = pd.DataFrame({"ticker": ["X"], "시가총액": [10_000_000_000]})
        assert uf._filter_market_cap(df).empty

    def test_exact_boundary_min(self, uf):
        df = pd.DataFrame({
            "ticker": ["A"],
            "시가총액": [100_000_000_000],  # 정확히 최솟값
            "상장주식수": [1_000_000],
            "종가": [100_000],
        })
        result = uf._filter_market_cap(df)
        assert len(result) == 1

    def test_exact_boundary_max(self, uf):
        df = pd.DataFrame({
            "ticker": ["A"],
            "시가총액": [300_000_000_000],  # 정확히 최댓값
            "상장주식수": [1_000_000],
            "종가": [300_000],
        })
        result = uf._filter_market_cap(df)
        assert len(result) == 1


# ── 섹터 분류 테스트 ─────────────────────────────────────────────────────

class TestSectorClassification:

    @pytest.mark.parametrize("name,expected_sector", [
        ("한국바이오",    "바이오"),
        ("셀트리온제약",  "바이오"),
        ("현대로봇",      "로봇"),
        ("AI솔루션",      "AI"),
        ("인공지능테크",  "AI"),
        ("2차전지제조",   "2차전지"),
        ("리튬배터리",    "2차전지"),
        ("삼성반도체",    "반도체"),
        ("일반제조업",    "기타"),
    ])
    def test_classify_sector(self, uf, name, expected_sector):
        sector, _ = uf._classify_one(name)
        assert sector == expected_sector, f"'{name}' → expected '{expected_sector}', got '{sector}'"

    @pytest.mark.parametrize("name,expected_weight", [
        ("바이오테크",  1.5),
        ("로봇산업",    1.5),
        ("AI클라우드",  1.5),
        ("전기차배터리", 1.5),
        ("일반기업",    1.0),
    ])
    def test_sector_weight(self, uf, name, expected_weight):
        _, weight = uf._classify_one(name)
        assert weight == expected_weight

    def test_classify_adds_columns(self, uf, sample_mktcap):
        filtered = uf._filter_market_cap(sample_mktcap)
        result = uf._classify_sector(filtered)
        assert "sector" in result.columns
        assert "sector_weight" in result.columns

    def test_bio_gets_high_weight(self, uf, sample_mktcap):
        filtered = uf._filter_market_cap(sample_mktcap)
        classified = uf._classify_sector(filtered)
        bio_row = classified[classified["name"] == "한국바이오"]
        assert not bio_row.empty
        assert bio_row["sector"].iloc[0] == "바이오"
        assert bio_row["sector_weight"].iloc[0] == 1.5


# ── 정렬 테스트 ──────────────────────────────────────────────────────────

class TestSorting:

    def test_sorted_by_sector_weight_desc(self, uf, sample_mktcap):
        filtered = uf._filter_market_cap(sample_mktcap)
        classified = uf._classify_sector(filtered)
        sorted_df = classified.sort_values(
            ["sector_weight", "시가총액"], ascending=[False, False]
        ).reset_index(drop=True)
        # 첫 번째 행의 가중치 ≥ 마지막 행
        assert sorted_df["sector_weight"].iloc[0] >= sorted_df["sector_weight"].iloc[-1]

    def test_adds_억_column(self, uf, sample_mktcap):
        result = uf.filter("20240101", sample_mktcap)
        assert "시가총액_억" in result.columns
        # 1500억이면 1500
        bio_row = result[result["name"] == "한국바이오"]
        if not bio_row.empty:
            assert bio_row["시가총액_억"].iloc[0] == 1500


# ── 전체 파이프라인 테스트 ────────────────────────────────────────────────

class TestFilterPipeline:

    def test_full_pipeline_returns_dataframe(self, uf, sample_mktcap):
        result = uf.filter("20240101", sample_mktcap)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_required_columns_present(self, uf, sample_mktcap):
        result = uf.filter("20240101", sample_mktcap)
        required = ["ticker", "name", "시가총액", "sector", "sector_weight", "시가총액_억"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_empty_mktcap_returns_empty(self, uf):
        result = uf.filter("20240101", pd.DataFrame())
        assert result.empty

    def test_count_after_filter(self, uf, sample_mktcap):
        result = uf.filter("20240101", sample_mktcap)
        # 800억(3번) 과 5000억(6번) 제외 → 4개
        assert len(result) == 4
