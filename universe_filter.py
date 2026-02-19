"""
universe_filter.py
──────────────────
코스닥 투자 유니버스 필터링 모듈.

- 시총 1,000억~3,000억 필터
- 유동주식비율 40~60% 필터 (pykrx 미지원 → DB 기반 근사)
- 섹터 분류 및 가중치 부여 (바이오/로봇/AI/2차전지 우대)
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from db_manager import DBManager

logger = logging.getLogger(__name__)

# ── 섹터 키워드 ────────────────────────────────────────────────────────────

SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "바이오":   ["바이오", "제약", "의약", "헬스케어", "메디", "셀", "진", "유전체"],
    "로봇":     ["로봇", "자동화", "드론", "무인", "협동로봇"],
    "AI":       ["인공지능", "AI", "딥러닝", "머신러닝", "데이터", "클라우드", "소프트웨어"],
    "2차전지":  ["2차전지", "배터리", "전지", "전기차", "EV", "충전", "양극재", "음극재", "리튬"],
    "반도체":   ["반도체", "웨이퍼", "칩", "파운드리", "팹"],
    "방위":     ["방산", "방위", "무기", "항공우주"],
}


class UniverseFilter:
    """코스닥 투자 유니버스 필터"""

    def __init__(self, config: dict, db: DBManager):
        self.config = config
        self.db = db

        uf = config.get("universe", {})
        self.mktcap_min = uf.get("market_cap_min", 100_000_000_000)
        self.mktcap_max = uf.get("market_cap_max", 300_000_000_000)
        self.float_min = uf.get("float_ratio_min", 0.40)
        self.float_max = uf.get("float_ratio_max", 0.60)
        self.sector_weights: Dict[str, float] = uf.get("sector_weights", {})

    # ── 공개 API ───────────────────────────────────────────────────────────

    def filter(self, date: str, mktcap_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        유니버스 필터 파이프라인 실행.

        Args:
            date: 기준 날짜 YYYYMMDD
            mktcap_df: DataCollector.collect_market_cap() 결과.
                       None 이면 DB에서 로드.

        Returns:
            필터된 종목 DataFrame
            columns: ticker, name, 시가총액, 상장주식수, 종가,
                     sector, sector_weight, 시가총액_억
        """
        logger.info(f"=== Universe Filter: {date} ===")

        # 1. 시총 데이터 확보
        if mktcap_df is None or mktcap_df.empty:
            mktcap_df = self._load_mktcap_from_db(date)

        if mktcap_df is None or mktcap_df.empty:
            logger.error("No market cap data available")
            return pd.DataFrame()

        df = mktcap_df.copy()
        self._normalise_columns(df)

        # 2. 시총 필터
        df = self._filter_market_cap(df)
        if df.empty:
            return df

        # 3. 유동주식비율 필터 (근사 계산)
        df = self._filter_float_ratio(df)
        if df.empty:
            return df

        # 4. 섹터 분류
        df = self._classify_sector(df)

        # 5. 정렬 (섹터 가중치 ↓, 시총 ↓)
        df = df.sort_values(
            ["sector_weight", "시가총액"], ascending=[False, False]
        ).reset_index(drop=True)

        df["시가총액_억"] = (df["시가총액"] / 1e8).round(0).astype(int)

        logger.info(
            f"Universe filter complete: {len(df)} stocks "
            f"({df['sector'].value_counts().to_dict()})"
        )
        return df

    # ── 내부 메서드 ────────────────────────────────────────────────────────

    def _load_mktcap_from_db(self, date: str) -> Optional[pd.DataFrame]:
        """DB에서 시총 데이터 로드"""
        try:
            df = self.db.load_market_cap(date)
            if df.empty:
                logger.warning(f"No market cap in DB for {date}")
                return None
            return df
        except Exception as e:
            logger.error(f"DB market cap load failed: {e}")
            return None

    def _normalise_columns(self, df: pd.DataFrame) -> None:
        """컬럼명 표준화 (pykrx / DB 혼용)"""
        rename = {
            "market_cap": "시가총액",
            "shares_listed": "상장주식수",
            "close": "종가",
        }
        for old, new in rename.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)

        for col in ["시가총액", "상장주식수", "종가"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    def _filter_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        if "시가총액" not in df.columns:
            logger.error("시가총액 column missing")
            return pd.DataFrame()
        before = len(df)
        filtered = df[
            (df["시가총액"] >= self.mktcap_min) &
            (df["시가총액"] <= self.mktcap_max)
        ].copy()
        logger.info(
            f"MarketCap filter ({self.mktcap_min/1e8:.0f}억~{self.mktcap_max/1e8:.0f}억): "
            f"{before} → {len(filtered)}"
        )
        return filtered

    def _filter_float_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        유동주식비율 필터.
        pykrx는 직접 제공하지 않으므로 상장주식수 기반 근사값 사용.
        실제 운영 시: DART 또는 KRX OpenAPI의 유동주식수 데이터 활용 권장.
        """
        if "상장주식수" not in df.columns or "종가" not in df.columns:
            logger.warning("Float ratio filter skipped (no shares/price data)")
            return df

        # 근사: float_ratio = 1.0 (pykrx 미지원 → 필터 패스)
        # 실 데이터가 있으면 아래 주석 해제:
        # df["float_ratio"] = df["float_shares"] / df["상장주식수"]
        # df = df[(df["float_ratio"] >= self.float_min) & (df["float_ratio"] <= self.float_max)]

        logger.info(
            f"Float ratio filter: SKIPPED (pykrx doesn't provide float shares directly). "
            f"Pass-through: {len(df)} stocks"
        )
        return df

    def _classify_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """섹터 분류 및 가중치 부여"""
        if "name" not in df.columns:
            df["sector"] = "기타"
            df["sector_weight"] = self.sector_weights.get("기타", 1.0)
            return df

        sectors, weights = [], []
        for name in df["name"]:
            s, w = self._classify_one(str(name))
            sectors.append(s)
            weights.append(w)

        df = df.copy()
        df["sector"] = sectors
        df["sector_weight"] = weights
        return df

    def _classify_one(self, name: str) -> Tuple[str, float]:
        name_up = name.upper()
        for sector, keywords in SECTOR_KEYWORDS.items():
            for kw in keywords:
                if kw.upper() in name_up:
                    w = self.sector_weights.get(sector, 1.0)
                    return sector, w
        return "기타", self.sector_weights.get("기타", 1.0)
