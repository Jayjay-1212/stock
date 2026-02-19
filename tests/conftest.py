"""
conftest.py – pytest 공통 설정 및 pykrx mock
"""
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ── pykrx가 없는 테스트 환경에서 mock 주입 ───────────────────────────────
if "pykrx" not in sys.modules:
    _mock_stock = MagicMock()
    _mock_stock.get_market_ticker_list.return_value = []
    _mock_stock.get_market_ticker_name.return_value = "테스트종목"
    _mock_stock.get_market_cap_by_ticker.return_value = pd.DataFrame()
    _mock_stock.get_market_ohlcv.return_value = pd.DataFrame()
    _mock_stock.get_market_trading_volume_by_investor.return_value = pd.DataFrame()

    _pykrx_mock = MagicMock()
    _pykrx_mock.stock = _mock_stock
    sys.modules["pykrx"] = _pykrx_mock
    sys.modules["pykrx.stock"] = _mock_stock

# ── OpenDartReader mock ───────────────────────────────────────────────────
if "OpenDartReader" not in sys.modules:
    sys.modules["OpenDartReader"] = MagicMock()
