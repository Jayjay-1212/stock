
try:
    from pykrx import stock
    import pandas as pd
    from datetime import datetime

    date = datetime.now().strftime("%Y%m%d")
    df = stock.get_market_cap_by_ticker(date, market="KOSDAQ")
    print("Columns:", df.columns)
    if "종목명" in df.columns:
        print("Name found in columns!")
    else:
        print("Name NOT found in columns.")
except Exception as e:
    print(e)

try:
    from tqdm import tqdm
    print("tqdm is installed")
except ImportError:
    print("tqdm is NOT installed")
