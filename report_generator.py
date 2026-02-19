"""
report_generator.py
────────────────────
Jinja2 HTML 리포트 + matplotlib 차트 + Telegram 전송.

차트 구성 (종목별 60일):
  subplot 1: 주가 + MA5/20/60 + 박스권 음영
  subplot 2: 거래량 + 거래량 MA20
  subplot 3: OBV
"""
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from db_manager import DBManager
from signal_engine import StockScore

matplotlib.use("Agg")  # GUI 없이 파일 저장
import platform
_FONT_FAMILY = (
    ["Malgun Gothic", "DejaVu Sans"]   # Windows
    if platform.system() == "Windows"
    else ["AppleGothic", "Malgun Gothic", "DejaVu Sans"]
)
matplotlib.rcParams["font.family"] = _FONT_FAMILY
matplotlib.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


# ── 차트 생성 ─────────────────────────────────────────────────────────────

class ChartGenerator:
    """종목별 60일 분석 차트 생성"""

    def __init__(self, config: dict):
        self.rpt_cfg = config.get("report", {})
        self.dpi    = self.rpt_cfg.get("chart_dpi", 100)
        self.width  = self.rpt_cfg.get("chart_width", 14)
        self.height = self.rpt_cfg.get("chart_height", 10)
        self.out_dir = Path(self.rpt_cfg.get("output_dir", "reports"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _calc_obv(self, df: pd.DataFrame) -> pd.Series:
        close = df["종가"].astype(float)
        volume = df["거래량"].astype(float)
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (direction * volume).cumsum()

    def _detect_box(self, df: pd.DataFrame, window: int = 60):
        """박스권 최근 구간의 고점/저점 반환"""
        roll = df.tail(window)
        return roll["고가"].max(), roll["저가"].min()

    def generate(self, stock_score: StockScore, df: pd.DataFrame) -> Optional[Path]:
        """
        차트 생성 및 PNG 저장.

        Args:
            stock_score: StockScore 객체
            df: OHLCV DataFrame (index=datetime, 컬럼: 시가 고가 저가 종가 거래량)

        Returns:
            저장된 파일 경로
        """
        if df.empty or len(df) < 20:
            logger.warning(f"[Chart] {stock_score.ticker}: 데이터 부족")
            return None

        # 최근 60일만 사용
        plot_df = df.tail(60).copy()

        try:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(self.width, self.height),
                gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
                sharex=True,
            )
            fig.suptitle(
                f"[{stock_score.ticker}] {stock_score.name}  "
                f"총점={stock_score.total_score:.1f}  "
                f"섹터={stock_score.sector}  "
                f"시총={stock_score.market_cap/1e8:.0f}억",
                fontsize=13, fontweight="bold",
            )

            x = range(len(plot_df))
            x_labels = plot_df.index.strftime("%m/%d") if hasattr(plot_df.index, "strftime") else range(len(plot_df))

            close = plot_df["종가"].astype(float).values
            high  = plot_df["고가"].astype(float).values
            low   = plot_df["저가"].astype(float).values
            vol   = plot_df["거래량"].astype(float).values

            # ── subplot 1: 주가 + MA + 박스권 ──────────────────────────────
            ax1.plot(x, close, color="black", linewidth=1.2, label="종가", zorder=3)

            # 이동평균선
            for period, color in [(5, "blue"), (20, "orange"), (60, "red")]:
                if len(close) >= period:
                    ma = pd.Series(close).rolling(period).mean().values
                    ax1.plot(x, ma, linewidth=0.8, color=color,
                             label=f"MA{period}", alpha=0.8)

            # 박스권 음영
            box_high, box_low = self._detect_box(plot_df)
            ax1.axhspan(box_low, box_high, alpha=0.08, color="green",
                        label=f"박스권({box_low:,.0f}~{box_high:,.0f})")
            ax1.axhline(box_high, color="green", linewidth=0.6, linestyle="--", alpha=0.5)
            ax1.axhline(box_low,  color="green", linewidth=0.6, linestyle="--", alpha=0.5)

            ax1.set_ylabel("주가 (원)")
            ax1.legend(loc="upper left", fontsize=8, ncol=3)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda val, _: f"{val:,.0f}")
            )

            # ── subplot 2: 거래량 ───────────────────────────────────────────
            # 상승/하락일 색상 구분
            colors = [
                "red" if close[i] >= (plot_df["시가"].astype(float).values[i])
                else "blue"
                for i in range(len(close))
            ]
            ax2.bar(x, vol, color=colors, alpha=0.6, width=0.8, label="거래량")

            # 거래량 MA20
            vol_ma = pd.Series(vol).rolling(20).mean().values
            ax2.plot(x, vol_ma, color="black", linewidth=1.0, label="거래량MA20")

            ax2.set_ylabel("거래량")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda val, _: f"{val/1e4:.0f}만")
            )

            # ── subplot 3: OBV ─────────────────────────────────────────────
            obv = self._calc_obv(plot_df).values
            ax3.plot(x, obv, color="purple", linewidth=1.0, label="OBV")
            obv_ma = pd.Series(obv).rolling(20).mean().values
            ax3.plot(x, obv_ma, color="gray", linewidth=0.8, linestyle="--",
                     label="OBV MA20")
            ax3.axhline(0, color="black", linewidth=0.4, alpha=0.5)
            ax3.set_ylabel("OBV")
            ax3.legend(loc="upper left", fontsize=8)
            ax3.grid(True, alpha=0.3)

            # X축 레이블 (10개만 표시)
            step = max(1, len(x) // 10)
            ax3.set_xticks(list(x)[::step])
            ax3.set_xticklabels(list(x_labels)[::step], rotation=45, fontsize=8)

            # 신호 점수 텍스트 박스
            sig_text = "\n".join(
                f"{s.name}: {s.score:.1f}  {s.detail[:30]}"
                for s in stock_score.signals if s.weight > 0
            )
            ax1.text(
                0.01, 0.02, sig_text,
                transform=ax1.transAxes,
                fontsize=7, verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()

            save_path = self.out_dir / f"chart_{stock_score.ticker}_{stock_score.date}.png"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"[Chart] Saved: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"[Chart] {stock_score.ticker} failed: {e}")
            plt.close("all")
            return None

    def generate_all(
        self, scores: List[StockScore], db: DBManager, max_n: int = 5
    ) -> List[Path]:
        """상위 N개 종목 차트 일괄 생성"""
        paths = []
        for score in scores[:max_n]:
            df = db.load_ohlcv(score.ticker, "20200101", score.date)
            path = self.generate(score, df)
            if path:
                paths.append(path)
        return paths


# ── HTML 리포트 ──────────────────────────────────────────────────────────

class HTMLReportGenerator:
    """Jinja2 기반 HTML 리포트 생성"""

    def __init__(self, config: dict):
        self.rpt_cfg = config.get("report", {})
        self.out_dir  = Path(self.rpt_cfg.get("output_dir", "reports"))
        tmpl_dir = Path(self.rpt_cfg.get("template_dir", "templates"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Jinja2 환경 설정
        if tmpl_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(tmpl_dir)),
                autoescape=select_autoescape(["html"]),
            )
        else:
            # 템플릿 디렉토리가 없으면 문자열 템플릿 사용
            from jinja2 import DictLoader
            self.env = Environment(
                loader=DictLoader({"report.html": DEFAULT_TEMPLATE}),
                autoescape=select_autoescape(["html"]),
            )

    def generate(
        self, scores: List[StockScore], scan_date: str, chart_paths: List[Path] = None
    ) -> Path:
        """HTML 리포트 생성"""
        rows = [s.to_dict() for s in scores]
        for row in rows:
            row["signals_fmt"] = {
                k: v for k, v in row.get("signals", {}).items()
            }

        chart_files = {}
        if chart_paths:
            for p in chart_paths:
                # ticker 추출: chart_000001_20240101.png
                parts = p.stem.split("_")
                if len(parts) >= 2:
                    chart_files[parts[1]] = p.name

        try:
            tmpl = self.env.get_template("report.html")
        except Exception:
            from jinja2 import Template
            tmpl = Template(DEFAULT_TEMPLATE)

        html = tmpl.render(
            scan_date=scan_date,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_count=len(scores),
            rows=rows,
            chart_files=chart_files,
        )

        out_path = self.out_dir / f"report_{scan_date}.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info(f"[HTML] Report saved: {out_path}")
        return out_path


class BacktestReportGenerator:
    """백테스트 결과 HTML 리포트 생성"""

    def __init__(self, config: dict):
        self.rpt_cfg = config.get("report", {})
        self.out_dir = Path(self.rpt_cfg.get("output_dir", "reports"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, result_df: pd.DataFrame, summary: dict) -> Path:
        """백테스트 결과 리포트 생성"""
        if result_df.empty:
            return None

        # 템플릿 데이터 준비
        scan_date = datetime.now().strftime("%Y-%m-%d")
        
        # 통계 데이터 정리 (10, 20, 30일)
        stats_list = []
        # summary keys are '10d', '20d', '30d' etc.
        # We need to ensure we handle the keys present in summary
        for period, data in summary.items():
            stats_list.append({
                "period": period,
                "count": data.get("count", 0),
                "win_rate": data.get("win_rate", 0),
                "avg_return": data.get("avg_return", 0),
                "median_return": data.get("median_return", 0),
                "best": data.get("best", 0),
                "worst": data.get("worst", 0)
            })

        # 상세 데이터 (Top 200 by Score)
        # Ensure we have the columns we want to display
        display_cols = ["signal_date", "ticker", "name", "score", "return_10d", "return_20d", "return_30d"]
        available_cols = [c for c in display_cols if c in result_df.columns]
        
        top_results = result_df.sort_values(by="score", ascending=False).head(200)[available_cols].to_dict("records")
        
        # ECharts용 데이터 준비
        periods = [s['period'] for s in stats_list]
        win_rates = [s['win_rate'] for s in stats_list]
        avg_returns = [s['avg_return'] for s in stats_list]

        html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>백테스트 결과 리포트 ({{ scan_date }})</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
  body { font-family: 'Malgun Gothic', sans-serif; margin: 20px; background: #f0f2f5; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
  .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
  
  table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
  th { background: #f8f9fa; padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; }
  td { padding: 10px; border-bottom: 1px solid #eee; }
  tr:hover { background-color: #f8f9fa; }
  
  .win-rate { font-weight: bold; color: #e74c3c; }
  .profit { color: #e74c3c; font-weight: bold; }
  .loss { color: #2980b9; font-weight: bold; }
  
  .chart-container { height: 400px; width: 100%; }
</style>
</head>
<body>
<div class="container">
  <h1>📊 백테스트 분석 리포트</h1>
  
  <div class="card">
    <h3>📈 기간별 성과 비교</h3>
    <div id="main_chart" class="chart-container"></div>
  </div>

  <div class="stats-grid">
    {% for stat in stats %}
    <div class="card">
      <h3>📅 {{ stat.period }} 보유 성과</h3>
      <table>
        <tr><td>샘플 수</td><td><b>{{ stat.count }}</b>건</td></tr>
        <tr><td>승률</td><td class="win-rate">{{ stat.win_rate }}%</td></tr>
        <tr><td>평균수익률</td><td class="{{ 'profit' if stat.avg_return > 0 else 'loss' }}">{{ stat.avg_return }}%</td></tr>
        <tr><td>중앙값</td><td>{{ stat.median_return }}%</td></tr>
        <tr><td>최고/최저</td><td>{{ stat.best }}% / {{ stat.worst }}%</td></tr>
      </table>
    </div>
    {% endfor %}
  </div>

  <div class="card">
    <h3>📋 상위 시그널 내역 (Top 200)</h3>
    <table>
      <thead>
        <tr>
            <th>날짜</th><th>종목</th><th>종목명</th><th>점수</th>
            <th>10일후</th><th>20일후</th><th>30일후</th>
        </tr>
      </thead>
      <tbody>
        {% for row in rows %}
        <tr>
            <td>{{ row.get('signal_date', '-') }}</td>
            <td><b>{{ row.get('ticker', '-') }}</b></td>
            <td>{{ row.get('name', '-') }}</td>
            <td>{{ row.get('score', 0) }}</td>
            <td class="{{ 'profit' if row.get('return_10d', 0) > 0 else 'loss' }}">{{ row.get('return_10d', '-') }}%</td>
            <td class="{{ 'profit' if row.get('return_20d', 0) > 0 else 'loss' }}">{{ row.get('return_20d', '-') }}%</td>
            <td class="{{ 'profit' if row.get('return_30d', 0) > 0 else 'loss' }}">{{ row.get('return_30d', '-') }}%</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<script>
  var chartDom = document.getElementById('main_chart');
  var myChart = echarts.init(chartDom);
  var option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' }
    },
    legend: { data: ['승률(%)', '평균수익률(%)'] },
    xAxis: [
      {
        type: 'category',
        data: {{ periods }},
        axisPointer: { type: 'shadow' }
      }
    ],
    yAxis: [
      {
        type: 'value',
        name: '승률',
        min: 0,
        max: 100,
        axisLabel: { formatter: '{value} %' }
      },
      {
        type: 'value',
        name: '수익률',
        axisLabel: { formatter: '{value} %' }
      }
    ],
    series: [
      {
        name: '승률(%)',
        type: 'bar',
        data: {{ win_rates }}
      },
      {
        name: '평균수익률(%)',
        type: 'line',
        yAxisIndex: 1,
        data: {{ avg_returns }}
      }
    ]
  };
  myChart.setOption(option);
</script>
</body>
</html>"""
        
        from jinja2 import Template
        template = Template(html)
        rendered_html = template.render(
            stats=stats_list, 
            rows=top_results,
            scan_date=scan_date,
            periods=periods,
            win_rates=win_rates,
            avg_returns=avg_returns
        )

        out_path = self.out_dir / f"backtest_report_v2_{scan_date}.html"
        out_path.write_text(rendered_html, encoding="utf-8")
        logger.info(f"[Backtest] Report saved: {out_path}")
        return out_path


# ── Telegram 전송 ─────────────────────────────────────────────────────────

class TelegramNotifier:
    """Telegram 봇 알림 전송"""

    def __init__(self, config: dict):
        tg = config.get("telegram", {})
        self.enabled      = tg.get("enabled", False)
        self.token        = tg.get("token", "")
        self.chat_id      = tg.get("chat_id", "")
        self.send_charts  = tg.get("send_charts", True)
        self.max_charts   = tg.get("max_chart_stocks", 5)

    def _make_message(self, scores: List[StockScore], scan_date: str) -> str:
        """텔레그램 메시지 포맷"""
        if not scores:
            return f"📊 [{scan_date}] 코스닥 세력 매집 스캔\n조건 충족 종목 없음"

        lines = [
            f"📊 <b>코스닥 세력 매집 감지</b> [{scan_date}]",
            f"━━━━━━━━━━━━━━━━",
            f"총 <b>{len(scores)}</b>개 종목 감지",
            "",
        ]
        medals = ["🥇", "🥈", "🥉"]
        for i, s in enumerate(scores[:10]):
            m = medals[i] if i < 3 else f"{i+1}."
            lines.append(
                f"{m} <b>[{s.ticker}] {s.name}</b>"
                f" | 점수: <b>{s.total_score:.1f}</b>"
            )
            lines.append(
                f"   {s.sector} | 시총 {s.market_cap/1e8:.0f}억 | "
                f"현재가 {s.close:,}원"
            )
            # 상위 3개 신호
            top3 = sorted(s.signals, key=lambda x: x.weighted, reverse=True)[:3]
            sig_str = " | ".join(
                f"{x.name[:6]}: {x.score:.1f}" for x in top3 if x.weight > 0
            )
            lines.append(f"   <i>{sig_str}</i>")
            lines.append("")

        if len(scores) > 10:
            lines.append(f"... 외 {len(scores)-10}개")

        lines += [
            "━━━━━━━━━━━━━━━━",
            "⚠️ <i>투자 참고용. 직접 분석 필수.</i>",
        ]
        return "\n".join(lines)

    def send_message(self, text: str) -> bool:
        if not self.enabled or not self.token or not self.chat_id:
            logger.info("[Telegram] Disabled or not configured")
            return False
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            r = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=30,
            )
            r.raise_for_status()
            logger.info("[Telegram] Message sent")
            return True
        except Exception as e:
            logger.error(f"[Telegram] sendMessage failed: {e}")
            return False

    def send_photo(self, photo_path: Path, caption: str = "") -> bool:
        if not self.enabled or not self.token or not self.chat_id:
            return False
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            with open(photo_path, "rb") as f:
                r = requests.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption[:200]},
                    files={"photo": f},
                    timeout=60,
                )
            r.raise_for_status()
            logger.info(f"[Telegram] Photo sent: {photo_path.name}")
            return True
        except Exception as e:
            logger.error(f"[Telegram] sendPhoto failed: {e}")
            return False

    def notify(
        self, scores: List[StockScore], scan_date: str, chart_paths: List[Path] = None
    ) -> None:
        """요약 메시지 + 차트 전송"""
        if not self.enabled:
            return

        # 1. 텍스트 메시지
        text = self._make_message(scores, scan_date)
        self.send_message(text)

        # 2. 차트 이미지 전송
        if self.send_charts and chart_paths:
            for i, path in enumerate(chart_paths[: self.max_charts]):
                ticker = scores[i].ticker if i < len(scores) else ""
                name   = scores[i].name   if i < len(scores) else ""
                caption = f"[{ticker}] {name} – 60일 분석차트"
                self.send_photo(path, caption)


# ── DEFAULT HTML 템플릿 ───────────────────────────────────────────────────

DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>코스닥 세력 매집 리포트 {{ scan_date }}</title>
<style>
  body { font-family: 'Malgun Gothic', sans-serif; margin: 20px; background: #f8f9fa; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }
  th { background: #2c3e50; color: white; padding: 12px 10px; text-align: left; font-size: 13px; }
  td { padding: 10px; border-bottom: 1px solid #eee; font-size: 13px; }
  tr:hover { background: #f0f4f8; }
  .score { font-weight: bold; color: #e74c3c; font-size: 15px; }
  .rank { color: #888; }
  .sector { display: inline-block; padding: 2px 8px; border-radius: 12px;
            background: #eaf4fe; color: #2980b9; font-size: 11px; }
  .signal-bar { height: 6px; background: #3498db; border-radius: 3px; display: inline-block; }
  .chart-img { max-width: 100%; border-radius: 6px; margin-top: 10px; }
  .card { background: white; border-radius: 8px; padding: 15px; margin: 15px 0;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
</head>
<body>
<h1>📊 코스닥 세력 매집 감지 리포트</h1>
<div class="meta">
  스캔일: <b>{{ scan_date }}</b> &nbsp;|&nbsp;
  생성: {{ generated_at }} &nbsp;|&nbsp;
  감지 종목: <b>{{ total_count }}</b>개
</div>
<table>
  <thead>
    <tr>
      <th>#</th><th>종목코드</th><th>종목명</th><th>총점</th><th>섹터</th>
      <th>시총(억)</th><th>OBV</th><th>거래량비대칭</th><th>박스권</th>
      <th>MA정배열</th><th>기관매수</th><th>DART</th>
    </tr>
  </thead>
  <tbody>
  {% for row in rows %}
  <tr>
    <td class="rank">{{ loop.index }}</td>
    <td><b>{{ row.ticker }}</b></td>
    <td>{{ row.name }}</td>
    <td class="score">{{ row.total_score }}</td>
    <td><span class="sector">{{ row.sector }}</span></td>
    <td>{{ row['market_cap_억'] }}</td>
    <td>{{ row.signals.get('obv_divergence', '-') }}</td>
    <td>{{ row.signals.get('volume_asymmetry', '-') }}</td>
    <td>{{ row.signals.get('box_range', '-') }}</td>
    <td>{{ row.signals.get('ma_alignment', '-') }}</td>
    <td>{{ row.signals.get('institutional_buying', '-') }}</td>
    <td>{{ row.signals.get('dart_signal', '-') }}</td>
  </tr>
  {% endfor %}
  </tbody>
</table>

{% if chart_files %}
<h2 style="margin-top:30px;">📈 상위 종목 차트</h2>
{% for row in rows %}
  {% if row.ticker in chart_files %}
  <div class="card">
    <h3>[{{ row.ticker }}] {{ row.name }}  –  {{ row.total_score }}점</h3>
    <img class="chart-img" src="{{ chart_files[row.ticker] }}" alt="{{ row.name }} 차트">
  </div>
  {% endif %}
{% endfor %}
{% endif %}

<p style="color:#aaa; font-size:12px; margin-top:30px;">
⚠ 본 리포트는 투자 참고용이며, 실제 투자 손익에 대한 책임은 본인에게 있습니다.
</p>
</body>
</html>
"""
