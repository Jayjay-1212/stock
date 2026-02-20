# 코스닥 세력 매집 감지 시스템

코스닥 중소형주(시총 1,000억~3,000억)에서 세력 매집 패턴을 자동 감지하는 시스템입니다.
매일 장 마감 후 자동 스캔하여 Telegram 알림과 HTML 리포트를 생성합니다.

---

## 주요 기능

- **6개 신호 엔진**: OBV Divergence, 거래량 비대칭, 박스권, MA 정배열, 기관매수, DART CB/BW
- **자동 스캔**: APScheduler로 매일 18:00 자동 실행
- **유니버스 필터**: 시총 1,000억~3,000억 + 섹터 가중치 (AI/바이오/2차전지 우대)
- **리포트**: matplotlib 차트 + Jinja2 HTML + Telegram 알림
- **백테스터**: 과거 시그널 정확도 검증 (점수 구간별 승률 분석)

---

## 신호 엔진

| 신호 | 가중치 | 설명 |
|------|:------:|------|
| OBV Divergence | 25 | 주가 하락 + OBV 상승 → 강한 매집 신호 |
| 거래량 비대칭 | 20 | 상승봉 거래량 / 하락봉 거래량 비율 |
| 박스권 감지 | 15 | Rolling 60일 고저폭 + Bollinger Band Width |
| MA 정배열 | 15 | MA5/20/60 정배열 + 골든크로스 준비 |
| 기관 순매수 | 15 | pykrx 기관+외국인 5일 순매수 합산 |
| DART CB/BW | 10 | 전환사채/신주인수권부사채 공시 감지 |

**종합 점수** = `Σ(신호점수 × 가중치) / Σ(가중치) × 10 × 섹터가중치` (0~100점)

---

## 설치

```bash
# 의존성 설치
pip install pykrx --no-deps
pip install multipledispatch tqdm Deprecated
pip install pandas numpy ta APScheduler Jinja2 matplotlib
pip install requests pyyaml OpenDartReader pytest pytest-mock
```

> **참고**: pykrx는 numpy < 2.0 요구사항으로 인해 `--no-deps` 옵션으로 설치합니다.

---

## 설정

```bash
# .env 파일 생성 후 토큰 입력
cp .env.example .env
```

```env
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
DART_API_KEY=your_dart_api_key
```

`config.yaml`에서 주요 파라미터를 조정할 수 있습니다.

```yaml
universe:
  market_cap_min: 100000000000   # 시총 최소 1,000억
  market_cap_max: 300000000000   # 시총 최대 3,000억

signals:
  min_score: 50    # 최소 종합 점수
  top_n: 20        # 상위 N개 반환
```

---

## 실행

```bash
# 최초 실행: 과거 1년 데이터 수집
python main.py --history

# 즉시 1회 스캔
python main.py --run-now

# 특정 날짜 스캔
python main.py --date 20260213

# 백테스트 실행
python main.py --backtest

# 자동 스케줄러 시작 (매일 18:00)
python main.py
```

---

## 테스트

```bash
python -m pytest tests/
# 66 passed
```

---

## 프로젝트 구조

```
stock/
├── main.py              # 진입점 (APScheduler + CLI)
├── data_collector.py    # pykrx + OpenDartReader 수집
├── db_manager.py        # SQLite CRUD + 증분 업데이트
├── universe_filter.py   # 시총/섹터 필터링
├── signal_engine.py     # 6개 신호 계산 + 종합 점수
├── backtester.py        # 시그널 정확도 검증
├── report_generator.py  # 차트 + HTML + Telegram
├── config.yaml          # 전체 파라미터
├── .env.example         # 환경변수 예시
└── tests/               # 자동화 테스트 (66개)
```

---

## 실운영 결과 예시

> 2026-02-13 (설날 연휴 전 마지막 거래일)

| 항목 | 값 |
|------|-----|
| 전체 코스닥 종목 | 1,821개 |
| 유니버스 필터 통과 | 496개 |
| 시그널 감지 | 54개 |
| 1위 | AI 섹터 80.47점 |
| 스캔 소요시간 | ~12분 |

---

## 주의사항

- 본 시스템은 **투자 참고용**이며, 실제 투자 손익에 대한 책임은 본인에게 있습니다.
- DART API 키는 [dart.fss.or.kr](https://dart.fss.or.kr)에서 무료 발급 가능합니다.
- Telegram 봇 토큰은 [@BotFather](https://t.me/BotFather)에서 발급합니다.
