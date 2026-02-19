# PDCA 완료 리포트 – 코스닥 세력 매집 감지 시스템

**Feature**: stock
**완료일**: 2026-02-20
**레벨**: Dynamic
**Match Rate**: 92%
**테스트**: 66/66 통과

---

## 1. 프로젝트 개요

코스닥 중소형주(시총 1,000억~3,000억)에서 **세력 매집 패턴을 자동 감지**하는 시스템.
매일 장 마감 후(18:00) 자동 스캔하여 Telegram 알림 + HTML 리포트를 생성한다.

---

## 2. PDCA 사이클 요약

```
[Plan] ✅ → [Design] ✅ → [Do] ✅ → [Check] ✅ (92%) → [Act] ✅ → [Report] ✅
```

| 단계 | 내용 | 결과 |
|------|------|------|
| Plan | 요구사항 정의, 6개 신호 설계 | 완료 |
| Design | 아키텍처 설계, DB 스키마, 신호 가중치 결정 | 완료 |
| Do | 8개 모듈 구현 + 66개 테스트 | 완료 |
| Check | Gap 분석 (Match Rate 92%) + 보안 이슈 발견 | 완료 |
| Act | Telegram 토큰 .env 분리, 미사용 import 제거 | 완료 |

---

## 3. 구현 완성도

### 3.1 모듈별 상태

| 모듈 | 역할 | 달성도 |
|------|------|:------:|
| `signal_engine.py` | 6개 신호 계산 + 종합 점수 | 100% |
| `db_manager.py` | SQLite CRUD + 증분 업데이트 | 100% |
| `data_collector.py` | pykrx + OpenDartReader 수집 | 100% |
| `report_generator.py` | matplotlib 차트 + Jinja2 HTML + Telegram | 100% |
| `main.py` | APScheduler + CLI 진입점 | 100% |
| `backtester.py` | 과거 시그널 정확도 검증 | 90% |
| `universe_filter.py` | 시총/섹터 필터링 | 95% |
| `tests/` | 자동화 테스트 | 66/66 ✅ |

### 3.2 핵심 신호 엔진

| 신호 | 가중치 | 로직 요약 |
|------|:------:|-----------|
| OBV Divergence | 25 | 주가 하락 + OBV 상승 → 강한 매집 신호 |
| Volume Asymmetry | 20 | 상승봉 거래량 / 하락봉 거래량 비율 + 윗꼬리 패널티 |
| Box Range | 15 | 60일 Rolling 고저폭 + Bollinger Band Width |
| MA Alignment | 15 | MA5/20/60 정배열 + 골든크로스 준비 |
| Institutional Buying | 15 | pykrx 기관+외국인 순매수 5일 합산 |
| DART Signal | 10 | CB/BW 공시 감지 + 최근 30일 가산점 |

**점수 공식**: `min(100, Σ(신호점수 × 가중치) / Σ(가중치) × 10 × 섹터가중치)`

---

## 4. 주요 기술적 도전과 해결

### 4.1 pykrx 시가총액 0 반환
- **문제**: 2025년 10월 이후 날짜에서 시총 = 0 반환
- **해결**: `db_manager.load_market_cap()` – OHLCV JOIN fallback (`shares_listed × close`)
- **위치**: `db_manager.py:256-278`

### 4.2 휴일/장 마감 직후 날짜 오탐
- **문제**: 종목 리스트는 있지만 실제 가격 데이터가 없는 날 반환
- **해결**: `_latest_trading_date()` – 실 OHLCV 데이터 존재 여부 확인 후 반환
- **위치**: `data_collector.py:62-85`

### 4.3 SQLite :memory: 테스트 격리
- **문제**: 매 커넥션마다 새 DB 초기화 → 테스트 데이터 소멸
- **해결**: `_memory_conn` 단일 연결 유지 패턴
- **위치**: `db_manager.py:119-135`

### 4.4 pykrx 설치 이슈
- **문제**: numpy<2.0 요구, C 컴파일러 없는 환경에서 빌드 실패
- **해결**: `pip install pykrx --no-deps` + `multipledispatch tqdm Deprecated` 별도 설치

---

## 5. 실제 운영 결과

### 2026-02-13 스캔 (설날 연휴 전 마지막 거래일)

| 지표 | 값 |
|------|-----|
| 전체 코스닥 종목 | 1,821개 |
| 유니버스 필터 통과 | 496개 (1,000억~3,000억) |
| 시그널 통과 (min_score=50) | 54개 |
| 최종 반환 | 상위 20개 |
| 1위 | AI 섹터 80.47점 |
| 2위 | AI 섹터 75.41점 |
| 스캔 소요시간 | ~713초 (~12분) |

---

## 6. 추가 구현된 기능 (설계 초과)

| 기능 | 위치 | 가치 |
|------|------|------|
| `return_tracking` 테이블 | `db_manager.py` | 시그널 종목 수익률 사후 추적 |
| `BacktestReportGenerator` | `report_generator.py` | ECharts 백테스트 결과 시각화 |
| `accuracy_by_score_band()` | `backtester.py` | 점수 구간별 승률 분석 |
| `Normalizer` 유틸 클래스 | `signal_engine.py` | minmax/sigmoid/threshold 정규화 |
| `--date`, `--limit`, `--config` CLI | `main.py` | 운영 편의성 향상 |
| `tqdm` 진행바 | `data_collector.py` | 수집 진행상황 시각화 |

---

## 7. 보안 조치 (Act 단계 완료)

| 항목 | 조치 내용 |
|------|-----------|
| Telegram 토큰 평문 노출 | `.env` 파일로 분리, `config.yaml`에서 제거 |
| 환경변수 자동 로드 | `load_config()` – `.env` 자동 파싱 + 오버라이드 |
| `.gitignore` | `.env`, `*.db`, `logs/`, `reports/`, `cache/` 제외 |
| `.env.example` | 팀 공유용 키 구조 문서화 |

---

## 8. 잔여 과제 (Backlog)

| 우선순위 | 내용 | 예상 작업량 |
|----------|------|:-----------:|
| 중간 | `db_manager.py` / `backtester.py` 단위 테스트 추가 | 소 |
| 중간 | `data_collector.py` 통합 테스트 (mock API) | 중 |
| 낮음 | 유동주식비율 필터 실데이터 연동 (DART/KRX) | 중 |
| 낮음 | 종목 병렬 수집 성능 최적화 (asyncio) | 중 |
| 낮음 | E2E 통합 테스트 추가 | 소 |

---

## 9. 실행 방법

```bash
# 환경 설정
cp .env.example .env          # 토큰/API키 입력
pip install pykrx --no-deps
pip install multipledispatch tqdm Deprecated
pip install pandas numpy ta APScheduler Jinja2 matplotlib requests pyyaml OpenDartReader

# 최초 실행 (과거 1년 데이터 수집)
python main.py --history

# 즉시 1회 스캔
python main.py --run-now

# 특정 날짜 스캔
python main.py --date 20260213

# 백테스트
python main.py --backtest

# 자동 스케줄러 시작 (매일 18:00)
python main.py

# 테스트
python -m pytest tests/
```

---

## 10. 아키텍처 다이어그램

```
pykrx API ──────────────────────────┐
                                     ▼
DART API ─────────────────► DataCollector ──► DBManager (SQLite)
                                     │               │
                                     ▼               ▼
                             UniverseFilter   SignalEngine
                             (1000억~3000억)  (6개 신호 × 가중치)
                                     │               │
                                     └───────────────┘
                                             │
                                             ▼
                                     StockScanner (main.py)
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                        ChartGen      HTMLReport      TelegramBot
                        (matplotlib)  (Jinja2)        (알림)
```

---

*PDCA 완료 – 코스닥 세력 매집 감지 시스템 v1.0*
*생성: 2026-02-20 | Match Rate: 92% | 테스트: 66/66*
