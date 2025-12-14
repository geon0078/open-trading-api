# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KIS Open Trading API는 한국투자증권 Open API를 LLM과 Python 개발자 모두가 쉽게 활용할 수 있도록 구성된 샘플 코드 저장소입니다.

**핵심 구조:**
- `examples_llm/`: LLM이 단일 API 기능을 탐색하도록 구성된 개별 함수 파일 (669개)
- `examples_user/`: 개발자용 통합 예제 (상품별 functions.py + examples.py)

## Common Commands

```bash
# 의존성 설치 (uv 권장)
uv sync

# 또는 pip 사용
pip install -r requirements.txt

# REST API 예제 실행
python examples_user/domestic_stock/domestic_stock_examples.py

# WebSocket 예제 실행
python examples_user/domestic_stock/domestic_stock_examples_ws.py

# 개별 API 테스트 (examples_llm)
python examples_llm/domestic_stock/inquire_price/chk_inquire_price.py
```

## Architecture

### 인증 흐름
```
kis_devlp.yaml (credentials) → kis_auth.py (token) → API 호출
```

`kis_auth.py`가 토큰 발급/관리, 환경 전환(실전/모의), WebSocket 연결을 담당합니다.

### API 카테고리 (7개)
| 폴더명 | 설명 |
|--------|------|
| `domestic_stock` | 국내주식 시세, 주문, 잔고 |
| `domestic_bond` | 국내채권 |
| `domestic_futureoption` | 국내선물옵션 |
| `overseas_stock` | 해외주식 |
| `overseas_futureoption` | 해외선물옵션 |
| `elw` | ELW |
| `etfetn` | ETF/ETN |

### examples_llm vs examples_user 구조

**examples_llm/** - 단일 API별 폴더:
```
domestic_stock/inquire_price/
├── inquire_price.py      # 한줄 호출 함수
└── chk_inquire_price.py  # 테스트 파일
```

**examples_user/** - 상품별 통합:
```
domestic_stock/
├── domestic_stock_functions.py     # REST 통합 함수
├── domestic_stock_examples.py      # REST 예제
├── domestic_stock_functions_ws.py  # WebSocket 통합 함수
└── domestic_stock_examples_ws.py   # WebSocket 예제
```

### MCP 서버
`MCP/Kis Trading MCP/`: Docker 기반 FastMCP 서버로 166+ API 함수를 Claude에서 직접 호출 가능

## Code Conventions

### 네이밍
- 모듈/변수/함수: `snake_case`
- 클래스: `PascalCase`
- 상수: `UPPER_SNAKE_CASE`

### 파일명 규칙
API URL에서 파생:
```
URL: /uapi/domestic-stock/v1/quotations/inquire-price
폴더: domestic_stock
파일: inquire_price.py
테스트: chk_inquire_price.py
```

### 필수 사항
- 타입 힌트 명시 (파라미터, 리턴값)
- Docstring 작성 (Google/NumPy 스타일)
- 명시적 import (wildcard 지양)
- try-except에 구체적 예외 타입 사용
- logging 모듈로 이벤트 기록

## Key Files

- `kis_devlp.yaml`: API 인증 정보 (앱키, 계좌번호 등)
- `kis_auth.py`: 인증 및 공통 함수 (각 examples 폴더에 존재)
- `stocks_info/`: 종목 마스터 데이터

## Authentication

```python
import kis_auth as ka

# 모의투자 인증
ka.auth(svr="vps", product="01")

# 실전투자 인증
ka.auth(svr="prod", product="01")

# WebSocket 인증 추가
ka.auth_ws()
```

토큰은 `~/.KIS/config/` (또는 설정된 경로)에 저장됩니다.

## News Trading System

`examples_user/news_trading/` 폴더에 LLM 기반 뉴스 트레이딩 시스템이 구축되어 있습니다.

### 핵심 구성요소

```
examples_user/news_trading/
├── ARCHITECTURE.md              # 상세 아키텍처 문서
├── config/hybrid_llm_32gb.yaml  # LLM 설정
├── modules/
│   ├── ensemble_analyzer.py     # 앙상블 LLM 분석기
│   ├── llm_analyzer.py          # 단일 LLM 분석기
│   └── llm/hybrid_llm_32gb.py   # 하이브리드 LLM
└── run_scalping_scanner.py      # 스캘핑 스캐너
```

### LLM 앙상블 프리셋

| 프리셋 | 모델 | VRAM | 용도 |
|--------|------|------|------|
| `deepseek` (권장) | DeepSeek-R1 + Qwen3 | ~12GB | 금융 추론 특화 |
| `default` | EXAONE + Fin-R1 + Qwen3 | ~31GB | 고정밀 분석 |
| `lightweight` | DeepSeek-R1 8B/1.5B + Qwen3 | ~12GB | 빠른 응답 |

### 빠른 시작

```bash
# Ollama 모델 설치
ollama pull deepseek-r1:8b
ollama pull qwen3:8b

# 스캘핑 스캐너 실행
cd examples_user/news_trading
python run_scalping_scanner.py
```

### LLM 사용 예시

```python
from modules.llm import FinancialHybridLLM

llm = FinancialHybridLLM()
llm.set_preset("deepseek")  # DeepSeek 앙상블 사용
result = llm.analyze("삼성전자, 3분기 영업이익 10조", "005930")
print(result.recommendation)  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
```

자세한 내용은 `examples_user/news_trading/ARCHITECTURE.md` 참조.
