# 뉴스 트레이딩 대시보드

LLM 기반 자동 트레이딩 시스템을 웹에서 관리하는 대시보드입니다.

## 구성

```
dashboard/
├── backend/          # FastAPI 백엔드 (Python)
│   ├── main.py       # 진입점
│   ├── api/          # API 라우터
│   ├── core/         # 핵심 서비스
│   └── models/       # Pydantic 모델
└── frontend/         # React 프론트엔드 (Vite + TypeScript)
    ├── src/
    │   ├── components/   # UI 컴포넌트
    │   ├── api/          # API 클라이언트
    │   └── store/        # Zustand 상태 관리
    └── vite.config.ts    # Vite 설정
```

## 요구사항

### 백엔드
- Python 3.10+
- FastAPI, Uvicorn
- 기타 의존성: `pydantic`, `python-multipart`, `sse-starlette`

### 프론트엔드
- Node.js 18+
- npm 또는 yarn

## 서버 실행 방법

### 1. 백엔드 실행

```bash
# 프로젝트 루트에서 실행
cd examples_user/news_trading/dashboard/backend

# 의존성 설치 (최초 1회)
pip install fastapi uvicorn pydantic python-multipart sse-starlette aiohttp

# 서버 실행 (포트 8000)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

또는 직접 실행:
```bash
python main.py
```

백엔드가 정상 실행되면:
- API 문서: http://localhost:8000/docs
- OpenAPI 스펙: http://localhost:8000/openapi.json

### 2. 프론트엔드 실행

```bash
# 프로젝트 루트에서 실행
cd examples_user/news_trading/dashboard/frontend

# 의존성 설치 (최초 1회)
npm install

# 개발 서버 실행 (포트 5173)
npm run dev
```

프론트엔드가 정상 실행되면:
- 대시보드: http://localhost:5173

### 3. 동시 실행 (권장)

두 개의 터미널을 열어 각각 실행:

**터미널 1 (백엔드):**
```bash
cd examples_user/news_trading/dashboard/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**터미널 2 (프론트엔드):**
```bash
cd examples_user/news_trading/dashboard/frontend
npm run dev
```

## 주요 기능

### 자동매매 관리
- **시작/중지**: 자동 트레이딩 활성화/비활성화
- **설정 변경**: 주문 한도, 신뢰도, 손절/익절률 등 조정
- **실시간 모니터링**: 매매 결과 실시간 확인
- **히스토리**: 과거 매매 내역 조회

### 수동 스캔
- **급등주 스캔**: 급등 종목 탐지 및 분석
- **스캘핑 모드**: 장 초반 스캘핑 전략 실행

## API 엔드포인트

### 자동매매 API (`/api/auto-trade`)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/status` | 자동매매 상태 조회 |
| POST | `/start` | 자동매매 시작 |
| POST | `/stop` | 자동매매 중지 |
| PUT | `/config` | 설정 변경 |
| POST | `/scan` | 급등주 스캔 및 매매 |
| POST | `/scalping` | 스캘핑 매매 실행 |
| GET | `/history` | 매매 히스토리 조회 |

### LLM 분석 API (`/api/llm`)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/models` | 사용 가능한 LLM 목록 |
| POST | `/analyze` | 뉴스 기반 종목 분석 |
| POST | `/ensemble` | 앙상블 분석 실행 |

## 설정 파일

### 자동매매 설정 (`config/auto_trade.yaml`)

```yaml
order:
  max_order_amount: 100000  # 1회 주문 한도 (원)
  ord_dvsn: "00"            # 주문 구분 (시장가)

threshold:
  min_confidence: 0.7       # 최소 신뢰도 (70%)
  min_consensus: 0.67       # 최소 합의도 (67%)

risk:
  stop_loss_pct: 0.5        # 손절률 (%)
  take_profit_pct: 1.5      # 익절률 (%)
  max_daily_trades: 10      # 일일 최대 거래 횟수
  max_daily_loss: 50000     # 일일 최대 손실 (원)
```

## 문제 해결

### 연결 오류
프론트엔드에서 "연결 끊김"이 표시되는 경우:
1. 백엔드가 포트 8000에서 실행 중인지 확인
2. `vite.config.ts`의 프록시 설정이 `http://localhost:8000`인지 확인

### 모듈 오류
`ModuleNotFoundError`가 발생하는 경우:
```bash
# 프로젝트 루트에서 실행
pip install -r requirements.txt
```

### 포트 충돌
이미 사용 중인 포트가 있는 경우:
```bash
# 다른 포트로 백엔드 실행
python -m uvicorn main:app --port 8001

# vite.config.ts에서 프록시 대상 포트 변경
```

## 참고 문서

- [매수/매도 조건](../docs/TRADING_CONDITIONS.md)
- [시스템 아키텍처](../ARCHITECTURE.md)
- [LLM 앙상블 설정](../config/hybrid_llm_32gb.yaml)
