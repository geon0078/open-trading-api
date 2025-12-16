# 한국투자증권 API 연동 뉴스 기반 자동매매 시스템

## 개요

이 모듈은 **한국투자증권(KIS) Open API**와 **Local LLM**을 연동하여 뉴스 기반 자동매매를 수행하는 시스템입니다.

### 핵심 기능
1. **KIS API 뉴스 수집**: `news_title` API로 실시간 뉴스 폴링
2. **Local LLM 분석**: 32GB VRAM 하이브리드 앙상블로 뉴스 감성 분석
3. **KIS API 주문 실행**: `order_cash` API로 실제 매수/매도 주문

---

## 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 한국투자증권 API 연동 뉴스 기반 자동매매 시스템                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  [1단계] KIS Open API - 뉴스 수집                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐        REST API 폴링         ┌─────────────────┐      │
│   │  KIS 뉴스 서버   │ ◄──────────────────────────► │  NewsCollector  │      │
│   │  (news_title)   │    TR: FHKST01011800         │  (30초 주기)     │      │
│   └─────────────────┘                              └────────┬────────┘      │
│                                                             │               │
│   Response:                                                 ▼               │
│   - data_dt: 뉴스 일자                              ┌─────────────────┐      │
│   - data_tm: 뉴스 시간                              │  pd.DataFrame   │      │
│   - titl: 뉴스 제목 ◄───────────────────────────────│  (뉴스 데이터)   │      │
│   - stck_shrn_iscd: 종목코드                        └────────┬────────┘      │
│                                                             │               │
└─────────────────────────────────────────────────────────────┼───────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [2단계] Local LLM - 뉴스 분석 (32GB VRAM)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input:                                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ news_title: "삼성전자, 3분기 영업이익 10조 원 돌파...HBM 수요 급증"   │       │
│   │ stock_code: "005930"                                            │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│   │  EXAONE 4.0     │ │  Fin-R1         │ │  Qwen3 8B       │              │
│   │  32B (20GB)     │ │  7B (6GB)       │ │  (5GB)          │              │
│   │                 │ │                 │ │                 │              │
│   │  한국어 이해     │ │  금융 전문 분석  │ │  빠른 스크리닝   │              │
│   │  Weight: 0.4    │ │  Weight: 0.4    │ │  Weight: 0.2    │              │
│   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘              │
│            │                   │                   │                        │
│            └───────────────────┼───────────────────┘                        │
│                                ▼                                            │
│                     ┌─────────────────┐                                     │
│                     │   앙상블 투표    │                                     │
│                     │   (가중 평균)    │                                     │
│                     └────────┬────────┘                                     │
│                              │                                              │
│   Output:                    ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ {                                                                │       │
│   │   "sentiment": "positive",                                       │       │
│   │   "confidence": 0.85,                                            │       │
│   │   "impact": "high",                                              │       │
│   │   "recommendation": "BUY",        ◄──── 매매 신호                 │       │
│   │   "stock_code": "005930",                                        │       │
│   │   "reasoning": "HBM 수요 급증으로 반도체 실적 개선 전망"            │       │
│   │ }                                                                │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [3단계] KIS Open API - 주문 실행                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                              ┌─────────────────┐      │
│   │  OrderExecutor  │                              │  KIS 주문 서버   │      │
│   │                 │ ─────────────────────────────►│  (order_cash)   │      │
│   │  매수: BUY      │    POST /uapi/.../order-cash │                 │      │
│   │  매도: SELL     │    TR: TTTC0012U (매수)       │                 │      │
│   │  대기: HOLD     │    TR: TTTC0011U (매도)       │                 │      │
│   └─────────────────┘                              └─────────────────┘      │
│                                                                             │
│   Request Body:                        Response:                            │
│   - CANO: 계좌번호                      - KRX_FWDG_ORD_ORGNO: 주문조직번호    │
│   - PDNO: 종목코드 (005930)             - ODNO: 주문번호                      │
│   - ORD_DVSN: 주문구분 (00: 지정가)     - ORD_TMD: 주문시각                   │
│   - ORD_QTY: 주문수량                                                        │
│   - ORD_UNPR: 주문단가                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 사용하는 KIS Open API

### 1. 뉴스 수집 API

| 항목 | 값 |
|------|-----|
| **API명** | 종합 시황/공시(제목) |
| **TR ID** | FHKST01011800 |
| **URL** | `/uapi/domestic-stock/v1/quotations/news-title` |
| **함수명** | `news_title()` |

```python
# 뉴스 수집 예시
from domestic_stock.news_title import news_title

df = news_title(
    fid_news_ofer_entp_code="",      # 뉴스 제공 업체 (공백: 전체)
    fid_cond_mrkt_cls_code="",       # 시장 구분 (공백: 전체)
    fid_input_iscd="005930",         # 종목코드 (삼성전자)
    fid_titl_cntt="",                # 제목 검색어
    fid_input_date_1="",             # 날짜
    fid_input_hour_1="",             # 시간
    fid_rank_sort_cls_code="",       # 정렬 코드
    fid_input_srno="",               # 일련번호
    max_depth=1                      # 1페이지만
)
```

**응답 필드:**
- `titl`: 뉴스 제목 → LLM 입력
- `data_dt`: 뉴스 일자
- `data_tm`: 뉴스 시간
- `stck_shrn_iscd`: 종목코드

---

### 2. 주식 주문 API (현금)

| 항목 | 값 |
|------|-----|
| **API명** | 주식주문(현금) |
| **TR ID** | TTTC0012U (매수) / TTTC0011U (매도) |
| **URL** | `/uapi/domestic-stock/v1/trading/order-cash` |
| **함수명** | `order_cash()` |

```python
# 주문 실행 예시
from domestic_stock.order_cash import order_cash

# 매수 주문
result = order_cash(
    env_dv="demo",              # 실전: "real", 모의: "demo"
    ord_dv="buy",               # 매수: "buy", 매도: "sell"
    cano="12345678",            # 계좌번호 앞 8자리
    acnt_prdt_cd="01",          # 계좌상품코드 (뒤 2자리)
    pdno="005930",              # 종목코드 (삼성전자)
    ord_dvsn="00",              # 주문구분 (00: 지정가)
    ord_qty="10",               # 주문수량
    ord_unpr="70000",           # 주문단가
    excg_id_dvsn_cd="KRX"       # 거래소 코드
)
```

**주문구분 코드:**
| 코드 | 설명 |
|------|------|
| 00 | 지정가 |
| 01 | 시장가 |
| 02 | 조건부지정가 |
| 03 | 최유리지정가 |
| 04 | 최우선지정가 |

---

### 3. 잔고 조회 API

| 항목 | 값 |
|------|-----|
| **API명** | 주식잔고조회 |
| **TR ID** | TTTC8434R (실전) / VTTC8434R (모의) |
| **URL** | `/uapi/domestic-stock/v1/trading/inquire-balance` |
| **함수명** | `inquire_balance()` |

```python
# 잔고 조회 예시
from domestic_stock.inquire_balance import inquire_balance

df1, df2 = inquire_balance(
    env_dv="demo",
    cano="12345678",
    acnt_prdt_cd="01",
    afhr_flpr_yn="N",
    inqr_dvsn="02",              # 종목별 조회
    unpr_dvsn="01",
    fund_sttl_icld_yn="N",
    fncg_amt_auto_rdpt_yn="N",
    prcs_dvsn="00"
)
```

---

## LLM 입출력 스키마

### 입력 (KIS API 뉴스 → LLM)

```python
@dataclass
class LLMInput:
    """LLM 분석 입력"""
    news_title: str        # KIS API titl 필드
    stock_code: str        # KIS API stck_shrn_iscd 필드
    news_datetime: str     # data_dt + data_tm

    # 추가 컨텍스트 (선택)
    current_price: Optional[float] = None
    holding_qty: Optional[int] = None
```

### 출력 (LLM → KIS API 주문)

```python
@dataclass
class LLMOutput:
    """LLM 분석 결과 → 주문 변환"""

    # 분석 결과
    sentiment: str          # "positive" | "negative" | "neutral"
    confidence: float       # 0.0 ~ 1.0
    impact: str             # "high" | "medium" | "low"
    reasoning: str          # 판단 근거

    # 매매 추천
    recommendation: str     # "BUY" | "SELL" | "HOLD"

    # KIS API 주문 파라미터로 변환
    def to_order_params(self, account_info: dict) -> Optional[dict]:
        """KIS API order_cash 파라미터로 변환"""
        if self.recommendation == "HOLD":
            return None

        return {
            "env_dv": account_info["env_dv"],
            "ord_dv": "buy" if self.recommendation == "BUY" else "sell",
            "cano": account_info["cano"],
            "acnt_prdt_cd": account_info["acnt_prdt_cd"],
            "pdno": self.stock_code,
            "ord_dvsn": "00",  # 지정가
            "ord_qty": str(account_info["order_qty"]),
            "ord_unpr": str(account_info["order_price"]),
            "excg_id_dvsn_cd": "KRX"
        }
```

---

## 처리 흐름 상세

### Step 1: 뉴스 수집 (NewsCollector)

```python
class NewsCollector:
    """KIS API를 통한 뉴스 수집"""

    def collect(self, stock_codes: List[str]) -> pd.DataFrame:
        """뉴스 수집 후 DataFrame 반환"""
        from domestic_stock.news_title import news_title

        all_news = []
        for stock_code in stock_codes:
            df = news_title(
                fid_input_iscd=stock_code,
                # ... 기타 파라미터
            )
            if not df.empty:
                all_news.append(df)

        return pd.concat(all_news) if all_news else pd.DataFrame()
```

### Step 2: LLM 분석 (FinancialHybridLLM)

```python
class FinancialHybridLLM:
    """32GB VRAM 하이브리드 LLM"""

    def analyze_kis_news(self, news_df: pd.DataFrame) -> List[LLMOutput]:
        """KIS API 뉴스 DataFrame을 분석"""
        results = []

        for _, row in news_df.iterrows():
            # KIS API 응답 필드 추출
            news_title = row.get('titl', '')
            stock_code = row.get('stck_shrn_iscd', '')

            # LLM 분석 수행
            result = self.analyze(
                news_title=news_title,
                stock_code=stock_code
            )

            results.append(LLMOutput(
                sentiment=result.final_sentiment.value,
                confidence=result.final_confidence,
                impact=result.final_impact.value,
                reasoning=self._extract_reasoning(result),
                recommendation=result.recommendation,
                stock_code=stock_code
            ))

        return results
```

### Step 3: 주문 실행 (OrderExecutor)

```python
class OrderExecutor:
    """KIS API를 통한 주문 실행"""

    def execute(self, llm_output: LLMOutput, account_info: dict) -> Optional[pd.DataFrame]:
        """LLM 분석 결과를 기반으로 주문 실행"""
        from domestic_stock.order_cash import order_cash

        # HOLD이면 주문 없음
        if llm_output.recommendation == "HOLD":
            return None

        # 주문 파라미터 생성
        order_params = llm_output.to_order_params(account_info)

        # KIS API 주문 실행
        result = order_cash(**order_params)

        return result
```

---

## 폴더 구조

```
examples_user/news_trading/
├── config/
│   ├── trading_config.yaml      # 매매 설정
│   └── hybrid_llm_32gb.yaml     # LLM 설정
├── modules/
│   ├── __init__.py
│   ├── llm/
│   │   ├── ARCHITECTURE.md      # 이 문서
│   │   ├── __init__.py
│   │   ├── hybrid_llm_32gb.py   # 하이브리드 LLM 클래스
│   │   └── utils/
│   │       ├── prompt_templates.py
│   │       └── ensemble.py
│   ├── news_collector.py        # KIS API 뉴스 수집
│   ├── order_executor.py        # KIS API 주문 실행
│   └── trading_engine.py        # 통합 매매 엔진
├── example_auto_trading.py      # 실행 예제
└── README.md
```

---

## 실행 예제

```python
# example_auto_trading.py
import sys
sys.path.extend(['../..', '.'])
import kis_auth as ka

from modules.llm import FinancialHybridLLM
from modules.news_collector import NewsCollector
from modules.order_executor import OrderExecutor

def main():
    # 1. KIS API 인증
    ka.auth()

    # 2. 모듈 초기화
    llm = FinancialHybridLLM()
    news_collector = NewsCollector(polling_interval=30)
    order_executor = OrderExecutor(
        env_dv="demo",  # 모의투자
        cano=ka.trenv.my_acct,
        acnt_prdt_cd=ka.trenv.my_prod
    )

    # 3. 모니터링 종목
    watch_list = ["005930", "000660", "035720"]  # 삼성전자, SK하이닉스, 카카오

    # 4. 뉴스 수집 → LLM 분석 → 주문 실행 루프
    while True:
        # 뉴스 수집 (KIS API)
        news_df = news_collector.collect(watch_list)

        if not news_df.empty:
            # LLM 분석
            analysis_results = llm.analyze_kis_news(news_df)

            for result in analysis_results:
                print(f"[{result.stock_code}] {result.recommendation} "
                      f"(신뢰도: {result.confidence:.1%})")

                # 주문 실행 (KIS API)
                if result.recommendation in ["BUY", "SELL"]:
                    order_result = order_executor.execute(result)
                    if order_result is not None:
                        print(f"  → 주문 완료: {order_result}")

        time.sleep(30)  # 30초 대기

if __name__ == "__main__":
    main()
```

---

## 주의사항

### 1. 투자 위험
- 이 시스템은 **자동매매** 시스템으로, 실제 금전적 손실이 발생할 수 있습니다.
- 반드시 **모의투자(demo)** 환경에서 충분히 테스트 후 실전 사용하세요.

### 2. API 호출 제한
- KIS API는 초당 호출 수 제한이 있습니다.
- `ka.smart_sleep()` 함수를 사용하여 적절한 대기 시간을 확보하세요.

### 3. 모델 로딩
- 32GB VRAM 환경에서 3개 모델 동시 로딩 시 약 31GB 사용
- 첫 실행 시 모델 로딩에 1-2분 소요

---

## 참고 자료

- [한국투자증권 Open API 문서](https://apiportal.koreainvestment.com/)
- [KIS Open Trading API GitHub](https://github.com/koreainvestment/open-trading-api)
- [Ollama 공식 문서](https://ollama.com/docs)
