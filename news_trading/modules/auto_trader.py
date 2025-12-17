# -*- coding: utf-8 -*-
"""
2단계 앙상블 LLM 자동 매매기

EXAONE 4.0을 메인 모델로, 3개 하위 모델(deepseek, qwen3, solar)이
데이터를 탐색하고 EXAONE이 최종 판단하여 자동 매매를 실행합니다.

워크플로우:
1. 급등 종목 탐지 (SurgeDetector)
2. OHLCV + 보조지표 조회 (OHLCVFetcher, TechnicalAnalyzer)
3. 1단계: 하위 모델 앙상블 분석
4. 2단계: EXAONE 최종 판단
5. 리스크 체크 (RiskManager)
6. 주문 실행 (OrderExecutor)

사용 예시:
    >>> from modules.auto_trader import AutoTrader, AutoTradeConfig
    >>> config = AutoTradeConfig(env_dv="real", max_order_amount=100000)
    >>> trader = AutoTrader(config)
    >>> results = trader.run_scan_and_trade()
"""

import sys
import logging
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import deque

# 경로 설정
sys.path.extend(['..', '.', '../..'])

logger = logging.getLogger(__name__)


@dataclass
class AutoTradeConfig:
    """자동 매매 설정"""
    # 환경 설정
    env_dv: str = "real"                    # 실전/모의 구분 ("real" 또는 "demo")
    cano: str = ""                          # 계좌번호 앞 8자리
    acnt_prdt_cd: str = ""                  # 계좌상품코드 (뒤 2자리)

    # 주문 설정
    max_order_amount: int = 100000          # 1회 주문 한도 (기본: 10만원)
    ord_dvsn: str = "00"                    # 주문구분 ("00": 지정가, "01": 시장가)

    # 신뢰도/합의도 설정
    min_confidence: float = 0.7             # 최소 신뢰도 (70%)
    min_consensus: float = 0.67             # 최소 합의도 (67%, 모델 2/3 이상)

    # 매매 시그널 설정
    allowed_buy_signals: List[str] = field(default_factory=lambda: ["STRONG_BUY", "BUY"])
    allowed_sell_signals: List[str] = field(default_factory=lambda: ["STRONG_SELL", "SELL"])

    # 리스크 관리
    stop_loss_pct: float = 0.5              # 손절률 (0.5%)
    take_profit_pct: float = 1.5            # 익절률 (1.5%)
    max_daily_trades: int = 0               # 일일 최대 거래 횟수 (0 = 무제한)
    max_daily_loss: int = 50000             # 일일 최대 손실 (5만원)

    # 급등 종목 스캔 설정
    min_surge_score: float = 50.0           # 최소 급등 점수
    max_stocks_per_scan: int = 5            # 1회 스캔 시 분석할 최대 종목 수
    min_technical_score: float = -10.0      # 최소 기술점수 (-10 미만 제외)
    min_price: int = 5000                   # 최소 가격 (저가주 제외)

    # 시장 시간 설정
    market_start: str = "09:00"             # 장 시작 시간
    market_end: str = "15:20"               # 장 마감 시간

    # 스캘핑 모드 설정
    scalping_enabled: bool = True           # 스캘핑 모드 활성화
    scalping_start: str = "09:00"           # 스캘핑 시작 시간
    scalping_end: str = "09:30"             # 스캘핑 종료 시간
    scalping_min_confidence: float = 0.65   # 스캘핑 최소 신뢰도 (빠른 결정)
    scalping_max_order_amount: int = 50000  # 스캘핑 1회 주문 한도 (리스크 감소)
    scalping_stop_loss_pct: float = 0.3     # 스캘핑 손절률 (0.3%, 타이트)
    scalping_take_profit_pct: float = 0.8   # 스캘핑 익절률 (0.8%, 빠른 익절)

    # 앙상블 설정
    use_parallel: bool = False              # 모델 병렬 실행 여부
    auto_unload: bool = True                # 분석 후 모델 언로드

    # 저장 설정
    save_to_db: bool = True                 # SQLite DB에 저장
    save_llm_logs: bool = True              # LLM 상세 로그 저장 (JSON)
    compress_logs: bool = False             # LLM 로그 압축 (.json.gz)


@dataclass
class AutoTradeResult:
    """자동 매매 결과"""
    success: bool
    action: str                             # BUY, SELL, HOLD, SKIP, ERROR
    stock_code: str
    stock_name: str
    current_price: int

    # 분석 결과
    ensemble_signal: str
    confidence: float
    consensus: float

    # 주문 정보
    order_qty: int = 0
    order_price: int = 0
    order_no: Optional[str] = None

    # 메시지
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 기술적 지표 요약
    technical_score: float = 0.0
    trend: str = ""


class AutoTrader:
    """
    2단계 앙상블 LLM 기반 자동 매매기

    워크플로우:
    1. 급등 종목 탐지 (SurgeDetector)
    2. OHLCV + 보조지표 조회 (OHLCVFetcher, TechnicalAnalyzer)
    3. 1단계: 하위 모델 앙상블 분석
    4. 2단계: EXAONE 최종 판단
    5. 리스크 체크 (RiskManager)
    6. 주문 실행 (OrderExecutor)
    """

    def __init__(self, config: AutoTradeConfig):
        """
        Args:
            config: AutoTradeConfig 설정 객체
        """
        self.config = config
        self._initialized = False
        self._trade_history: deque = deque(maxlen=100)
        self._today_trades: int = 0
        self._today_pnl: int = 0
        self._last_trade_date: str = ""

        # 컴포넌트 (lazy 초기화)
        self._ensemble_analyzer = None
        self._order_executor = None
        self._surge_detector = None

        # 마지막 앙상블 분석 결과 (LLM 입출력 저장용)
        self._last_ensemble_result = None

        # 스캔 중 발생한 모든 앙상블 결과 (stock_code -> ensemble_result)
        self._scan_ensemble_results = {}

        # 저장소 초기화 (lazy)
        self._trade_db = None
        self._llm_storage = None

    def _ensure_initialized(self):
        """컴포넌트 초기화 확인"""
        if self._initialized:
            return

        try:
            # KIS API 인증
            import kis_auth as ka
            ka.auth()

            # 계좌 정보 설정
            if not self.config.cano:
                self.config.cano = ka.getTREnv().my_acct
            if not self.config.acnt_prdt_cd:
                self.config.acnt_prdt_cd = ka.getTREnv().my_prod

            logger.info(f"KIS API 인증 완료: 계좌={self.config.cano}-{self.config.acnt_prdt_cd}")

            # 앙상블 분석기 초기화
            from .ensemble_analyzer import EnsembleLLMAnalyzer
            self._ensemble_analyzer = EnsembleLLMAnalyzer(
                keep_alive="5m",
                auto_unload=self.config.auto_unload
            )
            self._ensemble_analyzer.discover_models()
            self._ensemble_analyzer.setup_ensemble(use_financial_ensemble=True)
            logger.info(f"앙상블 모델 설정: {self._ensemble_analyzer.ensemble_models}")

            # 주문 실행기 초기화
            from .order_executor import OrderExecutor
            self._order_executor = OrderExecutor(
                env_dv=self.config.env_dv,
                cano=self.config.cano,
                acnt_prdt_cd=self.config.acnt_prdt_cd,
                default_ord_dvsn=self.config.ord_dvsn
            )
            logger.info(f"주문 실행기 초기화: env={self.config.env_dv}")

            # 급등 탐지기 초기화
            try:
                from .surge_detector import SurgeDetector
                self._surge_detector = SurgeDetector()
                logger.info("급등 탐지기 초기화 완료")
            except ImportError:
                logger.warning("SurgeDetector 모듈 없음 - 급등 종목 스캔 비활성화")
                self._surge_detector = None

            # 저장소 초기화
            if self.config.save_to_db:
                try:
                    from .storage import get_trade_history_db
                    self._trade_db = get_trade_history_db()
                    logger.info("거래 내역 DB 초기화 완료")
                except ImportError:
                    logger.warning("storage 모듈 없음 - DB 저장 비활성화")
                    self._trade_db = None

            if self.config.save_llm_logs:
                try:
                    from .storage import get_llm_log_storage
                    self._llm_storage = get_llm_log_storage(compress=self.config.compress_logs)
                    logger.info("LLM 로그 저장소 초기화 완료")
                except ImportError:
                    logger.warning("storage 모듈 없음 - LLM 로그 저장 비활성화")
                    self._llm_storage = None

            self._initialized = True
            logger.info("AutoTrader 초기화 완료")

        except Exception as e:
            logger.error(f"AutoTrader 초기화 실패: {e}")
            raise

    def _reset_daily_counters(self):
        """일일 카운터 리셋 (날짜 변경 시)"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_trade_date != today:
            self._today_trades = 0
            self._today_pnl = 0
            self._last_trade_date = today
            logger.info(f"일일 카운터 리셋: {today}")

    def _check_market_hours(self) -> Tuple[bool, str]:
        """
        시장 시간 체크

        Returns:
            (bool, str): (거래 가능 여부, 사유)
        """
        now = datetime.now()

        # 주말 체크
        if now.weekday() >= 5:  # 토(5), 일(6)
            return False, "주말은 거래 불가"

        # 시간 체크
        market_start = dt_time(*map(int, self.config.market_start.split(":")))
        market_end = dt_time(*map(int, self.config.market_end.split(":")))

        if now.time() < market_start:
            return False, f"장 시작 전 ({self.config.market_start})"
        if now.time() > market_end:
            return False, f"장 마감 후 ({self.config.market_end})"

        return True, "거래 가능 시간"

    def _check_risk_limits(self) -> Tuple[bool, str]:
        """
        리스크 한도 체크

        Returns:
            (bool, str): (거래 가능 여부, 사유)
        """
        self._reset_daily_counters()

        # 일일 거래 횟수 체크 (max_daily_trades <= 0 이면 무제한)
        if self.config.max_daily_trades > 0 and self._today_trades >= self.config.max_daily_trades:
            return False, f"일일 거래 횟수 초과: {self._today_trades}/{self.config.max_daily_trades}"

        # 일일 손실 한도 체크
        if self._today_pnl <= -self.config.max_daily_loss:
            return False, f"일일 손실 한도 초과: {self._today_pnl:,}원"

        return True, "리스크 한도 내"

    def _is_scalping_time(self) -> bool:
        """
        스캘핑 시간인지 확인 (09:00 ~ 09:30)

        Returns:
            bool: 스캘핑 시간 여부
        """
        if not self.config.scalping_enabled:
            return False

        now = datetime.now()

        # 주말 체크
        if now.weekday() >= 5:
            return False

        scalping_start = dt_time(*map(int, self.config.scalping_start.split(":")))
        scalping_end = dt_time(*map(int, self.config.scalping_end.split(":")))

        return scalping_start <= now.time() <= scalping_end

    def _collect_overnight_news(self, stock_codes: List[str] = None) -> List[str]:
        """
        야간 뉴스 수집 (전일 장 마감 ~ 금일 장 시작)

        전일 15:30 이후부터 오늘 09:00 이전의 모든 뉴스를 수집합니다.

        Args:
            stock_codes: 종목코드 리스트 (None이면 전체 뉴스)

        Returns:
            List[str]: 뉴스 제목 리스트
        """
        try:
            from .news_collector import NewsCollector
            collector = NewsCollector()

            # 뉴스 수집 (최대 5페이지 깊이로 야간 뉴스 확보)
            news_df = collector.collect(
                stock_codes=stock_codes,
                filter_duplicates=False,
                max_depth=5
            )

            if news_df.empty:
                logger.info("[Scalping] 야간 뉴스 없음")
                return []

            # 야간 시간대 뉴스 필터링
            overnight_news = []
            now = datetime.now()
            today_str = now.strftime("%Y%m%d")
            yesterday = (now - timedelta(days=1))
            yesterday_str = yesterday.strftime("%Y%m%d")

            for _, row in news_df.iterrows():
                news_date = str(row.get('data_dt', ''))
                news_time = str(row.get('data_tm', ''))[:4]  # HHMM

                # 전일 15:30 이후 뉴스
                if news_date == yesterday_str and news_time >= "1530":
                    overnight_news.append(row.get('titl', ''))

                # 금일 09:00 이전 뉴스
                elif news_date == today_str and news_time < "0900":
                    overnight_news.append(row.get('titl', ''))

            logger.info(f"[Scalping] 야간 뉴스 {len(overnight_news)}건 수집")
            return overnight_news

        except ImportError:
            logger.warning("[Scalping] NewsCollector 모듈 없음")
            return []
        except Exception as e:
            logger.error(f"[Scalping] 뉴스 수집 오류: {e}")
            return []

    def _build_scalping_context(self, overnight_news: List[str]) -> str:
        """
        스캘핑용 시황 컨텍스트 생성

        Args:
            overnight_news: 야간 뉴스 리스트

        Returns:
            str: 시황 요약 컨텍스트
        """
        if not overnight_news:
            return "야간 주요 뉴스 없음"

        # 뉴스 요약 (최대 20개)
        news_summary = "\n".join([f"- {title}" for title in overnight_news[:20]])

        context = f"""
=== 야간 시황 (전일 장 마감 ~ 금일 장 시작) ===
수집된 뉴스: {len(overnight_news)}건

주요 헤드라인:
{news_summary}

[분석 지침]
1. 야간 뉴스의 전반적인 시장 분위기를 파악하세요.
2. 호재/악재 뉴스의 비중을 평가하세요.
3. 장 초반 갭 상승/하락 가능성을 예측하세요.
4. 스캘핑에 적합한 종목인지 판단하세요.
"""
        return context

    def run_news_analysis(self, max_news: int = 20) -> Dict:
        """
        뉴스 분석 모드 실행 (장 시작 전 분석용)

        전체 뉴스를 수집하고 LLM으로 분석하여 시장 전망 및 주목 종목을 도출합니다.

        Args:
            max_news: 분석할 최대 뉴스 수

        Returns:
            Dict: 뉴스 분석 결과
                - news_count: 분석한 뉴스 수
                - news_list: 뉴스 제목 리스트
                - market_sentiment: 시장 심리 (BULLISH/BEARISH/NEUTRAL)
                - key_themes: 주요 테마 리스트
                - attention_stocks: 주목 종목 리스트
                - llm_analysis: LLM 분석 결과
        """
        self._ensure_initialized()

        logger.info("=" * 50)
        logger.info("[뉴스 분석] 장 시작 전 뉴스 분석 시작")
        logger.info("=" * 50)

        result = {
            "timestamp": datetime.now().isoformat(),
            "news_count": 0,
            "news_list": [],
            "market_sentiment": "NEUTRAL",
            "key_themes": [],
            "attention_stocks": [],
            "llm_analysis": None
        }

        # 1. 뉴스 수집
        try:
            from .news_collector import NewsCollector
            collector = NewsCollector()

            # 전체 뉴스 수집 (최대 5페이지)
            news_df = collector.collect(stock_codes=None, filter_duplicates=False, max_depth=5)

            if news_df.empty:
                logger.info("[뉴스 분석] 수집된 뉴스 없음")
                return result

            # 뉴스 제목 컬럼 찾기 (API 응답에 따라 다름)
            title_col = None
            for col in ['hts_pbnt_titl_cntt', 'titl', 'titl_cntt', 'cntt', 'news_titl', 'title']:
                if col in news_df.columns:
                    title_col = col
                    break

            if title_col is None:
                logger.warning(f"[뉴스 분석] 제목 컬럼을 찾을 수 없음. 컬럼: {list(news_df.columns)}")
                # 첫 번째 문자열 컬럼 사용
                for col in news_df.columns:
                    if news_df[col].dtype == 'object':
                        title_col = col
                        logger.info(f"[뉴스 분석] '{col}' 컬럼을 제목으로 사용")
                        break

            # 뉴스 제목 추출
            if title_col:
                news_titles = news_df[title_col].tolist()[:max_news]
            else:
                news_titles = []

            result["news_count"] = len(news_titles)
            result["news_list"] = news_titles

            logger.info(f"[뉴스 분석] {len(news_titles)}건 뉴스 수집 완료")

        except ImportError:
            logger.warning("[뉴스 분석] NewsCollector 모듈 없음")
            return result
        except Exception as e:
            logger.error(f"[뉴스 분석] 뉴스 수집 오류: {e}")
            return result

        if not news_titles:
            return result

        # 2. LLM 뉴스 분석
        try:
            news_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(news_titles)])

            analysis_prompt = f"""당신은 한국 주식 시장 전문 애널리스트입니다.
아래의 최신 뉴스 헤드라인을 분석하고, 오늘 장 시작 전 투자자들이 알아야 할 핵심 정보를 정리해주세요.

=== 오늘의 뉴스 헤드라인 ({len(news_titles)}건) ===
{news_text}

=== 분석 요청 ===
1. 시장 심리 판단: BULLISH(강세) / BEARISH(약세) / NEUTRAL(중립) 중 하나
2. 주요 테마 3가지 (예: 반도체, AI, 금리 등)
3. 주목해야 할 종목 최대 5개 (종목명과 이유)
4. 오늘 장 전망 요약 (2-3문장)

반드시 아래 JSON 형식으로 응답하세요:
{{
    "market_sentiment": "BULLISH/BEARISH/NEUTRAL 중 하나",
    "key_themes": ["테마1", "테마2", "테마3"],
    "attention_stocks": [
        {{"name": "종목명", "code": "종목코드(알면)", "reason": "주목 이유"}},
    ],
    "market_outlook": "오늘 장 전망 요약"
}}
"""

            # Ollama API로 직접 뉴스 분석
            logger.info("[뉴스 분석] LLM 분석 시작...")

            import requests

            model_name = self._ensemble_analyzer.ensemble_models[0] if self._ensemble_analyzer.ensemble_models else "qwen3:8b"
            ollama_url = self._ensemble_analyzer.ollama_url

            payload = {
                "model": model_name,
                "prompt": analysis_prompt,
                "stream": False,
                "keep_alive": "5m",
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1000,
                    "num_ctx": 4096
                }
            }

            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=180
            )

            if response.status_code == 200:
                raw_output = response.json().get("response", "").strip()
                logger.info(f"[뉴스 분석] LLM 응답 수신 ({len(raw_output)} chars)")

                # JSON 파싱 시도
                import json
                import re

                # <think> 태그 제거
                raw_output_clean = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)

                # JSON 블록 추출
                json_match = re.search(r'\{[\s\S]*\}', raw_output_clean)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group())
                        result["market_sentiment"] = analysis.get("market_sentiment", "NEUTRAL")
                        result["key_themes"] = analysis.get("key_themes", [])
                        result["attention_stocks"] = analysis.get("attention_stocks", [])
                        result["llm_analysis"] = {
                            "market_outlook": analysis.get("market_outlook", ""),
                            "raw_output": raw_output,
                            "model_used": model_name
                        }

                        logger.info(f"[뉴스 분석] 시장 심리: {result['market_sentiment']}")
                        logger.info(f"[뉴스 분석] 주요 테마: {result['key_themes']}")
                        logger.info(f"[뉴스 분석] 주목 종목: {len(result['attention_stocks'])}개")

                    except json.JSONDecodeError:
                        logger.warning("[뉴스 분석] JSON 파싱 실패, 원본 저장")
                        result["llm_analysis"] = {"raw_output": raw_output, "model_used": model_name}
                else:
                    result["llm_analysis"] = {"raw_output": raw_output, "model_used": model_name}
            else:
                logger.error(f"[뉴스 분석] Ollama API 오류: {response.status_code}")

        except Exception as e:
            logger.error(f"[뉴스 분석] LLM 분석 오류: {e}")

        logger.info("[뉴스 분석] 분석 완료")
        logger.info("=" * 50)

        return result

    def run_scalping_trade(
        self,
        stock_codes: List[str] = None,
        min_score: float = None,
        max_stocks: int = None
    ) -> List[AutoTradeResult]:
        """
        스캘핑 매매 실행 (09:00 ~ 09:30)

        야간 뉴스를 분석하고 하위 모델이 시황을 판단한 뒤,
        메인 모델(EXAONE)이 최종 스캘핑 매매 결정을 내립니다.

        Args:
            stock_codes: 스캘핑 대상 종목코드 리스트 (None이면 급등 종목 스캔)
            min_score: 최소 급등 점수 (None이면 config 사용)
            max_stocks: 분석할 최대 종목 수 (None이면 config 사용)

        Returns:
            List[AutoTradeResult]: 스캘핑 매매 결과 리스트
        """
        self._ensure_initialized()

        # 스캔 시작 시 앙상블 결과 초기화
        self._scan_ensemble_results = {}

        if not self._is_scalping_time():
            logger.info("[Scalping] 스캘핑 시간이 아닙니다")
            return []

        logger.info("=" * 50)
        logger.info("[Scalping] 스캘핑 모드 시작 (09:00 ~ 09:30)")
        logger.info("=" * 50)

        # 리스크 한도 체크
        can_trade, reason = self._check_risk_limits()
        if not can_trade:
            logger.info(f"[Scalping] 거래 불가: {reason}")
            return []

        # 1. 야간 뉴스 수집
        logger.info("[Scalping] Step 1: 야간 뉴스 수집...")
        overnight_news = self._collect_overnight_news(stock_codes)
        scalping_context = self._build_scalping_context(overnight_news)

        # 2. 대상 종목 선정
        if stock_codes:
            target_stocks = []
            for code in stock_codes:
                # 종목 정보 조회 필요 시 여기서 처리
                target_stocks.append({
                    'code': code,
                    'name': code,  # 이름은 나중에 조회
                    'price': 0
                })
        else:
            # 급등 종목 스캔
            if self._surge_detector is None:
                logger.warning("[Scalping] SurgeDetector 없음")
                return []

            if min_score is None:
                min_score = self.config.min_surge_score
            if max_stocks is None:
                max_stocks = self.config.max_stocks_per_scan

            logger.info(f"[Scalping] Step 2: 급등 종목 스캔 (최소 점수: {min_score})...")
            try:
                target_stocks = self._surge_detector.scan_surge_stocks(min_score=min_score)
            except Exception as e:
                logger.error(f"[Scalping] 급등 종목 스캔 실패: {e}")
                return []

        if not target_stocks:
            logger.info("[Scalping] 대상 종목 없음")
            return []

        logger.info(f"[Scalping] 대상 종목 {len(target_stocks)}개, 상위 {max_stocks}개 분석")

        # 3. 스캘핑 분석 및 매매 실행
        results = []
        for stock in target_stocks[:max_stocks]:
            # 리스크 한도 재확인
            can_trade, reason = self._check_risk_limits()
            if not can_trade:
                logger.info(f"[Scalping] 거래 중단: {reason}")
                break

            # SurgeCandidate 객체와 dict 모두 처리
            if hasattr(stock, 'code'):
                # SurgeCandidate 객체
                stock_code = stock.code
                stock_name = stock.name
                current_price = int(stock.price)
            else:
                # 딕셔너리
                stock_code = stock.get('code', '')
                stock_name = stock.get('name', '')
                current_price = int(stock.get('price', 0))

            if not stock_code:
                continue

            # 스캘핑 분석 (야간 뉴스 컨텍스트 포함)
            result = self._execute_scalping_analysis(
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                stock_data=stock,
                overnight_news=overnight_news,
                scalping_context=scalping_context
            )
            results.append(result)

            if result.success:
                logger.info(f"[Scalping] {stock_name}: {result.action} 성공")

        success_count = sum(1 for r in results if r.success)
        logger.info(f"[Scalping] 완료: {len(results)}개 분석, {success_count}개 매매 실행")
        logger.info("=" * 50)

        return results

    def _execute_scalping_analysis(
        self,
        stock_code: str,
        stock_name: str,
        current_price: int,
        stock_data: Dict = None,
        overnight_news: List[str] = None,
        scalping_context: str = ""
    ) -> AutoTradeResult:
        """
        스캘핑 분석 및 매매 실행

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            current_price: 현재가
            stock_data: 추가 종목 정보
            overnight_news: 야간 뉴스 리스트
            scalping_context: 시황 컨텍스트

        Returns:
            AutoTradeResult: 스캘핑 매매 결과
        """
        try:
            logger.info(f"[Scalping] {stock_name}({stock_code}) 분석 시작...")

            # 앙상블 분석 (스캘핑 모드)
            ensemble_result = self._ensemble_analyzer.analyze_with_technical_data(
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                stock_data=stock_data,
                news_list=overnight_news,
                additional_context=scalping_context,
                analysis_mode="scalping",  # 스캘핑 모드 플래그
                parallel=self.config.use_parallel,
                unload_after=self.config.auto_unload
            )

            # LLM 입출력 저장용으로 마지막 분석 결과 보관
            self._last_ensemble_result = ensemble_result
            # 스캔 결과에도 저장 (stock_code를 key로)
            self._scan_ensemble_results[stock_code] = ensemble_result

            # 기술적 지표 요약
            tech_summary = ensemble_result.input_data.get('technical_summary', {})
            tech_score = tech_summary.get('total_score', 0)
            trend = tech_summary.get('trend', 'N/A')

            logger.info(f"[Scalping] 분석 완료: 시그널={ensemble_result.ensemble_signal}, "
                       f"신뢰도={ensemble_result.ensemble_confidence:.0%}, "
                       f"합의도={ensemble_result.consensus_score:.0%}, "
                       f"기술점수={tech_score:+.1f}")

            # ========== 기술점수/가격 필터링 ==========
            # 기술점수가 너무 낮으면 매수 제외
            if tech_score < self.config.min_technical_score:
                logger.warning(f"[Scalping] 기술점수 미달로 매수 제외: {tech_score:.1f} < {self.config.min_technical_score:.1f}")
                return AutoTradeResult(
                    success=False,
                    action="SKIP",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal=ensemble_result.ensemble_signal,
                    confidence=ensemble_result.ensemble_confidence,
                    consensus=ensemble_result.consensus_score,
                    reason=f"[스캘핑] 기술점수 미달 ({tech_score:.1f} < {self.config.min_technical_score:.1f})",
                    technical_score=tech_score,
                    trend=trend
                )

            # 저가주 필터링
            if current_price < self.config.min_price:
                logger.warning(f"[Scalping] 저가주 제외: {current_price:,}원 < {self.config.min_price:,}원")
                return AutoTradeResult(
                    success=False,
                    action="SKIP",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal=ensemble_result.ensemble_signal,
                    confidence=ensemble_result.ensemble_confidence,
                    consensus=ensemble_result.consensus_score,
                    reason=f"[스캘핑] 저가주 제외 ({current_price:,}원 < {self.config.min_price:,}원)",
                    technical_score=tech_score,
                    trend=trend
                )

            # 스캘핑 설정으로 자동 매매 실행
            order_result = self._order_executor.execute_auto_trade(
                ensemble_result=ensemble_result,
                max_order_amount=self.config.scalping_max_order_amount,
                min_confidence=self.config.scalping_min_confidence,
                min_consensus=self.config.min_consensus,
                allowed_buy_signals=self.config.allowed_buy_signals,
                allowed_sell_signals=self.config.allowed_sell_signals,
                use_entry_price=True,
                ord_dvsn="01"  # 스캘핑은 시장가 주문
            )

            # 결과 생성
            if order_result.success:
                action = order_result.order_type.upper()
                self._today_trades += 1
                logger.info(f"[Scalping] 주문 성공: {action} {order_result.order_qty}주 @ {order_result.order_price:,}원")
            else:
                action = order_result.order_type.upper() if order_result.order_type else "SKIP"

            result = AutoTradeResult(
                success=order_result.success,
                action=action,
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                ensemble_signal=ensemble_result.ensemble_signal,
                confidence=ensemble_result.ensemble_confidence,
                consensus=ensemble_result.consensus_score,
                order_qty=order_result.order_qty,
                order_price=order_result.order_price,
                order_no=order_result.order_no,
                reason=f"[스캘핑] {order_result.message}",
                technical_score=tech_score,
                trend=trend
            )

            self._trade_history.append(result)

            # DB 및 LLM 로그 저장
            self._save_analysis_result(result, ensemble_result, overnight_news)

            return result

        except Exception as e:
            logger.error(f"[Scalping] 분석/매매 오류: {e}")
            return AutoTradeResult(
                success=False,
                action="ERROR",
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                ensemble_signal="N/A",
                confidence=0,
                consensus=0,
                reason=f"[스캘핑] 오류: {str(e)}"
            )

    def analyze_and_trade(
        self,
        stock_code: str,
        stock_name: str,
        current_price: int,
        stock_data: Dict = None,
        news_list: List[str] = None,
        check_market_hours: bool = True,
        check_risk_limits: bool = True,
        analysis_only: bool = False
    ) -> AutoTradeResult:
        """
        분석 및 자동 매매 실행

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            current_price: 현재가
            stock_data: 추가 종목 정보 (체결강도, 호가비 등)
            news_list: 관련 뉴스 리스트
            check_market_hours: 장 시간 체크 여부
            check_risk_limits: 리스크 한도 체크 여부
            analysis_only: True이면 분석만 실행하고 주문은 건너뜀 (장 시작 전 분석 모드)

        Returns:
            AutoTradeResult: 자동 매매 결과
        """
        self._ensure_initialized()

        # 장 시간 체크
        if check_market_hours:
            can_trade, reason = self._check_market_hours()
            if not can_trade:
                return AutoTradeResult(
                    success=False,
                    action="SKIP",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal="N/A",
                    confidence=0,
                    consensus=0,
                    reason=reason
                )

        # 리스크 한도 체크
        if check_risk_limits:
            can_trade, reason = self._check_risk_limits()
            if not can_trade:
                return AutoTradeResult(
                    success=False,
                    action="SKIP",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal="N/A",
                    confidence=0,
                    consensus=0,
                    reason=reason
                )

        try:
            # ========== 앙상블 분석 ==========
            logger.info(f"[AutoTrader] {stock_name}({stock_code}) 분석 시작...")

            ensemble_result = self._ensemble_analyzer.analyze_with_technical_data(
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                stock_data=stock_data,
                news_list=news_list,
                parallel=self.config.use_parallel,
                unload_after=self.config.auto_unload
            )

            # LLM 입출력 저장용으로 마지막 분석 결과 보관
            self._last_ensemble_result = ensemble_result
            # 스캔 결과에도 저장 (stock_code를 key로)
            self._scan_ensemble_results[stock_code] = ensemble_result

            # 기술적 지표 요약
            tech_summary = ensemble_result.input_data.get('technical_summary', {})
            tech_score = tech_summary.get('total_score', 0)
            trend = tech_summary.get('trend', 'N/A')

            logger.info(f"[AutoTrader] 분석 완료: 시그널={ensemble_result.ensemble_signal}, "
                       f"신뢰도={ensemble_result.ensemble_confidence:.0%}, "
                       f"합의도={ensemble_result.consensus_score:.0%}, "
                       f"기술점수={tech_score:+.1f}")

            # ========== 기술점수/가격 필터링 ==========
            # 기술점수가 너무 낮으면 매수 제외
            if tech_score < self.config.min_technical_score:
                logger.warning(f"[AutoTrader] 기술점수 미달로 매수 제외: {tech_score:.1f} < {self.config.min_technical_score:.1f}")
                return AutoTradeResult(
                    success=False,
                    action="SKIP",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal=ensemble_result.ensemble_signal,
                    confidence=ensemble_result.ensemble_confidence,
                    consensus=ensemble_result.consensus_score,
                    reason=f"기술점수 미달 ({tech_score:.1f} < {self.config.min_technical_score:.1f})",
                    technical_score=tech_score,
                    trend=trend
                )

            # 저가주 필터링
            if current_price < self.config.min_price:
                logger.warning(f"[AutoTrader] 저가주 제외: {current_price:,}원 < {self.config.min_price:,}원")
                return AutoTradeResult(
                    success=False,
                    action="SKIP",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal=ensemble_result.ensemble_signal,
                    confidence=ensemble_result.ensemble_confidence,
                    consensus=ensemble_result.consensus_score,
                    reason=f"저가주 제외 ({current_price:,}원 < {self.config.min_price:,}원)",
                    technical_score=tech_score,
                    trend=trend
                )

            # ========== 자동 매매 실행 ==========
            if analysis_only:
                # 분석 전용 모드: 주문 없이 분석 결과만 반환
                logger.info(f"[AutoTrader] 분석 전용 모드 - 주문 건너뜀")
                result = AutoTradeResult(
                    success=False,
                    action="ANALYSIS",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal=ensemble_result.ensemble_signal,
                    confidence=ensemble_result.ensemble_confidence,
                    consensus=ensemble_result.consensus_score,
                    order_qty=0,
                    order_price=0,
                    order_no=None,
                    reason="[분석 전용] 장 시작 전 분석 완료",
                    technical_score=tech_score,
                    trend=trend
                )
            else:
                order_result = self._order_executor.execute_auto_trade(
                    ensemble_result=ensemble_result,
                    max_order_amount=self.config.max_order_amount,
                    min_confidence=self.config.min_confidence,
                    min_consensus=self.config.min_consensus,
                    allowed_buy_signals=self.config.allowed_buy_signals,
                    allowed_sell_signals=self.config.allowed_sell_signals,
                    use_entry_price=True,
                    ord_dvsn=self.config.ord_dvsn
                )

                # 결과 생성
                if order_result.success:
                    action = order_result.order_type.upper()
                    self._today_trades += 1
                    logger.info(f"[AutoTrader] 주문 성공: {action} {order_result.order_qty}주 @ {order_result.order_price:,}원")
                else:
                    action = order_result.order_type.upper() if order_result.order_type else "SKIP"

                result = AutoTradeResult(
                    success=order_result.success,
                    action=action,
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    ensemble_signal=ensemble_result.ensemble_signal,
                    confidence=ensemble_result.ensemble_confidence,
                    consensus=ensemble_result.consensus_score,
                    order_qty=order_result.order_qty,
                    order_price=order_result.order_price,
                    order_no=order_result.order_no,
                    reason=order_result.message,
                    technical_score=tech_score,
                    trend=trend
                )

            # 히스토리 저장
            self._trade_history.append(result)

            # DB 및 LLM 로그 저장
            self._save_analysis_result(result, ensemble_result, news_list)

            return result

        except Exception as e:
            logger.error(f"[AutoTrader] 분석/매매 오류: {e}")
            return AutoTradeResult(
                success=False,
                action="ERROR",
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                ensemble_signal="N/A",
                confidence=0,
                consensus=0,
                reason=str(e)
            )

    def run_scan_and_trade(
        self,
        min_score: float = None,
        max_stocks: int = None,
        check_market_hours: bool = True,
        analysis_only: bool = False
    ) -> List[AutoTradeResult]:
        """
        급등 종목 스캔 후 자동 매매

        Args:
            min_score: 최소 급등 점수 (None이면 config 사용)
            max_stocks: 분석할 최대 종목 수 (None이면 config 사용)
            check_market_hours: 장 시간 체크 여부
            analysis_only: True이면 분석만 실행하고 주문은 건너뜀 (장 시작 전 분석 모드)

        Returns:
            List[AutoTradeResult]: 자동 매매 결과 리스트
        """
        self._ensure_initialized()

        # 스캔 시작 시 앙상블 결과 초기화
        self._scan_ensemble_results = {}

        if min_score is None:
            min_score = self.config.min_surge_score
        if max_stocks is None:
            max_stocks = self.config.max_stocks_per_scan

        # 장 시간 체크 (분석 전용 모드에서는 건너뜀)
        if check_market_hours and not analysis_only:
            can_trade, reason = self._check_market_hours()
            if not can_trade:
                logger.info(f"[AutoTrader] 거래 불가: {reason}")
                return []

        # 리스크 한도 체크 (분석 전용 모드에서는 건너뜀)
        if not analysis_only:
            can_trade, reason = self._check_risk_limits()
            if not can_trade:
                logger.info(f"[AutoTrader] 거래 불가: {reason}")
                return []

        # 급등 종목 스캔
        if self._surge_detector is None:
            logger.warning("[AutoTrader] SurgeDetector 없음 - 급등 종목 스캔 불가")
            return []

        logger.info(f"[AutoTrader] 급등 종목 스캔 시작 (최소 점수: {min_score})")

        try:
            surge_stocks = self._surge_detector.scan_surge_stocks(min_score=min_score)
        except Exception as e:
            logger.error(f"[AutoTrader] 급등 종목 스캔 실패: {e}")
            return []

        if not surge_stocks:
            logger.info("[AutoTrader] 급등 종목 없음")
            return []

        logger.info(f"[AutoTrader] {len(surge_stocks)}개 급등 종목 발견, 상위 {max_stocks}개 분석")

        # 상위 종목만 분석
        results = []
        for stock in surge_stocks[:max_stocks]:
            # 리스크 한도 재확인 (분석 전용 모드에서는 건너뜀)
            if not analysis_only:
                can_trade, reason = self._check_risk_limits()
                if not can_trade:
                    logger.info(f"[AutoTrader] 거래 중단: {reason}")
                    break

            # SurgeCandidate 객체와 dict 모두 처리
            if hasattr(stock, 'code'):
                # SurgeCandidate 객체
                stock_code = stock.code
                stock_name = stock.name
                current_price = int(stock.price)
            else:
                # 딕셔너리
                stock_code = stock.get('code', '')
                stock_name = stock.get('name', '')
                current_price = int(stock.get('price', 0))

            if not stock_code or current_price <= 0:
                continue

            result = self.analyze_and_trade(
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                stock_data=stock,
                check_market_hours=False,  # 이미 체크함
                check_risk_limits=False,   # 이미 체크함
                analysis_only=analysis_only
            )
            results.append(result)

            # 매매 성공 시 로그
            if result.success:
                logger.info(f"[AutoTrader] {stock_name}: {result.action} 완료")

        logger.info(f"[AutoTrader] 스캔 완료: {len(results)}개 분석, "
                   f"{sum(1 for r in results if r.success)}개 매매 실행")

        return results

    def get_trade_history(self, limit: int = 20) -> List[AutoTradeResult]:
        """거래 히스토리 조회"""
        return list(self._trade_history)[-limit:]

    def get_ensemble_result(self, stock_code: str):
        """
        특정 종목의 앙상블 분석 결과 조회 (LLM 입출력 저장용)

        Args:
            stock_code: 종목코드

        Returns:
            앙상블 분석 결과 또는 None
        """
        return self._scan_ensemble_results.get(stock_code)

    def get_all_ensemble_results(self) -> Dict:
        """
        현재 스캔의 모든 앙상블 분석 결과 조회 (LLM 입출력 저장용)

        Returns:
            Dict[stock_code, ensemble_result]
        """
        return self._scan_ensemble_results.copy()

    def get_status(self) -> Dict:
        """현재 상태 조회"""
        self._reset_daily_counters()

        can_trade_market, market_reason = self._check_market_hours()
        can_trade_risk, risk_reason = self._check_risk_limits()

        return {
            "initialized": self._initialized,
            "config": {
                "env_dv": self.config.env_dv,
                "max_order_amount": self.config.max_order_amount,
                "min_confidence": self.config.min_confidence,
                "min_consensus": self.config.min_consensus,
                "max_daily_trades": self.config.max_daily_trades,
                "max_daily_loss": self.config.max_daily_loss,
            },
            "daily_stats": {
                "today_trades": self._today_trades,
                "today_pnl": self._today_pnl,
                "date": self._last_trade_date,
            },
            "can_trade": can_trade_market and can_trade_risk,
            "market_status": {
                "can_trade": can_trade_market,
                "reason": market_reason,
            },
            "risk_status": {
                "can_trade": can_trade_risk,
                "reason": risk_reason,
            },
            "ensemble_models": self._ensemble_analyzer.ensemble_models if self._ensemble_analyzer else [],
            "main_model": self._ensemble_analyzer.MAIN_JUDGE_MODEL if self._ensemble_analyzer else "",
        }

    def _save_analysis_result(
        self,
        result: AutoTradeResult,
        ensemble_result,
        news_list: List[str] = None
    ):
        """
        분석 결과를 DB 및 LLM 로그에 저장

        Args:
            result: 자동 매매 결과
            ensemble_result: 앙상블 분석 결과
            news_list: 뉴스 리스트
        """
        if ensemble_result is None:
            return

        llm_log_path = None
        analysis_id = None

        try:
            # 1. LLM 상세 로그 저장 (JSON)
            if self._llm_storage and ensemble_result:
                from dataclasses import asdict, is_dataclass

                # 모델별 출력 준비
                model_outputs = []
                for model_result in ensemble_result.model_results:
                    if is_dataclass(model_result):
                        model_outputs.append(asdict(model_result))
                    elif isinstance(model_result, dict):
                        model_outputs.append(model_result)
                    else:
                        model_outputs.append({
                            "model_name": getattr(model_result, 'model_name', 'unknown'),
                            "raw_output": getattr(model_result, 'raw_output', ''),
                            "signal": getattr(model_result, 'signal', 'HOLD'),
                            "confidence": getattr(model_result, 'confidence', 0),
                            "reasoning": getattr(model_result, 'reasoning', ''),
                            "processing_time": getattr(model_result, 'processing_time', 0),
                            "success": getattr(model_result, 'success', False)
                        })

                # 앙상블 결과 준비
                ensemble_data = {
                    "ensemble_signal": ensemble_result.ensemble_signal,
                    "ensemble_confidence": ensemble_result.ensemble_confidence,
                    "ensemble_trend": ensemble_result.ensemble_trend,
                    "consensus_score": ensemble_result.consensus_score,
                    "signal_votes": ensemble_result.signal_votes,
                    "trend_votes": ensemble_result.trend_votes,
                    "avg_entry_price": ensemble_result.avg_entry_price,
                    "avg_stop_loss": ensemble_result.avg_stop_loss,
                    "avg_take_profit": ensemble_result.avg_take_profit,
                    "total_processing_time": ensemble_result.total_processing_time,
                    "models_used": ensemble_result.models_used,
                    "models_agreed": ensemble_result.models_agreed,
                    "total_models": ensemble_result.total_models
                }

                # LLM 로그 저장
                llm_log_path = self._llm_storage.save_ensemble_log(
                    stock_code=result.stock_code,
                    stock_name=result.stock_name,
                    input_prompt=ensemble_result.input_prompt,
                    stock_data=ensemble_result.input_data.get('stock', {}),
                    news_list=news_list or [],
                    technical_summary=ensemble_result.input_data.get('technical_summary', {}),
                    model_outputs=model_outputs,
                    ensemble_result=ensemble_data,
                    analysis_type="scalping" if result.reason and "[스캘핑]" in result.reason else "ensemble"
                )
                logger.debug(f"LLM 로그 저장: {llm_log_path}")

            # 2. 분석 요약 저장 (SQLite)
            if self._trade_db and ensemble_result:
                analysis_id = self._trade_db.save_analysis(
                    stock_code=result.stock_code,
                    stock_name=result.stock_name,
                    ensemble_signal=ensemble_result.ensemble_signal,
                    ensemble_confidence=ensemble_result.ensemble_confidence,
                    ensemble_trend=ensemble_result.ensemble_trend,
                    consensus_score=ensemble_result.consensus_score,
                    current_price=result.current_price,
                    avg_entry_price=ensemble_result.avg_entry_price,
                    avg_stop_loss=ensemble_result.avg_stop_loss,
                    avg_take_profit=ensemble_result.avg_take_profit,
                    models_used=ensemble_result.models_used,
                    models_agreed=ensemble_result.models_agreed,
                    total_models=ensemble_result.total_models,
                    signal_votes=ensemble_result.signal_votes,
                    trend_votes=ensemble_result.trend_votes,
                    technical_summary=ensemble_result.input_data.get('technical_summary', {}),
                    total_processing_time=ensemble_result.total_processing_time,
                    llm_log_path=llm_log_path
                )

            # 3. 거래 내역 저장 (성공한 주문만)
            if self._trade_db and result.success and result.order_qty > 0:
                self._trade_db.save_trade(
                    stock_code=result.stock_code,
                    stock_name=result.stock_name,
                    action=result.action,
                    order_qty=result.order_qty,
                    order_price=result.order_price,
                    order_no=result.order_no,
                    success=result.success,
                    ensemble_signal=result.ensemble_signal,
                    confidence=result.confidence,
                    consensus=result.consensus,
                    technical_score=result.technical_score,
                    trend=result.trend,
                    reason=result.reason,
                    analysis_id=analysis_id,
                    llm_log_path=llm_log_path,
                    timestamp=result.timestamp
                )
                logger.info(f"거래 내역 저장: {result.stock_name} {result.action}")

        except Exception as e:
            logger.error(f"분석 결과 저장 실패: {e}")

    def save_news_analysis(self, news_result: Dict):
        """
        뉴스 분석 결과 저장

        Args:
            news_result: run_news_analysis() 결과
        """
        if not self._llm_storage:
            return

        try:
            llm_analysis = news_result.get('llm_analysis', {})
            self._llm_storage.save_news_analysis_log(
                news_list=news_result.get('news_list', []),
                llm_analysis=llm_analysis,
                market_sentiment=news_result.get('market_sentiment', 'NEUTRAL'),
                key_themes=news_result.get('key_themes', []),
                attention_stocks=news_result.get('attention_stocks', []),
                model_used=llm_analysis.get('model_used', 'unknown'),
                raw_output=llm_analysis.get('raw_output', ''),
                processing_time=0
            )
            logger.info("뉴스 분석 로그 저장 완료")
        except Exception as e:
            logger.error(f"뉴스 분석 저장 실패: {e}")


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 설정 생성
    config = AutoTradeConfig(
        env_dv="real",                # 실전투자
        max_order_amount=100000,      # 10만원 한도
        min_confidence=0.7,           # 신뢰도 70%
        min_consensus=0.67,           # 합의도 67%
        max_daily_trades=10,          # 일일 10회
        max_daily_loss=50000,         # 일일 손실 5만원
        min_surge_score=50.0,         # 급등 점수 50점
        max_stocks_per_scan=5,        # 최대 5종목
    )

    print("=== AutoTrader 설정 ===")
    print(f"환경: {config.env_dv}")
    print(f"주문 한도: {config.max_order_amount:,}원")
    print(f"최소 신뢰도: {config.min_confidence:.0%}")
    print(f"최소 합의도: {config.min_consensus:.0%}")

    # AutoTrader 생성
    trader = AutoTrader(config)

    # 상태 확인
    status = trader.get_status()
    print("\n=== 현재 상태 ===")
    print(f"초기화: {status['initialized']}")
    print(f"거래 가능: {status['can_trade']}")
    print(f"시장 상태: {status['market_status']['reason']}")
    print(f"리스크 상태: {status['risk_status']['reason']}")

    # 단일 종목 분석 테스트
    print("\n=== 단일 종목 분석 테스트 ===")
    result = trader.analyze_and_trade(
        stock_code="005930",
        stock_name="삼성전자",
        current_price=55000,
        check_market_hours=False  # 테스트용으로 장 시간 체크 비활성화
    )

    print(f"종목: {result.stock_name} ({result.stock_code})")
    print(f"시그널: {result.ensemble_signal}")
    print(f"신뢰도: {result.confidence:.0%}")
    print(f"합의도: {result.consensus:.0%}")
    print(f"기술점수: {result.technical_score:+.1f}")
    print(f"행동: {result.action}")
    print(f"결과: {result.reason}")
