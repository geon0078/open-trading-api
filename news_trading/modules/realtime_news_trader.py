# -*- coding: utf-8 -*-
"""
실시간 뉴스 기반 트레이딩 시스템

9:30 이후 정규장에서 실시간으로 들어오는 뉴스를 LLM이 분석하고,
기술적 분석과 결합하여 자동 매매 결정을 내립니다.

주요 기능:
1. 실시간 뉴스 스트림 모니터링
2. 뉴스 발생 시 관련 종목 식별
3. LLM이 뉴스 + 기술적 분석 종합 판단
4. 자동 매수/매도 결정

사용 예시:
    >>> from modules.realtime_news_trader import RealtimeNewsTrader
    >>> trader = RealtimeNewsTrader()
    >>> await trader.start()
"""

import sys
import asyncio
import logging
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Set
from collections import deque

sys.path.extend(['..', '.', '../..'])

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """뉴스 아이템"""
    title: str
    content: str = ""
    source: str = ""
    timestamp: str = ""
    stock_codes: List[str] = field(default_factory=list)
    stock_names: List[str] = field(default_factory=list)
    sentiment: str = ""  # POSITIVE, NEGATIVE, NEUTRAL
    impact_level: str = ""  # HIGH, MEDIUM, LOW
    is_processed: bool = False


@dataclass
class NewsTradeSignal:
    """뉴스 기반 매매 시그널"""
    stock_code: str
    stock_name: str
    action: str  # BUY, SELL, HOLD, WATCH
    confidence: float = 0
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH, CRITICAL
    reason: str = ""

    # 뉴스 정보
    news_title: str = ""
    news_sentiment: str = ""
    news_impact: str = ""

    # 기술적 분석
    technical_score: float = 0
    trend: str = ""
    support_price: int = 0
    resistance_price: int = 0

    # 주문 정보
    suggested_price: int = 0
    suggested_quantity: int = 0
    stop_loss_reason: str = ""  # LLM이 제안하는 손절 조건
    take_profit_reason: str = ""  # LLM이 제안하는 익절 조건

    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class RealtimeNewsTrader:
    """
    실시간 뉴스 기반 트레이딩

    9:30 이후 정규장에서:
    1. 실시간 뉴스 모니터링
    2. 새 뉴스 발생 시 LLM 분석
    3. 기술적 분석과 결합
    4. 매매 결정 및 실행
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        analysis_model: str = "deepseek-r1:8b",
        secondary_model: str = "qwen3:8b",
        news_check_interval: int = 30,  # 30초마다 뉴스 체크
        on_trade_signal: Optional[Callable] = None,
        on_news_analyzed: Optional[Callable] = None,
        auto_execute: bool = False  # 자동 주문 실행 여부
    ):
        self.ollama_url = ollama_url
        self.analysis_model = analysis_model
        self.secondary_model = secondary_model
        self.news_check_interval = news_check_interval
        self.on_trade_signal = on_trade_signal
        self.on_news_analyzed = on_news_analyzed
        self.auto_execute = auto_execute

        self._running = False
        self._monitor_task = None

        # 처리된 뉴스 추적 (중복 방지)
        self._processed_news: Set[str] = set()
        self._news_history: deque = deque(maxlen=200)
        self._signal_history: deque = deque(maxlen=100)

        # 관심 종목 (이 종목들의 뉴스를 특히 주시)
        self._watchlist: Dict[str, str] = {}  # code -> name

        # 시장 컨텍스트 (전체 시장 분위기)
        self._market_context: str = ""
        self._market_sentiment: str = "NEUTRAL"

        # 컴포넌트
        self._news_collector = None
        self._ohlcv_fetcher = None
        self._technical_analyzer = None
        self._order_executor = None

    def _ensure_components(self):
        """컴포넌트 초기화"""
        try:
            if self._news_collector is None:
                from .news_collector import NewsCollector
                self._news_collector = NewsCollector()
                logger.info("NewsCollector 초기화 완료")
        except ImportError as e:
            logger.warning(f"NewsCollector 로드 실패: {e}")

        try:
            if self._ohlcv_fetcher is None:
                from .ohlcv_fetcher import OHLCVFetcher
                from .technical_indicators import TechnicalAnalyzer
                self._ohlcv_fetcher = OHLCVFetcher()
                self._technical_analyzer = TechnicalAnalyzer()
                logger.info("기술적 분석 컴포넌트 초기화 완료")
        except ImportError as e:
            logger.warning(f"기술적 분석 컴포넌트 로드 실패: {e}")

        try:
            if self._order_executor is None and self.auto_execute:
                from .order_executor import OrderExecutor
                self._order_executor = OrderExecutor()
                logger.info("OrderExecutor 초기화 완료")
        except ImportError as e:
            logger.warning(f"OrderExecutor 로드 실패: {e}")

    def add_to_watchlist(self, stock_code: str, stock_name: str):
        """관심 종목 추가"""
        self._watchlist[stock_code] = stock_name
        logger.info(f"[관심종목 추가] {stock_name}({stock_code})")

    def remove_from_watchlist(self, stock_code: str):
        """관심 종목 제거"""
        if stock_code in self._watchlist:
            name = self._watchlist.pop(stock_code)
            logger.info(f"[관심종목 제거] {name}({stock_code})")

    def set_market_context(self, context: str, sentiment: str = "NEUTRAL"):
        """시장 컨텍스트 설정"""
        self._market_context = context
        self._market_sentiment = sentiment

    async def start(self):
        """실시간 뉴스 트레이딩 시작"""
        if self._running:
            return

        self._ensure_components()
        self._running = True
        self._monitor_task = asyncio.create_task(self._news_monitoring_loop())
        logger.info(f"[실시간 뉴스 트레이딩] 시작 (주기: {self.news_check_interval}초)")

    async def stop(self):
        """중지"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[실시간 뉴스 트레이딩] 중지")

    async def _news_monitoring_loop(self):
        """뉴스 모니터링 메인 루프"""
        while self._running:
            try:
                # 1. 새로운 뉴스 수집
                new_news = await self._fetch_new_news()

                if new_news:
                    logger.info(f"[뉴스] 새 뉴스 {len(new_news)}건 발견")

                    # 2. 각 뉴스 분석 및 매매 판단
                    for news in new_news:
                        signals = await self._analyze_news_and_decide(news)

                        for signal in signals:
                            if signal.action in ["BUY", "SELL"]:
                                logger.info(
                                    f"[매매 시그널] {signal.stock_name}: {signal.action} "
                                    f"(신뢰도: {signal.confidence:.0%}, 긴급도: {signal.urgency})"
                                )
                                logger.info(f"  뉴스: {signal.news_title[:50]}...")
                                logger.info(f"  사유: {signal.reason}")

                                # 콜백 호출
                                if self.on_trade_signal:
                                    await self._safe_callback(self.on_trade_signal, signal)

                                # 자동 실행
                                if self.auto_execute and signal.confidence >= 0.7:
                                    await self._execute_trade(signal)

                            self._signal_history.append(signal)

                await asyncio.sleep(self.news_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[뉴스 모니터링 오류] {e}")
                await asyncio.sleep(10)

    async def _fetch_new_news(self) -> List[NewsItem]:
        """새로운 뉴스 수집"""
        if not self._news_collector:
            return []

        try:
            loop = asyncio.get_event_loop()

            # 전체 뉴스 + 관심 종목 뉴스 수집
            stock_codes = list(self._watchlist.keys()) if self._watchlist else None

            news_df = await loop.run_in_executor(
                None,
                lambda: self._news_collector.collect(
                    stock_codes=stock_codes,
                    filter_duplicates=False,
                    max_depth=2
                )
            )

            if news_df is None or news_df.empty:
                return []

            # 뉴스 아이템으로 변환
            new_items = []
            for _, row in news_df.iterrows():
                # 제목 컬럼 찾기
                title = ""
                for col in ['hts_pbnt_titl_cntt', 'titl', 'news_titl', 'title']:
                    if col in row and row[col]:
                        title = str(row[col])
                        break

                if not title:
                    continue

                # 중복 체크
                news_key = f"{row.get('data_dt', '')}_{title[:30]}"
                if news_key in self._processed_news:
                    continue

                self._processed_news.add(news_key)

                # 관련 종목 추출
                stock_codes = []
                stock_names = []
                if 'stck_shrn_iscd' in row and row['stck_shrn_iscd']:
                    stock_codes.append(str(row['stck_shrn_iscd']))
                if 'hts_kor_isnm' in row and row['hts_kor_isnm']:
                    stock_names.append(str(row['hts_kor_isnm']))

                news_item = NewsItem(
                    title=title,
                    content=str(row.get('cntt', '')),
                    source=str(row.get('info_prv_nm', '')),
                    timestamp=f"{row.get('data_dt', '')} {row.get('data_tm', '')}",
                    stock_codes=stock_codes,
                    stock_names=stock_names
                )

                new_items.append(news_item)
                self._news_history.append(news_item)

            # 최대 10개만 처리
            return new_items[:10]

        except Exception as e:
            logger.error(f"뉴스 수집 오류: {e}")
            return []

    async def _analyze_news_and_decide(self, news: NewsItem) -> List[NewsTradeSignal]:
        """
        뉴스 분석 및 매매 결정

        LLM이 뉴스를 분석하고 기술적 분석과 결합하여 매매 판단
        """
        signals = []

        try:
            # 1. 뉴스에서 관련 종목 식별 (LLM 사용)
            related_stocks = await self._identify_related_stocks(news)

            if not related_stocks:
                return []

            # 2. 각 관련 종목에 대해 분석
            for stock_info in related_stocks[:3]:  # 최대 3종목
                stock_code = stock_info.get('code', '')
                stock_name = stock_info.get('name', '')

                if not stock_code:
                    continue

                # 3. 기술적 분석 데이터 수집
                tech_data = await self._get_technical_data(stock_code)

                # 4. LLM 종합 분석 및 매매 결정
                signal = await self._llm_trade_decision(
                    news=news,
                    stock_code=stock_code,
                    stock_name=stock_name,
                    tech_data=tech_data,
                    news_impact=stock_info.get('impact', 'MEDIUM')
                )

                if signal:
                    signals.append(signal)

            # 콜백 호출
            if self.on_news_analyzed and signals:
                await self._safe_callback(self.on_news_analyzed, news, signals)

            news.is_processed = True

        except Exception as e:
            logger.error(f"뉴스 분석 오류: {e}")

        return signals

    async def _identify_related_stocks(self, news: NewsItem) -> List[Dict]:
        """
        뉴스에서 관련 종목 식별 (LLM 사용)
        """
        # 이미 종목 정보가 있으면 사용
        if news.stock_codes and news.stock_names:
            return [{"code": news.stock_codes[0], "name": news.stock_names[0], "impact": "HIGH"}]

        prompt = f"""다음 뉴스 헤드라인을 분석하고 관련된 한국 주식 종목을 식별하세요.

뉴스: {news.title}

관심 종목 리스트:
{json.dumps(self._watchlist, ensure_ascii=False, indent=2) if self._watchlist else '(없음)'}

반드시 아래 JSON 형식으로 응답하세요:
{{
    "related_stocks": [
        {{
            "code": "종목코드 (6자리)",
            "name": "종목명",
            "impact": "HIGH/MEDIUM/LOW (뉴스가 이 종목에 미치는 영향도)",
            "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
            "reason": "관련 이유"
        }}
    ],
    "news_category": "경제/산업/기업/정책/시장/기타",
    "overall_sentiment": "POSITIVE/NEGATIVE/NEUTRAL"
}}

관련 종목이 없으면 related_stocks를 빈 배열로 응답하세요.
"""

        try:
            result = await self._call_llm(prompt, model=self.secondary_model)
            clean_result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)

            json_match = re.search(r'\{[\s\S]*\}', clean_result)
            if json_match:
                data = json.loads(json_match.group())
                news.sentiment = data.get('overall_sentiment', 'NEUTRAL')
                return data.get('related_stocks', [])

        except Exception as e:
            logger.warning(f"종목 식별 오류: {e}")

        return []

    async def _get_technical_data(self, stock_code: str) -> Dict[str, Any]:
        """기술적 분석 데이터 조회"""
        if not self._ohlcv_fetcher or not self._technical_analyzer:
            return {}

        try:
            loop = asyncio.get_event_loop()

            ohlcv = await loop.run_in_executor(
                None,
                lambda: self._ohlcv_fetcher.fetch(stock_code, period="D", count=20)
            )

            if ohlcv is None or ohlcv.empty:
                return {}

            analysis = await loop.run_in_executor(
                None,
                lambda: self._technical_analyzer.analyze(ohlcv)
            )

            # 현재가 조회
            current_price = int(ohlcv['close'].iloc[-1]) if not ohlcv.empty else 0

            return {
                **analysis,
                "current_price": current_price
            }

        except Exception as e:
            logger.warning(f"기술적 데이터 조회 실패 ({stock_code}): {e}")
            return {}

    async def _llm_trade_decision(
        self,
        news: NewsItem,
        stock_code: str,
        stock_name: str,
        tech_data: Dict,
        news_impact: str
    ) -> Optional[NewsTradeSignal]:
        """
        LLM 종합 분석 및 매매 결정

        뉴스 + 기술적 분석 + 시장 상황을 종합하여 판단
        """
        # 기술적 지표 텍스트
        tech_text = ""
        if tech_data:
            tech_text = f"""
기술적 분석:
  - 현재가: {tech_data.get('current_price', 'N/A'):,}원
  - RSI: {tech_data.get('rsi', 'N/A')}
  - MACD 시그널: {tech_data.get('macd_signal', 'N/A')}
  - 이동평균 추세: {tech_data.get('ma_trend', 'N/A')}
  - 볼린저밴드 위치: {tech_data.get('bb_position', 'N/A')}
  - 거래량 변화: {tech_data.get('volume_change', 'N/A')}
  - 지지선: {tech_data.get('support', 'N/A')}
  - 저항선: {tech_data.get('resistance', 'N/A')}
"""

        prompt = f"""당신은 전문 트레이더입니다. 아래 정보를 종합 분석하여 매매 결정을 내리세요.

=== 뉴스 정보 ===
헤드라인: {news.title}
내용: {news.content[:500] if news.content else '(없음)'}
출처: {news.source}
시간: {news.timestamp}
뉴스 심리: {news.sentiment}
영향도: {news_impact}

=== 분석 대상 종목 ===
종목명: {stock_name}
종목코드: {stock_code}
{tech_text}

=== 시장 상황 ===
{self._market_context or '(정보 없음)'}
시장 심리: {self._market_sentiment}

=== 분석 요청 ===
위 뉴스가 {stock_name}에 미치는 영향과 기술적 상황을 종합하여 매매 결정을 내리세요.

** 중요 **
- 뉴스의 호재/악재 정도를 정확히 평가하세요
- 기술적 분석과 뉴스가 같은 방향이면 강한 시그널
- 반대 방향이면 신중하게 판단
- 매수 시 손절/익절 조건도 제안하세요 (고정 %가 아닌 상황 기반)

반드시 아래 JSON 형식으로 응답하세요:
{{
    "action": "BUY/SELL/HOLD/WATCH 중 하나",
    "confidence": 0.0~1.0,
    "urgency": "LOW/NORMAL/HIGH/CRITICAL",
    "reason": "결정 이유 (2-3문장)",
    "news_analysis": "뉴스가 종목에 미치는 영향 분석",
    "technical_view": "기술적 분석 관점",
    "suggested_price": 0 (제안 매매가, 0이면 시장가),
    "stop_loss_condition": "손절 조건 설명 (예: '호재 효과 소멸 시', '지지선 이탈 시')",
    "take_profit_condition": "익절 조건 설명 (예: '저항선 도달 시', '거래량 급감 시')",
    "risk_factors": ["리스크 1", "리스크 2"],
    "holding_period": "예상 보유 기간 (예: '단기 1-2일', '스윙 1주')"
}}

action 설명:
- BUY: 매수 진입 추천
- SELL: 매도/청산 추천
- HOLD: 기존 포지션 유지
- WATCH: 관망 (관심 종목으로 추가)
"""

        try:
            result = await self._call_llm(prompt, model=self.analysis_model)
            clean_result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)

            json_match = re.search(r'\{[\s\S]*\}', clean_result)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            return NewsTradeSignal(
                stock_code=stock_code,
                stock_name=stock_name,
                action=data.get('action', 'HOLD'),
                confidence=float(data.get('confidence', 0.5)),
                urgency=data.get('urgency', 'NORMAL'),
                reason=data.get('reason', ''),
                news_title=news.title,
                news_sentiment=news.sentiment,
                news_impact=news_impact,
                technical_score=tech_data.get('total_score', 0) if tech_data else 0,
                trend=tech_data.get('trend', '') if tech_data else '',
                support_price=int(tech_data.get('support', 0) or 0) if tech_data else 0,
                resistance_price=int(tech_data.get('resistance', 0) or 0) if tech_data else 0,
                suggested_price=int(data.get('suggested_price', 0) or 0),
                stop_loss_reason=data.get('stop_loss_condition', ''),
                take_profit_reason=data.get('take_profit_condition', '')
            )

        except Exception as e:
            logger.error(f"LLM 매매 결정 오류: {e}")
            return None

    async def _call_llm(self, prompt: str, model: str = None) -> str:
        """LLM API 호출"""
        import aiohttp

        payload = {
            "model": model or self.analysis_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1000,
                "num_ctx": 4096
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"LLM API 오류: {response.status}")

    async def _execute_trade(self, signal: NewsTradeSignal):
        """자동 매매 실행"""
        if not self._order_executor:
            logger.warning("OrderExecutor가 초기화되지 않음 - 자동 실행 불가")
            return

        try:
            if signal.action == "BUY":
                # 매수 주문
                order_result = self._order_executor.buy(
                    stock_code=signal.stock_code,
                    price=signal.suggested_price or 0,  # 0이면 시장가
                    quantity=signal.suggested_quantity or 0  # 금액 기준으로 계산 필요
                )
                logger.info(f"[자동매수] {signal.stock_name}: {order_result}")

            elif signal.action == "SELL":
                # 매도 주문
                order_result = self._order_executor.sell(
                    stock_code=signal.stock_code,
                    price=signal.suggested_price or 0,
                    quantity=signal.suggested_quantity or 0
                )
                logger.info(f"[자동매도] {signal.stock_name}: {order_result}")

        except Exception as e:
            logger.error(f"자동 매매 실행 오류: {e}")

    async def _safe_callback(self, callback, *args):
        """안전한 콜백 실행"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"콜백 실행 오류: {e}")

    async def analyze_news_now(self, news_title: str, stock_code: str = None) -> List[NewsTradeSignal]:
        """특정 뉴스 즉시 분석"""
        self._ensure_components()

        news = NewsItem(
            title=news_title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            stock_codes=[stock_code] if stock_code else []
        )

        return await self._analyze_news_and_decide(news)

    def get_signal_history(self, limit: int = 20) -> List[NewsTradeSignal]:
        """시그널 기록 조회"""
        return list(self._signal_history)[-limit:]

    def get_news_history(self, limit: int = 50) -> List[NewsItem]:
        """뉴스 기록 조회"""
        return list(self._news_history)[-limit:]


# 테스트 코드
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    async def on_signal(signal: NewsTradeSignal):
        print(f"\n{'='*60}")
        print(f"[매매 시그널] {signal.stock_name} ({signal.stock_code})")
        print(f"  행동: {signal.action}")
        print(f"  신뢰도: {signal.confidence:.0%}")
        print(f"  뉴스: {signal.news_title[:50]}...")
        print(f"  사유: {signal.reason}")
        print(f"  손절 조건: {signal.stop_loss_reason}")
        print(f"  익절 조건: {signal.take_profit_reason}")
        print(f"{'='*60}\n")

    async def main():
        trader = RealtimeNewsTrader(
            news_check_interval=30,
            on_trade_signal=on_signal,
            auto_execute=False
        )

        # 관심 종목 추가
        trader.add_to_watchlist("005930", "삼성전자")
        trader.add_to_watchlist("000660", "SK하이닉스")

        # 시장 컨텍스트 설정
        trader.set_market_context(
            "코스피 상승세, 반도체 섹터 강세",
            "BULLISH"
        )

        # 특정 뉴스 즉시 분석
        signals = await trader.analyze_news_now(
            "삼성전자, AI 반도체 대규모 수주 확정... 내년 실적 기대",
            stock_code="005930"
        )

        for signal in signals:
            print(f"분석 결과: {signal.action} (신뢰도: {signal.confidence:.0%})")
            print(f"사유: {signal.reason}")

    asyncio.run(main())
