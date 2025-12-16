# -*- coding: utf-8 -*-
"""
LLM 기반 실시간 포지션 관리 시스템

고정된 % 손절/익절이 아닌, LLM이 실시간으로 시장 상황을 분석하여
포지션 유지/청산을 판단합니다.

주요 기능:
1. 보유 포지션 실시간 모니터링
2. LLM이 현재 상황(뉴스, 기술적 지표, 가격 변동)을 분석
3. 청산 여부 및 타이밍 결정
4. 부분 익절/손절 전략 지원

사용 예시:
    >>> from modules.llm_position_manager import LLMPositionManager
    >>> manager = LLMPositionManager()
    >>> await manager.start_monitoring()
"""

import sys
import asyncio
import logging
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from collections import deque

sys.path.extend(['..', '.', '../..'])

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """보유 포지션 정보"""
    stock_code: str
    stock_name: str
    quantity: int
    avg_price: float
    current_price: float = 0
    entry_time: str = ""
    entry_reason: str = ""  # 매수 사유

    # 실시간 업데이트
    pnl: float = 0
    pnl_rate: float = 0
    max_profit_rate: float = 0  # 최대 수익률 (고점 대비)
    min_profit_rate: float = 0  # 최저 수익률 (저점 대비)

    # 관련 뉴스/이벤트
    recent_news: List[str] = field(default_factory=list)
    market_context: str = ""

    # LLM 분석 결과
    last_llm_analysis: Optional[Dict] = None
    last_analysis_time: str = ""


@dataclass
class ExitDecision:
    """LLM 청산 결정"""
    should_exit: bool
    exit_type: str  # "HOLD", "TAKE_PROFIT", "STOP_LOSS", "PARTIAL_EXIT", "URGENT_EXIT"
    exit_ratio: float = 1.0  # 청산 비율 (1.0 = 전량, 0.5 = 반량)
    confidence: float = 0
    reason: str = ""
    urgency: str = "NORMAL"  # "LOW", "NORMAL", "HIGH", "CRITICAL"
    suggested_price: int = 0  # 제안 청산가 (0이면 시장가)

    # LLM 분석 상세
    market_assessment: str = ""
    risk_factors: List[str] = field(default_factory=list)
    opportunity_factors: List[str] = field(default_factory=list)


class LLMPositionManager:
    """
    LLM 기반 실시간 포지션 관리

    고정 % 대신 LLM이 다음을 종합 분석하여 청산 결정:
    - 현재 손익률
    - 가격 추세 (상승/하락/횡보)
    - 거래량 변화
    - 실시간 뉴스
    - 시장 전체 분위기
    - 해당 종목의 기술적 지표
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        analysis_model: str = "deepseek-r1:8b",
        monitor_interval: int = 30,  # 30초마다 분석
        on_exit_decision: Optional[Callable] = None,
        on_position_update: Optional[Callable] = None
    ):
        self.ollama_url = ollama_url
        self.analysis_model = analysis_model
        self.monitor_interval = monitor_interval
        self.on_exit_decision = on_exit_decision
        self.on_position_update = on_position_update

        self.positions: Dict[str, Position] = {}
        self._running = False
        self._monitor_task = None
        self._decision_history: deque = deque(maxlen=100)

        # 뉴스 스트림 연결
        self._news_buffer: Dict[str, List[str]] = {}  # stock_code -> recent news

        # 기술적 분석기
        self._ohlcv_fetcher = None
        self._technical_analyzer = None

    def _ensure_components(self):
        """컴포넌트 초기화"""
        if self._ohlcv_fetcher is None:
            try:
                from .ohlcv_fetcher import OHLCVFetcher
                from .technical_indicators import TechnicalAnalyzer
                self._ohlcv_fetcher = OHLCVFetcher()
                self._technical_analyzer = TechnicalAnalyzer()
                logger.info("기술적 분석 컴포넌트 초기화 완료")
            except ImportError as e:
                logger.warning(f"기술적 분석 컴포넌트 로드 실패: {e}")

    async def add_position(
        self,
        stock_code: str,
        stock_name: str,
        quantity: int,
        avg_price: float,
        entry_reason: str = ""
    ):
        """포지션 추가"""
        position = Position(
            stock_code=stock_code,
            stock_name=stock_name,
            quantity=quantity,
            avg_price=avg_price,
            current_price=avg_price,
            entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            entry_reason=entry_reason
        )
        self.positions[stock_code] = position
        self._news_buffer[stock_code] = []
        logger.info(f"[포지션 추가] {stock_name}({stock_code}) {quantity}주 @ {avg_price:,.0f}원")

    async def remove_position(self, stock_code: str):
        """포지션 제거"""
        if stock_code in self.positions:
            pos = self.positions.pop(stock_code)
            logger.info(f"[포지션 제거] {pos.stock_name}({stock_code})")
        if stock_code in self._news_buffer:
            del self._news_buffer[stock_code]

    async def update_price(self, stock_code: str, current_price: float):
        """현재가 업데이트"""
        if stock_code not in self.positions:
            return

        pos = self.positions[stock_code]
        pos.current_price = current_price
        pos.pnl = (current_price - pos.avg_price) * pos.quantity
        pos.pnl_rate = (current_price - pos.avg_price) / pos.avg_price

        # 최대/최소 수익률 갱신
        if pos.pnl_rate > pos.max_profit_rate:
            pos.max_profit_rate = pos.pnl_rate
        if pos.pnl_rate < pos.min_profit_rate:
            pos.min_profit_rate = pos.pnl_rate

        if self.on_position_update:
            await self._safe_callback(self.on_position_update, pos)

    async def add_news(self, stock_code: str, news_title: str):
        """종목 관련 뉴스 추가"""
        if stock_code in self._news_buffer:
            self._news_buffer[stock_code].append(news_title)
            # 최근 10개만 유지
            self._news_buffer[stock_code] = self._news_buffer[stock_code][-10:]

            # 포지션에도 반영
            if stock_code in self.positions:
                self.positions[stock_code].recent_news = self._news_buffer[stock_code]

    async def set_market_context(self, context: str):
        """시장 전체 컨텍스트 설정"""
        for pos in self.positions.values():
            pos.market_context = context

    async def start_monitoring(self):
        """포지션 모니터링 시작"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"[LLM 포지션 관리] 모니터링 시작 (주기: {self.monitor_interval}초)")

    async def stop_monitoring(self):
        """포지션 모니터링 중지"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[LLM 포지션 관리] 모니터링 중지")

    async def _monitoring_loop(self):
        """메인 모니터링 루프"""
        self._ensure_components()

        while self._running:
            try:
                for stock_code, position in list(self.positions.items()):
                    # LLM 분석 실행
                    decision = await self._analyze_position(position)

                    if decision.should_exit:
                        logger.info(
                            f"[LLM 청산 결정] {position.stock_name}: {decision.exit_type} "
                            f"(신뢰도: {decision.confidence:.0%}, 긴급도: {decision.urgency})"
                        )
                        logger.info(f"  사유: {decision.reason}")

                        # 콜백 호출
                        if self.on_exit_decision:
                            await self._safe_callback(
                                self.on_exit_decision,
                                position,
                                decision
                            )

                    # 결정 기록
                    self._decision_history.append({
                        "stock_code": stock_code,
                        "decision": decision,
                        "timestamp": datetime.now().isoformat()
                    })

                await asyncio.sleep(self.monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[LLM 포지션 관리] 모니터링 오류: {e}")
                await asyncio.sleep(5)

    async def _analyze_position(self, position: Position) -> ExitDecision:
        """
        LLM으로 포지션 분석

        고정 % 대신 다양한 요소를 종합 분석:
        - 현재 손익 상황
        - 가격 추세
        - 실시간 뉴스
        - 시장 컨텍스트
        - 기술적 지표
        """
        try:
            # 1. 기술적 데이터 수집
            tech_data = await self._get_technical_data(position.stock_code)

            # 2. 분석 프롬프트 생성
            prompt = self._build_analysis_prompt(position, tech_data)

            # 3. LLM 분석
            result = await self._call_llm(prompt)

            # 4. 결과 파싱
            decision = self._parse_exit_decision(result, position)

            # 포지션에 분석 결과 저장
            position.last_llm_analysis = {
                "decision": decision.exit_type,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "raw_output": result
            }
            position.last_analysis_time = datetime.now().strftime("%H:%M:%S")

            return decision

        except Exception as e:
            logger.error(f"[LLM 분석 오류] {position.stock_name}: {e}")
            return ExitDecision(
                should_exit=False,
                exit_type="HOLD",
                reason=f"분석 오류: {str(e)}"
            )

    async def _get_technical_data(self, stock_code: str) -> Dict[str, Any]:
        """기술적 분석 데이터 조회"""
        if not self._ohlcv_fetcher or not self._technical_analyzer:
            return {}

        try:
            loop = asyncio.get_event_loop()

            # OHLCV 조회
            ohlcv = await loop.run_in_executor(
                None,
                lambda: self._ohlcv_fetcher.fetch(stock_code, period="D", count=20)
            )

            if ohlcv is None or ohlcv.empty:
                return {}

            # 기술적 분석
            analysis = await loop.run_in_executor(
                None,
                lambda: self._technical_analyzer.analyze(ohlcv)
            )

            return analysis

        except Exception as e:
            logger.warning(f"기술적 데이터 조회 실패 ({stock_code}): {e}")
            return {}

    def _build_analysis_prompt(self, position: Position, tech_data: Dict) -> str:
        """LLM 분석 프롬프트 생성"""

        # 보유 시간 계산
        try:
            entry_dt = datetime.strptime(position.entry_time, "%Y-%m-%d %H:%M:%S")
            hold_minutes = (datetime.now() - entry_dt).total_seconds() / 60
        except:
            hold_minutes = 0

        # 뉴스 요약
        news_text = "\n".join([f"  - {n}" for n in position.recent_news[-5:]]) if position.recent_news else "  (최근 뉴스 없음)"

        # 기술적 지표 요약
        tech_summary = ""
        if tech_data:
            tech_summary = f"""
기술적 지표:
  - RSI: {tech_data.get('rsi', 'N/A')}
  - MACD: {tech_data.get('macd_signal', 'N/A')}
  - 볼린저밴드: {tech_data.get('bb_position', 'N/A')}
  - 이동평균 추세: {tech_data.get('ma_trend', 'N/A')}
  - 거래량 변화: {tech_data.get('volume_change', 'N/A')}
"""

        prompt = f"""당신은 전문 트레이더입니다. 아래 보유 포지션을 분석하고 청산 여부를 결정하세요.

=== 포지션 정보 ===
종목: {position.stock_name} ({position.stock_code})
매수가: {position.avg_price:,.0f}원
현재가: {position.current_price:,.0f}원
수량: {position.quantity}주
손익: {position.pnl:+,.0f}원 ({position.pnl_rate:+.2%})
보유시간: {hold_minutes:.0f}분
매수사유: {position.entry_reason or '(기록 없음)'}

고점 대비 수익률: {position.max_profit_rate:+.2%}
저점 대비 수익률: {position.min_profit_rate:+.2%}
{tech_summary}
=== 최근 뉴스 ===
{news_text}

=== 시장 상황 ===
{position.market_context or '(정보 없음)'}

=== 분석 요청 ===
위 정보를 종합하여 이 포지션을 어떻게 해야 할지 결정하세요.

** 중요: 고정된 손절/익절 %가 아닌, 현재 상황을 종합적으로 판단하세요 **

고려할 요소:
1. 현재 손익 상태와 추세
2. 뉴스가 긍정적인지 부정적인지
3. 기술적 지표가 시사하는 방향
4. 추가 상승/하락 가능성
5. 리스크 대비 보상 (R/R)

반드시 아래 JSON 형식으로 응답하세요:
{{
    "decision": "HOLD/TAKE_PROFIT/STOP_LOSS/PARTIAL_EXIT/URGENT_EXIT 중 하나",
    "exit_ratio": 0.0~1.0 (청산 비율, HOLD면 0),
    "confidence": 0.0~1.0 (결정 확신도),
    "urgency": "LOW/NORMAL/HIGH/CRITICAL",
    "reason": "결정 이유 (2-3문장)",
    "market_assessment": "현재 시장/종목 상황 평가",
    "risk_factors": ["리스크 요인 1", "리스크 요인 2"],
    "opportunity_factors": ["기회 요인 1", "기회 요인 2"]
}}

decision 설명:
- HOLD: 보유 유지 (아직 청산 시점 아님)
- TAKE_PROFIT: 익절 (목표 달성 또는 상승 모멘텀 약화)
- STOP_LOSS: 손절 (추가 하락 예상 또는 악재)
- PARTIAL_EXIT: 부분 청산 (일부 익절 후 나머지 보유)
- URGENT_EXIT: 긴급 청산 (급락, 중대 악재 등)
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        import aiohttp

        payload = {
            "model": self.analysis_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 800,
                "num_ctx": 4096
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"LLM API 오류: {response.status}")

    def _parse_exit_decision(self, llm_output: str, position: Position) -> ExitDecision:
        """LLM 출력 파싱"""
        try:
            # <think> 태그 제거
            clean_output = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL)

            # JSON 추출
            json_match = re.search(r'\{[\s\S]*\}', clean_output)
            if not json_match:
                return ExitDecision(
                    should_exit=False,
                    exit_type="HOLD",
                    reason="JSON 파싱 실패"
                )

            data = json.loads(json_match.group())

            decision_type = data.get("decision", "HOLD").upper()
            exit_ratio = float(data.get("exit_ratio", 0))

            return ExitDecision(
                should_exit=decision_type != "HOLD",
                exit_type=decision_type,
                exit_ratio=exit_ratio if exit_ratio > 0 else 1.0,
                confidence=float(data.get("confidence", 0.5)),
                reason=data.get("reason", ""),
                urgency=data.get("urgency", "NORMAL"),
                market_assessment=data.get("market_assessment", ""),
                risk_factors=data.get("risk_factors", []),
                opportunity_factors=data.get("opportunity_factors", [])
            )

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 오류: {e}")
            return ExitDecision(
                should_exit=False,
                exit_type="HOLD",
                reason=f"파싱 오류: {str(e)}"
            )

    async def _safe_callback(self, callback, *args):
        """안전한 콜백 실행"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"콜백 실행 오류: {e}")

    async def analyze_now(self, stock_code: str) -> Optional[ExitDecision]:
        """특정 포지션 즉시 분석"""
        if stock_code not in self.positions:
            return None

        self._ensure_components()
        return await self._analyze_position(self.positions[stock_code])

    def get_all_positions(self) -> List[Position]:
        """모든 포지션 조회"""
        return list(self.positions.values())

    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        """결정 기록 조회"""
        return list(self._decision_history)[-limit:]


# 테스트 코드
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    async def on_exit(position: Position, decision: ExitDecision):
        print(f"\n{'='*50}")
        print(f"청산 결정: {position.stock_name}")
        print(f"유형: {decision.exit_type}")
        print(f"비율: {decision.exit_ratio:.0%}")
        print(f"사유: {decision.reason}")
        print(f"{'='*50}\n")

    async def main():
        manager = LLMPositionManager(
            monitor_interval=60,
            on_exit_decision=on_exit
        )

        # 테스트 포지션 추가
        await manager.add_position(
            stock_code="005930",
            stock_name="삼성전자",
            quantity=10,
            avg_price=55000,
            entry_reason="반도체 업황 개선 기대"
        )

        # 가격 업데이트
        await manager.update_price("005930", 56000)

        # 뉴스 추가
        await manager.add_news("005930", "삼성전자, AI 반도체 수주 확대")
        await manager.add_news("005930", "반도체 업황 내년 상반기 회복 전망")

        # 시장 컨텍스트 설정
        await manager.set_market_context("코스피 상승세, 반도체 섹터 강세")

        # 즉시 분석
        decision = await manager.analyze_now("005930")
        print(f"분석 결과: {decision}")

    asyncio.run(main())
