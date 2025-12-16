# -*- coding: utf-8 -*-
"""
통합 LLM 기반 자동 매매 시스템

모든 구성요소를 통합하여 완전한 자동 매매 시스템을 제공합니다:
1. 실시간 뉴스 분석 및 매매 결정
2. LLM 기반 포지션 관리 (동적 손절/익절)
3. 기술적 분석 연계
4. 시간대별 자동 모드 전환

사용 예시:
    >>> from modules.integrated_trader import IntegratedTrader
    >>> trader = IntegratedTrader()
    >>> await trader.start()
"""

import sys
import asyncio
import logging
from datetime import datetime, time as dt_time
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Callable

sys.path.extend(['..', '.', '../..'])

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """통합 트레이딩 설정"""
    # LLM 설정
    ollama_url: str = "http://localhost:11434"
    analysis_model: str = "deepseek-r1:8b"
    secondary_model: str = "qwen3:8b"

    # 거래 설정
    max_order_amount: int = 100000  # 1회 최대 주문 금액
    max_positions: int = 5  # 최대 동시 보유 종목 수
    min_confidence: float = 0.7  # 최소 신뢰도

    # 모니터링 설정
    news_check_interval: int = 30  # 뉴스 체크 주기 (초)
    position_check_interval: int = 30  # 포지션 체크 주기 (초)
    surge_scan_interval: int = 60  # 급등 종목 스캔 주기 (초)

    # 시간 설정
    market_start: str = "09:00"
    market_end: str = "15:20"
    scalping_end: str = "09:30"

    # 자동 실행
    auto_execute: bool = False  # 자동 주문 실행


class IntegratedTrader:
    """
    통합 LLM 기반 자동 매매 시스템

    시간대별 동작:
    - 장 시작 전: 뉴스 분석, 시장 심리 파악
    - 09:00~09:30: 스캘핑 모드 (야간 뉴스 기반 빠른 매매)
    - 09:30~15:20: 정규장 모드 (실시간 뉴스 + 기술적 분석)
    - 장 마감 후: 결산, 다음 거래일 준비
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()

        self._running = False
        self._current_mode = "INIT"

        # 핵심 컴포넌트
        self._news_trader = None
        self._position_manager = None
        self._surge_detector = None
        self._order_executor = None

        # 백그라운드 태스크
        self._main_task = None
        self._news_task = None
        self._position_task = None

        # 상태 정보
        self._market_context = ""
        self._market_sentiment = "NEUTRAL"
        self._today_trades = 0
        self._today_pnl = 0

        # 콜백
        self.on_trade_signal: Optional[Callable] = None
        self.on_exit_decision: Optional[Callable] = None
        self.on_mode_change: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None

    async def initialize(self):
        """컴포넌트 초기화"""
        logger.info("=" * 50)
        logger.info("[통합 트레이더] 초기화 시작")
        logger.info("=" * 50)

        try:
            # 1. 실시간 뉴스 트레이더
            from .realtime_news_trader import RealtimeNewsTrader
            self._news_trader = RealtimeNewsTrader(
                ollama_url=self.config.ollama_url,
                analysis_model=self.config.analysis_model,
                secondary_model=self.config.secondary_model,
                news_check_interval=self.config.news_check_interval,
                on_trade_signal=self._handle_trade_signal,
                auto_execute=False  # 자동 실행은 여기서 제어
            )
            logger.info("  - RealtimeNewsTrader 초기화 완료")

            # 2. LLM 포지션 관리자
            from .llm_position_manager import LLMPositionManager
            self._position_manager = LLMPositionManager(
                ollama_url=self.config.ollama_url,
                analysis_model=self.config.analysis_model,
                monitor_interval=self.config.position_check_interval,
                on_exit_decision=self._handle_exit_decision,
                on_position_update=self._handle_position_update
            )
            logger.info("  - LLMPositionManager 초기화 완료")

            # 3. 급등 탐지기
            try:
                from .surge_detector import SurgeDetector
                self._surge_detector = SurgeDetector()
                logger.info("  - SurgeDetector 초기화 완료")
            except ImportError:
                logger.warning("  - SurgeDetector 로드 실패")

            # 4. 주문 실행기
            if self.config.auto_execute:
                try:
                    from .order_executor import OrderExecutor
                    self._order_executor = OrderExecutor()
                    logger.info("  - OrderExecutor 초기화 완료")
                except ImportError:
                    logger.warning("  - OrderExecutor 로드 실패")

            logger.info("[통합 트레이더] 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"[통합 트레이더] 초기화 실패: {e}")
            return False

    async def start(self):
        """트레이딩 시작"""
        if self._running:
            return

        if not await self.initialize():
            return

        self._running = True

        # 메인 루프 시작
        self._main_task = asyncio.create_task(self._main_loop())

        logger.info("[통합 트레이더] 시작")

    async def stop(self):
        """트레이딩 중지"""
        self._running = False

        # 뉴스 트레이더 중지
        if self._news_trader:
            await self._news_trader.stop()

        # 포지션 관리자 중지
        if self._position_manager:
            await self._position_manager.stop_monitoring()

        # 메인 태스크 취소
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        logger.info("[통합 트레이더] 중지")

    async def _main_loop(self):
        """메인 트레이딩 루프"""
        while self._running:
            try:
                now = datetime.now()
                current_time = now.time()
                is_weekend = now.weekday() >= 5

                market_start = dt_time(*map(int, self.config.market_start.split(":")))
                market_end = dt_time(*map(int, self.config.market_end.split(":")))
                scalping_end = dt_time(*map(int, self.config.scalping_end.split(":")))

                # 모드 결정
                if is_weekend:
                    new_mode = "WEEKEND"
                elif current_time < market_start:
                    new_mode = "PRE_MARKET"
                elif current_time < scalping_end:
                    new_mode = "SCALPING"
                elif current_time < market_end:
                    new_mode = "REGULAR"
                else:
                    new_mode = "POST_MARKET"

                # 모드 변경 시 처리
                if new_mode != self._current_mode:
                    await self._handle_mode_change(new_mode)

                # 모드별 동작
                if new_mode == "PRE_MARKET":
                    await self._pre_market_routine()
                elif new_mode == "SCALPING":
                    await self._scalping_routine()
                elif new_mode == "REGULAR":
                    await self._regular_routine()
                elif new_mode == "POST_MARKET":
                    await self._post_market_routine()

                # 대기
                await asyncio.sleep(self.config.surge_scan_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[메인 루프 오류] {e}")
                await asyncio.sleep(10)

    async def _handle_mode_change(self, new_mode: str):
        """모드 변경 처리"""
        old_mode = self._current_mode
        self._current_mode = new_mode

        logger.info(f"[모드 변경] {old_mode} → {new_mode}")

        # 콜백 호출
        if self.on_mode_change:
            await self._safe_callback(
                self.on_mode_change,
                {"old_mode": old_mode, "new_mode": new_mode}
            )

        # 모드별 초기화
        if new_mode == "PRE_MARKET":
            # 장 시작 전: 뉴스 분석 준비
            pass

        elif new_mode == "SCALPING":
            # 스캘핑 모드: 뉴스 트레이더 시작
            if self._news_trader:
                await self._news_trader.start()

            # 포지션 관리자 시작
            if self._position_manager:
                await self._position_manager.start_monitoring()

        elif new_mode == "REGULAR":
            # 정규장: 뉴스 + 포지션 모니터링 지속
            if self._news_trader and not self._news_trader._running:
                await self._news_trader.start()

        elif new_mode == "POST_MARKET":
            # 장 마감 후: 모니터링 중지
            if self._news_trader:
                await self._news_trader.stop()

    async def _pre_market_routine(self):
        """장 시작 전 루틴"""
        logger.info("[장 시작 전] 뉴스 분석 실행")

        try:
            # 뉴스 분석
            if self._news_trader and self._news_trader._news_collector:
                loop = asyncio.get_event_loop()
                news_df = await loop.run_in_executor(
                    None,
                    lambda: self._news_trader._news_collector.collect(
                        stock_codes=None,
                        filter_duplicates=False,
                        max_depth=5
                    )
                )

                if news_df is not None and not news_df.empty:
                    # LLM으로 시장 분석
                    await self._analyze_market_sentiment(news_df)

        except Exception as e:
            logger.error(f"[장 시작 전 루틴 오류] {e}")

    async def _scalping_routine(self):
        """스캘핑 루틴 (09:00~09:30)"""
        logger.info("[스캘핑] 급등 종목 스캔")

        try:
            # 급등 종목 스캔
            if self._surge_detector:
                loop = asyncio.get_event_loop()
                surge_stocks = await loop.run_in_executor(
                    None,
                    lambda: self._surge_detector.scan_surge_stocks(min_score=50)
                )

                if surge_stocks:
                    logger.info(f"[스캘핑] {len(surge_stocks)}개 급등 종목 발견")

                    # 상위 종목 분석
                    for stock in surge_stocks[:3]:
                        if hasattr(stock, 'code'):
                            code, name = stock.code, stock.name
                        else:
                            code, name = stock.get('code', ''), stock.get('name', '')

                        if code and self._news_trader:
                            # 뉴스 + 기술적 분석 실행
                            signals = await self._news_trader.analyze_news_now(
                                f"{name} 급등 중 - 기술적 분석 요청",
                                stock_code=code
                            )

                            for signal in signals:
                                await self._handle_trade_signal(signal)

        except Exception as e:
            logger.error(f"[스캘핑 루틴 오류] {e}")

    async def _regular_routine(self):
        """정규장 루틴 (09:30~15:20)"""
        # 뉴스 트레이더가 자동으로 모니터링 중
        # 여기서는 추가적인 급등 종목 스캔만 수행

        try:
            if self._surge_detector:
                loop = asyncio.get_event_loop()
                surge_stocks = await loop.run_in_executor(
                    None,
                    lambda: self._surge_detector.scan_surge_stocks(min_score=60)
                )

                if surge_stocks:
                    logger.info(f"[정규장] {len(surge_stocks)}개 급등 종목 발견")

                    # 관심 종목에 추가
                    if self._news_trader:
                        for stock in surge_stocks[:5]:
                            if hasattr(stock, 'code'):
                                code, name = stock.code, stock.name
                            else:
                                code, name = stock.get('code', ''), stock.get('name', '')

                            if code:
                                self._news_trader.add_to_watchlist(code, name)

        except Exception as e:
            logger.error(f"[정규장 루틴 오류] {e}")

    async def _post_market_routine(self):
        """장 마감 후 루틴"""
        logger.info("[장 마감 후] 결산")

        # 결산 로직 (추후 구현)
        pass

    async def _analyze_market_sentiment(self, news_df):
        """시장 심리 분석"""
        try:
            # 뉴스 제목 추출
            title_col = None
            for col in ['hts_pbnt_titl_cntt', 'titl', 'news_titl', 'title']:
                if col in news_df.columns:
                    title_col = col
                    break

            if not title_col:
                return

            news_titles = news_df[title_col].tolist()[:30]
            news_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(news_titles)])

            prompt = f"""오늘의 뉴스를 분석하고 시장 심리를 판단하세요.

뉴스 헤드라인:
{news_text}

JSON 형식으로 응답:
{{
    "sentiment": "BULLISH/BEARISH/NEUTRAL",
    "key_themes": ["테마1", "테마2"],
    "outlook": "시장 전망 요약"
}}
"""

            if self._news_trader:
                result = await self._news_trader._call_llm(prompt)
                import re
                import json

                clean_result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                json_match = re.search(r'\{[\s\S]*\}', clean_result)

                if json_match:
                    data = json.loads(json_match.group())
                    self._market_sentiment = data.get('sentiment', 'NEUTRAL')
                    self._market_context = data.get('outlook', '')

                    # 뉴스 트레이더에 컨텍스트 설정
                    self._news_trader.set_market_context(
                        self._market_context,
                        self._market_sentiment
                    )

                    logger.info(f"[시장 분석] 심리: {self._market_sentiment}")
                    logger.info(f"[시장 분석] 전망: {self._market_context}")

        except Exception as e:
            logger.error(f"시장 심리 분석 오류: {e}")

    async def _handle_trade_signal(self, signal):
        """매매 시그널 처리"""
        from .realtime_news_trader import NewsTradeSignal

        if not isinstance(signal, NewsTradeSignal):
            return

        logger.info(f"[매매 시그널] {signal.stock_name}: {signal.action}")

        # 콜백 호출
        if self.on_trade_signal:
            await self._safe_callback(self.on_trade_signal, signal)

        # 신뢰도 체크
        if signal.confidence < self.config.min_confidence:
            logger.info(f"  → 신뢰도 미달 ({signal.confidence:.0%} < {self.config.min_confidence:.0%})")
            return

        # 자동 실행
        if self.config.auto_execute and self._order_executor:
            if signal.action == "BUY":
                await self._execute_buy(signal)
            elif signal.action == "SELL":
                await self._execute_sell(signal)

    async def _handle_exit_decision(self, position, decision):
        """청산 결정 처리"""
        logger.info(f"[청산 결정] {position.stock_name}: {decision.exit_type}")

        # 콜백 호출
        if self.on_exit_decision:
            await self._safe_callback(self.on_exit_decision, position, decision)

        # 자동 실행
        if self.config.auto_execute and decision.should_exit and self._order_executor:
            await self._execute_exit(position, decision)

    async def _handle_position_update(self, position):
        """포지션 업데이트 처리"""
        if self.on_position_update:
            await self._safe_callback(self.on_position_update, position)

    async def _execute_buy(self, signal):
        """매수 실행"""
        if not self._order_executor:
            return

        try:
            # 포지션 수 체크
            if len(self._position_manager.positions) >= self.config.max_positions:
                logger.warning("최대 포지션 수 초과")
                return

            # 주문 금액 계산
            order_amount = min(self.config.max_order_amount, 100000)

            # 매수 주문
            result = self._order_executor.buy(
                stock_code=signal.stock_code,
                price=signal.suggested_price or 0,
                amount=order_amount
            )

            if result.success:
                # 포지션 관리자에 추가
                await self._position_manager.add_position(
                    stock_code=signal.stock_code,
                    stock_name=signal.stock_name,
                    quantity=result.order_qty,
                    avg_price=result.order_price,
                    entry_reason=f"뉴스: {signal.news_title[:30]}... / {signal.reason}"
                )

                self._today_trades += 1
                logger.info(f"[매수 완료] {signal.stock_name} {result.order_qty}주 @ {result.order_price:,}원")

        except Exception as e:
            logger.error(f"매수 실행 오류: {e}")

    async def _execute_sell(self, signal):
        """매도 실행"""
        if not self._order_executor:
            return

        try:
            # 보유 포지션 확인
            if signal.stock_code not in self._position_manager.positions:
                logger.warning(f"{signal.stock_name} 보유 포지션 없음")
                return

            position = self._position_manager.positions[signal.stock_code]

            # 매도 주문
            result = self._order_executor.sell(
                stock_code=signal.stock_code,
                price=signal.suggested_price or 0,
                quantity=position.quantity
            )

            if result.success:
                # 손익 계산
                pnl = (result.order_price - position.avg_price) * position.quantity
                self._today_pnl += pnl

                # 포지션 제거
                await self._position_manager.remove_position(signal.stock_code)

                self._today_trades += 1
                logger.info(f"[매도 완료] {signal.stock_name} {position.quantity}주 @ {result.order_price:,}원 (손익: {pnl:+,}원)")

        except Exception as e:
            logger.error(f"매도 실행 오류: {e}")

    async def _execute_exit(self, position, decision):
        """청산 실행"""
        if not self._order_executor:
            return

        try:
            # 청산 수량 계산
            exit_qty = int(position.quantity * decision.exit_ratio)
            if exit_qty <= 0:
                return

            # 매도 주문
            result = self._order_executor.sell(
                stock_code=position.stock_code,
                price=decision.suggested_price or 0,
                quantity=exit_qty
            )

            if result.success:
                # 손익 계산
                pnl = (result.order_price - position.avg_price) * exit_qty
                self._today_pnl += pnl

                # 전량 청산이면 포지션 제거
                if decision.exit_ratio >= 1.0:
                    await self._position_manager.remove_position(position.stock_code)
                else:
                    # 부분 청산
                    position.quantity -= exit_qty

                self._today_trades += 1
                logger.info(
                    f"[{decision.exit_type}] {position.stock_name} {exit_qty}주 @ {result.order_price:,}원 "
                    f"(손익: {pnl:+,}원, 사유: {decision.reason})"
                )

        except Exception as e:
            logger.error(f"청산 실행 오류: {e}")

    async def _safe_callback(self, callback, *args):
        """안전한 콜백 실행"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"콜백 실행 오류: {e}")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "running": self._running,
            "current_mode": self._current_mode,
            "market_sentiment": self._market_sentiment,
            "market_context": self._market_context,
            "today_trades": self._today_trades,
            "today_pnl": self._today_pnl,
            "positions_count": len(self._position_manager.positions) if self._position_manager else 0,
            "positions": [
                {
                    "stock_code": p.stock_code,
                    "stock_name": p.stock_name,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "pnl": p.pnl,
                    "pnl_rate": p.pnl_rate
                }
                for p in self._position_manager.get_all_positions()
            ] if self._position_manager else []
        }


# 테스트 코드
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    async def on_signal(signal):
        print(f"\n[시그널] {signal.stock_name}: {signal.action}")
        print(f"  사유: {signal.reason}")
        print(f"  손절 조건: {signal.stop_loss_reason}")
        print(f"  익절 조건: {signal.take_profit_reason}")

    async def on_exit(position, decision):
        print(f"\n[청산] {position.stock_name}: {decision.exit_type}")
        print(f"  사유: {decision.reason}")

    async def main():
        config = TradingConfig(
            auto_execute=False,
            min_confidence=0.6
        )

        trader = IntegratedTrader(config)
        trader.on_trade_signal = on_signal
        trader.on_exit_decision = on_exit

        await trader.start()

        # 1분 동안 실행
        await asyncio.sleep(60)

        status = trader.get_status()
        print(f"\n상태: {status}")

        await trader.stop()

    asyncio.run(main())
