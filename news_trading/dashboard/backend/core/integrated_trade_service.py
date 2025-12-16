# -*- coding: utf-8 -*-
"""
통합 LLM 트레이딩 서비스

IntegratedTrader를 대시보드에서 비동기로 제어합니다.
- LLM 기반 동적 포지션 관리 (고정 % 손절/익절 아님)
- 실시간 뉴스 분석 + 기술적 분석 통합
- 시간대별 자동 모드 전환
"""

import sys
import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from collections import deque
from datetime import datetime
from dataclasses import asdict

# 경로 설정
sys.path.extend(['../../../modules', '../../../..', '../../..'])

logger = logging.getLogger(__name__)

# 콜백 저장용
_llm_output_callback: Optional[Callable] = None
_trade_signal_callback: Optional[Callable] = None
_exit_decision_callback: Optional[Callable] = None


def set_llm_output_callback(callback: Callable):
    """LLM 출력 콜백 설정"""
    global _llm_output_callback
    _llm_output_callback = callback


def set_trade_signal_callback(callback: Callable):
    """매매 시그널 콜백 설정"""
    global _trade_signal_callback
    _trade_signal_callback = callback


def set_exit_decision_callback(callback: Callable):
    """청산 결정 콜백 설정"""
    global _exit_decision_callback
    _exit_decision_callback = callback


class IntegratedTradeService:
    """
    통합 LLM 트레이딩 서비스

    특징:
    1. LLM이 동적으로 손절/익절 조건 결정 (고정 % 아님)
    2. 실시간 뉴스 + 기술적 분석 통합
    3. 시간대별 자동 모드 전환
    """

    def __init__(self):
        self._trader = None
        self._is_running = False
        self._lock = asyncio.Lock()

        # 이벤트 큐
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # 히스토리
        self._trade_signals: deque = deque(maxlen=100)
        self._exit_decisions: deque = deque(maxlen=100)
        self._position_updates: deque = deque(maxlen=50)

        # 현재 상태
        self._current_mode: str = "INIT"
        self._market_sentiment: str = "NEUTRAL"
        self._market_context: str = ""

    async def initialize(self, config_dict: Dict[str, Any] = None) -> bool:
        """
        IntegratedTrader 초기화
        """
        async with self._lock:
            try:
                from modules.integrated_trader import IntegratedTrader, TradingConfig

                # 설정 생성
                if config_dict:
                    config = TradingConfig(**config_dict)
                else:
                    config = TradingConfig(
                        auto_execute=False,  # 기본: 자동 실행 비활성화
                        min_confidence=0.7,
                        news_check_interval=30,
                        position_check_interval=30,
                        surge_scan_interval=60
                    )

                # IntegratedTrader 생성
                self._trader = IntegratedTrader(config)

                # 콜백 설정
                self._trader.on_trade_signal = self._handle_trade_signal
                self._trader.on_exit_decision = self._handle_exit_decision
                self._trader.on_mode_change = self._handle_mode_change
                self._trader.on_position_update = self._handle_position_update

                logger.info("IntegratedTradeService 초기화 완료")
                return True

            except Exception as e:
                logger.error(f"IntegratedTradeService 초기화 실패: {e}")
                import traceback
                traceback.print_exc()
                return False

    async def start(self) -> bool:
        """통합 트레이딩 시작"""
        if self._is_running:
            logger.warning("이미 실행 중입니다")
            return False

        if self._trader is None:
            if not await self.initialize():
                return False

        try:
            await self._trader.start()
            self._is_running = True

            await self._emit_event("status_changed", {"is_running": True})
            logger.info("IntegratedTradeService 시작")
            return True

        except Exception as e:
            logger.error(f"시작 실패: {e}")
            return False

    async def stop(self) -> bool:
        """통합 트레이딩 중지"""
        if not self._is_running:
            return True

        try:
            if self._trader:
                await self._trader.stop()

            self._is_running = False
            await self._emit_event("status_changed", {"is_running": False})
            logger.info("IntegratedTradeService 중지")
            return True

        except Exception as e:
            logger.error(f"중지 실패: {e}")
            return False

    async def _handle_trade_signal(self, signal):
        """매매 시그널 처리"""
        try:
            signal_dict = {
                "stock_code": signal.stock_code,
                "stock_name": signal.stock_name,
                "action": signal.action,
                "confidence": signal.confidence,
                "urgency": signal.urgency,
                "reason": signal.reason,
                "news_title": signal.news_title,
                "news_sentiment": signal.news_sentiment,
                "technical_score": signal.technical_score,
                "trend": signal.trend,
                "support_price": signal.support_price,
                "resistance_price": signal.resistance_price,
                "suggested_price": signal.suggested_price,
                "stop_loss_condition": signal.stop_loss_reason,  # LLM이 결정한 손절 조건 (텍스트)
                "take_profit_condition": signal.take_profit_reason,  # LLM이 결정한 익절 조건 (텍스트)
                "timestamp": signal.timestamp
            }

            self._trade_signals.append(signal_dict)

            # 이벤트 발행
            await self._emit_event("trade_signal", signal_dict)

            # 외부 콜백 호출
            if _trade_signal_callback:
                await self._safe_callback(_trade_signal_callback, signal_dict)

            logger.info(f"[매매 시그널] {signal.stock_name}: {signal.action} (신뢰도: {signal.confidence:.0%})")

        except Exception as e:
            logger.error(f"매매 시그널 처리 오류: {e}")

    async def _handle_exit_decision(self, position, decision):
        """청산 결정 처리"""
        try:
            decision_dict = {
                "stock_code": position.stock_code,
                "stock_name": position.stock_name,
                "current_pnl": position.pnl,
                "current_pnl_rate": position.pnl_rate,
                "exit_type": decision.exit_type,
                "exit_ratio": decision.exit_ratio,
                "should_exit": decision.should_exit,
                "reason": decision.reason,  # LLM이 결정한 청산 사유
                "suggested_price": decision.suggested_price,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self._exit_decisions.append(decision_dict)

            # 이벤트 발행
            await self._emit_event("exit_decision", decision_dict)

            # 외부 콜백 호출
            if _exit_decision_callback:
                await self._safe_callback(_exit_decision_callback, decision_dict)

            logger.info(f"[청산 결정] {position.stock_name}: {decision.exit_type} - {decision.reason}")

        except Exception as e:
            logger.error(f"청산 결정 처리 오류: {e}")

    async def _handle_mode_change(self, mode_info: Dict):
        """모드 변경 처리"""
        try:
            self._current_mode = mode_info.get("new_mode", "INIT")

            await self._emit_event("mode_changed", mode_info)
            logger.info(f"[모드 변경] {mode_info.get('old_mode')} → {mode_info.get('new_mode')}")

        except Exception as e:
            logger.error(f"모드 변경 처리 오류: {e}")

    async def _handle_position_update(self, position):
        """포지션 업데이트 처리"""
        try:
            update_dict = {
                "stock_code": position.stock_code,
                "stock_name": position.stock_name,
                "quantity": position.quantity,
                "avg_price": position.avg_price,
                "current_price": position.current_price,
                "pnl": position.pnl,
                "pnl_rate": position.pnl_rate,
                "entry_reason": position.entry_reason,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self._position_updates.append(update_dict)

            await self._emit_event("position_update", update_dict)

        except Exception as e:
            logger.error(f"포지션 업데이트 처리 오류: {e}")

    async def analyze_news_now(self, news_title: str, stock_code: str = None) -> Dict[str, Any]:
        """
        뉴스 즉시 분석 (LLM 기반)

        Returns:
            Dict: 분석 결과 및 매매 시그널
        """
        if self._trader is None:
            if not await self.initialize():
                return {"error": "초기화 실패"}

        try:
            # 뉴스 트레이더가 없으면 초기화
            if self._trader._news_trader is None:
                await self._trader.initialize()

            signals = await self._trader._news_trader.analyze_news_now(news_title, stock_code)

            results = []
            for signal in signals:
                signal_dict = {
                    "stock_code": signal.stock_code,
                    "stock_name": signal.stock_name,
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "urgency": signal.urgency,
                    "reason": signal.reason,
                    "news_title": signal.news_title,
                    "technical_score": signal.technical_score,
                    "stop_loss_condition": signal.stop_loss_reason,
                    "take_profit_condition": signal.take_profit_reason,
                    "timestamp": signal.timestamp
                }
                results.append(signal_dict)

            return {
                "news_title": news_title,
                "stock_code": stock_code,
                "signals": results,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            logger.error(f"뉴스 분석 오류: {e}")
            return {"error": str(e)}

    async def add_to_watchlist(self, stock_code: str, stock_name: str) -> bool:
        """관심 종목 추가"""
        if self._trader and self._trader._news_trader:
            self._trader._news_trader.add_to_watchlist(stock_code, stock_name)
            return True
        return False

    async def remove_from_watchlist(self, stock_code: str) -> bool:
        """관심 종목 제거"""
        if self._trader and self._trader._news_trader:
            self._trader._news_trader.remove_from_watchlist(stock_code)
            return True
        return False

    async def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        if self._trader is None:
            return {
                "is_running": False,
                "initialized": False,
                "current_mode": "INIT",
                "positions_count": 0,
                "positions": [],
                "market_sentiment": "NEUTRAL"
            }

        try:
            status = self._trader.get_status()
            return {
                "is_running": self._is_running,
                "initialized": True,
                "current_mode": status.get("current_mode", "INIT"),
                "market_sentiment": status.get("market_sentiment", "NEUTRAL"),
                "market_context": status.get("market_context", ""),
                "today_trades": status.get("today_trades", 0),
                "today_pnl": status.get("today_pnl", 0),
                "positions_count": status.get("positions_count", 0),
                "positions": status.get("positions", [])
            }
        except Exception as e:
            logger.error(f"상태 조회 오류: {e}")
            return {
                "is_running": self._is_running,
                "initialized": True,
                "error": str(e)
            }

    async def get_trade_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """매매 시그널 히스토리"""
        return list(self._trade_signals)[-limit:]

    async def get_exit_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """청산 결정 히스토리"""
        return list(self._exit_decisions)[-limit:]

    async def get_positions(self) -> List[Dict[str, Any]]:
        """현재 포지션 조회"""
        if self._trader and self._trader._position_manager:
            return [
                {
                    "stock_code": p.stock_code,
                    "stock_name": p.stock_name,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "pnl": p.pnl,
                    "pnl_rate": p.pnl_rate,
                    "entry_reason": p.entry_reason
                }
                for p in self._trader._position_manager.get_all_positions()
            ]
        return []

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """이벤트 발행"""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self._event_queue.put(event)

    async def get_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """SSE 이벤트 스트림"""
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=30.0
                )
                yield event
            except asyncio.TimeoutError:
                yield {
                    "event_type": "heartbeat",
                    "data": {"time": datetime.now().isoformat()},
                    "timestamp": datetime.now().isoformat()
                }

    async def _safe_callback(self, callback, *args):
        """안전한 콜백 실행"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"콜백 실행 오류: {e}")


# 전역 인스턴스
_integrated_trade_service: Optional[IntegratedTradeService] = None


def get_integrated_trade_service() -> IntegratedTradeService:
    """전역 IntegratedTradeService 인스턴스"""
    global _integrated_trade_service
    if _integrated_trade_service is None:
        _integrated_trade_service = IntegratedTradeService()
    return _integrated_trade_service
