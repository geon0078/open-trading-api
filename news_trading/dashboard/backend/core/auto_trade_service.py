# -*- coding: utf-8 -*-
"""
자동 매매 서비스

대시보드에서 AutoTrader를 비동기로 제어하는 서비스입니다.
"""

import sys
import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from collections import deque
from datetime import datetime, timedelta

# 경로 설정
sys.path.extend(['../../../modules', '../../../..', '../../..'])

logger = logging.getLogger(__name__)

# LLM 출력 콜백을 저장할 전역 변수
_llm_output_callback: Optional[Callable] = None


def set_llm_output_callback(callback: Callable):
    """LLM 출력 콜백 설정 (WebSocket 브로드캐스트용)"""
    global _llm_output_callback
    _llm_output_callback = callback
    logger.info("LLM 출력 콜백 설정됨")


class AutoTradeService:
    """
    자동 매매 서비스 (대시보드 연동)

    AutoTrader를 비동기 환경에서 제어합니다.
    """

    def __init__(self):
        self._auto_trader = None
        self._config = None
        self._is_running: bool = False
        self._trade_history: deque = deque(maxlen=100)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._background_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        # 뉴스 분석 상태
        self._current_mode: str = "INIT"
        self._last_news_analysis: Optional[Dict[str, Any]] = None
        self._next_scan_time: str = ""

    async def initialize(self, config_dict: Dict[str, Any] = None) -> bool:
        """
        AutoTrader 초기화

        Args:
            config_dict: 설정 딕셔너리

        Returns:
            bool: 성공 여부
        """
        logger.info("[Service] initialize 시작...")
        async with self._lock:
            try:
                logger.info("[Service] 모듈 import 시작...")
                from modules.auto_trader import AutoTrader, AutoTradeConfig
                logger.info("[Service] 모듈 import 완료")

                # 설정 생성
                if config_dict:
                    self._config = AutoTradeConfig(**config_dict)
                else:
                    self._config = AutoTradeConfig()
                logger.info("[Service] 설정 생성 완료")

                # AutoTrader 인스턴스만 생성 (초기화는 나중에)
                logger.info("[Service] AutoTrader 인스턴스 생성 시작...")
                self._auto_trader = AutoTrader(self._config)
                logger.info("[Service] AutoTrader 인스턴스 생성 완료 (lazy init)")
                return True

            except Exception as e:
                logger.error(f"AutoTradeService 초기화 실패: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False

    async def _ensure_trader_initialized(self) -> bool:
        """
        AutoTrader가 완전히 초기화되었는지 확인하고, 필요하면 초기화 수행

        Returns:
            bool: 초기화 성공 여부
        """
        if self._auto_trader is None:
            if not await self.initialize():
                return False

        # 이미 초기화됨
        if self._auto_trader._initialized:
            return True

        try:
            logger.info("AutoTrader 전체 초기화 시작 (KIS API + LLM)...")
            await self._emit_event("status_changed", {"status": "initializing", "message": "KIS API 인증 중..."})

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._auto_trader._ensure_initialized
            )

            # LLM 출력 콜백 설정 (WebSocket 브로드캐스트용)
            if _llm_output_callback and self._auto_trader._ensemble_analyzer:
                self._auto_trader._ensemble_analyzer.on_llm_output = _llm_output_callback
                logger.info("앙상블 분석기에 LLM 출력 콜백 설정됨")

            await self._emit_event("status_changed", {"status": "initialized", "message": "초기화 완료"})
            logger.info("AutoTrader 전체 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"AutoTrader 초기화 실패: {e}")
            await self._emit_event("error", {"error": f"초기화 실패: {str(e)}"})
            return False

    async def start_auto_trading(self, interval: int = 60) -> bool:
        """
        자동 매매 시작 (폴링 모드)

        백그라운드에서 초기화를 수행하고 즉시 응답합니다.

        Args:
            interval: 스캔 주기 (초)

        Returns:
            bool: 성공 여부
        """
        if self._is_running:
            logger.warning("자동 매매가 이미 실행 중입니다")
            return False

        # AutoTrader 인스턴스만 먼저 생성 (빠름)
        if self._auto_trader is None:
            if not await self.initialize():
                return False

        self._is_running = True

        # 이벤트 발행 - "시작 중" 상태
        await self._emit_event("status_changed", {
            "is_running": True,
            "status": "starting",
            "message": "자동 매매 시작 중... (초기화 진행)"
        })

        # 백그라운드 태스크 시작 (내부에서 초기화 수행)
        self._background_task = asyncio.create_task(
            self._polling_loop(interval)
        )

        logger.info(f"자동 매매 시작 (주기: {interval}초) - 백그라운드 초기화 진행")
        return True

    async def stop_auto_trading(self) -> bool:
        """
        자동 매매 중지

        Returns:
            bool: 성공 여부
        """
        if not self._is_running:
            return True

        self._is_running = False

        # 백그라운드 태스크 취소
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        # 이벤트 발행
        await self._emit_event("status_changed", {"is_running": False})

        logger.info("자동 매매 중지")
        return True

    async def _polling_loop(self, interval: int):
        """폴링 루프 (시간대별 자동 모드 전환)"""
        from datetime import time as dt_time

        # 먼저 전체 초기화 수행 (KIS API + LLM 모델)
        logger.info("[폴링 루프] 초기화 시작...")
        if not await self._ensure_trader_initialized():
            logger.error("[폴링 루프] 초기화 실패 - 자동 매매 중단")
            self._is_running = False
            await self._emit_event("error", {"error": "AutoTrader 초기화 실패"})
            return

        logger.info("[폴링 루프] 초기화 완료 - 자동 매매 시작")
        await self._emit_event("status_changed", {
            "is_running": True,
            "status": "running",
            "message": "자동 매매 실행 중"
        })

        market_start = dt_time(9, 0)
        market_end = dt_time(15, 20)
        scalping_end = dt_time(9, 30)

        while self._is_running:
            try:
                now = datetime.now()
                current_time = now.time()
                is_weekend = now.weekday() >= 5

                # 다음 스캔 시간 설정
                next_scan = now + timedelta(seconds=interval)
                self._next_scan_time = next_scan.strftime("%H:%M:%S")

                if is_weekend:
                    # 주말: 뉴스 분석 모드
                    self._current_mode = "NEWS"
                    logger.info("[자동매매] 주말 - 뉴스 분석 모드")
                    result = await self.run_news_analysis(max_news=30)
                    await self._emit_event("news_analysis_completed", result)

                elif current_time < market_start:
                    # 장 시작 전: 뉴스 분석 모드
                    self._current_mode = "NEWS"
                    logger.info(f"[자동매매] 장 시작 전 ({market_start}) - 뉴스 분석 모드")
                    result = await self.run_news_analysis(max_news=30)
                    await self._emit_event("news_analysis_completed", result)

                elif current_time > market_end:
                    # 장 마감 후: 뉴스 분석 모드
                    self._current_mode = "NEWS"
                    logger.info(f"[자동매매] 장 마감 후 ({market_end}) - 뉴스 분석 모드")
                    result = await self.run_news_analysis(max_news=30)
                    await self._emit_event("news_analysis_completed", result)

                elif current_time < scalping_end:
                    # 스캘핑 시간 (09:00 ~ 09:30): 스캘핑 매매
                    self._current_mode = "TRADING"
                    logger.info("[자동매매] 스캘핑 모드 (09:00~09:30) - 매매 + 분석")
                    results = await self.run_scalping_trade()
                    for result in results:
                        event_type = "trade_executed" if result.get('success') else "analysis_completed"
                        await self._emit_event(event_type, result)

                else:
                    # 정규 장 시간: 매매 + 분석 모드
                    self._current_mode = "TRADING"
                    logger.info("[자동매매] 정규장 - 매매 + 분석 모드")
                    results = await self.run_scan_and_trade()
                    for result in results:
                        event_type = "trade_executed" if result.get('success') else "analysis_completed"
                        await self._emit_event(event_type, result)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"폴링 루프 오류: {e}")
                await self._emit_event("error", {"error": str(e)})
                await asyncio.sleep(10)  # 오류 시 10초 대기

    async def analyze_single_stock(
        self,
        stock_code: str,
        stock_name: str,
        current_price: int,
        news_list: List[str] = None,
        skip_market_check: bool = False
    ) -> Dict[str, Any]:
        """
        단일 종목 분석 및 매매

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            current_price: 현재가
            news_list: 뉴스 리스트
            skip_market_check: 장 시간 체크 비활성화

        Returns:
            Dict: 분석 및 매매 결과
        """
        # 초기화 확인
        if not await self._ensure_trader_initialized():
            return {"success": False, "action": "ERROR", "reason": "초기화 실패"}

        try:
            # 동기 작업을 스레드에서 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._auto_trader.analyze_and_trade(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_price=current_price,
                    news_list=news_list,
                    check_market_hours=not skip_market_check
                )
            )

            # 결과를 dict로 변환
            result_dict = {
                "success": result.success,
                "action": result.action,
                "stock_code": result.stock_code,
                "stock_name": result.stock_name,
                "current_price": result.current_price,
                "ensemble_signal": result.ensemble_signal,
                "confidence": result.confidence,
                "consensus": result.consensus,
                "order_qty": result.order_qty,
                "order_price": result.order_price,
                "order_no": result.order_no,
                "reason": result.reason,
                "timestamp": result.timestamp,
                "technical_score": result.technical_score,
                "trend": result.trend,
            }

            # 히스토리 저장
            self._trade_history.append(result_dict)

            # 이벤트 발행
            event_type = "trade_executed" if result.success else "analysis_completed"
            await self._emit_event(event_type, result_dict)

            return result_dict

        except Exception as e:
            logger.error(f"단일 종목 분석 오류: {e}")
            return {
                "success": False,
                "action": "ERROR",
                "stock_code": stock_code,
                "stock_name": stock_name,
                "current_price": current_price,
                "ensemble_signal": "N/A",
                "confidence": 0,
                "consensus": 0,
                "reason": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    async def run_scan_and_trade(
        self,
        min_score: float = None,
        max_stocks: int = None,
        skip_market_check: bool = False
    ) -> List[Dict[str, Any]]:
        """
        급등 종목 스캔 후 매매

        Args:
            min_score: 최소 급등 점수
            max_stocks: 분석할 최대 종목 수
            skip_market_check: 장 시간 체크 비활성화

        Returns:
            List[Dict]: 분석 및 매매 결과 리스트
        """
        # 초기화 확인
        if not await self._ensure_trader_initialized():
            return []

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._auto_trader.run_scan_and_trade(
                    min_score=min_score,
                    max_stocks=max_stocks,
                    check_market_hours=not skip_market_check
                )
            )

            # 결과를 dict 리스트로 변환
            result_dicts = []
            for result in results:
                result_dict = {
                    "success": result.success,
                    "action": result.action,
                    "stock_code": result.stock_code,
                    "stock_name": result.stock_name,
                    "current_price": result.current_price,
                    "ensemble_signal": result.ensemble_signal,
                    "confidence": result.confidence,
                    "consensus": result.consensus,
                    "order_qty": result.order_qty,
                    "order_price": result.order_price,
                    "order_no": result.order_no,
                    "reason": result.reason,
                    "timestamp": result.timestamp,
                    "technical_score": result.technical_score,
                    "trend": result.trend,
                }
                result_dicts.append(result_dict)
                self._trade_history.append(result_dict)

            return result_dicts

        except Exception as e:
            logger.error(f"스캔 및 매매 오류: {e}")
            return []

    async def run_scalping_trade(
        self,
        min_score: float = None,
        max_stocks: int = None,
    ) -> List[Dict[str, Any]]:
        """
        스캘핑 매매 실행 (09:00 ~ 09:30)

        야간 뉴스를 분석하고 보조 모델이 시황을 판단한 뒤,
        메인 모델이 최종 스캘핑 결정을 내립니다.

        Args:
            min_score: 최소 급등 점수
            max_stocks: 분석할 최대 종목 수

        Returns:
            List[Dict]: 스캘핑 매매 결과 리스트
        """
        # 초기화 확인
        if not await self._ensure_trader_initialized():
            return []

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._auto_trader.run_scalping_trade(
                    min_score=min_score,
                    max_stocks=max_stocks,
                )
            )

            # 결과를 dict 리스트로 변환
            result_dicts = []
            for result in results:
                result_dict = {
                    "success": result.success,
                    "action": result.action,
                    "stock_code": result.stock_code,
                    "stock_name": result.stock_name,
                    "current_price": result.current_price,
                    "ensemble_signal": result.ensemble_signal,
                    "confidence": result.confidence,
                    "consensus": result.consensus,
                    "order_qty": result.order_qty,
                    "order_price": result.order_price,
                    "order_no": result.order_no,
                    "reason": result.reason,
                    "timestamp": result.timestamp,
                    "technical_score": result.technical_score,
                    "trend": result.trend,
                }
                result_dicts.append(result_dict)
                self._trade_history.append(result_dict)

                # 이벤트 발행
                event_type = "scalping_executed" if result.success else "scalping_analysis"
                await self._emit_event(event_type, result_dict)

            return result_dicts

        except Exception as e:
            logger.error(f"스캘핑 매매 오류: {e}")
            return []

    async def run_news_analysis(self, max_news: int = 30) -> Dict[str, Any]:
        """
        뉴스 분석 실행 (장 시작 전 분석용)

        Args:
            max_news: 분석할 최대 뉴스 수

        Returns:
            Dict: 뉴스 분석 결과
        """
        # 초기화 확인
        if not await self._ensure_trader_initialized():
            return {
                "news_count": 0,
                "market_sentiment": "NEUTRAL",
                "key_themes": [],
                "attention_stocks": [],
                "market_outlook": "분석 실패: 초기화 오류",
                "news_list": [],
            }

        try:
            self._current_mode = "NEWS"
            await self._emit_event("mode_changed", {"mode": "NEWS", "description": "뉴스 분석 모드"})

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._auto_trader.run_news_analysis(max_news=max_news)
            )

            # 결과 저장
            self._last_news_analysis = result
            self._last_news_analysis["analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 이벤트 발행
            await self._emit_event("news_analysis_completed", result)

            logger.info(f"뉴스 분석 완료: {result.get('news_count', 0)}건, 심리: {result.get('market_sentiment', 'N/A')}")
            return result

        except Exception as e:
            logger.error(f"뉴스 분석 오류: {e}")
            return {
                "news_count": 0,
                "market_sentiment": "NEUTRAL",
                "key_themes": [],
                "attention_stocks": [],
                "market_outlook": f"분석 오류: {str(e)}",
                "news_list": [],
            }

    async def get_trading_mode(self) -> Dict[str, Any]:
        """현재 트레이딩 모드 조회"""
        from datetime import time as dt_time

        now = datetime.now()
        current_time = now.time()
        is_weekend = now.weekday() >= 5

        market_start = dt_time(9, 0)
        market_end = dt_time(15, 20)

        if is_weekend:
            market_status = "주말"
            mode = "NEWS"
            mode_description = "주말 - 뉴스 분석 모드"
        elif current_time < market_start:
            market_status = "장 시작 전"
            mode = "NEWS"
            mode_description = "장 시작 전 - 뉴스 분석 모드"
        elif current_time > market_end:
            market_status = "장 마감 후"
            mode = "NEWS"
            mode_description = "장 마감 후 - 뉴스 분석 모드"
        else:
            market_status = "장중"
            mode = "TRADING"
            mode_description = "정규장 - 매매 + 분석 모드"

        return {
            "mode": mode,
            "mode_description": mode_description,
            "market_status": market_status,
            "next_scan_time": self._next_scan_time,
            "last_news_analysis": self._last_news_analysis,
            "is_running": self._is_running,
        }

    async def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        if self._auto_trader is None:
            return {
                "is_running": self._is_running,
                "initialized": False,
                "config": None,
                "today_trades": 0,
                "today_pnl": 0,
                "can_trade": False,
                "ensemble_models": [],
                "main_model": "",
                "current_mode": self._current_mode,
                "last_news_analysis": self._last_news_analysis,
            }

        try:
            loop = asyncio.get_event_loop()
            status = await loop.run_in_executor(
                None,
                self._auto_trader.get_status
            )

            return {
                "is_running": self._is_running,
                "initialized": status.get("initialized", False),
                "config": status.get("config", {}),
                "today_trades": status.get("daily_stats", {}).get("today_trades", 0),
                "today_pnl": status.get("daily_stats", {}).get("today_pnl", 0),
                "today_date": status.get("daily_stats", {}).get("date", ""),
                "can_trade": status.get("can_trade", False),
                "market_status": status.get("market_status", {}),
                "risk_status": status.get("risk_status", {}),
                "ensemble_models": status.get("ensemble_models", []),
                "main_model": status.get("main_model", ""),
                "current_mode": self._current_mode,
                "last_news_analysis": self._last_news_analysis,
            }

        except Exception as e:
            logger.error(f"상태 조회 오류: {e}")
            return {
                "is_running": self._is_running,
                "initialized": False,
                "error": str(e),
            }

    async def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """거래 히스토리 조회"""
        return list(self._trade_history)[-limit:]

    async def update_config(self, config_dict: Dict[str, Any]) -> bool:
        """설정 업데이트"""
        async with self._lock:
            was_running = self._is_running

            # 실행 중이면 중지
            if was_running:
                await self.stop_auto_trading()

            # 재초기화
            success = await self.initialize(config_dict)

            # 다시 시작
            if was_running and success:
                await self.start_auto_trading()

            return success

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """이벤트 발행"""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        await self._event_queue.put(event)

    async def get_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        SSE 이벤트 스트림

        Yields:
            Dict: 이벤트 데이터
        """
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=30.0
                )
                yield event
            except asyncio.TimeoutError:
                # 30초마다 heartbeat
                yield {
                    "event_type": "heartbeat",
                    "data": {"time": datetime.now().isoformat()},
                    "timestamp": datetime.now().isoformat(),
                }


# 전역 인스턴스
_auto_trade_service: Optional[AutoTradeService] = None


def get_auto_trade_service() -> AutoTradeService:
    """전역 AutoTradeService 인스턴스"""
    global _auto_trade_service
    if _auto_trade_service is None:
        _auto_trade_service = AutoTradeService()
    return _auto_trade_service
