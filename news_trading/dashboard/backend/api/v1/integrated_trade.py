# -*- coding: utf-8 -*-
"""
통합 LLM 트레이딩 API

LLM 기반 동적 포지션 관리 및 실시간 뉴스 트레이딩 API입니다.

특징:
- 고정 % 손절/익절이 아닌 LLM의 동적 판단
- 실시간 뉴스 + 기술적 분석 통합
- 시간대별 자동 모드 전환
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging

from core.integrated_trade_service import get_integrated_trade_service

logger = logging.getLogger(__name__)

router = APIRouter()


# === Pydantic Models ===

class TradeSignalResponse(BaseModel):
    """매매 시그널 응답"""
    stock_code: str
    stock_name: str
    action: str  # BUY, SELL, HOLD, WATCH
    confidence: float
    urgency: str  # LOW, NORMAL, HIGH, CRITICAL
    reason: str
    news_title: str = ""
    news_sentiment: str = ""
    technical_score: float = 0
    trend: str = ""
    support_price: int = 0
    resistance_price: int = 0
    suggested_price: int = 0
    stop_loss_condition: str = ""  # LLM이 결정한 손절 조건 (텍스트)
    take_profit_condition: str = ""  # LLM이 결정한 익절 조건 (텍스트)
    timestamp: str = ""


class ExitDecisionResponse(BaseModel):
    """청산 결정 응답"""
    stock_code: str
    stock_name: str
    current_pnl: float
    current_pnl_rate: float
    exit_type: str  # HOLD, TAKE_PROFIT, STOP_LOSS, PARTIAL_EXIT, URGENT_EXIT
    exit_ratio: float
    should_exit: bool
    reason: str  # LLM이 결정한 청산 사유
    suggested_price: int = 0
    timestamp: str = ""


class PositionResponse(BaseModel):
    """포지션 응답"""
    stock_code: str
    stock_name: str
    quantity: int
    avg_price: int
    current_price: int
    pnl: float
    pnl_rate: float
    entry_reason: str = ""


class StatusResponse(BaseModel):
    """상태 응답"""
    is_running: bool
    initialized: bool
    current_mode: str
    market_sentiment: str = "NEUTRAL"
    market_context: str = ""
    today_trades: int = 0
    today_pnl: float = 0
    positions_count: int = 0
    positions: List[PositionResponse] = []


class NewsAnalyzeRequest(BaseModel):
    """뉴스 분석 요청"""
    news_title: str
    stock_code: Optional[str] = None


class NewsAnalyzeResponse(BaseModel):
    """뉴스 분석 응답"""
    news_title: str
    stock_code: Optional[str] = None
    signals: List[TradeSignalResponse] = []
    timestamp: str = ""
    error: Optional[str] = None


class WatchlistRequest(BaseModel):
    """관심 종목 요청"""
    stock_code: str
    stock_name: str


class TradingConfigRequest(BaseModel):
    """트레이딩 설정 요청"""
    ollama_url: str = "http://localhost:11434"
    analysis_model: str = "deepseek-r1:8b"
    secondary_model: str = "qwen3:8b"
    max_order_amount: int = 100000
    max_positions: int = 5
    min_confidence: float = 0.7
    news_check_interval: int = 30
    position_check_interval: int = 30
    surge_scan_interval: int = 60
    auto_execute: bool = False


# === API Endpoints ===

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    통합 트레이딩 상태 조회

    Returns:
        StatusResponse: 현재 상태
            - is_running: 실행 중 여부
            - current_mode: 현재 모드 (INIT, PRE_MARKET, SCALPING, REGULAR, POST_MARKET)
            - market_sentiment: 시장 심리 (BULLISH, BEARISH, NEUTRAL)
            - positions: 현재 보유 포지션 목록
    """
    service = get_integrated_trade_service()
    status = await service.get_status()

    positions = [
        PositionResponse(**p) if isinstance(p, dict) else p
        for p in status.get("positions", [])
    ]

    return StatusResponse(
        is_running=status.get("is_running", False),
        initialized=status.get("initialized", False),
        current_mode=status.get("current_mode", "INIT"),
        market_sentiment=status.get("market_sentiment", "NEUTRAL"),
        market_context=status.get("market_context", ""),
        today_trades=status.get("today_trades", 0),
        today_pnl=status.get("today_pnl", 0),
        positions_count=status.get("positions_count", 0),
        positions=positions
    )


@router.post("/start")
async def start_trading(config: Optional[TradingConfigRequest] = None):
    """
    통합 LLM 트레이딩 시작

    특징:
    - 시간대별 자동 모드 전환
    - 실시간 뉴스 모니터링 + LLM 분석
    - LLM 기반 동적 포지션 관리

    Args:
        config: 트레이딩 설정 (선택적)

    Returns:
        Dict: 시작 결과
    """
    service = get_integrated_trade_service()

    # 설정 적용
    if config:
        config_dict = config.model_dump()
        success = await service.initialize(config_dict)
        if not success:
            raise HTTPException(status_code=500, detail="설정 초기화 실패")

    success = await service.start()

    if not success:
        raise HTTPException(status_code=400, detail="시작 실패 (이미 실행 중이거나 초기화 실패)")

    return {
        "success": True,
        "message": "통합 LLM 트레이딩 시작"
    }


@router.post("/stop")
async def stop_trading():
    """
    통합 LLM 트레이딩 중지

    Returns:
        Dict: 중지 결과
    """
    service = get_integrated_trade_service()
    success = await service.stop()

    return {
        "success": success,
        "message": "통합 트레이딩 중지" if success else "중지 실패"
    }


@router.post("/analyze-news", response_model=NewsAnalyzeResponse)
async def analyze_news(request: NewsAnalyzeRequest):
    """
    뉴스 즉시 분석 (LLM 기반)

    뉴스를 LLM이 분석하고 관련 종목을 식별하여 매매 시그널을 생성합니다.

    특징:
    - LLM이 뉴스의 영향도 분석
    - 기술적 분석과 결합
    - 동적 손절/익절 조건 제안 (고정 % 아님)

    Args:
        request: 뉴스 분석 요청
            - news_title: 뉴스 헤드라인
            - stock_code: 종목코드 (선택적, 특정 종목 분석 시)

    Returns:
        NewsAnalyzeResponse: 분석 결과 및 매매 시그널
    """
    service = get_integrated_trade_service()

    result = await service.analyze_news_now(
        news_title=request.news_title,
        stock_code=request.stock_code
    )

    if "error" in result:
        return NewsAnalyzeResponse(
            news_title=request.news_title,
            stock_code=request.stock_code,
            error=result["error"]
        )

    signals = [
        TradeSignalResponse(**s) if isinstance(s, dict) else s
        for s in result.get("signals", [])
    ]

    return NewsAnalyzeResponse(
        news_title=result.get("news_title", request.news_title),
        stock_code=result.get("stock_code"),
        signals=signals,
        timestamp=result.get("timestamp", "")
    )


@router.get("/signals", response_model=List[TradeSignalResponse])
async def get_trade_signals(limit: int = 20):
    """
    매매 시그널 히스토리 조회

    LLM이 생성한 매매 시그널 히스토리를 반환합니다.
    각 시그널에는 동적 손절/익절 조건이 포함됩니다.

    Args:
        limit: 조회 개수 (기본: 20)

    Returns:
        List[TradeSignalResponse]: 매매 시그널 목록
    """
    service = get_integrated_trade_service()
    signals = await service.get_trade_signals(limit)

    return [
        TradeSignalResponse(**s) if isinstance(s, dict) else s
        for s in signals
    ]


@router.get("/exit-decisions", response_model=List[ExitDecisionResponse])
async def get_exit_decisions(limit: int = 20):
    """
    청산 결정 히스토리 조회

    LLM이 내린 포지션 청산 결정 히스토리를 반환합니다.
    고정 % 기반이 아닌 LLM의 동적 판단입니다.

    Args:
        limit: 조회 개수 (기본: 20)

    Returns:
        List[ExitDecisionResponse]: 청산 결정 목록
    """
    service = get_integrated_trade_service()
    decisions = await service.get_exit_decisions(limit)

    return [
        ExitDecisionResponse(**d) if isinstance(d, dict) else d
        for d in decisions
    ]


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions():
    """
    현재 포지션 조회

    LLM이 관리 중인 현재 포지션을 반환합니다.

    Returns:
        List[PositionResponse]: 포지션 목록
    """
    service = get_integrated_trade_service()
    positions = await service.get_positions()

    return [
        PositionResponse(**p) if isinstance(p, dict) else p
        for p in positions
    ]


@router.post("/watchlist/add")
async def add_to_watchlist(request: WatchlistRequest):
    """
    관심 종목 추가

    실시간 뉴스 모니터링 대상에 종목을 추가합니다.

    Args:
        request: 관심 종목 정보

    Returns:
        Dict: 추가 결과
    """
    service = get_integrated_trade_service()
    success = await service.add_to_watchlist(request.stock_code, request.stock_name)

    return {
        "success": success,
        "message": f"{request.stock_name} 관심 종목 추가" if success else "추가 실패"
    }


@router.delete("/watchlist/{stock_code}")
async def remove_from_watchlist(stock_code: str):
    """
    관심 종목 제거

    Args:
        stock_code: 종목코드

    Returns:
        Dict: 제거 결과
    """
    service = get_integrated_trade_service()
    success = await service.remove_from_watchlist(stock_code)

    return {
        "success": success,
        "message": f"{stock_code} 관심 종목 제거" if success else "제거 실패"
    }


@router.get("/stream")
async def stream_events():
    """
    SSE 이벤트 스트림

    실시간 이벤트를 스트리밍합니다:
    - trade_signal: 새 매매 시그널
    - exit_decision: 청산 결정
    - position_update: 포지션 업데이트
    - mode_changed: 모드 변경
    - status_changed: 상태 변경

    Returns:
        StreamingResponse: SSE 스트림
    """
    async def event_generator():
        service = get_integrated_trade_service()

        async for event in service.get_events():
            yield f"event: {event['event_type']}\n"
            yield f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
