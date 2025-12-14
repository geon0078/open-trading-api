# -*- coding: utf-8 -*-
"""
자동 매매 API 엔드포인트

2단계 앙상블 LLM 자동 매매 시스템의 REST API입니다.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
import asyncio
import json
import logging

from models.auto_trade import (
    AutoTradeConfigRequest,
    AutoTradeConfigResponse,
    AutoTradeResult,
    AutoTradeStatus,
    AutoTradeHistory,
    AutoTradeHistoryItem,
    SingleStockAnalyzeRequest,
    ScanAndTradeRequest,
)
from core.auto_trade_service import get_auto_trade_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status", response_model=AutoTradeStatus)
async def get_auto_trade_status():
    """
    자동 매매 상태 조회

    Returns:
        AutoTradeStatus: 현재 상태
    """
    service = get_auto_trade_service()
    status = await service.get_status()

    return AutoTradeStatus(
        is_running=status.get("is_running", False),
        initialized=status.get("initialized", False),
        config=AutoTradeConfigResponse(**status["config"]) if status.get("config") else None,
        today_trades=status.get("today_trades", 0),
        today_pnl=status.get("today_pnl", 0),
        today_date=status.get("today_date", ""),
        can_trade=status.get("can_trade", False),
        market_status=status.get("market_status", {}),
        risk_status=status.get("risk_status", {}),
        ensemble_models=status.get("ensemble_models", []),
        main_model=status.get("main_model", ""),
    )


@router.post("/start")
async def start_auto_trade(
    interval: int = 60,
    config: Optional[AutoTradeConfigRequest] = None
):
    """
    자동 매매 시작

    Args:
        interval: 스캔 주기 (초, 기본: 60)
        config: 설정 (선택적)

    Returns:
        Dict: 시작 결과
    """
    service = get_auto_trade_service()

    # 설정 적용
    if config:
        config_dict = config.model_dump()
        success = await service.update_config(config_dict)
        if not success:
            raise HTTPException(status_code=500, detail="설정 적용 실패")

    # 자동 매매 시작
    success = await service.start_auto_trading(interval=interval)

    if not success:
        raise HTTPException(status_code=400, detail="자동 매매 시작 실패 (이미 실행 중이거나 초기화 실패)")

    return {
        "success": True,
        "message": f"자동 매매 시작 (주기: {interval}초)",
        "interval": interval,
    }


@router.post("/stop")
async def stop_auto_trade():
    """
    자동 매매 중지

    Returns:
        Dict: 중지 결과
    """
    service = get_auto_trade_service()
    success = await service.stop_auto_trading()

    return {
        "success": success,
        "message": "자동 매매 중지" if success else "중지 실패",
    }


@router.post("/analyze", response_model=AutoTradeResult)
async def analyze_single_stock(request: SingleStockAnalyzeRequest):
    """
    단일 종목 분석 및 매매

    Args:
        request: 분석 요청

    Returns:
        AutoTradeResult: 분석 및 매매 결과
    """
    service = get_auto_trade_service()

    result = await service.analyze_single_stock(
        stock_code=request.stock_code,
        stock_name=request.stock_name,
        current_price=request.current_price,
        news_list=request.news_list,
        skip_market_check=request.skip_market_check,
    )

    return AutoTradeResult(**result)


@router.post("/scan", response_model=List[AutoTradeResult])
async def scan_and_trade(request: ScanAndTradeRequest = None):
    """
    급등 종목 스캔 후 매매

    Args:
        request: 스캔 요청 (선택적)

    Returns:
        List[AutoTradeResult]: 분석 및 매매 결과 리스트
    """
    service = get_auto_trade_service()

    if request:
        results = await service.run_scan_and_trade(
            min_score=request.min_score,
            max_stocks=request.max_stocks,
            skip_market_check=request.skip_market_check,
        )
    else:
        results = await service.run_scan_and_trade()

    return [AutoTradeResult(**r) for r in results]


@router.post("/scalping", response_model=List[AutoTradeResult])
async def run_scalping_trade(
    min_score: Optional[float] = None,
    max_stocks: Optional[int] = None
):
    """
    스캘핑 매매 실행 (09:00 ~ 09:30)

    야간 뉴스(전일 장 마감 ~ 금일 장 시작)를 분석하고
    보조 모델이 시황을 판단한 뒤, 메인 모델이 최종 스캘핑 결정을 내립니다.

    Args:
        min_score: 최소 급등 점수 (선택적)
        max_stocks: 분석할 최대 종목 수 (선택적)

    Returns:
        List[AutoTradeResult]: 스캘핑 매매 결과 리스트
    """
    service = get_auto_trade_service()

    results = await service.run_scalping_trade(
        min_score=min_score,
        max_stocks=max_stocks,
    )

    return [AutoTradeResult(**r) for r in results]


@router.get("/history", response_model=AutoTradeHistory)
async def get_trade_history(limit: int = 20):
    """
    거래 히스토리 조회

    Args:
        limit: 조회 개수 (기본: 20)

    Returns:
        AutoTradeHistory: 거래 히스토리
    """
    service = get_auto_trade_service()
    history = await service.get_history(limit=limit)

    items = []
    total_buy = 0
    total_sell = 0
    success_count = 0

    for h in history:
        item = AutoTradeHistoryItem(
            timestamp=h.get("timestamp", ""),
            stock_code=h.get("stock_code", ""),
            stock_name=h.get("stock_name", ""),
            action=h.get("action", ""),
            ensemble_signal=h.get("ensemble_signal", ""),
            confidence=h.get("confidence", 0),
            consensus=h.get("consensus", 0),
            order_qty=h.get("order_qty", 0),
            order_price=h.get("order_price", 0),
            order_amount=h.get("order_qty", 0) * h.get("order_price", 0),
            order_no=h.get("order_no"),
            success=h.get("success", False),
            reason=h.get("reason", ""),
            technical_score=h.get("technical_score", 0),
        )
        items.append(item)

        if h.get("success"):
            success_count += 1
            amount = h.get("order_qty", 0) * h.get("order_price", 0)
            if h.get("action") == "BUY":
                total_buy += amount
            elif h.get("action") == "SELL":
                total_sell += amount

    return AutoTradeHistory(
        items=items,
        total_count=len(items),
        success_count=success_count,
        total_buy_amount=total_buy,
        total_sell_amount=total_sell,
    )


@router.put("/config", response_model=AutoTradeConfigResponse)
async def update_config(config: AutoTradeConfigRequest):
    """
    설정 업데이트

    Args:
        config: 새 설정

    Returns:
        AutoTradeConfigResponse: 업데이트된 설정
    """
    service = get_auto_trade_service()

    config_dict = config.model_dump()
    success = await service.update_config(config_dict)

    if not success:
        raise HTTPException(status_code=500, detail="설정 업데이트 실패")

    status = await service.get_status()
    return AutoTradeConfigResponse(
        **status.get("config", {}),
        ensemble_models=status.get("ensemble_models", []),
        main_model=status.get("main_model", ""),
    )


@router.get("/stream")
async def stream_auto_trade_events():
    """
    자동 매매 SSE 스트림

    Returns:
        StreamingResponse: SSE 스트림
    """
    async def event_generator():
        service = get_auto_trade_service()

        async for event in service.get_events():
            yield f"event: {event['event_type']}\n"
            yield f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
