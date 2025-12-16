# -*- coding: utf-8 -*-
"""포지션 모니터 API 엔드포인트."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from core.position_monitor import get_position_monitor

router = APIRouter()


class PositionMonitorConfig(BaseModel):
    """포지션 모니터 설정"""
    stop_loss_pct: Optional[float] = Field(None, ge=0.1, le=10.0, description="손절률 (%)")
    take_profit_pct: Optional[float] = Field(None, ge=0.1, le=20.0, description="익절률 (%)")
    check_interval: Optional[int] = Field(None, ge=5, le=300, description="체크 주기 (초)")
    enabled: Optional[bool] = Field(None, description="활성화 여부")
    use_market_order: Optional[bool] = Field(None, description="시장가 주문 사용")


@router.get("/status")
async def get_monitor_status():
    """포지션 모니터 상태 조회"""
    monitor = get_position_monitor()
    config = monitor.get_config()

    return {
        "is_running": monitor.is_running(),
        "config": config
    }


@router.post("/start")
async def start_monitor():
    """포지션 모니터 시작"""
    monitor = get_position_monitor()

    if monitor.is_running():
        return {"success": True, "message": "이미 실행 중입니다"}

    success = await monitor.start()

    if success:
        return {"success": True, "message": "포지션 모니터 시작됨"}
    else:
        raise HTTPException(status_code=500, detail="포지션 모니터 시작 실패")


@router.post("/stop")
async def stop_monitor():
    """포지션 모니터 중지"""
    monitor = get_position_monitor()
    await monitor.stop()

    return {"success": True, "message": "포지션 모니터 중지됨"}


@router.put("/config")
async def update_config(config: PositionMonitorConfig):
    """포지션 모니터 설정 업데이트"""
    monitor = get_position_monitor()

    # None이 아닌 값만 업데이트
    update_dict = {k: v for k, v in config.model_dump().items() if v is not None}

    if update_dict:
        monitor.update_config(update_dict)

    return {
        "success": True,
        "config": monitor.get_config()
    }


@router.get("/check")
async def manual_check():
    """수동 포지션 체크 (현재 상태 확인)"""
    monitor = get_position_monitor()

    try:
        results = await monitor.manual_check()
        return {
            "holdings": results,
            "count": len(results),
            "config": monitor.get_config()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(limit: int = 20):
    """알림 히스토리 조회"""
    monitor = get_position_monitor()
    alerts = monitor.get_alert_history(limit)

    return {
        "alerts": alerts,
        "count": len(alerts)
    }


@router.post("/clear-alerts")
async def clear_alerts(stock_code: Optional[str] = None):
    """알림 상태 초기화"""
    monitor = get_position_monitor()
    monitor.clear_alert_state(stock_code)

    return {
        "success": True,
        "message": f"알림 상태 초기화됨 {'(종목: ' + stock_code + ')' if stock_code else '(전체)'}"
    }
