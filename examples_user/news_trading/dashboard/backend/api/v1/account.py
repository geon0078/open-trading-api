# -*- coding: utf-8 -*-
"""계좌 API 엔드포인트."""

from fastapi import APIRouter, HTTPException

from core.kis_service import kis_service
from models.account import AccountBalance, OrdersList, PnLSummary

router = APIRouter()


@router.get("/balance", response_model=AccountBalance)
async def get_account_balance():
    """계좌 잔고 조회."""
    try:
        return await kis_service.get_account_balance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/holdings")
async def get_holdings():
    """보유 종목 조회."""
    try:
        balance = await kis_service.get_account_balance()
        return {
            "holdings": [h.model_dump() for h in balance.holdings],
            "count": len(balance.holdings),
            "updated_at": balance.updated_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/today", response_model=OrdersList)
async def get_today_orders():
    """오늘 체결 내역 조회."""
    try:
        return await kis_service.get_today_orders()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pnl", response_model=PnLSummary)
async def get_pnl_summary():
    """손익 현황 조회."""
    try:
        balance = await kis_service.get_account_balance()
        orders = await kis_service.get_today_orders()

        # 당일 실현손익 계산 (매도 금액 - 매수 금액의 간단한 추정)
        today_realized = orders.sell_amount - orders.buy_amount

        return PnLSummary(
            realized_pnl=today_realized,
            unrealized_pnl=balance.total_pnl,
            total_pnl=balance.total_pnl + today_realized,
            total_pnl_rate=balance.pnl_rate,
            today_pnl=today_realized,
            today_pnl_rate=0.0  # 정확한 계산 필요
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_account_status():
    """계좌 상태 조회."""
    try:
        balance = await kis_service.get_account_balance()
        return {
            "connected": True,
            "deposit": balance.deposit,
            "total_eval": balance.total_eval,
            "holdings_count": balance.holdings_count,
            "updated_at": balance.updated_at.isoformat()
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }
