# -*- coding: utf-8 -*-
"""계좌 관련 Pydantic 모델."""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class Holding(BaseModel):
    """보유 종목."""
    code: str = Field(..., description="종목코드")
    name: str = Field(..., description="종목명")
    quantity: int = Field(..., description="보유수량")
    avg_price: float = Field(..., description="평균단가")
    current_price: int = Field(..., description="현재가")
    eval_amount: int = Field(..., description="평가금액")
    pnl: int = Field(..., description="평가손익")
    pnl_rate: float = Field(..., description="수익률 (%)")


class AccountBalance(BaseModel):
    """계좌 잔고."""
    deposit: int = Field(..., description="예수금")
    total_eval: int = Field(..., description="총평가금액")
    total_purchase: int = Field(0, description="총매입금액")
    total_pnl: int = Field(..., description="총평가손익")
    pnl_rate: float = Field(..., description="수익률 (%)")
    holdings_count: int = Field(0, description="보유종목 수")
    holdings: List[Holding] = Field(default_factory=list, description="보유종목 리스트")
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Order(BaseModel):
    """체결 내역."""
    order_id: str = Field("", description="주문번호")
    order_time: str = Field(..., description="주문시각 (HHMMSS)")
    stock_code: str = Field(..., description="종목코드")
    stock_name: str = Field(..., description="종목명")
    side: str = Field(..., description="매수/매도")
    order_qty: int = Field(0, description="주문수량")
    exec_qty: int = Field(..., description="체결수량")
    order_price: int = Field(0, description="주문가격")
    exec_price: int = Field(..., description="체결가격")
    exec_amount: int = Field(0, description="체결금액")
    status: str = Field("체결", description="상태")


class OrdersList(BaseModel):
    """체결 내역 리스트."""
    orders: List[Order]
    total_count: int
    buy_count: int = Field(0, description="매수 건수")
    sell_count: int = Field(0, description="매도 건수")
    buy_amount: int = Field(0, description="매수 금액")
    sell_amount: int = Field(0, description="매도 금액")
    updated_at: datetime = Field(default_factory=datetime.now)


class PnLSummary(BaseModel):
    """손익 현황 요약."""
    realized_pnl: int = Field(0, description="실현손익")
    unrealized_pnl: int = Field(0, description="평가손익")
    total_pnl: int = Field(0, description="총손익")
    total_pnl_rate: float = Field(0.0, description="총수익률 (%)")
    today_pnl: int = Field(0, description="당일손익")
    today_pnl_rate: float = Field(0.0, description="당일수익률 (%)")
