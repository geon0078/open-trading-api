# -*- coding: utf-8 -*-
"""급등 종목 관련 Pydantic 모델."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """매매 시그널 타입."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    NEUTRAL = "NEUTRAL"


class SurgeCandidate(BaseModel):
    """급등 후보 종목."""
    code: str = Field(..., description="종목코드", example="005930")
    name: str = Field(..., description="종목명", example="삼성전자")
    price: int = Field(..., description="현재가")
    change: int = Field(..., description="전일대비")
    change_rate: float = Field(..., description="등락률 (%)")
    volume: int = Field(..., description="누적거래량")
    volume_power: float = Field(..., description="체결강도")
    buy_volume: int = Field(0, description="매수체결량")
    sell_volume: int = Field(0, description="매도체결량")
    bid_balance: int = Field(0, description="총매수호가잔량")
    ask_balance: int = Field(0, description="총매도호가잔량")
    balance_ratio: float = Field(1.0, description="호가잔량비")
    surge_score: float = Field(..., description="급등점수 (0-100)")
    rank: int = Field(..., description="순위")
    signal: SignalType = Field(..., description="시그널")
    detected_at: datetime = Field(default_factory=datetime.now, description="탐지시간")
    reasons: List[str] = Field(default_factory=list, description="매수사유")

    # LLM 분석 결과 (선택)
    llm_recommendation: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_analysis: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SurgeCandidateList(BaseModel):
    """급등 종목 리스트 응답."""
    candidates: List[SurgeCandidate]
    timestamp: datetime = Field(default_factory=datetime.now)
    total_count: int
    scan_duration_ms: float = Field(0.0, description="스캔 소요시간 (ms)")


class ScanSettings(BaseModel):
    """급등 종목 스캔 설정."""
    min_score: float = Field(50.0, ge=0, le=100, description="최소 급등점수")
    min_volume_power: float = Field(120.0, ge=100, description="최소 체결강도")
    min_change_rate: float = Field(1.0, description="최소 등락률 (%)")
    max_change_rate: float = Field(15.0, description="최대 등락률 (%)")
    min_balance_ratio: float = Field(1.2, ge=1.0, description="최소 호가잔량비")
    min_volume: int = Field(100000, ge=0, description="최소 거래량")
    auto_scan: bool = Field(True, description="자동 스캔 활성화")
    scan_interval: int = Field(60, ge=10, description="스캔 주기 (초)")
    limit: int = Field(20, ge=1, le=100, description="최대 종목 수")


class ScanResult(BaseModel):
    """스캔 결과."""
    success: bool
    message: str
    candidates_count: int
    scan_duration_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)
