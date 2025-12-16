# -*- coding: utf-8 -*-
"""
자동 매매 Pydantic 모델

대시보드 API에서 사용하는 자동 매매 관련 데이터 모델들입니다.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class AutoTradeConfigRequest(BaseModel):
    """자동 매매 설정 요청"""
    env_dv: str = Field(default="real", description="실전투자 전용")
    max_order_amount: int = Field(default=100000, ge=1000, description="1회 주문 한도 (원)")
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="최소 신뢰도")
    min_consensus: float = Field(default=0.67, ge=0.0, le=1.0, description="최소 합의도")
    stop_loss_pct: float = Field(default=0.5, ge=0.1, le=5.0, description="손절률 (%)")
    take_profit_pct: float = Field(default=1.5, ge=0.1, le=10.0, description="익절률 (%)")
    max_daily_trades: int = Field(default=0, ge=0, le=100, description="일일 최대 거래 횟수 (0=무제한)")
    max_daily_loss: int = Field(default=50000, ge=1000, description="일일 최대 손실 (원)")
    min_surge_score: float = Field(default=50.0, ge=0.0, le=100.0, description="최소 급등 점수")
    max_stocks_per_scan: int = Field(default=5, ge=1, le=20, description="스캔당 최대 종목 수")


class AutoTradeConfigResponse(BaseModel):
    """자동 매매 설정 응답"""
    env_dv: str = Field(default="real")
    max_order_amount: int = Field(default=100000)
    min_confidence: float = Field(default=0.7)
    min_consensus: float = Field(default=0.67)
    stop_loss_pct: float = Field(default=0.5)
    take_profit_pct: float = Field(default=1.5)
    max_daily_trades: int = Field(default=0)
    max_daily_loss: int = Field(default=50000)
    min_surge_score: float = Field(default=50.0)
    max_stocks_per_scan: int = Field(default=5)
    ensemble_models: List[str] = Field(default_factory=list)
    main_model: str = ""


class AutoTradeResult(BaseModel):
    """자동 매매 결과"""
    success: bool
    action: str = Field(description="BUY, SELL, HOLD, SKIP, ERROR")
    stock_code: str
    stock_name: str
    current_price: int

    # 분석 결과
    ensemble_signal: str
    confidence: float
    consensus: float

    # 주문 정보
    order_qty: int = 0
    order_price: int = 0
    order_no: Optional[str] = None

    # 기술적 지표
    technical_score: float = 0.0
    trend: str = ""

    # 메타
    reason: str = ""
    timestamp: str = ""


class AutoTradeStatus(BaseModel):
    """자동 매매 상태"""
    is_running: bool = False
    initialized: bool = False

    # 설정
    config: Optional[AutoTradeConfigResponse] = None

    # 일일 통계
    today_trades: int = 0
    today_pnl: int = 0
    today_date: str = ""

    # 거래 가능 여부
    can_trade: bool = False
    market_status: Dict[str, Any] = Field(default_factory=dict)
    risk_status: Dict[str, Any] = Field(default_factory=dict)

    # 모델 정보
    ensemble_models: List[str] = Field(default_factory=list)
    main_model: str = ""


class AutoTradeHistoryItem(BaseModel):
    """자동 매매 히스토리 항목"""
    timestamp: str
    stock_code: str
    stock_name: str
    action: str
    ensemble_signal: str
    confidence: float
    consensus: float
    order_qty: int
    order_price: int
    order_amount: int
    order_no: Optional[str] = None
    success: bool
    reason: str
    technical_score: float = 0.0


class AutoTradeHistory(BaseModel):
    """자동 매매 히스토리"""
    items: List[AutoTradeHistoryItem] = Field(default_factory=list)
    total_count: int = 0
    success_count: int = 0
    total_buy_amount: int = 0
    total_sell_amount: int = 0


class AutoTradeEvent(BaseModel):
    """자동 매매 SSE 이벤트"""
    event_type: str = Field(description="trade_executed, analysis_completed, status_changed, error")
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SingleStockAnalyzeRequest(BaseModel):
    """단일 종목 분석 요청"""
    stock_code: str = Field(min_length=6, max_length=6, description="종목코드 (6자리)")
    stock_name: str = Field(min_length=1, description="종목명")
    current_price: int = Field(ge=1, description="현재가")
    news_list: Optional[List[str]] = Field(default=None, description="관련 뉴스 리스트")
    skip_market_check: bool = Field(default=False, description="장 시간 체크 비활성화")


class ScanAndTradeRequest(BaseModel):
    """스캔 및 매매 요청"""
    min_score: Optional[float] = Field(default=None, description="최소 급등 점수")
    max_stocks: Optional[int] = Field(default=None, description="분석할 최대 종목 수")
    skip_market_check: bool = Field(default=False, description="장 시간 체크 비활성화")


class AttentionStock(BaseModel):
    """주목 종목"""
    code: str = Field(description="종목코드")
    name: str = Field(description="종목명")
    reason: str = Field(description="주목 사유")


class NewsAnalysisResult(BaseModel):
    """뉴스 분석 결과"""
    news_count: int = Field(default=0, description="분석된 뉴스 수")
    market_sentiment: str = Field(default="NEUTRAL", description="시장 심리 (BULLISH, BEARISH, NEUTRAL)")
    key_themes: List[str] = Field(default_factory=list, description="주요 테마")
    attention_stocks: List[AttentionStock] = Field(default_factory=list, description="주목 종목")
    market_outlook: str = Field(default="", description="시장 전망")
    news_list: List[str] = Field(default_factory=list, description="수집된 뉴스 헤드라인")
    analysis_time: str = Field(default="", description="분석 시간")
    llm_raw_output: str = Field(default="", description="LLM 원본 출력")


class TradingMode(BaseModel):
    """현재 트레이딩 모드"""
    mode: str = Field(default="INIT", description="현재 모드 (INIT, NEWS, TRADING, IDLE)")
    mode_description: str = Field(default="초기화 중", description="모드 설명")
    market_status: str = Field(default="장 시작 전", description="시장 상태")
    next_scan_time: str = Field(default="", description="다음 스캔 시간")
    last_news_analysis: Optional[NewsAnalysisResult] = Field(default=None, description="마지막 뉴스 분석 결과")
