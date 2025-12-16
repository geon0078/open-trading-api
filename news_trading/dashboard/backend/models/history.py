# -*- coding: utf-8 -*-
"""
히스토리 API 모델
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TradeRecord(BaseModel):
    """거래 기록"""
    id: int
    timestamp: str
    date: str
    stock_code: str
    stock_name: str
    action: str
    order_qty: int = 0
    order_price: int = 0
    order_amount: int = 0
    order_no: Optional[str] = None
    success: bool = False

    # 분석 결과
    ensemble_signal: Optional[str] = None
    confidence: float = 0
    consensus: float = 0
    technical_score: float = 0
    trend: Optional[str] = None

    # 메타
    reason: Optional[str] = None
    analysis_id: Optional[str] = None
    llm_log_path: Optional[str] = None


class AnalysisSummary(BaseModel):
    """분석 요약"""
    id: int
    analysis_id: str
    timestamp: str
    date: str
    stock_code: str
    stock_name: str

    # 앙상블 결과
    ensemble_signal: Optional[str] = None
    ensemble_confidence: float = 0
    ensemble_trend: Optional[str] = None
    consensus_score: float = 0

    # 가격 정보
    current_price: int = 0
    avg_entry_price: float = 0
    avg_stop_loss: float = 0
    avg_take_profit: float = 0

    # 모델 정보
    models_used: List[str] = Field(default_factory=list)
    models_agreed: int = 0
    total_models: int = 0
    signal_votes: Dict[str, int] = Field(default_factory=dict)
    trend_votes: Dict[str, int] = Field(default_factory=dict)

    # 기술적 지표
    technical_summary: Dict[str, Any] = Field(default_factory=dict)

    # 처리 시간
    total_processing_time: float = 0

    # LLM 로그 참조
    llm_log_path: Optional[str] = None


class DailyStats(BaseModel):
    """일별 통계"""
    date: str
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    success_count: int = 0
    total_buy_amount: int = 0
    total_sell_amount: int = 0
    realized_pnl: int = 0
    total_analyses: int = 0
    avg_confidence: float = 0
    avg_consensus: float = 0
    signal_distribution: Dict[str, int] = Field(default_factory=dict)


class LLMLogEntry(BaseModel):
    """LLM 로그 엔트리 (인덱스)"""
    log_id: str
    timestamp: str
    stock_code: str
    stock_name: str
    analysis_type: str
    signal: Optional[str] = None
    confidence: float = 0


class LLMLogIndex(BaseModel):
    """LLM 로그 일별 인덱스"""
    date: str
    entries: List[LLMLogEntry] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    """개별 모델 출력"""
    model_name: str
    raw_output: str = ""
    parsed_result: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = 0
    success: bool = False
    error_message: str = ""


class LLMLogDetail(BaseModel):
    """LLM 로그 상세"""
    analysis_id: str
    timestamp: str
    stock_code: str
    stock_name: str
    analysis_type: str

    # 입력
    input: Dict[str, Any] = Field(default_factory=dict)

    # 모델별 출력
    model_outputs: List[ModelOutput] = Field(default_factory=list)

    # 앙상블 결과
    ensemble_result: Dict[str, Any] = Field(default_factory=dict)

    # 메타
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelPerformance(BaseModel):
    """모델 성능 통계"""
    model_name: str
    total_calls: int = 0
    success_rate: float = 0
    avg_processing_time: float = 0
    signal_distribution: Dict[str, int] = Field(default_factory=dict)


class StorageStats(BaseModel):
    """저장소 통계"""
    base_dir: str
    total_files: int = 0
    total_size_mb: float = 0
    date_range: Dict[str, Optional[str]] = Field(default_factory=dict)
    compression: bool = False


class HistoryOverview(BaseModel):
    """히스토리 개요"""
    today_stats: DailyStats
    recent_trades: List[TradeRecord] = Field(default_factory=list)
    recent_analyses: List[AnalysisSummary] = Field(default_factory=list)
    storage_stats: Optional[StorageStats] = None
