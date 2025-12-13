# -*- coding: utf-8 -*-
"""LLM 분석 관련 Pydantic 모델."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    """개별 모델 분석 결과."""
    model_name: str = Field(..., description="모델명")
    signal: str = Field(..., description="매매 시그널")
    confidence: float = Field(..., ge=0, le=1, description="신뢰도")
    trend_prediction: str = Field(..., description="추세 예측 (UP/DOWN/SIDEWAYS)")
    entry_price: float = Field(0.0, description="진입가")
    stop_loss: float = Field(0.0, description="손절가")
    take_profit: float = Field(0.0, description="익절가")
    reasoning: str = Field("", description="분석 근거")
    news_impact: str = Field("NEUTRAL", description="뉴스 영향 (POSITIVE/NEGATIVE/NEUTRAL)")
    risk_factors: List[str] = Field(default_factory=list, description="리스크 요인")
    processing_time: float = Field(0.0, description="처리 시간 (초)")
    success: bool = Field(True, description="성공 여부")
    raw_output: str = Field("", description="모델 원본 출력")
    error_message: str = Field("", description="에러 메시지")


class EnsembleAnalysisResult(BaseModel):
    """앙상블 분석 결과."""
    id: str = Field(..., description="분석 ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    stock_code: str = Field(..., description="종목코드")
    stock_name: str = Field(..., description="종목명")

    # 앙상블 결과
    ensemble_signal: str = Field(..., description="최종 시그널")
    ensemble_confidence: float = Field(..., ge=0, le=1, description="최종 신뢰도")
    ensemble_trend: str = Field(..., description="최종 추세")

    # 가격 정보
    current_price: int = Field(0, description="현재가")
    avg_entry_price: float = Field(0.0, description="평균 진입가")
    avg_stop_loss: float = Field(0.0, description="평균 손절가")
    avg_take_profit: float = Field(0.0, description="평균 익절가")

    # 투표 결과
    signal_votes: Dict[str, int] = Field(default_factory=dict, description="시그널 투표")
    trend_votes: Dict[str, int] = Field(default_factory=dict, description="추세 투표")

    # 모델별 결과
    model_results: List[ModelResult] = Field(default_factory=list)
    models_used: List[str] = Field(default_factory=list, description="사용된 모델 목록")
    models_agreed: int = Field(0, description="합의한 모델 수")
    total_models: int = Field(0, description="전체 모델 수")
    consensus_score: float = Field(0.0, ge=0, le=1, description="합의도")

    # 입력 데이터
    input_prompt: str = Field("", description="입력 프롬프트")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="입력 데이터")
    news_list: List[str] = Field(default_factory=list, description="관련 뉴스")

    # 메타
    total_processing_time: float = Field(0.0, description="총 처리 시간")
    success: bool = Field(True, description="성공 여부")
    error_message: str = Field("", description="에러 메시지")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalyzeRequest(BaseModel):
    """분석 요청."""
    stock_code: str = Field(..., description="종목코드")
    stock_name: Optional[str] = Field(None, description="종목명")
    include_news: bool = Field(True, description="뉴스 포함 여부")
    include_technical: bool = Field(True, description="기술적 지표 포함 여부")
    parallel: bool = Field(False, description="병렬 실행 여부")


class LLMSettings(BaseModel):
    """LLM 설정."""
    preset: str = Field("deepseek", description="프리셋 (deepseek/default/lightweight)")
    auto_analyze: bool = Field(True, description="자동 분석 활성화")
    analyze_interval: int = Field(120, ge=30, description="분석 주기 (초)")
    max_analyze_count: int = Field(5, ge=1, le=20, description="최대 분석 종목 수")
    parallel_execution: bool = Field(False, description="병렬 실행")
    keep_alive: str = Field("5m", description="모델 유지 시간")
    max_vram_gb: float = Field(24.0, description="최대 VRAM (GB)")
    models: List[str] = Field(
        default_factory=lambda: ["deepseek-r1:8b", "qwen3:8b"],
        description="사용할 모델 목록"
    )


class AnalysisHistoryItem(BaseModel):
    """분석 히스토리 아이템."""
    id: str
    timestamp: datetime
    stock_code: str
    stock_name: str
    ensemble_signal: str
    ensemble_confidence: float
    consensus_score: float
    models_used: List[str]
    success: bool

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
