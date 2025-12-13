# -*- coding: utf-8 -*-
"""LLM 분석 API 엔드포인트."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from core.llm_service import llm_service
from models.llm import (
    AnalysisHistoryItem,
    AnalyzeRequest,
    EnsembleAnalysisResult,
    LLMSettings,
)

router = APIRouter()


@router.post("/analyze", response_model=EnsembleAnalysisResult)
async def analyze_stock(request: AnalyzeRequest):
    """단일 종목 앙상블 분석."""
    if llm_service.is_analyzing:
        raise HTTPException(status_code=409, detail="이미 분석이 진행 중입니다.")

    try:
        result = await llm_service.analyze_stock(
            stock_code=request.stock_code,
            stock_name=request.stock_name or "",
            parallel=request.parallel
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[AnalysisHistoryItem])
async def get_analysis_history(
    limit: int = Query(20, ge=1, le=100, description="조회 개수")
):
    """분석 히스토리 조회."""
    return llm_service.get_history(limit=limit)


@router.get("/history/{analysis_id}", response_model=EnsembleAnalysisResult)
async def get_analysis_detail(analysis_id: str):
    """분석 상세 조회."""
    result = llm_service.get_analysis_by_id(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
    return result


@router.get("/settings", response_model=LLMSettings)
async def get_llm_settings():
    """LLM 설정 조회."""
    return llm_service.get_settings()


@router.put("/settings", response_model=LLMSettings)
async def update_llm_settings(settings: LLMSettings):
    """LLM 설정 업데이트."""
    return llm_service.update_settings(settings)


@router.post("/preset")
async def set_preset(preset: str = Query(..., description="프리셋 (deepseek/default/lightweight)")):
    """프리셋 변경."""
    current = llm_service.get_settings()
    current.preset = preset
    llm_service.update_settings(current)
    return {"success": True, "preset": preset}


@router.get("/status")
async def get_llm_status():
    """LLM 상태 조회."""
    return {
        "is_analyzing": llm_service.is_analyzing,
        "history_count": len(llm_service._history),
        "current_analysis": llm_service.current_analysis.id if llm_service.current_analysis else None,
        "settings": llm_service.get_settings().model_dump()
    }


@router.get("/models")
async def get_available_models():
    """사용 가능한 모델 목록."""
    settings = llm_service.get_settings()
    return {
        "models": settings.models,
        "preset": settings.preset
    }
