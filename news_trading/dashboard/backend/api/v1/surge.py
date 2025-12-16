# -*- coding: utf-8 -*-
"""급등 종목 API 엔드포인트."""

from fastapi import APIRouter, HTTPException, Query

from core.surge_service import surge_service
from models.surge import ScanResult, ScanSettings, SurgeCandidateList

router = APIRouter()


@router.get("/candidates", response_model=SurgeCandidateList)
async def get_surge_candidates(
    min_score: float = Query(50.0, ge=0, le=100, description="최소 급등점수"),
    limit: int = Query(20, ge=1, le=100, description="최대 종목 수"),
    refresh: bool = Query(False, description="새로 스캔 여부")
):
    """급등 종목 조회."""
    if refresh:
        return await surge_service.scan_stocks(min_score=min_score, limit=limit)
    else:
        cached = surge_service.get_cached_candidates()
        if cached.total_count == 0:
            # 캐시가 비어있으면 새로 스캔
            return await surge_service.scan_stocks(min_score=min_score, limit=limit)
        return cached


@router.post("/scan", response_model=ScanResult)
async def scan_surge_stocks(
    min_score: float = Query(50.0, ge=0, le=100, description="최소 급등점수"),
    limit: int = Query(20, ge=1, le=100, description="최대 종목 수")
):
    """수동 급등 종목 스캔."""
    try:
        result = await surge_service.scan_stocks(min_score=min_score, limit=limit)
        return ScanResult(
            success=True,
            message="스캔 완료",
            candidates_count=result.total_count,
            scan_duration_ms=result.scan_duration_ms,
            timestamp=result.timestamp
        )
    except Exception as e:
        return ScanResult(
            success=False,
            message=str(e),
            candidates_count=0,
            scan_duration_ms=0
        )


@router.get("/settings", response_model=ScanSettings)
async def get_scan_settings():
    """스캔 설정 조회."""
    return surge_service.get_settings()


@router.put("/settings", response_model=ScanSettings)
async def update_scan_settings(settings: ScanSettings):
    """스캔 설정 업데이트."""
    return surge_service.update_settings(settings)


@router.get("/status")
async def get_scan_status():
    """스캔 상태 조회."""
    return {
        "is_scanning": surge_service.is_scanning,
        "last_scan_time": surge_service.last_scan_time.isoformat() if surge_service.last_scan_time else None,
        "cached_count": len(surge_service._cached_candidates)
    }
