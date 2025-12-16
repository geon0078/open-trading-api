# -*- coding: utf-8 -*-
"""
거래/분석 히스토리 API 엔드포인트

SQLite DB와 JSON LLM 로그에 저장된 데이터를 조회합니다.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.history import (
    TradeRecord,
    AnalysisSummary,
    DailyStats,
    LLMLogIndex,
    LLMLogDetail,
    ModelPerformance,
    StorageStats,
    HistoryOverview,
)

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


def _get_trade_db():
    """TradeHistoryDB 인스턴스 반환"""
    try:
        from modules.storage import get_trade_history_db
        return get_trade_history_db()
    except ImportError as e:
        logger.error(f"storage 모듈 import 실패: {e}")
        raise HTTPException(status_code=500, detail="Storage module not available")


def _get_llm_storage():
    """LLMLogStorage 인스턴스 반환"""
    try:
        from modules.storage import get_llm_log_storage
        return get_llm_log_storage()
    except ImportError as e:
        logger.error(f"storage 모듈 import 실패: {e}")
        raise HTTPException(status_code=500, detail="Storage module not available")


# =========================================================================
# 개요
# =========================================================================

@router.get("/overview", response_model=HistoryOverview)
async def get_history_overview():
    """
    히스토리 개요 조회

    오늘 통계, 최근 거래/분석 내역, 저장소 상태를 한번에 조회합니다.
    """
    db = _get_trade_db()
    storage = _get_llm_storage()

    today = datetime.now().strftime("%Y-%m-%d")

    # 오늘 통계
    today_stats_data = db.get_daily_stats(today)
    today_stats = DailyStats(**today_stats_data)

    # 최근 거래 (10건)
    recent_trades_data = db.get_trades(limit=10)
    recent_trades = [TradeRecord(**t) for t in recent_trades_data]

    # 최근 분석 (10건)
    recent_analyses_data = db.get_analyses(limit=10)
    recent_analyses = [AnalysisSummary(**a) for a in recent_analyses_data]

    # 저장소 통계
    storage_stats_data = storage.get_storage_stats()
    storage_stats = StorageStats(**storage_stats_data)

    return HistoryOverview(
        today_stats=today_stats,
        recent_trades=recent_trades,
        recent_analyses=recent_analyses,
        storage_stats=storage_stats
    )


# =========================================================================
# 거래 내역
# =========================================================================

@router.get("/trades", response_model=List[TradeRecord])
async def get_trades(
    date: Optional[str] = Query(None, description="조회 날짜 (YYYY-MM-DD)"),
    stock_code: Optional[str] = Query(None, description="종목코드"),
    action: Optional[str] = Query(None, description="거래 유형 (BUY/SELL)"),
    success_only: bool = Query(False, description="성공한 거래만"),
    limit: int = Query(50, ge=1, le=500, description="조회 개수"),
    offset: int = Query(0, ge=0, description="오프셋")
):
    """
    거래 내역 조회
    """
    db = _get_trade_db()
    trades = db.get_trades(
        date=date,
        stock_code=stock_code,
        action=action,
        success_only=success_only,
        limit=limit,
        offset=offset
    )
    return [TradeRecord(**t) for t in trades]


@router.get("/trades/today", response_model=List[TradeRecord])
async def get_today_trades(success_only: bool = Query(False)):
    """오늘 거래 내역 조회"""
    db = _get_trade_db()
    trades = db.get_today_trades(success_only=success_only)
    return [TradeRecord(**t) for t in trades]


@router.get("/trades/stock/{stock_code}", response_model=List[TradeRecord])
async def get_stock_trades(
    stock_code: str,
    limit: int = Query(50, ge=1, le=500)
):
    """특정 종목의 거래 내역 조회"""
    db = _get_trade_db()
    trades = db.get_trades(stock_code=stock_code, limit=limit)
    return [TradeRecord(**t) for t in trades]


# =========================================================================
# 분석 내역
# =========================================================================

@router.get("/analyses", response_model=List[AnalysisSummary])
async def get_analyses(
    date: Optional[str] = Query(None, description="조회 날짜 (YYYY-MM-DD)"),
    stock_code: Optional[str] = Query(None, description="종목코드"),
    signal: Optional[str] = Query(None, description="앙상블 시그널 필터"),
    limit: int = Query(50, ge=1, le=500, description="조회 개수"),
    offset: int = Query(0, ge=0, description="오프셋")
):
    """
    분석 요약 조회
    """
    db = _get_trade_db()
    analyses = db.get_analyses(
        date=date,
        stock_code=stock_code,
        signal=signal,
        limit=limit,
        offset=offset
    )
    return [AnalysisSummary(**a) for a in analyses]


@router.get("/analyses/today", response_model=List[AnalysisSummary])
async def get_today_analyses():
    """오늘 분석 내역 조회"""
    db = _get_trade_db()
    analyses = db.get_today_analyses()
    return [AnalysisSummary(**a) for a in analyses]


@router.get("/analyses/stock/{stock_code}")
async def get_stock_history(stock_code: str, days: int = Query(30, ge=1, le=365)):
    """
    특정 종목의 거래/분석 히스토리 조회
    """
    db = _get_trade_db()
    history = db.get_stock_history(stock_code=stock_code, days=days)
    return history


# =========================================================================
# 일별 통계
# =========================================================================

@router.get("/stats/daily", response_model=DailyStats)
async def get_daily_stats(
    date: Optional[str] = Query(None, description="조회 날짜 (YYYY-MM-DD, 기본: 오늘)")
):
    """일별 통계 조회"""
    db = _get_trade_db()
    stats = db.get_daily_stats(date)
    return DailyStats(**stats)


@router.get("/stats/range", response_model=List[DailyStats])
async def get_stats_range(
    start_date: str = Query(..., description="시작 날짜 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="종료 날짜 (YYYY-MM-DD)")
):
    """기간별 통계 조회"""
    db = _get_trade_db()
    stats_list = db.get_stats_range(start_date, end_date)
    return [DailyStats(**s) for s in stats_list]


@router.get("/stats/weekly", response_model=List[DailyStats])
async def get_weekly_stats():
    """최근 7일 통계 조회"""
    db = _get_trade_db()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    stats_list = db.get_stats_range(start_date, end_date)
    return [DailyStats(**s) for s in stats_list]


# =========================================================================
# LLM 로그
# =========================================================================

@router.get("/llm-logs/index", response_model=LLMLogIndex)
async def get_llm_log_index(
    date: Optional[str] = Query(None, description="조회 날짜 (YYYY-MM-DD, 기본: 오늘)")
):
    """LLM 로그 일별 인덱스 조회"""
    storage = _get_llm_storage()

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    index = storage.get_index(date)
    return LLMLogIndex(**index)


@router.get("/llm-logs/detail")
async def get_llm_log_detail(
    log_path: str = Query(..., description="로그 파일 경로")
):
    """LLM 로그 상세 조회"""
    storage = _get_llm_storage()

    try:
        log_data = storage.get_log(log_path)
        return log_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-logs/by-date")
async def get_llm_logs_by_date(
    date: str = Query(..., description="조회 날짜 (YYYY-MM-DD)")
):
    """특정 날짜의 모든 LLM 로그 조회"""
    storage = _get_llm_storage()
    logs = storage.get_logs_by_date(date)
    return logs


@router.get("/llm-logs/by-stock/{stock_code}")
async def get_llm_logs_by_stock(
    stock_code: str,
    days: int = Query(7, ge=1, le=30, description="조회 일수")
):
    """특정 종목의 LLM 로그 조회"""
    storage = _get_llm_storage()
    logs = storage.get_logs_by_stock(stock_code=stock_code, days=days)
    return logs


# =========================================================================
# 모델 성능
# =========================================================================

@router.get("/model-performance", response_model=List[ModelPerformance])
async def get_model_performance(
    days: int = Query(7, ge=1, le=30, description="분석 기간 (일)")
):
    """모델별 성능 통계 조회"""
    storage = _get_llm_storage()
    performance = storage.get_model_performance(days=days)

    result = []
    for model_name, stats in performance.items():
        result.append(ModelPerformance(
            model_name=model_name,
            **stats
        ))

    return result


# =========================================================================
# 저장소 관리
# =========================================================================

@router.get("/storage/stats", response_model=StorageStats)
async def get_storage_stats():
    """저장소 통계 조회"""
    storage = _get_llm_storage()
    stats = storage.get_storage_stats()
    return StorageStats(**stats)


@router.post("/export/csv")
async def export_to_csv(
    date: Optional[str] = Query(None, description="내보낼 날짜 (None이면 전체)")
):
    """
    데이터를 CSV로 내보내기

    Returns:
        Dict: 생성된 파일 경로
    """
    db = _get_trade_db()

    # 내보내기 디렉토리
    export_dir = str(Path(__file__).parent.parent.parent.parent / "data" / "export")

    try:
        exported = db.export_to_csv(export_dir, date)
        return {
            "success": True,
            "files": exported,
            "export_dir": export_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/llm-logs/cleanup")
async def cleanup_old_logs(
    keep_days: int = Query(30, ge=7, le=365, description="보관 일수")
):
    """
    오래된 LLM 로그 삭제

    Returns:
        Dict: 삭제된 파일 수
    """
    storage = _get_llm_storage()
    deleted = storage.cleanup_old_logs(keep_days=keep_days)

    return {
        "success": True,
        "deleted_folders": deleted,
        "keep_days": keep_days
    }
