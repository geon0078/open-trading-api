# -*- coding: utf-8 -*-
"""Trading Dashboard FastAPI 백엔드."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 기존 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))  # news_trading
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # examples_user

from config import settings
from api.v1 import api_router
from core.surge_service import surge_service
from core.kis_service import kis_service

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# 백그라운드 태스크
background_tasks = set()


async def auto_scan_task():
    """자동 급등 종목 스캔 태스크."""
    logger.info("자동 스캔 태스크 시작")

    while True:
        try:
            scan_settings = surge_service.get_settings()

            if scan_settings.auto_scan:
                logger.debug("자동 급등 종목 스캔 실행 중...")
                await surge_service.scan_stocks(
                    min_score=scan_settings.min_score,
                    limit=scan_settings.limit
                )

            await asyncio.sleep(scan_settings.scan_interval)

        except asyncio.CancelledError:
            logger.info("자동 스캔 태스크 종료")
            break
        except Exception as e:
            logger.error(f"자동 스캔 오류: {e}")
            await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리."""
    logger.info("=" * 60)
    logger.info(f"Trading Dashboard API v{settings.app_version} 시작")
    logger.info("=" * 60)

    # KIS API 인증 시도
    try:
        await kis_service.ensure_auth()
        logger.info("KIS API 인증 완료")
    except Exception as e:
        logger.warning(f"KIS API 인증 실패 (나중에 재시도): {e}")

    # 초기 스캔 실행
    try:
        await surge_service.scan_stocks()
        logger.info("초기 급등 종목 스캔 완료")
    except Exception as e:
        logger.warning(f"초기 스캔 실패: {e}")

    # 백그라운드 태스크 시작
    task = asyncio.create_task(auto_scan_task())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    yield

    # 백그라운드 태스크 종료
    logger.info("백그라운드 태스크 종료 중...")
    for task in background_tasks:
        task.cancel()

    await asyncio.gather(*background_tasks, return_exceptions=True)
    logger.info("Trading Dashboard API 종료")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="실시간 트레이딩 대시보드 API - 급등 종목 탐지, LLM 분석, 계좌 관리",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router)


@app.get("/")
async def root():
    """루트 엔드포인트."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """헬스 체크."""
    return {
        "status": "healthy",
        "kis_authenticated": kis_service._authenticated,
        "surge_scanning": surge_service.is_scanning,
        "surge_cached_count": len(surge_service._cached_candidates),
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 핸들러."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
