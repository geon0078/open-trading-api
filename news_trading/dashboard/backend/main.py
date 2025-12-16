# -*- coding: utf-8 -*-
"""Trading Dashboard FastAPI 백엔드."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 기존 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))  # news_trading
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # open-trading-api
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples_user"))  # examples_user (for domestic_stock)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples_user" / "domestic_stock"))  # kis_auth

from config import settings
from api.v1 import api_router
from core.surge_service import surge_service
from core.kis_service import kis_service
from core.position_monitor import get_position_monitor
from core.auto_trade_service import set_llm_output_callback
from api.v1.ws import broadcast_event, broadcast_account_update, broadcast_execution, broadcast_llm_output

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


# 전역 이벤트 루프 참조 저장
_main_event_loop = None


async def setup_llm_output_callback():
    """LLM 출력 WebSocket 콜백 설정 (async 버전)"""
    global _main_event_loop

    # 현재 실행 중인 이벤트 루프를 저장
    _main_event_loop = asyncio.get_running_loop()
    logger.info(f"이벤트 루프 저장됨: {_main_event_loop}")

    def on_llm_output(stock_code: str, stock_name: str, model_name: str,
                      output_type: str, content: str, **kwargs):
        """LLM 출력 브로드캐스트 (동기 함수 -> 비동기 호출)"""
        global _main_event_loop

        if _main_event_loop is None:
            logger.warning("이벤트 루프가 설정되지 않음")
            return

        try:
            coro = broadcast_llm_output(
                stock_code=stock_code,
                stock_name=stock_name,
                model_name=model_name,
                output_type=output_type,
                content=content,
                **kwargs
            )

            # 다른 스레드에서 호출되므로 run_coroutine_threadsafe 사용
            # 이벤트 루프가 닫혔는지 확인
            if not _main_event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(coro, _main_event_loop)
            else:
                logger.warning("이벤트 루프가 닫혀있음")

        except Exception as e:
            logger.error(f"LLM 출력 브로드캐스트 오류: {e}", exc_info=True)

    set_llm_output_callback(on_llm_output)
    logger.info("LLM 출력 WebSocket 콜백 설정 완료")


async def setup_position_monitor_callbacks():
    """포지션 모니터 WebSocket 콜백 설정"""
    position_monitor = get_position_monitor()

    async def on_alert(alert):
        """손절/익절 알림 브로드캐스트"""
        from dataclasses import asdict
        await broadcast_event("position_alert", asdict(alert))

    async def on_sell_executed(data: Dict):
        """매도 실행 브로드캐스트"""
        await broadcast_execution(data)
        # 계좌 정보 갱신 트리거
        await broadcast_event("refresh_account", {"reason": f"{data.get('alert_type', 'unknown')}_sell"})

    async def on_position_update(holdings: List[Dict]):
        """포지션 업데이트 브로드캐스트"""
        await broadcast_account_update({
            "holdings": holdings,
            "count": len(holdings),
            "updated_at": datetime.now().isoformat()
        })

    position_monitor.set_callbacks(
        on_alert=on_alert,
        on_sell_executed=on_sell_executed,
        on_position_update=on_position_update
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리."""
    logger.info("=" * 60)
    logger.info(f"Trading Dashboard API v{settings.app_version} 시작")
    logger.info("=" * 60)

    # LLM 출력 콜백 설정 (AutoTrader 초기화 전에 설정)
    await setup_llm_output_callback()

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

    # 포지션 모니터 설정 및 시작
    try:
        await setup_position_monitor_callbacks()
        position_monitor = get_position_monitor()
        await position_monitor.start()
        logger.info("포지션 모니터 시작 완료")
    except Exception as e:
        logger.warning(f"포지션 모니터 시작 실패: {e}")

    yield

    # 포지션 모니터 종료
    try:
        position_monitor = get_position_monitor()
        await position_monitor.stop()
        logger.info("포지션 모니터 종료")
    except Exception as e:
        logger.warning(f"포지션 모니터 종료 오류: {e}")

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
    position_monitor = get_position_monitor()

    return {
        "status": "healthy",
        "kis_authenticated": kis_service._authenticated,
        "surge_scanning": surge_service.is_scanning,
        "surge_cached_count": len(surge_service._cached_candidates),
        "position_monitor_running": position_monitor.is_running(),
        "position_monitor_config": position_monitor.get_config(),
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
