# -*- coding: utf-8 -*-
"""SSE 스트리밍 API 엔드포인트."""

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from core.llm_service import llm_service
from core.surge_service import surge_service
from models.llm import AnalyzeRequest

router = APIRouter()


async def surge_event_generator() -> AsyncGenerator[dict, None]:
    """급등 종목 SSE 이벤트 생성기."""
    last_scan = None

    while True:
        try:
            # 마지막 스캔 시간이 변경되었으면 업데이트 전송
            if surge_service.last_scan_time != last_scan:
                last_scan = surge_service.last_scan_time
                candidates = surge_service.get_cached_candidates()

                yield {
                    "event": "surge_update",
                    "data": json.dumps({
                        "candidates": [c.model_dump() for c in candidates.candidates],
                        "total_count": candidates.total_count,
                        "timestamp": candidates.timestamp.isoformat()
                    }, default=str, ensure_ascii=False)
                }

            # 15초마다 heartbeat
            yield {
                "event": "heartbeat",
                "data": json.dumps({"time": datetime.now().isoformat()})
            }

            await asyncio.sleep(5)

        except asyncio.CancelledError:
            break
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
            await asyncio.sleep(5)


@router.get("/surge")
async def stream_surge():
    """급등 종목 SSE 스트림."""
    return EventSourceResponse(surge_event_generator())


async def llm_analysis_generator(
    stock_code: str,
    stock_name: str = ""
) -> AsyncGenerator[dict, None]:
    """LLM 분석 SSE 이벤트 생성기."""
    async for event in llm_service.analyze_stock_streaming(
        stock_code=stock_code,
        stock_name=stock_name
    ):
        yield {
            "event": event.get("event", "message"),
            "data": json.dumps(event.get("data", {}), default=str, ensure_ascii=False)
        }


@router.get("/llm/analyze")
async def stream_llm_analysis(
    stock_code: str = Query(..., description="종목코드"),
    stock_name: str = Query("", description="종목명")
):
    """LLM 분석 SSE 스트림."""
    if llm_service.is_analyzing:
        async def error_generator():
            yield {
                "event": "error",
                "data": json.dumps({"error": "이미 분석이 진행 중입니다."})
            }
        return EventSourceResponse(error_generator())

    return EventSourceResponse(
        llm_analysis_generator(stock_code=stock_code, stock_name=stock_name)
    )


async def all_events_generator() -> AsyncGenerator[dict, None]:
    """통합 SSE 이벤트 생성기."""
    last_surge_scan = None
    last_llm_analysis = None

    while True:
        try:
            # 급등 종목 업데이트 확인
            if surge_service.last_scan_time != last_surge_scan:
                last_surge_scan = surge_service.last_scan_time
                candidates = surge_service.get_cached_candidates()

                yield {
                    "event": "surge_update",
                    "data": json.dumps({
                        "candidates": [c.model_dump() for c in candidates.candidates[:10]],
                        "total_count": candidates.total_count,
                        "timestamp": candidates.timestamp.isoformat()
                    }, default=str, ensure_ascii=False)
                }

            # LLM 분석 업데이트 확인
            if llm_service.current_analysis:
                if llm_service.current_analysis.id != last_llm_analysis:
                    last_llm_analysis = llm_service.current_analysis.id

                    yield {
                        "event": "llm_update",
                        "data": json.dumps(
                            llm_service.current_analysis.model_dump(),
                            default=str,
                            ensure_ascii=False
                        )
                    }

            # 상태 업데이트
            yield {
                "event": "status",
                "data": json.dumps({
                    "surge_scanning": surge_service.is_scanning,
                    "llm_analyzing": llm_service.is_analyzing,
                    "timestamp": datetime.now().isoformat()
                })
            }

            await asyncio.sleep(3)

        except asyncio.CancelledError:
            break
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
            await asyncio.sleep(5)


@router.get("/all")
async def stream_all():
    """통합 SSE 스트림."""
    return EventSourceResponse(all_events_generator())
