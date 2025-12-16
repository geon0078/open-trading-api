# -*- coding: utf-8 -*-
"""API v1 routers."""

from fastapi import APIRouter

from .surge import router as surge_router
from .llm import router as llm_router
from .account import router as account_router
from .stream import router as stream_router
from .auto_trade import router as auto_trade_router
from .ws import router as ws_router
from .position import router as position_router
from .integrated_trade import router as integrated_trade_router
from .history import router as history_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(surge_router, prefix="/surge", tags=["surge"])
api_router.include_router(llm_router, prefix="/llm", tags=["llm"])
api_router.include_router(account_router, prefix="/account", tags=["account"])
api_router.include_router(stream_router, prefix="/stream", tags=["stream"])
api_router.include_router(auto_trade_router, prefix="/auto-trade", tags=["auto-trade"])
api_router.include_router(ws_router, prefix="/ws", tags=["websocket"])
api_router.include_router(position_router, prefix="/position", tags=["position"])
api_router.include_router(integrated_trade_router, prefix="/integrated", tags=["integrated-llm"])
api_router.include_router(history_router, prefix="/history", tags=["history"])
