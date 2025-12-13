# -*- coding: utf-8 -*-
"""API v1 routers."""

from fastapi import APIRouter

from .surge import router as surge_router
from .llm import router as llm_router
from .account import router as account_router
from .stream import router as stream_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(surge_router, prefix="/surge", tags=["surge"])
api_router.include_router(llm_router, prefix="/llm", tags=["llm"])
api_router.include_router(account_router, prefix="/account", tags=["account"])
api_router.include_router(stream_router, prefix="/stream", tags=["stream"])
