# -*- coding: utf-8 -*-
"""애플리케이션 설정."""

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정."""

    # 서버 설정
    app_name: str = "Trading Dashboard API"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS 설정
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # KIS API 설정
    kis_env: str = "prod"  # prod 또는 vps (모의투자)

    # 프로젝트 경로
    project_root: Path = Path(__file__).parent.parent.parent  # news_trading 폴더

    # 급등 종목 스캔 설정
    surge_min_score: float = 50.0
    surge_scan_interval: int = 60
    surge_auto_scan: bool = True

    # LLM 설정
    llm_preset: str = "deepseek"
    llm_auto_analyze: bool = True
    llm_analyze_interval: int = 120
    llm_max_analyze_count: int = 5
    llm_parallel: bool = False
    llm_keep_alive: str = "5m"

    # Ollama 설정
    ollama_host: str = "http://localhost:11434"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# 전역 설정 인스턴스
settings = Settings()
