# -*- coding: utf-8 -*-
"""
하이브리드 LLM 앙상블 모듈

금융 뉴스 분석을 위한 하이브리드 LLM 앙상블 시스템

프리셋 (preset):
- "default": EXAONE 4.0 32B + Fin-R1 + Qwen3 8B (32GB VRAM)
- "deepseek": DeepSeek-R1 중심 앙상블 (금융 추론 특화, 권장)
- "lightweight": 경량 앙상블 (16GB VRAM 이하)

사용 예시:
    >>> from modules.llm import FinancialHybridLLM
    >>> llm = FinancialHybridLLM()
    >>> llm.set_preset("deepseek")  # DeepSeek 중심으로 전환
    >>> result = llm.analyze("삼성전자 실적 호조", "005930")
"""

from .hybrid_llm_32gb import (
    FinancialHybridLLM,
    AnalysisResult,
    EnsembleResult,
    SentimentType,
    ImpactLevel
)

__all__ = [
    "FinancialHybridLLM",
    "AnalysisResult",
    "EnsembleResult",
    "SentimentType",
    "ImpactLevel"
]

__version__ = "1.1.0"  # DeepSeek 프리셋 추가
