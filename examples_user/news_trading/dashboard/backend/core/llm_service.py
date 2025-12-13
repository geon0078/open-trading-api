# -*- coding: utf-8 -*-
"""LLM 분석 서비스."""

import asyncio
import logging
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# 기존 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # news_trading

from config import settings
from models.llm import (
    AnalysisHistoryItem,
    AnalyzeRequest,
    EnsembleAnalysisResult,
    LLMSettings,
    ModelResult,
)

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 분석 서비스."""

    def __init__(self):
        self._ensemble = None
        self._single = None
        self._history: deque = deque(maxlen=100)
        self._is_analyzing = False
        self._settings = LLMSettings()
        self._current_analysis: Optional[EnsembleAnalysisResult] = None

    def _ensure_ensemble(self):
        """EnsembleLLMAnalyzer 초기화."""
        if self._ensemble is None:
            try:
                from modules.ensemble_analyzer import EnsembleLLMAnalyzer
                self._ensemble = EnsembleLLMAnalyzer(
                    keep_alive=self._settings.keep_alive,
                    auto_unload=False
                )
                logger.info("EnsembleLLMAnalyzer 초기화 완료")
            except ImportError as e:
                logger.error(f"EnsembleLLMAnalyzer 임포트 실패: {e}")
                raise

    def _ensure_single(self):
        """LLMAnalyzer 초기화."""
        if self._single is None:
            try:
                from modules.llm_analyzer import LLMAnalyzer
                self._single = LLMAnalyzer()
                logger.info("LLMAnalyzer 초기화 완료")
            except ImportError as e:
                logger.error(f"LLMAnalyzer 임포트 실패: {e}")
                raise

    async def analyze_stock(
        self,
        stock_code: str,
        stock_name: str = "",
        stock_data: Optional[Dict[str, Any]] = None,
        news_list: Optional[List[str]] = None,
        parallel: bool = False
    ) -> EnsembleAnalysisResult:
        """종목 앙상블 분석."""
        if self._is_analyzing:
            raise RuntimeError("이미 분석이 진행 중입니다.")

        self._is_analyzing = True
        analysis_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            self._ensure_ensemble()

            # 기본 종목 데이터 구성
            if stock_data is None:
                stock_data = {
                    "code": stock_code,
                    "name": stock_name,
                }

            # 뉴스 조회
            if news_list is None:
                news_list = await self._get_news_for_stock(stock_code, stock_name)

            loop = asyncio.get_event_loop()

            def _run_analysis():
                # 앙상블 분석 실행
                if not self._ensemble.ensemble_models:
                    self._ensemble.setup_ensemble(use_financial_ensemble=True)

                result = self._ensemble.analyze_stock(
                    stock_data=stock_data,
                    news_list=news_list,
                    parallel=parallel
                )
                return result

            original_result = await loop.run_in_executor(None, _run_analysis)

            # Pydantic 모델로 변환
            result = self._convert_to_pydantic(original_result, analysis_id, stock_data, news_list)
            result.total_processing_time = time.time() - start_time

            # 히스토리에 저장
            self._history.append(result)
            self._current_analysis = result

            logger.info(
                f"앙상블 분석 완료: {stock_code} - {result.ensemble_signal} "
                f"(신뢰도: {result.ensemble_confidence:.2f}, 합의도: {result.consensus_score:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"앙상블 분석 실패: {e}")
            return EnsembleAnalysisResult(
                id=analysis_id,
                timestamp=datetime.now(),
                stock_code=stock_code,
                stock_name=stock_name,
                ensemble_signal="ERROR",
                ensemble_confidence=0.0,
                ensemble_trend="UNKNOWN",
                success=False,
                error_message=str(e)
            )
        finally:
            self._is_analyzing = False

    async def analyze_stock_streaming(
        self,
        stock_code: str,
        stock_name: str = "",
        stock_data: Optional[Dict[str, Any]] = None,
        news_list: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """실시간 스트리밍 앙상블 분석."""
        analysis_id = str(uuid.uuid4())
        self._is_analyzing = True

        try:
            self._ensure_ensemble()

            # 분석 시작 이벤트
            yield {
                "event": "analysis_started",
                "data": {
                    "id": analysis_id,
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "timestamp": datetime.now().isoformat(),
                    "models": self._settings.models
                }
            }

            # 기본 종목 데이터 구성
            if stock_data is None:
                stock_data = {"code": stock_code, "name": stock_name}

            # 뉴스 조회
            if news_list is None:
                news_list = await self._get_news_for_stock(stock_code, stock_name)

            # 앙상블 설정
            if not self._ensemble.ensemble_models:
                self._ensemble.setup_ensemble(use_financial_ensemble=True)

            # 프롬프트 생성
            prompt = self._ensemble.build_prompt(stock_data, news_list)

            yield {
                "event": "prompt_generated",
                "data": {
                    "prompt": prompt,
                    "input_data": stock_data,
                    "news_list": news_list
                }
            }

            # 각 모델 순차 호출
            model_results = []
            for model_name in self._ensemble.ensemble_models:
                # 모델 처리 시작
                yield {
                    "event": "model_started",
                    "data": {
                        "model": model_name,
                        "status": "processing"
                    }
                }

                start_time = time.time()

                try:
                    # 동기 호출을 비동기로 래핑
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda m=model_name: self._ensemble._call_single_model(m, prompt)
                    )

                    processing_time = time.time() - start_time
                    model_results.append(result)

                    # 모델 결과 전송
                    yield {
                        "event": "model_completed",
                        "data": {
                            "model": model_name,
                            "result": {
                                "signal": getattr(result, "signal", "HOLD"),
                                "confidence": getattr(result, "confidence", 0.0),
                                "trend_prediction": getattr(result, "trend_prediction", "SIDEWAYS"),
                                "reasoning": getattr(result, "reasoning", ""),
                                "processing_time": processing_time,
                                "success": getattr(result, "success", True),
                                "raw_output": getattr(result, "raw_output", ""),
                                "error_message": getattr(result, "error_message", "")
                            }
                        }
                    }
                except Exception as e:
                    yield {
                        "event": "model_error",
                        "data": {
                            "model": model_name,
                            "error": str(e)
                        }
                    }

            # 앙상블 집계
            try:
                ensemble_result = self._ensemble._aggregate_results(
                    model_results, stock_data, prompt, news_list
                )

                result_model = self._convert_to_pydantic(
                    ensemble_result, analysis_id, stock_data, news_list
                )

                # 히스토리에 저장
                self._history.append(result_model)
                self._current_analysis = result_model

                # 최종 결과 전송
                yield {
                    "event": "analysis_completed",
                    "data": result_model.model_dump()
                }

            except Exception as e:
                yield {
                    "event": "analysis_error",
                    "data": {"error": str(e)}
                }

        except Exception as e:
            yield {
                "event": "analysis_error",
                "data": {"id": analysis_id, "error": str(e)}
            }
        finally:
            self._is_analyzing = False

    async def _get_news_for_stock(self, stock_code: str, stock_name: str) -> List[str]:
        """종목 관련 뉴스 조회."""
        try:
            self._ensure_single()
            loop = asyncio.get_event_loop()

            def _fetch_news():
                return self._single.get_news_for_stock(stock_code, stock_name)

            news_list = await loop.run_in_executor(None, _fetch_news)
            return news_list or []
        except Exception as e:
            logger.warning(f"뉴스 조회 실패: {e}")
            return []

    def _convert_to_pydantic(
        self,
        original,
        analysis_id: str,
        stock_data: Dict[str, Any],
        news_list: List[str]
    ) -> EnsembleAnalysisResult:
        """기존 결과를 Pydantic 모델로 변환."""
        # 모델별 결과 변환
        model_results = []
        for mr in getattr(original, "model_results", []):
            model_results.append(ModelResult(
                model_name=getattr(mr, "model_name", "unknown"),
                signal=getattr(mr, "signal", "HOLD"),
                confidence=float(getattr(mr, "confidence", 0.0)),
                trend_prediction=getattr(mr, "trend_prediction", "SIDEWAYS"),
                entry_price=float(getattr(mr, "entry_price", 0)),
                stop_loss=float(getattr(mr, "stop_loss", 0)),
                take_profit=float(getattr(mr, "take_profit", 0)),
                reasoning=getattr(mr, "reasoning", ""),
                news_impact=getattr(mr, "news_impact", "NEUTRAL"),
                risk_factors=list(getattr(mr, "risk_factors", [])),
                processing_time=float(getattr(mr, "processing_time", 0)),
                success=getattr(mr, "success", True),
                raw_output=getattr(mr, "raw_output", ""),
                error_message=getattr(mr, "error_message", "")
            ))

        return EnsembleAnalysisResult(
            id=analysis_id,
            timestamp=datetime.now(),
            stock_code=getattr(original, "stock_code", stock_data.get("code", "")),
            stock_name=getattr(original, "stock_name", stock_data.get("name", "")),
            ensemble_signal=getattr(original, "ensemble_signal", "HOLD"),
            ensemble_confidence=float(getattr(original, "ensemble_confidence", 0.0)),
            ensemble_trend=getattr(original, "ensemble_trend", "SIDEWAYS"),
            current_price=int(stock_data.get("price", 0)),
            avg_entry_price=float(getattr(original, "avg_entry_price", 0)),
            avg_stop_loss=float(getattr(original, "avg_stop_loss", 0)),
            avg_take_profit=float(getattr(original, "avg_take_profit", 0)),
            signal_votes=dict(getattr(original, "signal_votes", {})),
            trend_votes=dict(getattr(original, "trend_votes", {})),
            model_results=model_results,
            models_used=list(getattr(original, "models_used", [])),
            models_agreed=int(getattr(original, "models_agreed", 0)),
            total_models=int(getattr(original, "total_models", len(model_results))),
            consensus_score=float(getattr(original, "consensus_score", 0.0)),
            input_prompt=getattr(original, "input_prompt", ""),
            input_data=stock_data,
            news_list=news_list,
            success=getattr(original, "success", True),
            error_message=getattr(original, "error_message", "")
        )

    def get_history(self, limit: int = 20) -> List[AnalysisHistoryItem]:
        """분석 히스토리 조회."""
        items = []
        for result in list(self._history)[-limit:]:
            items.append(AnalysisHistoryItem(
                id=result.id,
                timestamp=result.timestamp,
                stock_code=result.stock_code,
                stock_name=result.stock_name,
                ensemble_signal=result.ensemble_signal,
                ensemble_confidence=result.ensemble_confidence,
                consensus_score=result.consensus_score,
                models_used=result.models_used,
                success=result.success
            ))
        return list(reversed(items))

    def get_analysis_by_id(self, analysis_id: str) -> Optional[EnsembleAnalysisResult]:
        """ID로 분석 결과 조회."""
        for result in self._history:
            if result.id == analysis_id:
                return result
        return None

    def get_settings(self) -> LLMSettings:
        """LLM 설정 조회."""
        return self._settings

    def update_settings(self, new_settings: LLMSettings) -> LLMSettings:
        """LLM 설정 업데이트."""
        self._settings = new_settings
        logger.info(f"LLM 설정 업데이트: preset={new_settings.preset}")
        return self._settings

    @property
    def is_analyzing(self) -> bool:
        """분석 중 여부."""
        return self._is_analyzing

    @property
    def current_analysis(self) -> Optional[EnsembleAnalysisResult]:
        """현재 진행 중인 분석."""
        return self._current_analysis


# 싱글톤 인스턴스
llm_service = LLMService()
