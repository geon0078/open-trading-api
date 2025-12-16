# -*- coding: utf-8 -*-
"""
앙상블 LLM 분석기

여러 LLM 모델의 결과를 종합하여 더 정확한 매매 신호를 생성합니다.

앙상블 방식:
1. 다중 모델 호출 (병렬/순차)
2. 투표 기반 시그널 결정
3. 가중 평균 신뢰도 계산
4. 모델별 결과 비교 분석

"""

import os
import sys
import json
import logging
import time
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# OHLCV 및 기술적 지표 모듈
try:
    from .ohlcv_fetcher import OHLCVFetcher
    from .technical_indicators import TechnicalAnalyzer
except ImportError:
    try:
        from ohlcv_fetcher import OHLCVFetcher
        from technical_indicators import TechnicalAnalyzer
    except ImportError:
        OHLCVFetcher = None
        TechnicalAnalyzer = None

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """개별 모델 분석 결과"""
    model_name: str
    signal: str
    confidence: float
    trend_prediction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    news_impact: str
    risk_factors: List[str]
    processing_time: float
    success: bool
    raw_output: str
    error_message: str = ""


@dataclass
class EnsembleAnalysis:
    """앙상블 분석 결과"""
    timestamp: str
    stock_code: str
    stock_name: str

    # 앙상블 결과
    ensemble_signal: str  # 최종 시그널 (투표 결과)
    ensemble_confidence: float  # 평균 신뢰도
    ensemble_trend: str  # 다수결 추세

    # 가격 (평균값)
    avg_entry_price: float
    avg_stop_loss: float
    avg_take_profit: float

    # 투표 결과
    signal_votes: Dict[str, int]  # {"STRONG_BUY": 2, "BUY": 1}
    trend_votes: Dict[str, int]

    # 모델별 결과
    model_results: List[ModelResult]
    models_used: List[str]
    models_agreed: int  # 동의한 모델 수
    total_models: int

    # 합의도
    consensus_score: float  # 0~1, 모델 간 합의 정도

    # Input 데이터
    input_prompt: str
    input_data: Dict

    # 메타
    total_processing_time: float
    success: bool
    error_message: str = ""


class EnsembleLLMAnalyzer:
    """
    앙상블 LLM 분석기

    여러 모델의 결과를 종합하여 더 신뢰할 수 있는 매매 신호를 생성합니다.
    """

    # 시그널 가중치 (투표 집계용)
    SIGNAL_WEIGHTS = {
        "STRONG_BUY": 2,
        "BUY": 1,
        "HOLD": 0,
        "SELL": -1,
        "STRONG_SELL": -2
    }

    # 제외할 모델 (JSON 응답 불가 또는 너무 느린 모델)
    EXCLUDED_MODELS = [
        "exaone-1.2b:latest",  # JSON 대신 </think>만 반환
        "exaone-1.2b",
        # 30B 이상 모델은 너무 느림 (50초+ 소요)
        "packeting/Qwen2.5-VL-32B-Instruct:latest",
        "qwen2.5-coder:32b",
        "Qwen2.5-VL-32B",
        "qwen3-coder:30b",     # ~50초 (너무 느림)
    ]

    # 빠른 모델 (10초 이내 응답, 실시간 분석용)
    # DeepSeek 모델 추가 - 금융 추론에 강력한 성능
    FAST_MODELS = [
        "deepseek-r1:8b",      # ~6초, 금융 추론 최강
        "deepseek-r1:1.5b",    # ~2초, 경량 추론
        "qwen3:8b",            # ~5초
        "qwen2.5:7b",          # ~5초
        "qwen2.5:3b",          # ~2초
    ]

    # 금융 특화 앙상블 모델 (권장 조합)
    FINANCIAL_ENSEMBLE = [
        "deepseek-r1:8b",         # 금융 추론
        "qwen3:8b",               # 한국어 + 범용
        "solar:10.7b",            # 한국어 특화
    ]

    # 메인 판단 모델 (앙상블 결과를 최종 종합)
    MAIN_JUDGE_MODEL = "ingu627/exaone4.0:32b"

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        keep_alive: str = "5m",
        auto_unload: bool = True,
        on_llm_output: Optional[callable] = None
    ):
        """
        앙상블 LLM 분석기 초기화

        Args:
            ollama_url: Ollama API URL
            keep_alive: 모델 유지 시간 (기본: 5분, "0"=즉시 언로드, "-1"=영구 유지)
            auto_unload: 분석 완료 후 자동 언로드 여부
            on_llm_output: LLM 출력 콜백 함수 (WebSocket 브로드캐스트용)
        """
        self.ollama_url = ollama_url
        self.keep_alive = keep_alive
        self.auto_unload = auto_unload
        self.on_llm_output = on_llm_output  # LLM 출력 콜백
        self.available_models: List[str] = []
        self.ensemble_models: List[str] = []  # 앙상블에 사용할 모델
        self.model_weights: Dict[str, float] = {}  # 모델별 가중치
        self.analysis_history: List[EnsembleAnalysis] = []
        self.max_history = 50
        self._lock = threading.Lock()
        self._loaded_models: List[str] = []  # 현재 로드된 모델 추적

        # 현재 분석 중인 종목 정보 (WebSocket 브로드캐스트용)
        self._current_stock_code = ""
        self._current_stock_name = ""

    def discover_models(self) -> List[str]:
        """사용 가능한 모델 탐색 (제외 모델 필터링)"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                all_models = [m["name"] for m in models]

                # 제외 모델 필터링
                self.available_models = [
                    m for m in all_models
                    if not any(excluded in m for excluded in self.EXCLUDED_MODELS)
                ]

                excluded = [m for m in all_models if m not in self.available_models]
                if excluded:
                    logger.info(f"제외된 모델: {excluded}")
                logger.info(f"사용 가능 모델: {self.available_models}")
                return self.available_models
        except Exception as e:
            logger.error(f"모델 탐색 실패: {e}")
        return []

    # =========================================================================
    # GPU 메모리 관리
    # =========================================================================

    def get_gpu_status(self) -> Dict:
        """
        현재 GPU 메모리 상태 조회

        Returns:
            Dict: {
                "running_models": [{"name": str, "size": int, "vram": float}],
                "total_vram_gb": float,
                "available": bool
            }
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                running = []
                total_vram = 0
                for m in models:
                    size_gb = m.get("size", 0) / (1024**3)
                    vram_gb = m.get("size_vram", m.get("size", 0)) / (1024**3)
                    running.append({
                        "name": m.get("name", "unknown"),
                        "size_gb": round(size_gb, 2),
                        "vram_gb": round(vram_gb, 2)
                    })
                    total_vram += vram_gb

                self._loaded_models = [m["name"] for m in running]

                return {
                    "running_models": running,
                    "total_vram_gb": round(total_vram, 2),
                    "available": True
                }
        except Exception as e:
            logger.error(f"GPU 상태 조회 실패: {e}")

        return {"running_models": [], "total_vram_gb": 0, "available": False}

    def unload_model(self, model_name: str) -> bool:
        """
        특정 모델을 GPU 메모리에서 언로드

        Args:
            model_name: 언로드할 모델명

        Returns:
            bool: 성공 여부
        """
        try:
            # Ollama에서 모델 언로드 (keep_alive=0으로 즉시 해제)
            payload = {
                "model": model_name,
                "keep_alive": 0
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                logger.info(f"모델 언로드 완료: {model_name}")
                if model_name in self._loaded_models:
                    self._loaded_models.remove(model_name)
                return True
        except Exception as e:
            logger.error(f"모델 언로드 실패 ({model_name}): {e}")
        return False

    def unload_all_models(self) -> int:
        """
        모든 로드된 모델을 GPU 메모리에서 언로드

        Returns:
            int: 언로드된 모델 수
        """
        status = self.get_gpu_status()
        unloaded = 0

        for model in status["running_models"]:
            if self.unload_model(model["name"]):
                unloaded += 1

        logger.info(f"총 {unloaded}개 모델 언로드 완료")
        return unloaded

    def preload_model(self, model_name: str) -> bool:
        """
        모델을 미리 GPU 메모리에 로드 (워밍업)

        Args:
            model_name: 로드할 모델명

        Returns:
            bool: 성공 여부
        """
        try:
            payload = {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"num_predict": 1}
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                logger.info(f"모델 프리로드 완료: {model_name}")
                if model_name not in self._loaded_models:
                    self._loaded_models.append(model_name)
                return True
        except Exception as e:
            logger.error(f"모델 프리로드 실패 ({model_name}): {e}")
        return False

    def optimize_memory(self, max_vram_gb: float = 12.0) -> None:
        """
        GPU 메모리 최적화 - 지정된 VRAM 이하로 유지

        Args:
            max_vram_gb: 최대 허용 VRAM (GB)
        """
        status = self.get_gpu_status()

        if status["total_vram_gb"] > max_vram_gb:
            logger.warning(f"VRAM 초과 ({status['total_vram_gb']:.1f}GB > {max_vram_gb}GB), 최적화 시작")

            # 앙상블에 사용하지 않는 모델부터 언로드
            for model in status["running_models"]:
                if model["name"] not in self.ensemble_models:
                    self.unload_model(model["name"])

                    # 다시 확인
                    new_status = self.get_gpu_status()
                    if new_status["total_vram_gb"] <= max_vram_gb:
                        break

    def setup_ensemble(
        self,
        models: List[str] = None,
        weights: Dict[str, float] = None,
        single_model: bool = False,
        use_financial_ensemble: bool = False
    ):
        """앙상블 모델 설정

        Args:
            models: 사용할 모델 목록 (None이면 자동 선택)
            weights: 모델별 가중치
            single_model: True면 단일 모델만 사용 (GPU 부하 최소화)
            use_financial_ensemble: True면 금융 특화 앙상블 사용 (DeepSeek + Qwen3 + Fin-R1)
        """
        if not self.available_models:
            self.discover_models()

        if models:
            # 지정된 모델 사용
            self.ensemble_models = [m for m in models if m in self.available_models]
        elif use_financial_ensemble:
            # 금융 특화 앙상블 (DeepSeek 중심)
            self.ensemble_models = []
            for target_model in self.FINANCIAL_ENSEMBLE:
                # 유연한 모델 매칭 (다양한 명명 규칙 처리)
                target_base = target_model.split(":")[0].lower()  # e.g., "solar"
                found = False

                for available in self.available_models:
                    available_lower = available.lower()

                    # 정확히 일치
                    if target_model == available or target_model.lower() == available_lower:
                        self.ensemble_models.append(available)
                        found = True
                        break

                    # 기본 이름으로 시작하는지 체크 (e.g., "solar" in "solar:10.7b")
                    if target_base in available_lower.split(":")[0]:
                        self.ensemble_models.append(available)
                        found = True
                        break

                    # 기본 이름이 포함되어 있는지 체크
                    if target_base in available_lower:
                        self.ensemble_models.append(available)
                        found = True
                        break

                if not found:
                    logger.warning(f"모델 '{target_model}' 찾을 수 없음")

            # 최소 2개 모델 보장
            if len(self.ensemble_models) < 2:
                logger.warning(f"금융 앙상블 모델 부족 ({len(self.ensemble_models)}개). 대체 모델 추가...")
                for m in self.FAST_MODELS:
                    if m in self.available_models and m not in self.ensemble_models:
                        self.ensemble_models.append(m)
                    if len(self.ensemble_models) >= 3:
                        break

            logger.info(f"금융 특화 앙상블 설정: {self.ensemble_models}")
        elif single_model:
            # 단일 모델 모드 - GPU 부하 최소화
            # 빠른 모델 우선 선택
            for m in self.FAST_MODELS:
                if m in self.available_models:
                    self.ensemble_models = [m]
                    break
            if not self.ensemble_models and self.available_models:
                self.ensemble_models = [self.available_models[0]]
        else:
            # 기본: 빠른 모델만 사용 (10초 이내 응답)
            # FAST_MODELS 목록에서 사용 가능한 모델 선택
            self.ensemble_models = []

            for m in self.FAST_MODELS:
                if m in self.available_models:
                    self.ensemble_models.append(m)
                if len(self.ensemble_models) >= 3:  # 최대 3개
                    break

            # FAST_MODELS에서 3개를 못 찾으면 나머지 모델 추가
            if len(self.ensemble_models) < 2:
                for m in self.available_models:
                    if m not in self.ensemble_models:
                        self.ensemble_models.append(m)
                    if len(self.ensemble_models) >= 3:
                        break

        # 가중치 설정 (기본: 균등)
        if weights:
            self.model_weights = weights
        else:
            # 모델 특성에 따른 가중치 (금융 특화 모델에 더 높은 가중치)
            self.model_weights = {}
            for model in self.ensemble_models:
                model_lower = model.lower()
                # DeepSeek-R1: 금융 추론 특화
                if "deepseek-r1" in model_lower:
                    if "32b" in model_lower:
                        self.model_weights[model] = 1.5
                    elif "8b" in model_lower:
                        self.model_weights[model] = 1.2
                    else:
                        self.model_weights[model] = 1.0
                # Solar: 한국어 특화
                elif "solar" in model_lower:
                    self.model_weights[model] = 1.0
                # Qwen3: 범용
                elif "qwen" in model_lower:
                    self.model_weights[model] = 1.0
                # 대형 모델
                elif "32b" in model_lower or "30b" in model_lower:
                    self.model_weights[model] = 1.2
                elif "8b" in model_lower or "7b" in model_lower:
                    self.model_weights[model] = 1.0
                else:
                    self.model_weights[model] = 0.8

            # 메인 판단 모델 (exaone4.0) - 최고 가중치
            self.model_weights[self.MAIN_JUDGE_MODEL] = 2.5

        logger.info(f"앙상블 모델 설정: {self.ensemble_models}")
        logger.info(f"모델 가중치: {self.model_weights}")

        return self.ensemble_models

    def build_prompt(self, stock_data: Dict, news_list: List[str] = None) -> str:
        """분석 프롬프트 생성"""
        prompt_parts = [
            "당신은 전문 스캘핑 트레이더입니다. 아래 데이터를 분석하여 3분 스캘핑 전략을 제시하세요.",
            "",
            "## 종목 정보",
            f"- 종목명: {stock_data.get('name', 'N/A')}",
            f"- 종목코드: {stock_data.get('code', 'N/A')}",
            f"- 현재가: {stock_data.get('price', 0):,}원",
            f"- 등락률: {stock_data.get('change_rate', 0):+.2f}%",
            f"- 체결강도: {stock_data.get('volume_power', 100):.1f} (100 이상 매수우세)",
            f"- 호가잔량비: {stock_data.get('balance_ratio', 1.0):.2f} (1 이상 매수우세)",
            f"- 거래량: {stock_data.get('volume', 0):,}주",
            f"- 급등점수: {stock_data.get('surge_score', 0):.1f}/100",
        ]

        if news_list:
            prompt_parts.extend([
                "",
                "## 관련 뉴스",
            ])
            for i, news in enumerate(news_list[:5], 1):
                prompt_parts.append(f"{i}. {news}")

        prompt_parts.extend([
            "",
            "## 분석 요청",
            "위 데이터를 종합하여 다음을 분석하세요:",
            "1. 현재 추세와 모멘텀 상태",
            "2. 뉴스가 주가에 미치는 영향",
            "3. 3분 내 스캘핑 진입 여부",
            "4. 구체적인 진입가, 손절가, 익절가",
            "",
            "다음 JSON 형식으로만 응답하세요:",
            "```json",
            "{",
            '    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",',
            '    "confidence": 0.0~1.0,',
            '    "trend_prediction": "UP|DOWN|SIDEWAYS",',
            '    "entry_price": 진입가(숫자),',
            '    "stop_loss": 손절가(숫자),',
            '    "take_profit": 익절가(숫자),',
            '    "reasoning": "분석 근거 (50자 이내)",',
            '    "news_impact": "POSITIVE|NEGATIVE|NEUTRAL",',
            '    "risk_factors": ["리스크1", "리스크2"]',
            "}",
            "```",
            "",
            "JSON만 출력하세요."
        ])

        return "\n".join(prompt_parts)

    def build_enhanced_prompt(
        self,
        stock_data: Dict,
        technical_summary: Dict,
        ohlcv_text: str,
        news_list: List[str] = None
    ) -> str:
        """
        OHLCV + 보조지표를 포함한 강화된 프롬프트 생성

        Args:
            stock_data: 종목 기본 정보
            technical_summary: TechnicalAnalyzer.get_scalping_summary() 결과
            ohlcv_text: TechnicalAnalyzer.format_for_llm() 결과
            news_list: 관련 뉴스 리스트
        """
        prompt_parts = [
            "당신은 전문 스캘핑 트레이더입니다. 아래 데이터를 분석하여 3분 스캘핑 전략을 제시하세요.",
            "",
            "## 종목 정보",
            f"- 종목명: {stock_data.get('name', 'N/A')}",
            f"- 종목코드: {stock_data.get('code', 'N/A')}",
            f"- 현재가: {stock_data.get('price', 0):,}원",
            f"- 등락률: {stock_data.get('change_rate', 0):+.2f}%",
            f"- 체결강도: {stock_data.get('volume_power', 100):.1f} (100 이상 매수우세)",
            f"- 호가잔량비: {stock_data.get('balance_ratio', 1.0):.2f} (1 이상 매수우세)",
            f"- 거래량: {stock_data.get('volume', 0):,}주",
            f"- 급등점수: {stock_data.get('surge_score', 0):.1f}/100",
        ]

        # 기술적 지표 추가
        if technical_summary:
            prompt_parts.extend([
                "",
                "## 기술적 지표 분석",
                f"- 추세: {technical_summary.get('trend', 'N/A')}",
                f"- 변동성: {technical_summary.get('volatility', 'N/A')}",
                f"- 종합점수: {technical_summary.get('total_score', 0):+.1f}/100",
                "",
                "### 모멘텀 지표",
                f"- RSI(14): {technical_summary.get('rsi_14', 0):.1f} (30이하 과매도, 70이상 과매수)",
                f"- RSI(7): {technical_summary.get('rsi_7', 0):.1f} (단기)",
                f"- 스토캐스틱 %K: {technical_summary.get('stoch_k', 0):.1f}",
                f"- 스토캐스틱 %D: {technical_summary.get('stoch_d', 0):.1f}",
                f"- MACD: {technical_summary.get('macd', 0):.2f}",
                f"- MACD Signal: {technical_summary.get('macd_signal', 0):.2f}",
                f"- MACD Histogram: {technical_summary.get('macd_histogram', 0):.2f}",
                "",
                "### 밴드/레벨",
                f"- 볼린저 상단: {technical_summary.get('bb_upper', 0):,.0f}원",
                f"- 볼린저 하단: {technical_summary.get('bb_lower', 0):,.0f}원",
                f"- 볼린저 %B: {technical_summary.get('bb_percent_b', 0):.2f} (0이하 하단돌파, 1이상 상단돌파)",
                f"- VWAP: {technical_summary.get('vwap', 0):,.0f}원",
                "",
                "### 이동평균",
                f"- EMA5: {technical_summary.get('ema_5', 0):,.0f}원",
                f"- EMA20: {technical_summary.get('ema_20', 0):,.0f}원",
                f"- 배열: {'정배열(상승)' if technical_summary.get('ema_5', 0) > technical_summary.get('ema_20', 0) else '역배열(하락)'}",
                "",
                "### 변동성/거래량",
                f"- ATR(14): {technical_summary.get('atr_14', 0):,.0f}원 ({technical_summary.get('atr_percent', 0):.2f}%)",
                f"- 거래량비율: {technical_summary.get('volume_ratio', 1):.2f}x (평균 대비)",
                "",
                "### 지지/저항",
                f"- 피봇: {technical_summary.get('pivot', 0):,.0f}원",
                f"- 저항1(R1): {technical_summary.get('resistance_1', 0):,.0f}원",
                f"- 지지1(S1): {technical_summary.get('support_1', 0):,.0f}원",
            ])

            # 발생한 신호들
            signals = technical_summary.get('signals', [])
            if signals:
                prompt_parts.extend(["", "### 발생 신호"])
                for sig in signals[:5]:  # 최대 5개
                    direction = "매수" if sig.get('direction') == "BUY" else "매도"
                    prompt_parts.append(f"- {sig.get('reason', '')}: {direction} (강도: {sig.get('strength', 0):.1f})")

        # OHLCV 캔들 데이터
        if ohlcv_text:
            prompt_parts.extend([
                "",
                "## 최근 캔들 데이터",
                ohlcv_text[:1000]  # 너무 길면 잘라냄
            ])

        # 뉴스
        if news_list:
            prompt_parts.extend([
                "",
                "## 관련 뉴스",
            ])
            for i, news in enumerate(news_list[:5], 1):
                prompt_parts.append(f"{i}. {news}")

        # 분석 요청
        prompt_parts.extend([
            "",
            "## 분석 요청",
            "위 데이터를 종합하여 다음을 분석하세요:",
            "1. 현재 추세와 모멘텀 상태 (기술적 지표 기반)",
            "2. 뉴스가 주가에 미치는 영향",
            "3. 3분 내 스캘핑 진입 여부",
            "4. 구체적인 진입가, 손절가(-0.5%), 익절가(+1.0~1.5%)",
            "",
            "다음 JSON 형식으로만 응답하세요:",
            "```json",
            "{",
            '    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",',
            '    "confidence": 0.0~1.0,',
            '    "trend_prediction": "STRONG_UP|UP|SIDEWAYS|DOWN|STRONG_DOWN",',
            '    "entry_price": 진입가(숫자),',
            '    "stop_loss": 손절가(숫자, 진입가 -0.5%),',
            '    "take_profit": 익절가(숫자, 진입가 +1.0~1.5%),',
            '    "reasoning": "분석 근거 (50자 이내)",',
            '    "news_impact": "POSITIVE|NEGATIVE|NEUTRAL",',
            '    "risk_factors": ["리스크1", "리스크2"]',
            "}",
            "```",
            "",
            "JSON만 출력하세요."
        ])

        return "\n".join(prompt_parts)

    def build_scalping_prompt(
        self,
        stock_data: Dict,
        technical_summary: Dict,
        ohlcv_text: str,
        news_list: List[str] = None,
        overnight_context: str = ""
    ) -> str:
        """
        스캘핑 전용 프롬프트 생성 (09:00 ~ 09:30 시간대)

        야간 뉴스 시황과 기술적 지표를 종합하여 장 초반 스캘핑 판단을 위한
        프롬프트를 생성합니다.

        Args:
            stock_data: 종목 기본 정보
            technical_summary: TechnicalAnalyzer.get_scalping_summary() 결과
            ohlcv_text: TechnicalAnalyzer.format_for_llm() 결과
            news_list: 야간 뉴스 리스트
            overnight_context: 야간 시황 컨텍스트

        Returns:
            str: 스캘핑 분석용 프롬프트
        """
        prompt_parts = [
            "=" * 60,
            "## 장 초반 스캘핑 분석 (09:00 ~ 09:30)",
            "=" * 60,
            "",
            "당신은 장 초반 스캘핑 전문 트레이더입니다.",
            "야간 뉴스와 기술적 지표를 종합하여 빠른 스캘핑 매매 전략을 제시하세요.",
            "",
            "## 핵심 목표",
            "- 장 초반 갭 상승/하락에서의 빠른 진입/청산",
            "- 손절(-0.3%), 익절(+0.8%) 타이트하게 설정",
            "- 30초~3분 내 빠른 판단 필요",
            "",
        ]

        # 종목 정보
        prompt_parts.extend([
            "## 종목 정보",
            f"- 종목명: {stock_data.get('name', 'N/A')}",
            f"- 종목코드: {stock_data.get('code', 'N/A')}",
            f"- 현재가: {stock_data.get('price', 0):,}원",
            f"- 등락률: {stock_data.get('change_rate', 0):+.2f}%",
            f"- 체결강도: {stock_data.get('volume_power', 100):.1f} (100 이상 매수우세)",
            f"- 호가잔량비: {stock_data.get('balance_ratio', 1.0):.2f} (1 이상 매수우세)",
            f"- 급등점수: {stock_data.get('surge_score', 0):.1f}/100",
            "",
        ])

        # 야간 시황 컨텍스트
        if overnight_context:
            prompt_parts.extend([
                overnight_context,
                "",
            ])

        # 기술적 지표
        if technical_summary:
            prompt_parts.extend([
                "## 기술적 지표 (장 초반 판단 핵심)",
                f"- 추세: {technical_summary.get('trend', 'N/A')}",
                f"- 종합점수: {technical_summary.get('total_score', 0):+.1f}/100",
                "",
                "### 스캘핑 핵심 지표",
                f"- RSI(7): {technical_summary.get('rsi_7', 0):.1f} (단기 과매수/과매도)",
                f"- RSI(14): {technical_summary.get('rsi_14', 0):.1f}",
                f"- 스토캐스틱 %K: {technical_summary.get('stoch_k', 0):.1f}",
                f"- MACD Histogram: {technical_summary.get('macd_histogram', 0):.2f}",
                "",
                "### 가격 레벨",
                f"- VWAP: {technical_summary.get('vwap', 0):,.0f}원",
                f"- 볼린저 상단: {technical_summary.get('bb_upper', 0):,.0f}원",
                f"- 볼린저 하단: {technical_summary.get('bb_lower', 0):,.0f}원",
                f"- 피봇: {technical_summary.get('pivot', 0):,.0f}원",
                f"- R1: {technical_summary.get('resistance_1', 0):,.0f}원",
                f"- S1: {technical_summary.get('support_1', 0):,.0f}원",
                "",
                "### 변동성",
                f"- ATR: {technical_summary.get('atr_14', 0):,.0f}원 ({technical_summary.get('atr_percent', 0):.2f}%)",
                "",
            ])

            # 발생한 신호들
            signals = technical_summary.get('signals', [])
            if signals:
                prompt_parts.extend(["### 발생 신호"])
                for sig in signals[:5]:
                    direction = "매수" if sig.get('direction') == "BUY" else "매도"
                    prompt_parts.append(f"- {sig.get('reason', '')}: {direction}")
                prompt_parts.append("")

        # 야간 뉴스 (주요 헤드라인만)
        if news_list:
            prompt_parts.extend([
                "## 야간 뉴스 (전일 장 마감 ~ 금일 장 시작)",
            ])
            for i, news in enumerate(news_list[:10], 1):
                prompt_parts.append(f"{i}. {news}")
            prompt_parts.append("")

        # 스캘핑 분석 요청
        prompt_parts.extend([
            "## 스캘핑 분석 요청",
            "1. 야간 뉴스의 호재/악재 비중과 영향도",
            "2. 장 초반 갭 상승/하락 예상 방향",
            "3. 체결강도와 호가잔량에서 나타나는 수급 신호",
            "4. 스캘핑 진입 타이밍 (즉시/대기/불가)",
            "5. 빠른 손절(-0.3%), 익절(+0.8%) 가격",
            "",
            "반드시 아래 JSON 형식으로만 응답하세요:",
            "```json",
            "{",
            '    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",',
            '    "confidence": 0.0~1.0,',
            '    "trend_prediction": "GAP_UP|UP|SIDEWAYS|DOWN|GAP_DOWN",',
            '    "entry_price": 권장 진입가(숫자),',
            '    "stop_loss": 손절가(숫자, 진입가 -0.3%),',
            '    "take_profit": 익절가(숫자, 진입가 +0.8%),',
            '    "reasoning": "장 초반 스캘핑 판단 근거 (50자 이내)",',
            '    "news_impact": "POSITIVE|NEGATIVE|NEUTRAL",',
            '    "scalping_timing": "IMMEDIATE|WAIT|AVOID",',
            '    "risk_factors": ["위험1", "위험2"]',
            "}",
            "```",
            "",
            "JSON만 출력하세요. 스캘핑은 빠른 판단이 생명입니다!"
        ])

        return "\n".join(prompt_parts)

    def analyze_with_technical_data(
        self,
        stock_code: str,
        stock_name: str,
        current_price: int,
        stock_data: Dict = None,
        news_list: List[str] = None,
        additional_context: str = "",
        analysis_mode: str = "normal",
        parallel: bool = False,
        unload_after: bool = None
    ) -> EnsembleAnalysis:
        """
        OHLCV + 보조지표를 포함한 완전한 2단계 앙상블 분석

        1단계: 하위 모델들(deepseek, qwen3, solar)이 기술적 데이터 분석
        2단계: 메인 모델(exaone4.0)이 하위 모델 결과를 종합하여 최종 판단

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            current_price: 현재가
            stock_data: 종목 추가 정보 (체결강도, 호가비 등)
            news_list: 뉴스 리스트
            additional_context: 추가 컨텍스트 (스캘핑 시황 등)
            analysis_mode: 분석 모드 ("normal", "scalping")
            parallel: 병렬 실행 여부
            unload_after: 분석 후 모델 언로드 여부

        Returns:
            EnsembleAnalysis: 앙상블 분석 결과
        """
        # 현재 분석 중인 종목 정보 설정 (WebSocket 브로드캐스트용)
        self._current_stock_code = stock_code
        self._current_stock_name = stock_name

        # 분석 시작 브로드캐스트
        self._broadcast_llm_output("system", "analysis_start",
            f"{stock_name}({stock_code}) 분석 시작",
            current_price=current_price
        )

        # OHLCV 및 기술적 지표 조회
        technical_summary = {}
        ohlcv_text = ""

        if OHLCVFetcher is not None and TechnicalAnalyzer is not None:
            try:
                logger.info(f"[{stock_name}] OHLCV 및 기술적 지표 조회 중...")

                # OHLCV 데이터 조회
                fetcher = OHLCVFetcher(env_dv="prod")
                daily_df = fetcher.get_daily_data(stock_code, days=30)

                if not daily_df.empty:
                    # 기술적 지표 계산
                    analyzer = TechnicalAnalyzer()
                    technical_summary = analyzer.get_scalping_summary(daily_df)
                    ohlcv_text = analyzer.format_for_llm(daily_df, stock_name)
                    logger.info(f"[{stock_name}] 기술적 지표 계산 완료 (점수: {technical_summary.get('total_score', 0):+.1f})")
                else:
                    logger.warning(f"[{stock_name}] OHLCV 데이터 없음, 기본 분석 진행")

            except Exception as e:
                logger.error(f"[{stock_name}] 기술적 지표 조회 실패: {e}")

        # 기본 stock_data 구성 (SurgeCandidate 객체도 처리)
        if stock_data is None:
            stock_data = {}
        elif hasattr(stock_data, '__dataclass_fields__'):
            # dataclass 객체인 경우 딕셔너리로 변환
            from dataclasses import asdict
            stock_data = asdict(stock_data)
        elif not isinstance(stock_data, dict):
            # 기타 객체인 경우 새 딕셔너리 생성
            stock_data = {}

        stock_data.update({
            'code': stock_code,
            'name': stock_name,
            'price': current_price
        })

        # 분석 모드에 따른 프롬프트 생성
        if analysis_mode == "scalping":
            # 스캘핑 모드: 야간 뉴스 시황 + 기술적 지표
            prompt = self.build_scalping_prompt(
                stock_data, technical_summary, ohlcv_text, news_list, additional_context
            )
        elif technical_summary:
            prompt = self.build_enhanced_prompt(stock_data, technical_summary, ohlcv_text, news_list)
        else:
            prompt = self.build_prompt(stock_data, news_list)

        # 앙상블 모델 설정
        if not self.ensemble_models:
            self.setup_ensemble(use_financial_ensemble=True)

        if not self.ensemble_models:
            raise ValueError("사용 가능한 앙상블 모델이 없습니다")

        # ========== 1단계: 하위 모델 분석 ==========
        logger.info(f"[1단계] {stock_name} 하위 모델 분석 시작 - 모델: {self.ensemble_models}")

        model_results = []

        if parallel and len(self.ensemble_models) > 1:
            with ThreadPoolExecutor(max_workers=len(self.ensemble_models)) as executor:
                futures = {
                    executor.submit(self._call_single_model, model, prompt): model
                    for model in self.ensemble_models
                }

                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        result = future.result()
                        model_results.append(result)
                        logger.info(f"  [{model}] {result.signal} ({result.confidence*100:.0f}%) - {result.processing_time:.1f}s")
                    except Exception as e:
                        logger.error(f"  [{model}] 오류: {e}")
        else:
            for model in self.ensemble_models:
                result = self._call_single_model(model, prompt)
                model_results.append(result)
                logger.info(f"  [{model}] {result.signal} ({result.confidence*100:.0f}%) - {result.processing_time:.1f}s")

        # ========== 2단계: 메인 모델 최종 판단 ==========
        logger.info(f"[2단계] 메인 모델({self.MAIN_JUDGE_MODEL}) 최종 판단 시작")

        # 메인 모델에게 전달할 종합 프롬프트 생성
        final_result = self._call_main_judge(stock_data, news_list, model_results, prompt)

        if final_result and final_result.success:
            model_results.append(final_result)
            logger.info(f"  [{self.MAIN_JUDGE_MODEL}] {final_result.signal} ({final_result.confidence*100:.0f}%) - {final_result.processing_time:.1f}s")

        # 결과 집계 (메인 모델 결과 포함)
        ensemble_result = self._aggregate_results(model_results, stock_data, prompt, news_list)

        # 기술적 지표 정보 추가
        if technical_summary:
            ensemble_result.input_data['technical_summary'] = technical_summary

        logger.info(f"[앙상블] 최종 시그널: {ensemble_result.ensemble_signal} "
                   f"(신뢰도: {ensemble_result.ensemble_confidence*100:.0f}%, "
                   f"합의도: {ensemble_result.consensus_score*100:.0f}%)")

        # 히스토리 저장
        with self._lock:
            self.analysis_history.append(ensemble_result)
            if len(self.analysis_history) > self.max_history:
                self.analysis_history = self.analysis_history[-self.max_history:]

        # GPU 메모리 정리 (선택적)
        should_unload = unload_after if unload_after is not None else self.auto_unload
        if should_unload:
            self._cleanup_gpu_memory()

        return ensemble_result

    def _safe_float(self, value, default: float = 0.0) -> float:
        """안전한 float 변환 (None, 빈 문자열 등 처리)"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_str(self, value, default: str = "") -> str:
        """안전한 문자열 변환"""
        if value is None:
            return default
        return str(value)

    def _broadcast_llm_output(self, model_name: str, output_type: str, content: str, **kwargs):
        """LLM 출력을 콜백으로 전달 (WebSocket 브로드캐스트용)"""
        if self.on_llm_output:
            try:
                self.on_llm_output(
                    stock_code=self._current_stock_code,
                    stock_name=self._current_stock_name,
                    model_name=model_name,
                    output_type=output_type,
                    content=content,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"LLM 출력 브로드캐스트 실패: {e}")

    def _call_single_model(self, model_name: str, prompt: str, max_retries: int = 2) -> ModelResult:
        """단일 모델 호출 (재시도 로직 포함, GPU 메모리 관리 적용)"""
        start_time = time.time()

        # 분석 시작 브로드캐스트
        self._broadcast_llm_output(model_name, "start", f"{model_name} 분석 시작...")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,  # GPU 메모리 관리
            "options": {
                "temperature": 0.1,
                "num_predict": 500,
                "num_ctx": 4096
            }
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"  [{model_name}] API 호출 시작... (시도 {attempt + 1})")

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=180
                )

                if response.status_code == 200:
                    response_json = response.json()
                    raw_output = response_json.get("response", "").strip()

                    if not raw_output:
                        logger.warning(f"  [{model_name}] 빈 응답 수신, done={response_json.get('done')}")
                        self._broadcast_llm_output(model_name, "warning", "빈 응답 수신")
                    else:
                        logger.debug(f"  [{model_name}] 응답 수신: {len(raw_output)} chars")
                        # 원본 응답 브로드캐스트
                        self._broadcast_llm_output(model_name, "response", raw_output)

                    parsed = self._parse_json_response(raw_output)
                    processing_time = time.time() - start_time

                    # 파싱 결과 브로드캐스트
                    signal = self._safe_str(parsed.get("signal"), "HOLD")
                    confidence = self._safe_float(parsed.get("confidence"), 0.0)
                    reasoning = self._safe_str(parsed.get("reasoning"), "")

                    self._broadcast_llm_output(
                        model_name, "signal",
                        f"시그널: {signal}, 신뢰도: {confidence * 100:.0f}%",
                        signal=signal,
                        confidence=confidence,
                        reasoning=reasoning,
                        processing_time=processing_time
                    )

                    return ModelResult(
                        model_name=model_name,
                        signal=signal,
                        confidence=confidence,
                        trend_prediction=self._safe_str(parsed.get("trend_prediction"), "SIDEWAYS"),
                        entry_price=self._safe_float(parsed.get("entry_price"), 0.0),
                        stop_loss=self._safe_float(parsed.get("stop_loss"), 0.0),
                        take_profit=self._safe_float(parsed.get("take_profit"), 0.0),
                        reasoning=reasoning,
                        news_impact=self._safe_str(parsed.get("news_impact"), "NEUTRAL"),
                        risk_factors=parsed.get("risk_factors") or [],
                        processing_time=processing_time,
                        success=bool(parsed),
                        raw_output=raw_output,
                        error_message="" if parsed else "JSON 파싱 실패"
                    )

                elif response.status_code == 500:
                    last_error = f"서버 오류 (500): {response.text[:100] if response.text else 'No response'}"
                    logger.warning(f"  [{model_name}] {last_error} (시도 {attempt + 1}/{max_retries + 1})")
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:100] if response.text else 'No response'}"
                    break

            except requests.exceptions.Timeout:
                last_error = f"타임아웃 (180초 초과)"
                logger.warning(f"  [{model_name}] 타임아웃 (시도 {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    continue
            except requests.exceptions.ConnectionError:
                last_error = "Ollama 서버 연결 실패"
                break
            except Exception as e:
                last_error = str(e)
                break

        # 모든 재시도 실패
        processing_time = time.time() - start_time
        logger.error(f"  [{model_name}] 실패: {last_error}")
        return ModelResult(
            model_name=model_name,
            signal="HOLD",
            confidence=0,
            trend_prediction="SIDEWAYS",
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            reasoning="",
            news_impact="NEUTRAL",
            risk_factors=[],
            processing_time=processing_time,
            success=False,
            raw_output="",
            error_message=last_error
        )

    def _parse_json_response(self, text: str) -> Dict:
        """JSON 응답 파싱 (다양한 LLM 출력 형식 처리)"""
        if not text or not text.strip():
            logger.warning("  빈 응답 수신")
            return {}

        original_text = text
        try:
            # 1. <think>...</think> 태그 제거 (qwen3, deepseek 등)
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL)

            # 2. 기타 특수 태그 제거
            text = re.sub(r'<\|.*?\|>', '', text)  # <|im_start|> 등

            # 3. JSON 블록 추출 시도
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    # JSON이 포함된 코드 블록 찾기
                    for part in parts:
                        if "{" in part and "}" in part:
                            text = part
                            break

            # 4. 중괄호 찾기
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
            else:
                logger.warning(f"  JSON 구조 없음: {original_text[:100]}...")
                return {}

            # 5. JSON 문자열 정리
            text = text.strip()
            # 잘못된 이스케이프 시퀀스 수정
            text = text.replace('\n', ' ').replace('\r', ' ')
            # 연속된 공백 제거
            text = re.sub(r'\s+', ' ', text)

            result = json.loads(text)

            # 필수 필드 검증
            if "signal" not in result:
                # signal이 없으면 다른 필드에서 추론 시도
                if "recommendation" in result:
                    result["signal"] = result["recommendation"]
                elif "action" in result:
                    result["signal"] = result["action"]
                else:
                    logger.warning(f"  signal 필드 없음")
                    return {}

            # signal 값 정규화
            signal = str(result.get("signal", "")).upper().strip()
            valid_signals = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
            if signal not in valid_signals:
                # 부분 매칭 시도
                if "STRONG" in signal and "BUY" in signal:
                    result["signal"] = "STRONG_BUY"
                elif "BUY" in signal:
                    result["signal"] = "BUY"
                elif "STRONG" in signal and "SELL" in signal:
                    result["signal"] = "STRONG_SELL"
                elif "SELL" in signal:
                    result["signal"] = "SELL"
                else:
                    result["signal"] = "HOLD"

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"  JSON 파싱 오류: {e}")
            logger.debug(f"  원본 텍스트: {original_text[:300]}...")

            # 마지막 시도: 정규식으로 필드 추출
            try:
                import re
                signal_match = re.search(r'"signal"\s*:\s*"([^"]+)"', original_text)
                conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', original_text)

                if signal_match:
                    return {
                        "signal": signal_match.group(1).upper(),
                        "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                        "reasoning": "정규식 추출"
                    }
            except Exception:
                pass

            return {}

    def _aggregate_results(
        self,
        model_results: List[ModelResult],
        stock_data: Dict,
        prompt: str,
        news_list: List[str]
    ) -> EnsembleAnalysis:
        """모델 결과 집계"""

        successful_results = [r for r in model_results if r.success]

        if not successful_results:
            # 모든 모델 실패
            return EnsembleAnalysis(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                stock_code=stock_data.get("code", ""),
                stock_name=stock_data.get("name", ""),
                ensemble_signal="HOLD",
                ensemble_confidence=0,
                ensemble_trend="SIDEWAYS",
                avg_entry_price=0,
                avg_stop_loss=0,
                avg_take_profit=0,
                signal_votes={},
                trend_votes={},
                model_results=model_results,
                models_used=[r.model_name for r in model_results],
                models_agreed=0,
                total_models=len(model_results),
                consensus_score=0,
                input_prompt=prompt,
                input_data={"stock": stock_data, "news": news_list or []},
                total_processing_time=sum(r.processing_time for r in model_results),
                success=False,
                error_message="모든 모델 분석 실패"
            )

        # 시그널 투표 (가중치 적용)
        signal_votes = Counter()
        weighted_signal_score = 0
        total_weight = 0

        for result in successful_results:
            weight = self.model_weights.get(result.model_name, 1.0)
            signal_votes[result.signal] += 1
            weighted_signal_score += self.SIGNAL_WEIGHTS.get(result.signal, 0) * weight * result.confidence
            total_weight += weight

        # 최종 시그널 결정 (가중 점수 기반)
        avg_weighted_score = weighted_signal_score / total_weight if total_weight > 0 else 0

        if avg_weighted_score >= 1.5:
            ensemble_signal = "STRONG_BUY"
        elif avg_weighted_score >= 0.5:
            ensemble_signal = "BUY"
        elif avg_weighted_score <= -1.5:
            ensemble_signal = "STRONG_SELL"
        elif avg_weighted_score <= -0.5:
            ensemble_signal = "SELL"
        else:
            ensemble_signal = "HOLD"

        # 추세 투표
        trend_votes = Counter(r.trend_prediction for r in successful_results)
        ensemble_trend = trend_votes.most_common(1)[0][0] if trend_votes else "SIDEWAYS"

        # 평균 신뢰도 (가중치 적용)
        weighted_confidence = sum(
            r.confidence * self.model_weights.get(r.model_name, 1.0)
            for r in successful_results
        )
        ensemble_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

        # 평균 가격
        avg_entry = sum(r.entry_price for r in successful_results if r.entry_price > 0) / max(1, sum(1 for r in successful_results if r.entry_price > 0))
        avg_sl = sum(r.stop_loss for r in successful_results if r.stop_loss > 0) / max(1, sum(1 for r in successful_results if r.stop_loss > 0))
        avg_tp = sum(r.take_profit for r in successful_results if r.take_profit > 0) / max(1, sum(1 for r in successful_results if r.take_profit > 0))

        # 합의도 계산 (동일 시그널 비율)
        most_common_signal = signal_votes.most_common(1)[0] if signal_votes else ("HOLD", 0)
        models_agreed = most_common_signal[1]
        consensus_score = models_agreed / len(successful_results) if successful_results else 0

        return EnsembleAnalysis(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            stock_code=stock_data.get("code", ""),
            stock_name=stock_data.get("name", ""),
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            ensemble_trend=ensemble_trend,
            avg_entry_price=avg_entry,
            avg_stop_loss=avg_sl,
            avg_take_profit=avg_tp,
            signal_votes=dict(signal_votes),
            trend_votes=dict(trend_votes),
            model_results=model_results,
            models_used=[r.model_name for r in model_results],
            models_agreed=models_agreed,
            total_models=len(model_results),
            consensus_score=consensus_score,
            input_prompt=prompt,
            input_data={"stock": stock_data, "news": news_list or []},
            total_processing_time=sum(r.processing_time for r in model_results),
            success=True
        )

    def _call_main_judge(
        self,
        stock_data: Dict,
        news_list: List[str],
        sub_model_results: List[ModelResult],
        original_prompt: str
    ) -> ModelResult:
        """
        메인 판단 모델(exaone4.0) 호출

        하위 모델들의 분석 결과를 종합하여 최종 판단을 내립니다.
        """
        # 메인 모델이 사용 가능한지 확인
        if self.MAIN_JUDGE_MODEL not in self.available_models:
            logger.warning(f"메인 판단 모델 {self.MAIN_JUDGE_MODEL}이 사용 불가. 기본 집계 사용.")
            return None

        # 하위 모델 결과 요약
        sub_results_summary = []
        for result in sub_model_results:
            if result.success:
                sub_results_summary.append(
                    f"- {result.model_name}: {result.signal} (신뢰도: {result.confidence*100:.0f}%)\n"
                    f"  추론: {result.reasoning[:200] if result.reasoning else 'N/A'}..."
                )

        sub_results_text = "\n".join(sub_results_summary)

        # 메인 모델용 종합 프롬프트
        judge_prompt = f"""당신은 최고 수준의 금융 전문가입니다.
여러 AI 모델들이 아래 종목을 분석했습니다. 이들의 분석 결과를 종합하여 최종 투자 판단을 내려주세요.

## 종목 정보
- 종목코드: {stock_data.get('code', 'N/A')}
- 종목명: {stock_data.get('name', 'N/A')}
- 현재가: {stock_data.get('price', 'N/A')}
- 등락률: {stock_data.get('change_rate', 'N/A')}%
- 체결강도: {stock_data.get('volume_power', 'N/A')}

## 하위 모델 분석 결과
{sub_results_text}

## 원본 분석 데이터
{original_prompt[:1500]}...

## 당신의 임무
위 정보를 종합하여 최종 투자 판단을 내려주세요.
각 모델의 의견을 참고하되, 당신만의 전문적 판단을 추가하세요.

반드시 아래 JSON 형식으로만 응답:
{{
    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL 중 하나",
    "confidence": 0.0~1.0 사이 숫자,
    "trend_prediction": "STRONG_UP|UP|SIDEWAYS|DOWN|STRONG_DOWN 중 하나",
    "reasoning": "최종 판단 근거 (하위 모델 의견 종합 + 추가 분석)",
    "entry_price": 권장 진입가 (숫자),
    "stop_loss": 손절가 (숫자),
    "take_profit": 목표가 (숫자),
    "risk_factors": ["위험요소1", "위험요소2"]
}}
"""

        # 메인 모델 호출
        try:
            result = self._call_single_model(self.MAIN_JUDGE_MODEL, judge_prompt)
            return result
        except Exception as e:
            logger.error(f"메인 판단 모델 호출 실패: {e}")
            return None

    def analyze_stock(
        self,
        stock_data: Dict,
        news_list: List[str] = None,
        parallel: bool = True,
        unload_after: bool = None
    ) -> EnsembleAnalysis:
        """
        2단계 앙상블 분석 실행

        1단계: 하위 모델들(deepseek, qwen3, solar)이 데이터 분석
        2단계: 메인 모델(exaone4.0)이 하위 모델 결과를 종합하여 최종 판단

        Args:
            stock_data: 종목 데이터
            news_list: 뉴스 리스트
            parallel: 병렬 실행 여부
            unload_after: 분석 후 모델 언로드 여부 (None이면 self.auto_unload 사용)
        """
        if not self.ensemble_models:
            self.setup_ensemble()

        if not self.ensemble_models:
            raise ValueError("사용 가능한 앙상블 모델이 없습니다")

        prompt = self.build_prompt(stock_data, news_list)

        # ========== 1단계: 하위 모델 분석 ==========
        logger.info(f"[1단계] {stock_data.get('name')} 하위 모델 분석 시작 - 모델: {self.ensemble_models}")

        model_results = []

        if parallel and len(self.ensemble_models) > 1:
            # 병렬 실행 (주의: GPU 메모리 동시 사용)
            with ThreadPoolExecutor(max_workers=len(self.ensemble_models)) as executor:
                futures = {
                    executor.submit(self._call_single_model, model, prompt): model
                    for model in self.ensemble_models
                }

                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        result = future.result()
                        model_results.append(result)
                        logger.info(f"  [{model}] {result.signal} ({result.confidence*100:.0f}%) - {result.processing_time:.1f}s")
                    except Exception as e:
                        logger.error(f"  [{model}] 오류: {e}")
        else:
            # 순차 실행 (GPU 메모리 효율적 - 권장)
            for model in self.ensemble_models:
                result = self._call_single_model(model, prompt)
                model_results.append(result)
                logger.info(f"  [{model}] {result.signal} ({result.confidence*100:.0f}%) - {result.processing_time:.1f}s")

        # ========== 2단계: 메인 모델 최종 판단 ==========
        logger.info(f"[2단계] 메인 모델({self.MAIN_JUDGE_MODEL}) 최종 판단 시작")

        # 메인 모델에게 전달할 종합 프롬프트 생성
        final_result = self._call_main_judge(stock_data, news_list, model_results, prompt)

        if final_result and final_result.success:
            # 메인 모델 결과를 model_results에 추가
            model_results.append(final_result)
            logger.info(f"  [{self.MAIN_JUDGE_MODEL}] {final_result.signal} ({final_result.confidence*100:.0f}%) - {final_result.processing_time:.1f}s")

        # 결과 집계 (메인 모델 결과 포함)
        ensemble_result = self._aggregate_results(model_results, stock_data, prompt, news_list)

        logger.info(f"[앙상블] 최종 시그널: {ensemble_result.ensemble_signal} "
                   f"(신뢰도: {ensemble_result.ensemble_confidence*100:.0f}%, "
                   f"합의도: {ensemble_result.consensus_score*100:.0f}%)")

        # 히스토리 저장
        with self._lock:
            self.analysis_history.append(ensemble_result)
            if len(self.analysis_history) > self.max_history:
                self.analysis_history = self.analysis_history[-self.max_history:]

        # GPU 메모리 정리 (선택적)
        should_unload = unload_after if unload_after is not None else self.auto_unload
        if should_unload:
            self._cleanup_gpu_memory()

        return ensemble_result

    def _cleanup_gpu_memory(self) -> None:
        """분석 완료 후 GPU 메모리 정리"""
        # keep_alive가 "0"이면 자동으로 언로드되므로 별도 처리 불필요
        if self.keep_alive == "0":
            return

        # 앙상블에 사용하지 않는 모델만 정리
        status = self.get_gpu_status()
        for model in status["running_models"]:
            if model["name"] not in self.ensemble_models:
                self.unload_model(model["name"])

    def get_history(self, limit: int = 20) -> List[Dict]:
        """분석 히스토리 조회"""
        with self._lock:
            history = self.analysis_history[-limit:]

        result = []
        for h in reversed(history):
            result.append({
                "timestamp": h.timestamp,
                "stock_code": h.stock_code,
                "stock_name": h.stock_name,
                "ensemble_signal": h.ensemble_signal,
                "ensemble_confidence": h.ensemble_confidence,
                "ensemble_trend": h.ensemble_trend,
                "avg_entry_price": h.avg_entry_price,
                "avg_stop_loss": h.avg_stop_loss,
                "avg_take_profit": h.avg_take_profit,
                "signal_votes": h.signal_votes,
                "trend_votes": h.trend_votes,
                "models_used": h.models_used,
                "models_agreed": h.models_agreed,
                "total_models": h.total_models,
                "consensus_score": h.consensus_score,
                "total_processing_time": h.total_processing_time,
                "success": h.success,
                "input_prompt": h.input_prompt,
                "model_results": [
                    {
                        "model_name": r.model_name,
                        "signal": r.signal,
                        "confidence": r.confidence,
                        "trend_prediction": r.trend_prediction,
                        "entry_price": r.entry_price,
                        "stop_loss": r.stop_loss,
                        "take_profit": r.take_profit,
                        "reasoning": r.reasoning,
                        "processing_time": r.processing_time,
                        "success": r.success,
                        "raw_output": r.raw_output,
                        "error_message": r.error_message
                    }
                    for r in h.model_results
                ]
            })

        return result


# 전역 인스턴스
_ensemble_analyzer = None

def get_ensemble_analyzer() -> EnsembleLLMAnalyzer:
    """전역 앙상블 분석기 인스턴스"""
    global _ensemble_analyzer
    if _ensemble_analyzer is None:
        _ensemble_analyzer = EnsembleLLMAnalyzer()
        _ensemble_analyzer.setup_ensemble()
    return _ensemble_analyzer


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    analyzer = EnsembleLLMAnalyzer()

    # 모델 탐색
    models = analyzer.discover_models()
    print(f"사용 가능 모델: {models}")

    # 앙상블 설정
    ensemble = analyzer.setup_ensemble()
    print(f"앙상블 모델: {ensemble}")
    print(f"모델 가중치: {analyzer.model_weights}")

    # 테스트 분석
    test_stock = {
        'code': '005930',
        'name': '삼성전자',
        'price': 55000,
        'change_rate': 2.5,
        'volume_power': 180,
        'balance_ratio': 1.8,
        'surge_score': 75,
        'volume': 5000000
    }

    test_news = [
        "삼성전자, AI 반도체 수주 급증",
        "외국인 삼성전자 3거래일 연속 순매수"
    ]

    print("\n=== 앙상블 분석 시작 ===")
    result = analyzer.analyze_stock(test_stock, test_news)

    print("\n=== 앙상블 결과 ===")
    print(f"종목: {result.stock_name}")
    print(f"최종 시그널: {result.ensemble_signal}")
    print(f"신뢰도: {result.ensemble_confidence:.0%}")
    print(f"추세 예측: {result.ensemble_trend}")
    print(f"합의도: {result.consensus_score:.0%}")
    print(f"\n시그널 투표: {result.signal_votes}")
    print(f"추세 투표: {result.trend_votes}")

    print(f"\n평균 진입가: {result.avg_entry_price:,.0f}원")
    print(f"평균 손절가: {result.avg_stop_loss:,.0f}원")
    print(f"평균 익절가: {result.avg_take_profit:,.0f}원")

    print(f"\n처리 시간: {result.total_processing_time:.1f}초")

    print("\n=== 모델별 결과 ===")
    for r in result.model_results:
        status = "OK" if r.success else "FAIL"
        print(f"[{status}] {r.model_name}: {r.signal} ({r.confidence:.0%}) - {r.processing_time:.1f}s")
        if r.reasoning:
            print(f"       근거: {r.reasoning[:60]}...")
