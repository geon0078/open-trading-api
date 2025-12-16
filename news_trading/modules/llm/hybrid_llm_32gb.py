# -*- coding: utf-8 -*-
"""
32GB VRAM 최적화 금융 뉴스 분석 하이브리드 LLM 모듈

구성:
- EXAONE 4.0 32B: 한국어 이해 + 추론 (메인)
- Fin-R1 7B: 금융 전문 분석 (서브)
- Qwen3 8B: 빠른 필터링 (폴백)

총 VRAM 사용량: ~31GB (32GB에 최적화)

사용 예시:
    >>> from hybrid_llm_32gb import FinancialHybridLLM
    >>> llm = FinancialHybridLLM()
    >>> result = llm.analyze("삼성전자, 3분기 영업이익 10조 돌파", "005930")
    >>> print(result.recommendation)
    BUY
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SentimentType(Enum):
    """감성 분류 타입"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ImpactLevel(Enum):
    """영향도 레벨"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AnalysisResult:
    """
    단일 모델 분석 결과

    Attributes:
        sentiment: 감성 분류 (positive/negative/neutral)
        confidence: 신뢰도 (0.0~1.0)
        impact: 영향도 (high/medium/low)
        reasoning: 판단 근거
        model_source: 분석 수행 모델명
        processing_time: 처리 시간 (초)
    """
    sentiment: SentimentType
    confidence: float
    impact: ImpactLevel
    reasoning: str
    model_source: str
    processing_time: float


@dataclass
class EnsembleResult:
    """
    앙상블 최종 결과

    Attributes:
        final_sentiment: 최종 감성 결정
        final_confidence: 가중 평균 신뢰도
        final_impact: 최종 영향도
        individual_results: 개별 모델 결과 리스트
        consensus_score: 모델 간 합의도 (0.0~1.0)
        recommendation: 매매 추천 (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)
    """
    final_sentiment: SentimentType
    final_confidence: float
    final_impact: ImpactLevel
    individual_results: List[AnalysisResult]
    consensus_score: float
    recommendation: str


class FinancialHybridLLM:
    """
    32GB VRAM 최적화 금융 뉴스 분석 하이브리드 LLM

    3개 모델의 앙상블을 통해 정확도와 안정성을 확보합니다.
    각 모델은 특화된 역할을 수행하며, 가중 투표로 최종 결과를 도출합니다.

    모델 구성:
        - korean_reasoning (EXAONE 4.0 32B): 한국어 뉴스 이해 + 심층 추론
        - financial_expert (Fin-R1 7B): 금융 전문 분석 + 수치 추론
        - fast_filter (Qwen3 8B): 빠른 1차 스크리닝 + 폴백

    사용 예시:
        >>> llm = FinancialHybridLLM(api_url="http://localhost:11434")
        >>> result = llm.analyze("삼성전자, 3분기 영업이익 10조 돌파", "005930")
        >>> print(f"감성: {result.final_sentiment.value}")
        >>> print(f"추천: {result.recommendation}")
    """

    def __init__(
        self,
        api_url: str = "http://localhost:11434",
        timeout: int = 120,
        enable_parallel: bool = True,
        config_path: Optional[str] = None,
        keep_alive: str = "5m",
        max_vram_gb: float = 24.0
    ):
        """
        하이브리드 LLM 초기화

        Args:
            api_url: Ollama API 주소 (기본: http://localhost:11434)
            timeout: API 호출 타임아웃 (초, 기본: 120)
            enable_parallel: 병렬 처리 활성화 여부 (기본: True)
            config_path: 설정 파일 경로 (선택)
            keep_alive: 모델 유지 시간 (기본: 5m, "0"=즉시 언로드)
            max_vram_gb: 최대 VRAM 사용량 (GB, 기본: 24.0)

        Raises:
            ConnectionError: Ollama 서버 연결 실패 시
        """
        self.api_url = api_url
        self.timeout = timeout
        self.enable_parallel = enable_parallel
        self.keep_alive = keep_alive
        self.max_vram_gb = max_vram_gb
        self.auto_unload = False
        self.optimize_on_start = True
        self.preload_models = False
        self.model_keep_alive = {}

        # 모델 프리셋 정의
        self._model_presets = {
            # 기본: EXAONE + Fin-R1 + Qwen3 (32GB VRAM)
            "default": {
                "korean_reasoning": {
                    "name": "ingu627/exaone4.0:32b",
                    "role": "한국어 뉴스 이해 + 심층 추론",
                    "weight": 0.4,
                    "vram": "20GB",
                    "priority": 1,
                    "temperature": 0.2,
                },
                "financial_expert": {
                    "name": "fin-r1",
                    "role": "금융 전문 분석 + 수치 추론",
                    "weight": 0.4,
                    "vram": "6GB",
                    "priority": 1,
                    "temperature": 0.2,
                },
                "fast_filter": {
                    "name": "qwen3:8b",
                    "role": "빠른 1차 스크리닝 + 폴백",
                    "weight": 0.2,
                    "vram": "5GB",
                    "priority": 2,
                    "temperature": 0.3,
                }
            },
            # DeepSeek 중심 앙상블 (금융 추론 특화)
            "deepseek": {
                "financial_reasoning": {
                    "name": "deepseek-r1:32b",
                    "role": "금융 추론 + 수치 분석 (메인)",
                    "weight": 0.5,
                    "vram": "20GB",
                    "priority": 1,
                    "temperature": 0.1,
                },
                "korean_support": {
                    "name": "qwen3:8b",
                    "role": "한국어 이해 + 보조 분석",
                    "weight": 0.3,
                    "vram": "5GB",
                    "priority": 1,
                    "temperature": 0.2,
                },
                "fast_filter": {
                    "name": "deepseek-r1:8b",
                    "role": "빠른 1차 스크리닝",
                    "weight": 0.2,
                    "vram": "5GB",
                    "priority": 2,
                    "temperature": 0.2,
                }
            },
            # 경량 앙상블 (16GB VRAM 이하)
            "lightweight": {
                "main_reasoning": {
                    "name": "deepseek-r1:8b",
                    "role": "금융 추론 (메인)",
                    "weight": 0.5,
                    "vram": "5GB",
                    "priority": 1,
                    "temperature": 0.1,
                },
                "korean_support": {
                    "name": "qwen3:8b",
                    "role": "한국어 이해 + 보조",
                    "weight": 0.3,
                    "vram": "5GB",
                    "priority": 1,
                    "temperature": 0.2,
                },
                "fast_filter": {
                    "name": "deepseek-r1:1.5b",
                    "role": "빠른 필터링",
                    "weight": 0.2,
                    "vram": "2GB",
                    "priority": 2,
                    "temperature": 0.3,
                }
            }
        }

        # 기본 모델 설정 (32GB VRAM 최적화)
        self.models = self._model_presets["default"].copy()

        # 프롬프트 템플릿 초기화
        self.prompts = self._init_prompts()

        # 설정 파일 로드 (선택)
        if config_path:
            self._load_config(config_path)

        # 모델 상태 확인
        self._check_models()

        # 시작 시 메모리 최적화 (선택)
        if self.optimize_on_start:
            self.optimize_memory()

        # 시작 시 모델 프리로드 (선택)
        if self.preload_models:
            self._preload_ensemble_models()

    def _init_prompts(self) -> Dict[str, str]:
        """역할별 프롬프트 템플릿 초기화"""
        return {
            "korean_reasoning": """당신은 한국 증권시장 전문 애널리스트입니다.
다음 뉴스를 읽고 주가에 미칠 영향을 분석해주세요.

[뉴스 제목]
{news_title}

[종목코드]
{stock_code}

다음 JSON 형식으로만 응답하세요:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0~1.0,
    "impact": "high|medium|low",
    "reasoning": "판단 근거를 한글로 2-3문장으로 설명"
}}

판단 기준:
- positive: 실적 개선, 신규 수주, 투자 유치, 긍정적 전망
- negative: 실적 악화, 소송, 규제, 부정적 이슈
- neutral: 단순 공시, 인사 변동, 영향 미미

JSON만 출력하세요.""",

            "financial_expert": """You are a financial analyst expert.
Analyze the following Korean financial news and assess its impact on stock price.

[News Title]
{news_title}

[Stock Code]
{stock_code}

Respond ONLY in this JSON format:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0~1.0,
    "impact": "high|medium|low",
    "reasoning": "Brief explanation in 1-2 sentences"
}}

Assessment criteria:
- positive: Revenue growth, new contracts, expansion, favorable outlook
- negative: Revenue decline, lawsuits, regulatory issues
- neutral: Routine filings, minor announcements

Output JSON only.""",

            "fast_filter": """뉴스 감성 분석:

뉴스: {news_title}
종목: {stock_code}

JSON으로 응답:
{{"sentiment": "positive|negative|neutral", "confidence": 0.0~1.0, "impact": "high|medium|low", "reasoning": "한줄설명"}}"""
        }

    def set_preset(self, preset_name: str) -> None:
        """
        모델 프리셋 변경

        Args:
            preset_name: 프리셋 이름
                - "default": EXAONE + Fin-R1 + Qwen3 (32GB VRAM)
                - "deepseek": DeepSeek-R1 중심 앙상블 (금융 추론 특화)
                - "lightweight": 경량 앙상블 (16GB VRAM 이하)

        Example:
            >>> llm = FinancialHybridLLM()
            >>> llm.set_preset("deepseek")  # DeepSeek 중심으로 전환
        """
        if preset_name not in self._model_presets:
            available = list(self._model_presets.keys())
            raise ValueError(f"알 수 없는 프리셋: {preset_name}. 사용 가능: {available}")

        self.models = self._model_presets[preset_name].copy()
        self.prompts = self._init_prompts()
        logger.info(f"모델 프리셋 변경: {preset_name}")
        logger.info(f"  사용 모델: {[m['name'] for m in self.models.values()]}")

    def get_available_presets(self) -> Dict[str, List[str]]:
        """
        사용 가능한 프리셋 목록 반환

        Returns:
            Dict: {프리셋명: [모델 목록]}
        """
        return {
            name: [m["name"] for m in preset.values()]
            for name, preset in self._model_presets.items()
        }

    def _load_config(self, config_path: str) -> None:
        """YAML 설정 파일 로드"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Ollama 설정
            if 'ollama' in config:
                self.api_url = config['ollama'].get('api_url', self.api_url)
                self.timeout = config['ollama'].get('timeout', self.timeout)

            # 프리셋 선택 (새로 추가)
            if 'preset' in config:
                preset_name = config['preset']
                if preset_name in self._model_presets:
                    self.models = self._model_presets[preset_name].copy()
                    logger.info(f"프리셋 적용: {preset_name}")

            # 모델 설정 (개별 오버라이드)
            if 'models' in config:
                for key, model_config in config['models'].items():
                    if key in self.models:
                        self.models[key].update(model_config)

            # 메모리 관리 설정
            if 'memory' in config:
                mem_config = config['memory']
                self.max_vram_gb = mem_config.get('max_vram_gb', self.max_vram_gb)
                self.keep_alive = mem_config.get('keep_alive', self.keep_alive)
                self.auto_unload = mem_config.get('auto_unload', False)
                self.optimize_on_start = mem_config.get('optimize_on_start', True)
                self.preload_models = mem_config.get('preload_models', False)
                logger.info(f"메모리 설정 적용: max_vram={self.max_vram_gb}GB, keep_alive={self.keep_alive}")

            # 모델별 keep_alive 설정
            if 'model_keep_alive' in config:
                self.model_keep_alive = config['model_keep_alive']
            else:
                self.model_keep_alive = {}

            logger.info(f"설정 파일 로드 완료: {config_path}")
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패: {e}")

    # =========================================================================
    # GPU 메모리 관리
    # =========================================================================

    def get_gpu_status(self) -> Dict[str, Any]:
        """
        현재 GPU 메모리 상태 조회

        Returns:
            Dict: {"running_models": [...], "total_vram_gb": float}
        """
        try:
            response = requests.get(f"{self.api_url}/api/ps", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                running = []
                total_vram = 0
                for m in models:
                    vram_gb = m.get("size_vram", m.get("size", 0)) / (1024**3)
                    running.append({
                        "name": m.get("name", "unknown"),
                        "vram_gb": round(vram_gb, 2)
                    })
                    total_vram += vram_gb

                return {
                    "running_models": running,
                    "total_vram_gb": round(total_vram, 2)
                }
        except Exception as e:
            logger.error(f"GPU 상태 조회 실패: {e}")
        return {"running_models": [], "total_vram_gb": 0}

    def unload_model(self, model_name: str) -> bool:
        """특정 모델을 GPU에서 언로드"""
        try:
            payload = {"model": model_name, "keep_alive": 0}
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                logger.info(f"모델 언로드: {model_name}")
                return True
        except Exception as e:
            logger.error(f"모델 언로드 실패 ({model_name}): {e}")
        return False

    def unload_all(self) -> int:
        """모든 모델 언로드"""
        status = self.get_gpu_status()
        count = 0
        for model in status["running_models"]:
            if self.unload_model(model["name"]):
                count += 1
        logger.info(f"총 {count}개 모델 언로드 완료")
        return count

    def optimize_memory(self) -> None:
        """GPU 메모리 최적화 - max_vram_gb 이하로 유지"""
        status = self.get_gpu_status()
        if status["total_vram_gb"] > self.max_vram_gb:
            logger.warning(f"VRAM 초과 ({status['total_vram_gb']:.1f}GB), 최적화 시작")

            # 현재 앙상블에 사용하지 않는 모델 언로드
            ensemble_model_names = [m["name"] for m in self.models.values()]
            for model in status["running_models"]:
                if model["name"] not in ensemble_model_names:
                    self.unload_model(model["name"])

    def _preload_ensemble_models(self) -> None:
        """앙상블 모델 미리 로드 (워밍업)"""
        logger.info("앙상블 모델 프리로드 시작...")
        for key, config in self.models.items():
            model_name = config["name"]
            try:
                # 짧은 프롬프트로 모델 로드 유도
                payload = {
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1}
                }
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    logger.info(f"  {key} ({model_name}) 프리로드 완료")
            except Exception as e:
                logger.warning(f"  {key} ({model_name}) 프리로드 실패: {e}")

    def _check_models(self) -> None:
        """Ollama 서버 및 모델 상태 확인"""
        try:
            response = requests.get(
                f"{self.api_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                available = response.json().get("models", [])
                available_names = [m["name"] for m in available]

                logger.info("=== 모델 상태 확인 ===")
                for key, config in self.models.items():
                    model_name = config["name"]
                    # 부분 매칭 (태그 무시)
                    is_available = any(
                        model_name.split(":")[0] in name
                        for name in available_names
                    )
                    status = "✅ 로딩됨" if is_available else "⚠️ 미설치"
                    logger.info(f"  {key}: {model_name} - {status}")
            else:
                logger.warning("Ollama 서버 응답 이상")
        except Exception as e:
            logger.error(f"Ollama 서버 연결 실패: {e}")
            raise ConnectionError("Ollama 서버를 먼저 실행하세요: ollama serve")

    def _call_model(
        self,
        model_key: str,
        news_title: str,
        stock_code: str = ""
    ) -> Optional[AnalysisResult]:
        """
        단일 모델 호출

        Args:
            model_key: 모델 키 (korean_reasoning, financial_expert, fast_filter)
            news_title: 뉴스 제목
            stock_code: 종목코드

        Returns:
            AnalysisResult 또는 None (실패시)
        """
        config = self.models[model_key]
        prompt = self.prompts[model_key].format(
            news_title=news_title,
            stock_code=stock_code
        )

        start_time = time.time()

        try:
            payload = {
                "model": config["name"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.get("temperature", 0.2),
                    "num_predict": 300,
                    "num_ctx": 2048
                }
            }

            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.warning(f"{model_key} 호출 실패: {response.status_code}")
                return None

            result_text = response.json().get("response", "").strip()
            processing_time = time.time() - start_time

            # JSON 파싱
            parsed = self._parse_response(result_text)
            if not parsed:
                logger.warning(f"{model_key} JSON 파싱 실패")
                return None

            return AnalysisResult(
                sentiment=SentimentType(parsed.get("sentiment", "neutral")),
                confidence=float(parsed.get("confidence", 0.5)),
                impact=ImpactLevel(parsed.get("impact", "low")),
                reasoning=parsed.get("reasoning", "분석 불가"),
                model_source=model_key,
                processing_time=processing_time
            )

        except requests.exceptions.Timeout:
            logger.warning(f"{model_key} 타임아웃 ({self.timeout}초)")
            return None
        except Exception as e:
            logger.error(f"{model_key} 호출 오류: {e}")
            return None

    def _parse_response(self, text: str) -> Optional[Dict]:
        """LLM 응답에서 JSON 추출 및 파싱"""
        try:
            # ```json ... ``` 블록 제거
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            # JSON 객체 추출
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

            result = json.loads(text.strip())

            # 유효성 검증
            if result.get("sentiment") not in ["positive", "negative", "neutral"]:
                result["sentiment"] = "neutral"
            if result.get("impact") not in ["high", "medium", "low"]:
                result["impact"] = "low"

            confidence = result.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            result["confidence"] = max(0.0, min(1.0, float(confidence)))

            return result

        except json.JSONDecodeError as e:
            logger.debug(f"JSON 파싱 오류: {e}, 텍스트: {text[:100]}")
            return None

    def _ensemble_vote(
        self,
        results: List[AnalysisResult]
    ) -> EnsembleResult:
        """
        가중 투표 앙상블

        Args:
            results: 각 모델의 분석 결과 리스트

        Returns:
            EnsembleResult: 앙상블 최종 결과
        """
        if not results:
            return EnsembleResult(
                final_sentiment=SentimentType.NEUTRAL,
                final_confidence=0.0,
                final_impact=ImpactLevel.LOW,
                individual_results=[],
                consensus_score=0.0,
                recommendation="HOLD"
            )

        # 가중 투표 계산
        sentiment_scores: Dict[SentimentType, float] = {
            SentimentType.POSITIVE: 0.0,
            SentimentType.NEGATIVE: 0.0,
            SentimentType.NEUTRAL: 0.0
        }

        impact_scores: Dict[ImpactLevel, float] = {
            ImpactLevel.HIGH: 0.0,
            ImpactLevel.MEDIUM: 0.0,
            ImpactLevel.LOW: 0.0
        }

        total_weight = 0.0
        weighted_confidence = 0.0

        for result in results:
            weight = self.models[result.model_source]["weight"]
            sentiment_scores[result.sentiment] += weight * result.confidence
            impact_scores[result.impact] += weight
            weighted_confidence += weight * result.confidence
            total_weight += weight

        # 최종 감성 결정
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        final_impact = max(impact_scores, key=impact_scores.get)

        # 평균 신뢰도
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        # 합의도 계산 (모델들이 얼마나 일치하는지)
        sentiment_values = [r.sentiment for r in results]
        consensus_count = sentiment_values.count(final_sentiment)
        consensus_score = consensus_count / len(results) if results else 0.0

        # 매매 추천 결정
        recommendation = self._determine_recommendation(
            final_sentiment,
            final_confidence,
            final_impact,
            consensus_score
        )

        return EnsembleResult(
            final_sentiment=final_sentiment,
            final_confidence=round(final_confidence, 3),
            final_impact=final_impact,
            individual_results=results,
            consensus_score=round(consensus_score, 3),
            recommendation=recommendation
        )

    def _determine_recommendation(
        self,
        sentiment: SentimentType,
        confidence: float,
        impact: ImpactLevel,
        consensus: float
    ) -> str:
        """
        매매 추천 결정 로직

        Args:
            sentiment: 최종 감성
            confidence: 신뢰도
            impact: 영향도
            consensus: 합의도

        Returns:
            str: 매매 추천 (STRONG_BUY/BUY/WEAK_BUY/HOLD/WEAK_SELL/SELL/STRONG_SELL)
        """
        # 고신뢰도 + 높은 합의도일 때만 적극적 추천
        if confidence >= 0.7 and consensus >= 0.67:
            if sentiment == SentimentType.POSITIVE:
                if impact == ImpactLevel.HIGH:
                    return "STRONG_BUY"
                return "BUY"
            elif sentiment == SentimentType.NEGATIVE:
                if impact == ImpactLevel.HIGH:
                    return "STRONG_SELL"
                return "SELL"

        # 중간 신뢰도
        if confidence >= 0.5:
            if sentiment == SentimentType.POSITIVE:
                return "WEAK_BUY"
            elif sentiment == SentimentType.NEGATIVE:
                return "WEAK_SELL"

        return "HOLD"

    def analyze(
        self,
        news_title: str,
        stock_code: str = ""
    ) -> EnsembleResult:
        """
        뉴스 분석 (하이브리드 앙상블)

        3개 모델을 병렬로 호출하여 각각의 분석 결과를 가중 투표로 결합합니다.

        Args:
            news_title: 뉴스 제목
            stock_code: 종목코드 (선택)

        Returns:
            EnsembleResult: 앙상블 분석 결과

        Example:
            >>> result = llm.analyze("삼성전자, 3분기 영업이익 10조 돌파", "005930")
            >>> print(f"감성: {result.final_sentiment.value}")
            >>> print(f"신뢰도: {result.final_confidence:.1%}")
            >>> print(f"추천: {result.recommendation}")
        """
        logger.info(f"뉴스 분석 시작: {news_title[:50]}...")
        start_time = time.time()

        results: List[AnalysisResult] = []

        if self.enable_parallel:
            # 병렬 처리 (권장)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        self._call_model, key, news_title, stock_code
                    ): key
                    for key in self.models.keys()
                }

                for future in as_completed(futures, timeout=self.timeout):
                    model_key = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            logger.info(
                                f"  {model_key}: {result.sentiment.value} "
                                f"(신뢰도: {result.confidence:.2f}, "
                                f"{result.processing_time:.1f}초)"
                            )
                    except Exception as e:
                        logger.warning(f"  {model_key} 실패: {e}")
        else:
            # 순차 처리
            for key in self.models.keys():
                result = self._call_model(key, news_title, stock_code)
                if result:
                    results.append(result)
                    logger.info(
                        f"  {key}: {result.sentiment.value} "
                        f"(신뢰도: {result.confidence:.2f})"
                    )

        # 앙상블 투표
        ensemble = self._ensemble_vote(results)

        total_time = time.time() - start_time
        logger.info(
            f"분석 완료: {ensemble.final_sentiment.value} "
            f"(신뢰도: {ensemble.final_confidence:.2f}, "
            f"합의도: {ensemble.consensus_score:.2f}, "
            f"추천: {ensemble.recommendation}, "
            f"총 {total_time:.1f}초)"
        )

        # 자동 언로드 (VRAM 절약)
        if self.auto_unload:
            self.unload_all()
            logger.info("분석 완료 후 모델 자동 언로드")

        return ensemble

    def analyze_batch(
        self,
        news_list: List[Dict[str, str]],
        delay: float = 0.5
    ) -> List[EnsembleResult]:
        """
        뉴스 배치 분석

        Args:
            news_list: [{"title": "...", "stock_code": "..."}, ...]
            delay: 분석 간 대기 시간 (초, 기본: 0.5)

        Returns:
            List[EnsembleResult]: 분석 결과 리스트
        """
        results = []
        for i, news in enumerate(news_list, 1):
            logger.info(f"배치 분석 진행: {i}/{len(news_list)}")
            result = self.analyze(
                news_title=news.get("title", ""),
                stock_code=news.get("stock_code", "")
            )
            results.append(result)
            time.sleep(delay)  # API 부하 방지
        return results

    def get_model_status(self) -> Dict[str, Any]:
        """
        모델 상태 정보 반환

        Returns:
            Dict: {
                "running_models": [...],
                "total_vram_used": float (GB)
            }
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/ps",
                timeout=10
            )
            if response.status_code == 200:
                running = response.json().get("models", [])
                return {
                    "running_models": running,
                    "total_vram_used": sum(
                        m.get("size", 0) / (1024**3)
                        for m in running
                    )
                }
        except Exception:
            pass
        return {"running_models": [], "total_vram_used": 0}

    def warmup(self) -> bool:
        """
        모델 워밍업 (선택)

        첫 실행 시 모델 로딩 시간을 단축하기 위해 미리 호출합니다.

        Returns:
            bool: 성공 여부
        """
        logger.info("모델 워밍업 시작...")
        try:
            test_result = self.analyze("테스트 뉴스입니다.", "005930")
            success = len(test_result.individual_results) > 0
            if success:
                logger.info("모델 워밍업 완료")
            return success
        except Exception as e:
            logger.error(f"워밍업 실패: {e}")
            return False

    def analyze_kis_news(self, news_df) -> List[Dict[str, Any]]:
        """
        KIS API 뉴스 DataFrame을 분석하여 매매 신호 생성

        Args:
            news_df: KIS API news_title 응답 DataFrame
                     필수 컬럼: titl (뉴스 제목), stck_shrn_iscd (종목코드)

        Returns:
            List[Dict]: 각 뉴스에 대한 분석 결과 리스트
                {
                    "news_title": str,
                    "stock_code": str,
                    "sentiment": str,
                    "confidence": float,
                    "impact": str,
                    "recommendation": str,
                    "reasoning": str,
                    "consensus_score": float
                }

        Example:
            >>> from domestic_stock.news_title import news_title
            >>> news_df = news_title(fid_input_iscd="005930", ...)
            >>> results = llm.analyze_kis_news(news_df)
            >>> for r in results:
            ...     if r["recommendation"] in ["BUY", "SELL"]:
            ...         print(f"[{r['stock_code']}] {r['recommendation']}")
        """
        if news_df is None or news_df.empty:
            logger.warning("분석할 뉴스가 없습니다.")
            return []

        results = []

        for idx, row in news_df.iterrows():
            # KIS API 응답 필드 추출
            news_title = row.get('titl', '')
            stock_code = row.get('stck_shrn_iscd', '')

            if not news_title:
                continue

            # LLM 분석 수행
            try:
                ensemble = self.analyze(
                    news_title=news_title,
                    stock_code=stock_code
                )

                # 개별 모델의 reasoning 추출
                reasoning_list = [r.reasoning for r in ensemble.individual_results]
                combined_reasoning = " | ".join(reasoning_list) if reasoning_list else "분석 불가"

                results.append({
                    "news_title": news_title,
                    "stock_code": stock_code,
                    "news_date": row.get('data_dt', ''),
                    "news_time": row.get('data_tm', ''),
                    "sentiment": ensemble.final_sentiment.value,
                    "confidence": ensemble.final_confidence,
                    "impact": ensemble.final_impact.value,
                    "recommendation": ensemble.recommendation,
                    "reasoning": combined_reasoning,
                    "consensus_score": ensemble.consensus_score,
                    "individual_results": [
                        {
                            "model": r.model_source,
                            "sentiment": r.sentiment.value,
                            "confidence": r.confidence
                        }
                        for r in ensemble.individual_results
                    ]
                })

            except Exception as e:
                logger.error(f"뉴스 분석 실패: {news_title[:30]}... - {e}")
                continue

        logger.info(f"KIS 뉴스 {len(results)}건 분석 완료")
        return results


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 하이브리드 LLM 초기화
    llm = FinancialHybridLLM(
        api_url="http://localhost:11434",
        enable_parallel=True
    )

    # 단일 뉴스 분석
    test_news = [
        "삼성전자, 3분기 영업이익 10조 원 돌파...HBM 수요 급증",
        "SK하이닉스, 美 반도체 수출 규제 강화에 주가 급락",
        "현대차, 전기차 판매량 전년 대비 15% 증가"
    ]

    for news in test_news:
        print("\n" + "=" * 60)
        result = llm.analyze(news, stock_code="005930")

        print(f"\n뉴스: {news}")
        print(f"최종 감성: {result.final_sentiment.value}")
        print(f"신뢰도: {result.final_confidence:.1%}")
        print(f"영향도: {result.final_impact.value}")
        print(f"모델 합의도: {result.consensus_score:.1%}")
        print(f"추천: {result.recommendation}")

        print("\n[개별 모델 분석]")
        for r in result.individual_results:
            print(f"  - {r.model_source}: {r.sentiment.value} "
                  f"({r.confidence:.1%}) - {r.reasoning[:50]}")
