# -*- coding: utf-8 -*-
"""
앙상블 투표 유틸리티 모듈

여러 LLM 모델의 분석 결과를 가중 투표로 결합합니다.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

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
class ModelResult:
    """단일 모델 결과"""
    model_name: str
    sentiment: SentimentType
    confidence: float
    impact: ImpactLevel
    weight: float = 1.0


@dataclass
class VoteResult:
    """투표 결과"""
    final_sentiment: SentimentType
    final_confidence: float
    final_impact: ImpactLevel
    consensus_score: float
    recommendation: str
    vote_details: Dict[str, float]


class EnsembleVoter:
    """
    앙상블 투표 관리자

    여러 모델의 분석 결과를 가중 투표로 결합하여
    최종 감성, 신뢰도, 영향도를 결정합니다.

    사용 예시:
        >>> voter = EnsembleVoter()
        >>> results = [
        ...     ModelResult("model1", SentimentType.POSITIVE, 0.8, ImpactLevel.HIGH, 0.4),
        ...     ModelResult("model2", SentimentType.POSITIVE, 0.7, ImpactLevel.MEDIUM, 0.4),
        ...     ModelResult("model3", SentimentType.NEUTRAL, 0.6, ImpactLevel.LOW, 0.2),
        ... ]
        >>> vote = voter.vote(results)
        >>> print(vote.final_sentiment)  # POSITIVE
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        consensus_threshold: float = 0.67
    ):
        """
        Args:
            confidence_threshold: 적극적 추천을 위한 최소 신뢰도
            consensus_threshold: 적극적 추천을 위한 최소 합의도
        """
        self.confidence_threshold = confidence_threshold
        self.consensus_threshold = consensus_threshold

    def vote(self, results: List[ModelResult]) -> VoteResult:
        """
        가중 투표 수행

        Args:
            results: 모델 결과 리스트

        Returns:
            VoteResult: 투표 결과
        """
        if not results:
            return VoteResult(
                final_sentiment=SentimentType.NEUTRAL,
                final_confidence=0.0,
                final_impact=ImpactLevel.LOW,
                consensus_score=0.0,
                recommendation="HOLD",
                vote_details={}
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
            sentiment_scores[result.sentiment] += result.weight * result.confidence
            impact_scores[result.impact] += result.weight
            weighted_confidence += result.weight * result.confidence
            total_weight += result.weight

        # 최종 감성 결정
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        final_impact = max(impact_scores, key=impact_scores.get)

        # 평균 신뢰도
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        # 합의도 계산
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

        # 투표 상세
        vote_details = {
            f"{s.value}_score": round(v, 3)
            for s, v in sentiment_scores.items()
        }

        return VoteResult(
            final_sentiment=final_sentiment,
            final_confidence=round(final_confidence, 3),
            final_impact=final_impact,
            consensus_score=round(consensus_score, 3),
            recommendation=recommendation,
            vote_details=vote_details
        )

    def _determine_recommendation(
        self,
        sentiment: SentimentType,
        confidence: float,
        impact: ImpactLevel,
        consensus: float
    ) -> str:
        """매매 추천 결정 로직"""
        # 고신뢰도 + 높은 합의도
        if confidence >= self.confidence_threshold and consensus >= self.consensus_threshold:
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

    def calculate_agreement(
        self,
        results: List[ModelResult]
    ) -> Tuple[float, str]:
        """
        모델 간 합의도 계산

        Args:
            results: 모델 결과 리스트

        Returns:
            Tuple[float, str]: (합의도, 합의 레벨)
                - 합의 레벨: "unanimous", "majority", "split", "none"
        """
        if not results:
            return 0.0, "none"

        sentiments = [r.sentiment for r in results]
        unique_sentiments = set(sentiments)

        # 만장일치
        if len(unique_sentiments) == 1:
            return 1.0, "unanimous"

        # 다수결 확인
        for sentiment in unique_sentiments:
            count = sentiments.count(sentiment)
            ratio = count / len(sentiments)
            if ratio >= 0.67:
                return ratio, "majority"

        # 분열
        return max(sentiments.count(s) / len(sentiments) for s in unique_sentiments), "split"

    def weighted_average_confidence(
        self,
        results: List[ModelResult]
    ) -> float:
        """
        가중 평균 신뢰도 계산

        Args:
            results: 모델 결과 리스트

        Returns:
            float: 가중 평균 신뢰도
        """
        if not results:
            return 0.0

        total_weight = sum(r.weight for r in results)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(r.confidence * r.weight for r in results)
        return weighted_sum / total_weight
