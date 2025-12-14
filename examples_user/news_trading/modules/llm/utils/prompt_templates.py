# -*- coding: utf-8 -*-
"""
프롬프트 템플릿 관리 모듈

각 LLM 모델별 최적화된 프롬프트 템플릿을 제공합니다.
"""

from typing import Dict


class PromptTemplates:
    """
    금융 뉴스 분석용 프롬프트 템플릿 관리자

    각 모델의 특성에 맞게 최적화된 프롬프트를 제공합니다.
    - korean_reasoning: 한국어 심층 분석
    - financial_expert: 영어 기반 금융 전문 분석
    - fast_filter: 빠른 스크리닝용 간결한 프롬프트
    """

    # 한국어 추론 모델용 (EXAONE 4.0)
    KOREAN_REASONING = """당신은 한국 증권시장 전문 애널리스트입니다.
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

JSON만 출력하세요."""

    # 금융 전문가 모델용 (Fin-R1)
    FINANCIAL_EXPERT = """You are a financial analyst expert.
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

Output JSON only."""

    # 빠른 필터링 모델용 (Qwen3)
    FAST_FILTER = """뉴스 감성 분석:

뉴스: {news_title}
종목: {stock_code}

JSON으로 응답:
{{"sentiment": "positive|negative|neutral", "confidence": 0.0~1.0, "impact": "high|medium|low", "reasoning": "한줄설명"}}"""

    # 상세 분석용 (긴 뉴스 본문)
    DETAILED_ANALYSIS = """당신은 한국 증권시장 전문 애널리스트입니다.
다음 뉴스 기사 전문을 읽고 주가에 미칠 영향을 상세히 분석해주세요.

[뉴스 제목]
{news_title}

[뉴스 본문]
{news_content}

[종목코드]
{stock_code}

[관련 종목]
{related_stocks}

다음 JSON 형식으로만 응답하세요:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0~1.0,
    "impact": "high|medium|low",
    "short_term_impact": "주가에 대한 단기 영향 (1-3일)",
    "mid_term_impact": "주가에 대한 중기 영향 (1-4주)",
    "key_factors": ["핵심 요인 1", "핵심 요인 2"],
    "risk_factors": ["리스크 요인 1", "리스크 요인 2"],
    "reasoning": "상세한 판단 근거"
}}

JSON만 출력하세요."""

    # 비교 분석용 (여러 뉴스)
    COMPARATIVE_ANALYSIS = """당신은 한국 증권시장 전문 애널리스트입니다.
다음 여러 뉴스를 종합적으로 분석하여 시장 동향을 파악해주세요.

[뉴스 목록]
{news_list}

[분석 대상 종목]
{stock_code}

다음 JSON 형식으로만 응답하세요:
{{
    "overall_sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.0~1.0,
    "market_trend": "상승|하락|횡보|혼조",
    "key_themes": ["주요 테마 1", "주요 테마 2"],
    "recommendation": "BUY|HOLD|SELL",
    "reasoning": "종합 판단 근거"
}}

JSON만 출력하세요."""

    @classmethod
    def get_template(cls, template_name: str) -> str:
        """
        템플릿 이름으로 프롬프트 가져오기

        Args:
            template_name: 템플릿 이름
                - "korean_reasoning": 한국어 심층 분석
                - "financial_expert": 금융 전문 분석
                - "fast_filter": 빠른 스크리닝
                - "detailed_analysis": 상세 분석
                - "comparative_analysis": 비교 분석

        Returns:
            str: 프롬프트 템플릿
        """
        templates = {
            "korean_reasoning": cls.KOREAN_REASONING,
            "financial_expert": cls.FINANCIAL_EXPERT,
            "fast_filter": cls.FAST_FILTER,
            "detailed_analysis": cls.DETAILED_ANALYSIS,
            "comparative_analysis": cls.COMPARATIVE_ANALYSIS,
        }
        return templates.get(template_name, cls.FAST_FILTER)

    @classmethod
    def get_all_templates(cls) -> Dict[str, str]:
        """모든 템플릿 반환"""
        return {
            "korean_reasoning": cls.KOREAN_REASONING,
            "financial_expert": cls.FINANCIAL_EXPERT,
            "fast_filter": cls.FAST_FILTER,
            "detailed_analysis": cls.DETAILED_ANALYSIS,
            "comparative_analysis": cls.COMPARATIVE_ANALYSIS,
        }

    @staticmethod
    def format_prompt(
        template: str,
        news_title: str,
        stock_code: str = "",
        **kwargs
    ) -> str:
        """
        프롬프트 템플릿에 값 채우기

        Args:
            template: 프롬프트 템플릿
            news_title: 뉴스 제목
            stock_code: 종목코드
            **kwargs: 추가 파라미터

        Returns:
            str: 완성된 프롬프트
        """
        return template.format(
            news_title=news_title,
            stock_code=stock_code,
            **kwargs
        )
