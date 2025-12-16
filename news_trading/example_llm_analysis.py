# -*- coding: utf-8 -*-
"""
32GB VRAM 하이브리드 LLM 뉴스 분석 예제

이 스크립트는 FinancialHybridLLM을 사용하여
금융 뉴스를 분석하는 방법을 보여줍니다.

사전 요구사항:
1. Ollama 설치 및 실행 (ollama serve)
2. 모델 다운로드:
   - ollama pull ingu627/exaone4.0:32b
   - ollama pull qwen3:8b
   - fin-r1 (GGUF 변환 필요)

실행:
    python example_llm_analysis.py
"""

import logging
import sys

# 모듈 경로 추가
sys.path.insert(0, '.')

from modules.llm import FinancialHybridLLM


def main():
    """메인 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("32GB VRAM 하이브리드 LLM 뉴스 분석 예제")
    print("=" * 60)

    # 하이브리드 LLM 초기화
    try:
        llm = FinancialHybridLLM(
            api_url="http://localhost:11434",
            enable_parallel=True,
            timeout=120
        )
        print("\n✅ LLM 초기화 완료")
    except ConnectionError as e:
        print(f"\n❌ Ollama 서버 연결 실패: {e}")
        print("   'ollama serve' 명령어로 Ollama를 먼저 실행하세요.")
        return

    # 테스트 뉴스 목록
    test_news = [
        {
            "title": "삼성전자, 3분기 영업이익 10조 원 돌파...HBM 수요 급증",
            "stock_code": "005930",
            "expected": "POSITIVE"
        },
        {
            "title": "SK하이닉스, 美 반도체 수출 규제 강화에 주가 급락",
            "stock_code": "000660",
            "expected": "NEGATIVE"
        },
        {
            "title": "현대차, 전기차 판매량 전년 대비 15% 증가",
            "stock_code": "005380",
            "expected": "POSITIVE"
        },
        {
            "title": "카카오, 임시 주주총회 개최 예정...경영진 인사 변동 예고",
            "stock_code": "035720",
            "expected": "NEUTRAL"
        },
        {
            "title": "LG에너지솔루션, 美 애리조나 공장 착공...1조원 투자",
            "stock_code": "373220",
            "expected": "POSITIVE"
        }
    ]

    print("\n" + "-" * 60)
    print("뉴스 분석 시작")
    print("-" * 60)

    results = []

    for i, news in enumerate(test_news, 1):
        print(f"\n[{i}/{len(test_news)}] 분석 중...")
        print(f"📰 뉴스: {news['title']}")
        print(f"🏢 종목: {news['stock_code']}")

        # 분석 수행
        result = llm.analyze(
            news_title=news["title"],
            stock_code=news["stock_code"]
        )

        results.append({
            "news": news,
            "result": result
        })

        # 결과 출력
        print(f"\n📊 분석 결과:")
        print(f"   • 최종 감성: {result.final_sentiment.value.upper()}")
        print(f"   • 신뢰도: {result.final_confidence:.1%}")
        print(f"   • 영향도: {result.final_impact.value}")
        print(f"   • 모델 합의도: {result.consensus_score:.1%}")
        print(f"   • 매매 추천: {result.recommendation}")

        # 개별 모델 결과
        if result.individual_results:
            print(f"\n   [개별 모델 분석]")
            for r in result.individual_results:
                print(f"   • {r.model_source}: {r.sentiment.value} "
                      f"({r.confidence:.0%}) - {r.reasoning[:40]}...")

    # 요약
    print("\n" + "=" * 60)
    print("분석 요약")
    print("=" * 60)

    for item in results:
        news = item["news"]
        result = item["result"]
        match = "✅" if result.final_sentiment.value == news["expected"].lower() else "❌"

        print(f"\n{match} {news['title'][:30]}...")
        print(f"   예상: {news['expected']}, 결과: {result.final_sentiment.value.upper()}, 추천: {result.recommendation}")

    # 모델 상태 확인
    print("\n" + "-" * 60)
    print("모델 상태")
    print("-" * 60)
    status = llm.get_model_status()
    print(f"실행 중인 모델: {len(status['running_models'])}개")
    print(f"총 VRAM 사용량: {status['total_vram_used']:.1f} GB")


if __name__ == "__main__":
    main()
