# -*- coding: utf-8 -*-
"""
KIS Open API 뉴스 기반 자동매매 통합 예제

이 스크립트는 다음 흐름을 보여줍니다:
1. KIS API 인증
2. NewsCollector로 뉴스 수집 (news_title API)
3. FinancialHybridLLM으로 뉴스 분석
4. OrderExecutor로 매매 실행 (order_cash API)

사전 요구사항:
1. .env 파일에 KIS API 인증정보 설정
2. Ollama 설치 및 실행 (ollama serve)
3. 모델 다운로드:
   - ollama pull ingu627/exaone4.0:32b
   - ollama pull qwen3:8b

실행:
    python example_auto_trading.py
"""

import sys
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# 모듈 경로 추가
sys.path.extend(['../..', '.'])

from modules.news_collector import NewsCollector
from modules.order_executor import OrderExecutor
from modules.llm import FinancialHybridLLM

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsBasedAutoTrader:
    """
    뉴스 기반 자동매매 시스템

    KIS API 뉴스 수집 → LLM 분석 → 주문 실행의 전체 흐름을 관리합니다.

    Attributes:
        news_collector: 뉴스 수집기
        llm_analyzer: LLM 분석기
        order_executor: 주문 실행기
        watch_list: 모니터링 종목 리스트
    """

    def __init__(
        self,
        env_dv: str = "demo",
        cano: str = "",
        acnt_prdt_cd: str = "",
        watch_list: List[str] = None,
        order_qty: int = 1,
        min_confidence: float = 0.7,
        polling_interval: int = 60,
        dry_run: bool = True
    ):
        """
        Args:
            env_dv: 실전/모의 구분 ("real": 실전, "demo": 모의)
            cano: 계좌번호 앞 8자리
            acnt_prdt_cd: 계좌상품코드 (뒤 2자리)
            watch_list: 모니터링 종목 리스트
            order_qty: 기본 주문 수량
            min_confidence: 주문 실행 최소 신뢰도
            polling_interval: 뉴스 수집 주기 (초)
            dry_run: True면 실제 주문 없이 시뮬레이션만
        """
        self.env_dv = env_dv
        self.watch_list = watch_list or ["005930", "000660"]  # 기본: 삼성전자, SK하이닉스
        self.order_qty = order_qty
        self.min_confidence = min_confidence
        self.dry_run = dry_run

        # 1. 뉴스 수집기 초기화
        self.news_collector = NewsCollector(polling_interval=polling_interval)

        # 2. LLM 분석기 초기화
        try:
            self.llm_analyzer = FinancialHybridLLM(
                api_url="http://localhost:11434",
                enable_parallel=True,
                timeout=120
            )
            logger.info("LLM 분석기 초기화 완료")
        except ConnectionError as e:
            logger.warning(f"LLM 연결 실패: {e}")
            logger.warning("LLM 없이 실행됩니다. (분석 기능 비활성화)")
            self.llm_analyzer = None

        # 3. 주문 실행기 초기화
        self.order_executor = OrderExecutor(
            env_dv=env_dv,
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd
        )

        # 통계
        self.stats = {
            "news_collected": 0,
            "signals_generated": 0,
            "orders_executed": 0,
            "start_time": datetime.now()
        }

    def collect_and_analyze(self) -> List[Dict[str, Any]]:
        """
        뉴스 수집 및 분석

        Returns:
            분석 결과 리스트
        """
        logger.info(f"뉴스 수집 시작: {self.watch_list}")

        # 뉴스 수집
        news_df = self.news_collector.collect(
            stock_codes=self.watch_list,
            filter_duplicates=True
        )

        if news_df.empty:
            logger.info("새로운 뉴스 없음")
            return []

        self.stats["news_collected"] += len(news_df)
        logger.info(f"새 뉴스 {len(news_df)}건 수집됨")

        # LLM이 없으면 빈 결과 반환
        if self.llm_analyzer is None:
            logger.warning("LLM 분석기가 없어 분석을 건너뜁니다.")
            return []

        # LLM 분석
        logger.info("LLM 분석 시작...")
        analysis_results = self.llm_analyzer.analyze_kis_news(news_df)

        self.stats["signals_generated"] += len(analysis_results)
        return analysis_results

    def execute_signals(self, analysis_results: List[Dict[str, Any]]) -> List[Dict]:
        """
        분석 결과를 바탕으로 주문 실행

        Args:
            analysis_results: LLM 분석 결과 리스트

        Returns:
            주문 결과 리스트
        """
        order_results = []

        for result in analysis_results:
            recommendation = result.get("recommendation", "HOLD")
            confidence = result.get("confidence", 0.0)
            stock_code = result.get("stock_code", "")

            # HOLD거나 신뢰도 부족이면 스킵
            if recommendation == "HOLD":
                logger.info(f"[{stock_code}] HOLD - 주문 없음")
                continue

            if confidence < self.min_confidence:
                logger.info(f"[{stock_code}] 신뢰도 부족 ({confidence:.1%}) - 스킵")
                continue

            # 현재가 조회 (실제 구현 시 추가 필요)
            order_price = self._get_current_price(stock_code)

            if self.dry_run:
                logger.info(f"[DRY RUN] 주문: {stock_code} {recommendation} {self.order_qty}주 @ {order_price}원")
                order_results.append({
                    "stock_code": stock_code,
                    "recommendation": recommendation,
                    "dry_run": True,
                    "message": "시뮬레이션 모드"
                })
            else:
                # 실제 주문 실행
                order_result = self.order_executor.execute_from_llm_result(
                    llm_result=result,
                    order_qty=self.order_qty,
                    order_price=order_price,
                    min_confidence=self.min_confidence
                )

                if order_result:
                    self.stats["orders_executed"] += 1
                    order_results.append({
                        "stock_code": stock_code,
                        "order_no": order_result.order_no,
                        "success": order_result.success,
                        "message": order_result.message
                    })

        return order_results

    def _get_current_price(self, stock_code: str) -> int:
        """
        현재가 조회 (간단한 구현)

        실제로는 KIS API inquire_price를 사용해야 합니다.
        """
        # TODO: 실제 현재가 조회 API 연동
        # 여기서는 임시로 고정값 반환
        default_prices = {
            "005930": 70000,   # 삼성전자
            "000660": 150000,  # SK하이닉스
            "005380": 200000,  # 현대차
            "035720": 50000,   # 카카오
            "373220": 400000,  # LG에너지솔루션
        }
        return default_prices.get(stock_code, 50000)

    def run_once(self) -> Dict[str, Any]:
        """
        단일 실행 (수집 → 분석 → 주문)

        Returns:
            실행 결과 요약
        """
        logger.info("=" * 50)
        logger.info("자동매매 사이클 시작")
        logger.info("=" * 50)

        # 1. 수집 및 분석
        analysis_results = self.collect_and_analyze()

        # 2. 주문 실행
        order_results = self.execute_signals(analysis_results)

        # 결과 요약
        summary = {
            "timestamp": datetime.now().isoformat(),
            "news_count": len(analysis_results),
            "orders_count": len(order_results),
            "analysis_results": analysis_results,
            "order_results": order_results
        }

        logger.info(f"사이클 완료: 뉴스 {len(analysis_results)}건, 주문 {len(order_results)}건")
        return summary

    def run_polling(self, max_iterations: int = None):
        """
        폴링 모드 실행

        Args:
            max_iterations: 최대 반복 횟수 (None이면 무한)
        """
        logger.info("뉴스 기반 자동매매 시작")
        logger.info(f"모니터링 종목: {self.watch_list}")
        logger.info(f"폴링 주기: {self.news_collector.polling_interval}초")
        logger.info(f"최소 신뢰도: {self.min_confidence:.1%}")
        logger.info(f"Dry Run: {self.dry_run}")

        iteration = 0

        while True:
            try:
                if max_iterations and iteration >= max_iterations:
                    logger.info("최대 반복 횟수 도달")
                    break

                self.run_once()

                iteration += 1
                time.sleep(self.news_collector.polling_interval)

            except KeyboardInterrupt:
                logger.info("사용자 중단")
                break
            except Exception as e:
                logger.error(f"오류 발생: {e}")
                time.sleep(self.news_collector.polling_interval)

        self.print_stats()

    def print_stats(self):
        """통계 출력"""
        elapsed = datetime.now() - self.stats["start_time"]
        print("\n" + "=" * 50)
        print("자동매매 통계")
        print("=" * 50)
        print(f"실행 시간: {elapsed}")
        print(f"수집된 뉴스: {self.stats['news_collected']}건")
        print(f"생성된 신호: {self.stats['signals_generated']}건")
        print(f"실행된 주문: {self.stats['orders_executed']}건")
        print("=" * 50)


# =====================================================
# 메인 실행
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KIS Open API 뉴스 기반 자동매매 시스템")
    print("=" * 60)

    # 1. KIS API 인증
    try:
        import kis_auth as ka
        ka.auth()
        print("✅ KIS API 인증 성공")
    except Exception as e:
        print(f"❌ KIS API 인증 실패: {e}")
        print("   .env 파일 및 kis_auth 설정을 확인하세요.")
        sys.exit(1)

    # 2. 자동매매 시스템 초기화
    trader = NewsBasedAutoTrader(
        env_dv="demo",                    # 모의투자
        cano=ka.getTREnv().my_acct,       # 계좌번호
        acnt_prdt_cd=ka.getTREnv().my_prod,  # 계좌상품코드
        watch_list=["005930", "000660"],  # 삼성전자, SK하이닉스
        order_qty=1,                       # 주문수량: 1주
        min_confidence=0.7,               # 최소 신뢰도: 70%
        polling_interval=60,              # 폴링 주기: 60초
        dry_run=True                      # 시뮬레이션 모드 (실제 주문 X)
    )

    # 3. 실행 모드 선택
    print("\n[실행 모드 선택]")
    print("1. 단일 실행 (한 번만 수집/분석/주문)")
    print("2. 폴링 실행 (지속적 모니터링)")
    print("3. 잔고 조회만")

    mode = input("\n선택 (1/2/3): ").strip()

    if mode == "1":
        # 단일 실행
        result = trader.run_once()
        print("\n분석 결과:")
        for r in result.get("analysis_results", []):
            print(f"  [{r['stock_code']}] {r['sentiment']} ({r['confidence']:.1%}) - {r['recommendation']}")

    elif mode == "2":
        # 폴링 실행
        print("\n폴링 모드 시작 (Ctrl+C로 중단)")
        trader.run_polling(max_iterations=5)  # 테스트용: 5회만

    elif mode == "3":
        # 잔고 조회
        balance = trader.order_executor.get_balance()
        if balance is not None and not balance.empty:
            print("\n현재 잔고:")
            print(balance[['pdno', 'prdt_name', 'hldg_qty', 'pchs_avg_pric']].to_string())
        else:
            print("잔고가 없습니다.")

    else:
        print("잘못된 선택입니다.")

    print("\n프로그램 종료")
