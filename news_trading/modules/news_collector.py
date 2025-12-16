# -*- coding: utf-8 -*-
"""
KIS Open API 뉴스 수집 모듈

한국투자증권 Open API의 news_title API를 사용하여
실시간 뉴스를 폴링 방식으로 수집합니다.

사용 예시:
    >>> collector = NewsCollector(polling_interval=30)
    >>> news_df = collector.collect(["005930", "000660"])
    >>> print(news_df)
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Set
import pandas as pd

# KIS API 모듈 경로 추가
from pathlib import Path
_MODULE_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _MODULE_DIR.parent.parent  # open-trading-api
sys.path.insert(0, str(_PROJECT_ROOT / "examples_llm"))
sys.path.insert(0, str(_MODULE_DIR.parent))  # news_trading
sys.path.extend(['..', '.'])

logger = logging.getLogger(__name__)


class NewsCollector:
    """
    KIS Open API 뉴스 수집기

    news_title API를 사용하여 종목별 뉴스를 수집합니다.
    중복 뉴스 필터링 및 폴링 기능을 제공합니다.

    Attributes:
        polling_interval: 폴링 주기 (초)
        seen_news: 이미 수집된 뉴스 식별자 집합 (중복 방지)

    Example:
        >>> collector = NewsCollector(polling_interval=30)
        >>> # 단일 수집
        >>> news_df = collector.collect(["005930"])
        >>> # 폴링 모드
        >>> collector.start_polling(["005930"], callback=process_news)
    """

    def __init__(self, polling_interval: int = 30):
        """
        Args:
            polling_interval: 폴링 주기 (초, 기본값: 30)
        """
        self.polling_interval = polling_interval
        self.seen_news: Set[str] = set()
        self.last_collection_time = datetime.now()

    def collect(
        self,
        stock_codes: Optional[List[str]] = None,
        filter_duplicates: bool = True,
        max_depth: int = 1
    ) -> pd.DataFrame:
        """
        뉴스 수집

        Args:
            stock_codes: 종목코드 리스트 (None이면 전체 뉴스)
            filter_duplicates: 중복 뉴스 필터링 여부
            max_depth: API 페이지 깊이 (기본값: 1)

        Returns:
            pd.DataFrame: 수집된 뉴스 데이터
                - titl: 뉴스 제목
                - data_dt: 뉴스 일자
                - data_tm: 뉴스 시간
                - stck_shrn_iscd: 종목코드

        Example:
            >>> df = collector.collect(["005930", "000660"])
            >>> print(df[['titl', 'stck_shrn_iscd']])
        """
        try:
            from domestic_stock.news_title.news_title import news_title
        except ImportError as e:
            logger.error(f"domestic_stock.news_title 모듈을 찾을 수 없습니다: {e}")
            logger.error("examples_llm/domestic_stock 경로를 확인하세요.")
            return pd.DataFrame()

        all_news = []

        # 종목별 뉴스 수집
        if stock_codes:
            for stock_code in stock_codes:
                try:
                    df = news_title(
                        fid_news_ofer_entp_code="",
                        fid_cond_mrkt_cls_code="",
                        fid_input_iscd=stock_code,
                        fid_titl_cntt="",
                        fid_input_date_1="",
                        fid_input_hour_1="",
                        fid_rank_sort_cls_code="",
                        fid_input_srno="",
                        max_depth=max_depth
                    )

                    if df is not None and not df.empty:
                        # 종목코드 컬럼 추가 (없는 경우)
                        if 'stck_shrn_iscd' not in df.columns:
                            df['stck_shrn_iscd'] = stock_code
                        all_news.append(df)
                        logger.info(f"[{stock_code}] 뉴스 {len(df)}건 수집")

                    time.sleep(0.5)  # API 호출 간격

                except Exception as e:
                    logger.error(f"[{stock_code}] 뉴스 수집 실패: {e}")
                    continue
        else:
            # 전체 뉴스 수집
            try:
                df = news_title(
                    fid_news_ofer_entp_code="",
                    fid_cond_mrkt_cls_code="",
                    fid_input_iscd="",
                    fid_titl_cntt="",
                    fid_input_date_1="",
                    fid_input_hour_1="",
                    fid_rank_sort_cls_code="",
                    fid_input_srno="",
                    max_depth=max_depth
                )

                if df is not None and not df.empty:
                    all_news.append(df)
                    logger.info(f"전체 뉴스 {len(df)}건 수집")

            except Exception as e:
                logger.error(f"전체 뉴스 수집 실패: {e}")

        # 결과 병합
        if not all_news:
            return pd.DataFrame()

        result_df = pd.concat(all_news, ignore_index=True)

        # 중복 제거
        if filter_duplicates:
            result_df = self._filter_duplicates(result_df)

        self.last_collection_time = datetime.now()
        return result_df

    def _filter_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """중복 뉴스 필터링"""
        if df.empty:
            return df

        # 뉴스 고유 식별자 생성 (제목 + 일자 + 시간)
        new_rows = []
        for idx, row in df.iterrows():
            news_id = f"{row.get('titl', '')}_{row.get('data_dt', '')}_{row.get('data_tm', '')}"

            if news_id not in self.seen_news:
                self.seen_news.add(news_id)
                new_rows.append(row)

        if not new_rows:
            logger.info("새로운 뉴스가 없습니다.")
            return pd.DataFrame()

        new_df = pd.DataFrame(new_rows)
        logger.info(f"새로운 뉴스 {len(new_df)}건 (중복 {len(df) - len(new_df)}건 필터링)")
        return new_df

    def start_polling(
        self,
        stock_codes: List[str],
        callback=None,
        max_iterations: Optional[int] = None
    ):
        """
        폴링 모드 시작

        지정된 주기로 뉴스를 수집하고 콜백 함수를 호출합니다.

        Args:
            stock_codes: 모니터링할 종목코드 리스트
            callback: 뉴스 수집 후 호출할 콜백 함수 (news_df를 인자로 받음)
            max_iterations: 최대 반복 횟수 (None이면 무한 반복)

        Example:
            >>> def on_news(news_df):
            ...     for _, row in news_df.iterrows():
            ...         print(f"새 뉴스: {row['titl']}")
            >>> collector.start_polling(["005930"], callback=on_news)
        """
        logger.info(f"뉴스 폴링 시작 (주기: {self.polling_interval}초)")
        logger.info(f"모니터링 종목: {stock_codes}")

        iteration = 0

        while True:
            try:
                # 최대 반복 횟수 체크
                if max_iterations is not None and iteration >= max_iterations:
                    logger.info("최대 반복 횟수 도달. 폴링 종료.")
                    break

                # 뉴스 수집
                news_df = self.collect(stock_codes, filter_duplicates=True)

                # 콜백 호출
                if callback and not news_df.empty:
                    callback(news_df)

                # 대기
                time.sleep(self.polling_interval)
                iteration += 1

            except KeyboardInterrupt:
                logger.info("사용자 중단. 폴링 종료.")
                break
            except Exception as e:
                logger.error(f"폴링 중 오류: {e}")
                time.sleep(self.polling_interval)

    def clear_seen_news(self):
        """수집된 뉴스 기록 초기화"""
        self.seen_news.clear()
        logger.info("뉴스 수집 기록 초기화 완료")

    def get_stats(self) -> dict:
        """수집 통계 반환"""
        return {
            "total_seen_news": len(self.seen_news),
            "last_collection_time": self.last_collection_time.isoformat(),
            "polling_interval": self.polling_interval
        }


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # KIS API 인증
    try:
        import kis_auth as ka
        ka.auth()
    except Exception as e:
        print(f"KIS API 인증 실패: {e}")
        print("kis_auth 모듈 및 .env 설정을 확인하세요.")
        sys.exit(1)

    # 뉴스 수집기 초기화
    collector = NewsCollector(polling_interval=30)

    # 테스트: 단일 수집
    print("\n=== 뉴스 수집 테스트 ===")
    watch_list = ["005930", "000660"]  # 삼성전자, SK하이닉스

    news_df = collector.collect(watch_list)

    if not news_df.empty:
        print(f"\n수집된 뉴스: {len(news_df)}건")
        for _, row in news_df.head(5).iterrows():
            print(f"  [{row.get('stck_shrn_iscd', '')}] {row.get('titl', '')[:50]}...")
    else:
        print("수집된 뉴스가 없습니다.")

    print(f"\n수집 통계: {collector.get_stats()}")
