# -*- coding: utf-8 -*-
"""
KIS Open API OHLCV 데이터 조회 모듈

분봉/일봉/당일체결 데이터를 조회하여
기술적 분석에 필요한 OHLCV 데이터를 제공합니다.

사용 예시:
    >>> fetcher = OHLCVFetcher(env_dv="prod")
    >>> df = fetcher.get_minute_data("005930", minutes=30)
    >>> df = fetcher.get_daily_data("005930", days=60)
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class OHLCVFetcher:
    """
    KIS Open API OHLCV 데이터 조회기

    분봉, 일봉, 당일 체결 데이터를 조회합니다.
    """

    def __init__(self, env_dv: str = "prod"):
        """
        Args:
            env_dv: 환경 구분 ("prod": 실전, "vps": 모의)
        """
        self.env_dv = "real" if env_dv == "prod" else "demo"
        self._authenticated = False

    def _ensure_auth(self):
        """인증 확인 및 수행"""
        if self._authenticated:
            return

        try:
            sys.path.extend(['../..', '.', '../../..'])
            import kis_auth as ka
            ka.auth()
            self._authenticated = True
            logger.info("KIS API 인증 완료")
        except Exception as e:
            logger.error(f"KIS API 인증 실패: {e}")
            raise

    def get_minute_data(
        self,
        stock_code: str,
        date: Optional[str] = None,
        time: str = "153000",
        market_code: str = "J"
    ) -> pd.DataFrame:
        """
        분봉 데이터 조회 (주식일별분봉조회 API)

        Args:
            stock_code: 종목코드 (예: "005930")
            date: 조회 날짜 (YYYYMMDD, 기본: 오늘)
            time: 조회 시작 시간 (HHMMSS, 기본: "153000")
            market_code: 시장 코드 (J: KRX, NX: NXT, UN: 통합)

        Returns:
            DataFrame: OHLCV 데이터
                - datetime: 시간
                - open, high, low, close: 가격
                - volume: 거래량
        """
        self._ensure_auth()

        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        try:
            from domestic_stock.inquire_time_dailychartprice.inquire_time_dailychartprice import (
                inquire_time_dailychartprice
            )

            output1, output2 = inquire_time_dailychartprice(
                fid_cond_mrkt_div_code=market_code,
                fid_input_iscd=stock_code,
                fid_input_hour_1=time,
                fid_input_date_1=date,
                fid_pw_data_incu_yn="N"
            )

            if output2.empty:
                logger.warning(f"[{stock_code}] 분봉 데이터 없음")
                return pd.DataFrame()

            # 컬럼명 표준화
            df = output2.rename(columns={
                'stck_bsop_date': 'date',
                'stck_cntg_hour': 'time',
                'stck_oprc': 'open',
                'stck_hgpr': 'high',
                'stck_lwpr': 'low',
                'stck_prpr': 'close',
                'cntg_vol': 'volume',
                'acml_tr_pbmn': 'amount'
            })

            # 필요한 컬럼만 선택
            cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols].copy()

            # 숫자형 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # datetime 컬럼 생성
            if 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(
                    df['date'].astype(str) + df['time'].astype(str).str.zfill(6),
                    format='%Y%m%d%H%M%S',
                    errors='coerce'
                )

            # 시간순 정렬
            df = df.sort_values('datetime' if 'datetime' in df.columns else 'time')
            df = df.reset_index(drop=True)

            logger.info(f"[{stock_code}] 분봉 데이터 {len(df)}건 조회")
            return df

        except Exception as e:
            logger.error(f"[{stock_code}] 분봉 조회 오류: {e}")
            return pd.DataFrame()

    def get_daily_data(
        self,
        stock_code: str,
        days: int = 60,
        end_date: Optional[str] = None,
        period: str = "D",
        market_code: str = "J",
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        일봉/주봉/월봉 데이터 조회 (국내주식기간별시세 API)

        Args:
            stock_code: 종목코드
            days: 조회 기간 (일 수, 기본: 60)
            end_date: 종료일 (YYYYMMDD, 기본: 오늘)
            period: 기간 구분 (D: 일봉, W: 주봉, M: 월봉, Y: 년봉)
            market_code: 시장 코드
            adjusted: 수정주가 여부 (True: 수정주가, False: 원주가)

        Returns:
            DataFrame: OHLCV 데이터
        """
        self._ensure_auth()

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        # 시작일 계산
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=days)
        start_date = start_dt.strftime("%Y%m%d")

        try:
            from domestic_stock.inquire_daily_itemchartprice.inquire_daily_itemchartprice import (
                inquire_daily_itemchartprice
            )

            output1, output2 = inquire_daily_itemchartprice(
                env_dv=self.env_dv,
                fid_cond_mrkt_div_code=market_code,
                fid_input_iscd=stock_code,
                fid_input_date_1=start_date,
                fid_input_date_2=end_date,
                fid_period_div_code=period,
                fid_org_adj_prc="0" if adjusted else "1"
            )

            if output2.empty:
                logger.warning(f"[{stock_code}] 일봉 데이터 없음")
                return pd.DataFrame()

            # 컬럼명 표준화
            df = output2.rename(columns={
                'stck_bsop_date': 'date',
                'stck_oprc': 'open',
                'stck_hgpr': 'high',
                'stck_lwpr': 'low',
                'stck_clpr': 'close',
                'acml_vol': 'volume',
                'acml_tr_pbmn': 'amount',
                'prdy_vrss': 'change',
                'prdy_vrss_sign': 'change_sign',
                'prdy_ctrt': 'change_rate'
            })

            # 필요한 컬럼만 선택
            cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'change_rate']
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols].copy()

            # 숫자형 변환
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'change_rate']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # datetime 컬럼 생성
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

            # 날짜순 정렬
            df = df.sort_values('date')
            df = df.reset_index(drop=True)

            logger.info(f"[{stock_code}] 일봉 데이터 {len(df)}건 조회")
            return df

        except Exception as e:
            logger.error(f"[{stock_code}] 일봉 조회 오류: {e}")
            return pd.DataFrame()

    def get_intraday_ticks(
        self,
        stock_code: str,
        start_time: str = "090000",
        market_code: str = "J"
    ) -> pd.DataFrame:
        """
        당일 시간대별 체결 데이터 조회

        Args:
            stock_code: 종목코드
            start_time: 조회 시작 시간 (HHMMSS)
            market_code: 시장 코드

        Returns:
            DataFrame: 시간대별 체결 데이터
        """
        self._ensure_auth()

        try:
            from domestic_stock.inquire_time_itemconclusion.inquire_time_itemconclusion import (
                inquire_time_itemconclusion
            )

            output1, output2 = inquire_time_itemconclusion(
                env_dv=self.env_dv,
                fid_cond_mrkt_div_code=market_code,
                fid_input_iscd=stock_code,
                fid_input_hour_1=start_time
            )

            if output2.empty:
                logger.warning(f"[{stock_code}] 체결 데이터 없음")
                return pd.DataFrame()

            # 컬럼명 표준화
            df = output2.rename(columns={
                'stck_cntg_hour': 'time',
                'stck_prpr': 'price',
                'prdy_vrss': 'change',
                'prdy_vrss_sign': 'change_sign',
                'prdy_ctrt': 'change_rate',
                'acml_vol': 'volume',
                'cnqn': 'tick_volume',
                'tday_rltv': 'strength'  # 체결강도
            })

            # 숫자형 변환
            for col in ['price', 'change', 'change_rate', 'volume', 'tick_volume', 'strength']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 시간순 정렬
            df = df.sort_values('time')
            df = df.reset_index(drop=True)

            logger.info(f"[{stock_code}] 체결 데이터 {len(df)}건 조회")
            return df

        except Exception as e:
            logger.error(f"[{stock_code}] 체결 조회 오류: {e}")
            return pd.DataFrame()

    def get_scalping_data(
        self,
        stock_code: str,
        market_code: str = "J"
    ) -> Dict:
        """
        스캘핑용 종합 데이터 조회

        분봉, 일봉, 당일 체결 데이터를 한번에 조회합니다.

        Args:
            stock_code: 종목코드
            market_code: 시장 코드

        Returns:
            Dict: {
                'minute': 분봉 DataFrame,
                'daily': 일봉 DataFrame,
                'ticks': 체결 DataFrame
            }
        """
        today = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H%M%S")

        return {
            'minute': self.get_minute_data(stock_code, today, current_time, market_code),
            'daily': self.get_daily_data(stock_code, days=30, market_code=market_code),
            'ticks': self.get_intraday_ticks(stock_code, "090000", market_code)
        }


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 설정 로드
    try:
        from config_loader import setup_kis_config
        setup_kis_config()
    except ImportError:
        pass

    # OHLCV 조회기 초기화
    fetcher = OHLCVFetcher(env_dv="prod")

    # 테스트: 분봉 데이터
    print("\n=== 삼성전자 분봉 데이터 ===")
    minute_df = fetcher.get_minute_data("005930")
    if not minute_df.empty:
        print(minute_df.head(10))

    # 테스트: 일봉 데이터
    print("\n=== 삼성전자 일봉 데이터 (최근 30일) ===")
    daily_df = fetcher.get_daily_data("005930", days=30)
    if not daily_df.empty:
        print(daily_df.tail(10))

    # 테스트: 당일 체결
    print("\n=== 삼성전자 당일 체결 ===")
    ticks_df = fetcher.get_intraday_ticks("005930")
    if not ticks_df.empty:
        print(ticks_df.head(10))
