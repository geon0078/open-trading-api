# -*- coding: utf-8 -*-
"""
KIS Open API 현재가 조회 모듈

한국투자증권 Open API의 inquire_price API를 사용하여
주식 현재가 시세를 조회합니다.

사용 예시:
    >>> checker = PriceChecker(env_dv="prod")
    >>> price_info = checker.get_price("005930")  # 삼성전자
    >>> print(f"현재가: {price_info['current_price']:,}원")
"""

import sys
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PriceInfo:
    """주식 현재가 정보"""
    stock_code: str           # 종목코드
    stock_name: str           # 종목명
    current_price: int        # 현재가
    change: int               # 전일대비
    change_rate: float        # 등락률
    change_sign: str          # 등락부호 (1:상한, 2:상승, 3:보합, 4:하한, 5:하락)
    open_price: int           # 시가
    high_price: int           # 고가
    low_price: int            # 저가
    volume: int               # 거래량
    volume_amount: int        # 거래대금
    market_cap: int           # 시가총액
    per: float                # PER
    pbr: float                # PBR
    eps: float                # EPS
    bps: float                # BPS
    week52_high: int          # 52주 최고가
    week52_low: int           # 52주 최저가
    upper_limit: int          # 상한가
    lower_limit: int          # 하한가
    raw_data: dict            # 원본 데이터


class PriceChecker:
    """
    KIS Open API 현재가 조회기

    주식 현재가 시세를 조회합니다.
    실시간 시세가 필요하면 WebSocket API를 사용하세요.

    Attributes:
        env_dv: 환경 구분 (prod: 실전, vps: 모의)

    Example:
        >>> checker = PriceChecker(env_dv="prod")
        >>> # 단일 종목 조회
        >>> price = checker.get_price("005930")
        >>> print(f"삼성전자 현재가: {price.current_price:,}원")
        >>> # 여러 종목 조회
        >>> prices = checker.get_prices(["005930", "000660"])
    """

    # 등락 부호 매핑
    CHANGE_SIGNS = {
        "1": "상한",
        "2": "상승",
        "3": "보합",
        "4": "하한",
        "5": "하락",
    }

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
            # examples_llm 경로 추가
            sys.path.extend(['../..', '.', '../../..'])

            import kis_auth as ka
            ka.auth()
            self._authenticated = True
            logger.info("KIS API 인증 완료")
        except Exception as e:
            logger.error(f"KIS API 인증 실패: {e}")
            raise

    def get_price(
        self,
        stock_code: str,
        market_code: str = "J"
    ) -> Optional[PriceInfo]:
        """
        단일 종목 현재가 조회

        Args:
            stock_code: 종목코드 (예: "005930")
            market_code: 시장 구분 코드
                - "J": KRX (코스피/코스닥)
                - "NX": NXT
                - "UN": 통합

        Returns:
            PriceInfo: 현재가 정보 또는 None (실패시)

        Example:
            >>> price = checker.get_price("005930")
            >>> if price:
            ...     print(f"현재가: {price.current_price:,}원")
            ...     print(f"등락률: {price.change_rate:+.2f}%")
        """
        self._ensure_auth()

        try:
            # inquire_price API 호출
            sys.path.extend(['../..', '.'])
            from domestic_stock.inquire_price.inquire_price import inquire_price

            df = inquire_price(
                env_dv=self.env_dv,
                fid_cond_mrkt_div_code=market_code,
                fid_input_iscd=stock_code
            )

            if df is None or df.empty:
                logger.warning(f"[{stock_code}] 현재가 조회 실패: 데이터 없음")
                return None

            # 첫 번째 행 데이터 추출
            row = df.iloc[0].to_dict()

            # PriceInfo 객체 생성
            price_info = PriceInfo(
                stock_code=stock_code,
                stock_name=row.get("bstp_kor_isnm", ""),
                current_price=self._safe_int(row.get("stck_prpr", 0)),
                change=self._safe_int(row.get("prdy_vrss", 0)),
                change_rate=self._safe_float(row.get("prdy_ctrt", 0)),
                change_sign=row.get("prdy_vrss_sign", "3"),
                open_price=self._safe_int(row.get("stck_oprc", 0)),
                high_price=self._safe_int(row.get("stck_hgpr", 0)),
                low_price=self._safe_int(row.get("stck_lwpr", 0)),
                volume=self._safe_int(row.get("acml_vol", 0)),
                volume_amount=self._safe_int(row.get("acml_tr_pbmn", 0)),
                market_cap=self._safe_int(row.get("hts_avls", 0)),
                per=self._safe_float(row.get("per", 0)),
                pbr=self._safe_float(row.get("pbr", 0)),
                eps=self._safe_float(row.get("eps", 0)),
                bps=self._safe_float(row.get("bps", 0)),
                week52_high=self._safe_int(row.get("w52_hgpr", 0)),
                week52_low=self._safe_int(row.get("w52_lwpr", 0)),
                upper_limit=self._safe_int(row.get("stck_mxpr", 0)),
                lower_limit=self._safe_int(row.get("stck_llam", 0)),
                raw_data=row
            )

            logger.info(
                f"[{stock_code}] 현재가: {price_info.current_price:,}원 "
                f"({price_info.change_rate:+.2f}%)"
            )
            return price_info

        except Exception as e:
            logger.error(f"[{stock_code}] 현재가 조회 오류: {e}")
            return None

    def get_prices(
        self,
        stock_codes: List[str],
        market_code: str = "J"
    ) -> Dict[str, Optional[PriceInfo]]:
        """
        여러 종목 현재가 조회

        Args:
            stock_codes: 종목코드 리스트
            market_code: 시장 구분 코드

        Returns:
            Dict[str, PriceInfo]: 종목코드별 현재가 정보

        Example:
            >>> prices = checker.get_prices(["005930", "000660"])
            >>> for code, price in prices.items():
            ...     if price:
            ...         print(f"{price.stock_name}: {price.current_price:,}원")
        """
        import time

        results = {}
        for stock_code in stock_codes:
            results[stock_code] = self.get_price(stock_code, market_code)
            time.sleep(0.2)  # API 호출 간격

        return results

    def get_price_df(
        self,
        stock_codes: List[str],
        market_code: str = "J"
    ) -> pd.DataFrame:
        """
        여러 종목 현재가를 DataFrame으로 반환

        Args:
            stock_codes: 종목코드 리스트
            market_code: 시장 구분 코드

        Returns:
            pd.DataFrame: 현재가 정보 테이블

        Example:
            >>> df = checker.get_price_df(["005930", "000660", "035720"])
            >>> print(df[['stock_name', 'current_price', 'change_rate']])
        """
        prices = self.get_prices(stock_codes, market_code)

        rows = []
        for code, price in prices.items():
            if price:
                rows.append({
                    "stock_code": price.stock_code,
                    "stock_name": price.stock_name,
                    "current_price": price.current_price,
                    "change": price.change,
                    "change_rate": price.change_rate,
                    "change_sign": self.CHANGE_SIGNS.get(price.change_sign, ""),
                    "open_price": price.open_price,
                    "high_price": price.high_price,
                    "low_price": price.low_price,
                    "volume": price.volume,
                    "volume_amount": price.volume_amount,
                    "market_cap": price.market_cap,
                    "per": price.per,
                    "pbr": price.pbr,
                })

        return pd.DataFrame(rows)

    def get_simple_quote(
        self,
        stock_code: str,
        market_code: str = "J"
    ) -> Optional[Dict]:
        """
        간단한 시세 정보만 반환 (LLM 분석용)

        Args:
            stock_code: 종목코드
            market_code: 시장 구분 코드

        Returns:
            Dict: 간단한 시세 정보
                - current_price: 현재가
                - change_rate: 등락률
                - volume: 거래량
                - trend: 추세 (상승/보합/하락)
        """
        price = self.get_price(stock_code, market_code)

        if not price:
            return None

        # 추세 판단
        if price.change_sign in ["1", "2"]:
            trend = "상승"
        elif price.change_sign in ["4", "5"]:
            trend = "하락"
        else:
            trend = "보합"

        return {
            "stock_code": stock_code,
            "stock_name": price.stock_name,
            "current_price": price.current_price,
            "change_rate": price.change_rate,
            "volume": price.volume,
            "trend": trend,
            "week52_high": price.week52_high,
            "week52_low": price.week52_low,
            "per": price.per,
            "pbr": price.pbr,
        }

    @staticmethod
    def _safe_int(value) -> int:
        """안전한 정수 변환"""
        try:
            if pd.isna(value) or value == "":
                return 0
            return int(float(str(value).replace(",", "")))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _safe_float(value) -> float:
        """안전한 실수 변환"""
        try:
            if pd.isna(value) or value == "":
                return 0.0
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return 0.0


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

    # 현재가 조회기 초기화
    checker = PriceChecker(env_dv="prod")

    # 테스트: 삼성전자 현재가 조회
    print("\n=== 삼성전자 현재가 조회 ===")
    price = checker.get_price("005930")

    if price:
        print(f"종목명: {price.stock_name}")
        print(f"현재가: {price.current_price:,}원")
        print(f"전일대비: {price.change:+,}원 ({price.change_rate:+.2f}%)")
        print(f"시가: {price.open_price:,}원")
        print(f"고가: {price.high_price:,}원")
        print(f"저가: {price.low_price:,}원")
        print(f"거래량: {price.volume:,}주")
        print(f"52주 최고: {price.week52_high:,}원")
        print(f"52주 최저: {price.week52_low:,}원")
        print(f"PER: {price.per:.2f}")
        print(f"PBR: {price.pbr:.2f}")
    else:
        print("현재가 조회 실패")

    # 테스트: 여러 종목 조회
    print("\n=== 여러 종목 현재가 조회 ===")
    watch_list = ["005930", "000660", "035720"]  # 삼성전자, SK하이닉스, 카카오
    df = checker.get_price_df(watch_list)

    if not df.empty:
        print(df[['stock_code', 'stock_name', 'current_price', 'change_rate']].to_string())
