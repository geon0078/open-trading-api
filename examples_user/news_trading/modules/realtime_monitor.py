# -*- coding: utf-8 -*-
"""
KIS Open API 실시간 시세 모니터 모듈

한국투자증권 WebSocket API를 사용하여
실시간 체결가/호가 데이터를 모니터링합니다.

사용 예시:
    >>> monitor = RealtimeMonitor(env_dv="prod")
    >>> monitor.start(
    ...     stock_codes=["005930", "000660"],
    ...     on_price_change=my_callback
    ... )
"""

import sys
import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RealtimePrice:
    """실시간 체결 데이터"""
    stock_code: str           # 종목코드
    price: int                # 체결가
    change: int               # 전일대비
    change_rate: float        # 등락률
    change_sign: str          # 등락부호
    volume: int               # 체결수량
    acml_volume: int          # 누적거래량
    acml_amount: int          # 누적거래대금
    timestamp: str            # 체결시간
    raw_data: dict            # 원본 데이터


@dataclass
class RealtimeAskBid:
    """실시간 호가 데이터"""
    stock_code: str           # 종목코드
    ask_prices: List[int]     # 매도호가 (1~10)
    bid_prices: List[int]     # 매수호가 (1~10)
    ask_volumes: List[int]    # 매도잔량 (1~10)
    bid_volumes: List[int]    # 매수잔량 (1~10)
    total_ask_volume: int     # 총매도잔량
    total_bid_volume: int     # 총매수잔량
    timestamp: str            # 시간
    raw_data: dict            # 원본 데이터


class RealtimeMonitor:
    """
    KIS Open API 실시간 시세 모니터

    WebSocket을 통해 실시간 체결가/호가 데이터를 수신합니다.
    콜백 함수를 통해 데이터 변화에 반응할 수 있습니다.

    Attributes:
        env_dv: 환경 구분 (prod: 실전, vps: 모의)
        price_history: 종목별 최근 가격 히스토리

    Example:
        >>> def on_price(price: RealtimePrice):
        ...     print(f"{price.stock_code}: {price.price:,}원")
        >>>
        >>> monitor = RealtimeMonitor(env_dv="prod")
        >>> monitor.start(["005930"], on_price_change=on_price)
    """

    def __init__(self, env_dv: str = "prod"):
        """
        Args:
            env_dv: 환경 구분 ("prod": 실전, "vps": 모의)
        """
        self.env_dv = "real" if env_dv == "prod" else "demo"
        self.price_history: Dict[str, List[RealtimePrice]] = {}
        self.askbid_history: Dict[str, RealtimeAskBid] = {}
        self._authenticated = False
        self._ws = None
        self._callbacks = {
            "on_price_change": None,
            "on_askbid_change": None,
            "on_significant_change": None,
        }
        # 유의미한 변동 기준 (%)
        self.significant_change_threshold = 1.0

    def _ensure_auth(self):
        """WebSocket 인증 확인 및 수행"""
        if self._authenticated:
            return

        try:
            sys.path.extend(['../..', '.', '../../..'])

            import kis_auth as ka
            ka.auth(svr="prod" if self.env_dv == "real" else "vps")
            ka.auth_ws(svr="prod" if self.env_dv == "real" else "vps")
            self._authenticated = True
            logger.info("KIS WebSocket 인증 완료")
        except Exception as e:
            logger.error(f"KIS WebSocket 인증 실패: {e}")
            raise

    def _on_result(self, ws, tr_id: str, df: pd.DataFrame, data_map: dict):
        """WebSocket 데이터 수신 콜백"""
        if df.empty:
            return

        try:
            # 체결가 데이터 (H0STCNT0)
            if tr_id.startswith("H0STCNT") or tr_id == "H0STCNT0":
                self._handle_price_data(df)
            # 호가 데이터 (H0STASP0)
            elif tr_id.startswith("H0STASP") or tr_id == "H0STASP0":
                self._handle_askbid_data(df)
        except Exception as e:
            logger.error(f"데이터 처리 오류: {e}")

    def _handle_price_data(self, df: pd.DataFrame):
        """체결가 데이터 처리"""
        for _, row in df.iterrows():
            try:
                stock_code = str(row.get("MKSC_SHRN_ISCD", ""))
                if not stock_code:
                    continue

                price = RealtimePrice(
                    stock_code=stock_code,
                    price=self._safe_int(row.get("STCK_PRPR", 0)),
                    change=self._safe_int(row.get("PRDY_VRSS", 0)),
                    change_rate=self._safe_float(row.get("PRDY_CTRT", 0)),
                    change_sign=str(row.get("PRDY_VRSS_SIGN", "3")),
                    volume=self._safe_int(row.get("CNTG_VOL", 0)),
                    acml_volume=self._safe_int(row.get("ACML_VOL", 0)),
                    acml_amount=self._safe_int(row.get("ACML_TR_PBMN", 0)),
                    timestamp=str(row.get("STCK_CNTG_HOUR", "")),
                    raw_data=row.to_dict()
                )

                # 히스토리 저장
                if stock_code not in self.price_history:
                    self.price_history[stock_code] = []
                self.price_history[stock_code].append(price)

                # 최대 1000개 유지
                if len(self.price_history[stock_code]) > 1000:
                    self.price_history[stock_code] = self.price_history[stock_code][-500:]

                # 콜백 호출
                if self._callbacks["on_price_change"]:
                    self._callbacks["on_price_change"](price)

                # 유의미한 변동 체크
                if abs(price.change_rate) >= self.significant_change_threshold:
                    if self._callbacks["on_significant_change"]:
                        self._callbacks["on_significant_change"](price)

            except Exception as e:
                logger.error(f"체결가 데이터 처리 오류: {e}")

    def _handle_askbid_data(self, df: pd.DataFrame):
        """호가 데이터 처리"""
        for _, row in df.iterrows():
            try:
                stock_code = str(row.get("MKSC_SHRN_ISCD", ""))
                if not stock_code:
                    continue

                askbid = RealtimeAskBid(
                    stock_code=stock_code,
                    ask_prices=[
                        self._safe_int(row.get(f"ASKP{i}", 0))
                        for i in range(1, 11)
                    ],
                    bid_prices=[
                        self._safe_int(row.get(f"BIDP{i}", 0))
                        for i in range(1, 11)
                    ],
                    ask_volumes=[
                        self._safe_int(row.get(f"ASKP_RSQN{i}", 0))
                        for i in range(1, 11)
                    ],
                    bid_volumes=[
                        self._safe_int(row.get(f"BIDP_RSQN{i}", 0))
                        for i in range(1, 11)
                    ],
                    total_ask_volume=self._safe_int(row.get("TOTAL_ASKP_RSQN", 0)),
                    total_bid_volume=self._safe_int(row.get("TOTAL_BIDP_RSQN", 0)),
                    timestamp=str(row.get("BSOP_HOUR", "")),
                    raw_data=row.to_dict()
                )

                self.askbid_history[stock_code] = askbid

                # 콜백 호출
                if self._callbacks["on_askbid_change"]:
                    self._callbacks["on_askbid_change"](askbid)

            except Exception as e:
                logger.error(f"호가 데이터 처리 오류: {e}")

    def start(
        self,
        stock_codes: List[str],
        on_price_change: Optional[Callable[[RealtimePrice], None]] = None,
        on_askbid_change: Optional[Callable[[RealtimeAskBid], None]] = None,
        on_significant_change: Optional[Callable[[RealtimePrice], None]] = None,
        subscribe_price: bool = True,
        subscribe_askbid: bool = False,
    ):
        """
        실시간 모니터링 시작

        Args:
            stock_codes: 모니터링할 종목코드 리스트
            on_price_change: 체결가 변동시 콜백
            on_askbid_change: 호가 변동시 콜백
            on_significant_change: 유의미한 변동시 콜백 (threshold 이상)
            subscribe_price: 체결가 구독 여부
            subscribe_askbid: 호가 구독 여부

        Example:
            >>> monitor.start(
            ...     stock_codes=["005930", "000660"],
            ...     on_price_change=lambda p: print(p.price),
            ...     on_significant_change=lambda p: alert(p)
            ... )
        """
        self._ensure_auth()

        # 콜백 등록
        self._callbacks["on_price_change"] = on_price_change
        self._callbacks["on_askbid_change"] = on_askbid_change
        self._callbacks["on_significant_change"] = on_significant_change

        try:
            import kis_auth as ka

            # WebSocket 생성
            kws = ka.KISWebSocket(api_url="/tryitout")

            # 체결가 구독
            if subscribe_price:
                from domestic_stock.domestic_stock_functions_ws import ccnl_krx
                kws.subscribe(request=ccnl_krx, data=stock_codes)
                logger.info(f"실시간 체결가 구독: {stock_codes}")

            # 호가 구독
            if subscribe_askbid:
                from domestic_stock.domestic_stock_functions_ws import asking_price_krx
                kws.subscribe(request=asking_price_krx, data=stock_codes)
                logger.info(f"실시간 호가 구독: {stock_codes}")

            # WebSocket 시작
            logger.info("실시간 모니터링 시작...")
            kws.start(on_result=self._on_result, result_all_data=True)

        except KeyboardInterrupt:
            logger.info("사용자 중단")
        except Exception as e:
            logger.error(f"WebSocket 오류: {e}")
            raise

    def get_latest_price(self, stock_code: str) -> Optional[RealtimePrice]:
        """최근 체결가 조회"""
        if stock_code in self.price_history and self.price_history[stock_code]:
            return self.price_history[stock_code][-1]
        return None

    def get_latest_askbid(self, stock_code: str) -> Optional[RealtimeAskBid]:
        """최근 호가 조회"""
        return self.askbid_history.get(stock_code)

    def get_price_trend(
        self,
        stock_code: str,
        window: int = 10
    ) -> Optional[Dict]:
        """
        최근 가격 추세 분석

        Args:
            stock_code: 종목코드
            window: 분석 윈도우 크기

        Returns:
            Dict: 추세 정보 (trend, avg_price, price_range, volume)
        """
        if stock_code not in self.price_history:
            return None

        history = self.price_history[stock_code][-window:]
        if len(history) < 2:
            return None

        prices = [p.price for p in history]
        volumes = [p.volume for p in history]

        # 추세 판단
        if prices[-1] > prices[0]:
            trend = "up"
        elif prices[-1] < prices[0]:
            trend = "down"
        else:
            trend = "flat"

        return {
            "stock_code": stock_code,
            "trend": trend,
            "start_price": prices[0],
            "end_price": prices[-1],
            "avg_price": sum(prices) / len(prices),
            "high_price": max(prices),
            "low_price": min(prices),
            "total_volume": sum(volumes),
            "data_points": len(history),
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

    # 콜백 함수 정의
    def on_price(price: RealtimePrice):
        print(
            f"[{price.timestamp}] {price.stock_code}: "
            f"{price.price:,}원 ({price.change_rate:+.2f}%) "
            f"Vol: {price.volume:,}"
        )

    def on_significant(price: RealtimePrice):
        print(
            f"*** ALERT *** {price.stock_code}: "
            f"{price.change_rate:+.2f}% 변동!"
        )

    # 실시간 모니터 시작
    monitor = RealtimeMonitor(env_dv="prod")
    monitor.significant_change_threshold = 0.5  # 0.5% 이상 변동시 알림

    print("\n=== 실시간 시세 모니터링 ===")
    print("종목: 삼성전자(005930), SK하이닉스(000660)")
    print("Ctrl+C로 종료\n")

    monitor.start(
        stock_codes=["005930", "000660"],
        on_price_change=on_price,
        on_significant_change=on_significant
    )
