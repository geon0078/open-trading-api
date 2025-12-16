# -*- coding: utf-8 -*-
"""KIS API 서비스 래퍼."""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# 기존 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # news_trading
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))  # open-trading-api
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "examples_user"))  # examples_user (for domestic_stock)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "examples_user" / "domestic_stock"))  # kis_auth

from config import settings
from models.account import AccountBalance, Holding, Order, OrdersList

logger = logging.getLogger(__name__)


class KISService:
    """KIS API 서비스."""

    def __init__(self):
        self._authenticated = False
        self._ka = None
        self._trenv = None

    async def ensure_auth(self, svr: str = None) -> bool:
        """비동기 인증 확인."""
        if self._authenticated and self._ka is not None:
            return True

        svr = svr or settings.kis_env

        try:
            loop = asyncio.get_event_loop()

            def _auth():
                import kis_auth as ka
                ka.auth(svr=svr)
                return ka, ka.getTREnv()

            self._ka, self._trenv = await loop.run_in_executor(None, _auth)
            self._authenticated = True
            logger.info(f"KIS API 인증 성공 (환경: {svr})")
            return True
        except Exception as e:
            logger.error(f"KIS API 인증 실패: {e}")
            self._authenticated = False
            raise RuntimeError(f"KIS API 인증 실패: {e}")

    async def get_account_balance(self) -> AccountBalance:
        """계좌 잔고 조회."""
        await self.ensure_auth()

        loop = asyncio.get_event_loop()

        def _fetch_balance():
            from domestic_stock.domestic_stock_functions import inquire_balance

            df1, df2 = inquire_balance(
                env_dv="real" if settings.kis_env == "prod" else "demo",
                cano=self._trenv.my_acct,
                acnt_prdt_cd=self._trenv.my_prod,
                afhr_flpr_yn="N",
                inqr_dvsn="02",  # 종목별
                unpr_dvsn="01",
                fund_sttl_icld_yn="N",
                fncg_amt_auto_rdpt_yn="N",
                prcs_dvsn="00"
            )
            return df1, df2

        try:
            df1, df2 = await loop.run_in_executor(None, _fetch_balance)
            return self._convert_balance(df1, df2)
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            # 빈 결과 반환
            return AccountBalance(
                deposit=0,
                total_eval=0,
                total_pnl=0,
                pnl_rate=0.0,
                holdings=[],
                updated_at=datetime.now()
            )

    def _convert_balance(self, df1: pd.DataFrame, df2: pd.DataFrame) -> AccountBalance:
        """DataFrame을 AccountBalance 모델로 변환."""
        holdings = []

        if df1 is not None and not df1.empty:
            for _, row in df1.iterrows():
                try:
                    holding = Holding(
                        code=str(row.get("pdno", "")),
                        name=str(row.get("prdt_name", "")),
                        quantity=int(row.get("hldg_qty", 0)),
                        avg_price=float(row.get("pchs_avg_pric", 0)),
                        current_price=int(row.get("prpr", 0)),
                        eval_amount=int(row.get("evlu_amt", 0)),
                        pnl=int(row.get("evlu_pfls_amt", 0)),
                        pnl_rate=float(row.get("evlu_pfls_rt", 0))
                    )
                    if holding.quantity > 0:
                        holdings.append(holding)
                except Exception as e:
                    logger.warning(f"보유종목 변환 오류: {e}")

        # 계좌 요약 정보
        deposit = 0
        total_eval = 0
        total_pnl = 0
        pnl_rate = 0.0

        if df2 is not None and not df2.empty:
            row = df2.iloc[0]
            deposit = int(row.get("dnca_tot_amt", 0))
            total_eval = int(row.get("tot_evlu_amt", 0))
            total_pnl = int(row.get("evlu_pfls_smtl_amt", 0))
            pnl_rate = float(row.get("asst_icdc_erng_rt", 0))

        return AccountBalance(
            deposit=deposit,
            total_eval=total_eval,
            total_purchase=total_eval - total_pnl if total_eval > 0 else 0,
            total_pnl=total_pnl,
            pnl_rate=pnl_rate,
            holdings_count=len(holdings),
            holdings=holdings,
            updated_at=datetime.now()
        )

    async def get_today_orders(self) -> OrdersList:
        """오늘 체결 내역 조회."""
        await self.ensure_auth()

        loop = asyncio.get_event_loop()

        def _fetch_orders():
            from domestic_stock.domestic_stock_functions import inquire_daily_ccld

            today = datetime.now().strftime("%Y%m%d")
            df1, df2 = inquire_daily_ccld(
                env_dv="real" if settings.kis_env == "prod" else "demo",
                pd_dv="inner",  # 3개월 이내
                cano=self._trenv.my_acct,
                acnt_prdt_cd=self._trenv.my_prod,
                inqr_strt_dt=today,
                inqr_end_dt=today,
                sll_buy_dvsn_cd="00",  # 전체 (매수/매도)
                inqr_dvsn="00",  # 역순
                pdno="",  # 전체 종목
                ccld_dvsn="01",  # 체결만
                inqr_dvsn_3="00"  # 전체
            )
            return df1, df2

        try:
            df1, _ = await loop.run_in_executor(None, _fetch_orders)
            return self._convert_orders(df1)
        except Exception as e:
            logger.error(f"체결 내역 조회 실패: {e}")
            return OrdersList(orders=[], total_count=0)

    def _convert_orders(self, df: pd.DataFrame) -> OrdersList:
        """DataFrame을 OrdersList 모델로 변환."""
        orders = []
        buy_count = 0
        sell_count = 0
        buy_amount = 0
        sell_amount = 0

        if df is not None and not df.empty:
            for _, row in df.iterrows():
                try:
                    side_code = str(row.get("sll_buy_dvsn_cd", ""))
                    side = "매수" if side_code == "02" else "매도"
                    exec_qty = int(row.get("tot_ccld_qty", 0))
                    exec_price = int(float(row.get("avg_prvs", 0)))
                    exec_amount = exec_qty * exec_price

                    order = Order(
                        order_id=str(row.get("odno", "")),
                        order_time=str(row.get("ord_tmd", "")),
                        stock_code=str(row.get("pdno", "")),
                        stock_name=str(row.get("prdt_name", "")),
                        side=side,
                        order_qty=int(row.get("ord_qty", 0)),
                        exec_qty=exec_qty,
                        order_price=int(float(row.get("ord_unpr", 0))),
                        exec_price=exec_price,
                        exec_amount=exec_amount,
                        status="체결"
                    )
                    orders.append(order)

                    if side == "매수":
                        buy_count += 1
                        buy_amount += exec_amount
                    else:
                        sell_count += 1
                        sell_amount += exec_amount
                except Exception as e:
                    logger.warning(f"체결 내역 변환 오류: {e}")

        return OrdersList(
            orders=orders,
            total_count=len(orders),
            buy_count=buy_count,
            sell_count=sell_count,
            buy_amount=buy_amount,
            sell_amount=sell_amount,
            updated_at=datetime.now()
        )

    async def get_stock_price(self, stock_code: str) -> dict:
        """종목 현재가 조회."""
        await self.ensure_auth()

        loop = asyncio.get_event_loop()

        def _fetch_price():
            from domestic_stock.domestic_stock_functions import inquire_price

            df = inquire_price(
                env_dv="real" if settings.kis_env == "prod" else "demo",
                fid_cond_mrkt_div_code="J",
                fid_input_iscd=stock_code
            )
            return df

        try:
            df = await loop.run_in_executor(None, _fetch_price)
            if df is not None and not df.empty:
                row = df.iloc[0]
                return {
                    "code": stock_code,
                    "price": int(row.get("stck_prpr", 0)),
                    "change": int(row.get("prdy_vrss", 0)),
                    "change_rate": float(row.get("prdy_ctrt", 0)),
                    "volume": int(row.get("acml_vol", 0)),
                    "high": int(row.get("stck_hgpr", 0)),
                    "low": int(row.get("stck_lwpr", 0)),
                    "open": int(row.get("stck_oprc", 0)),
                }
            return {}
        except Exception as e:
            logger.error(f"현재가 조회 실패: {e}")
            return {}


# 싱글톤 인스턴스
kis_service = KISService()
