# -*- coding: utf-8 -*-
"""
포지션 모니터링 서비스

보유 종목의 손절/익절 자동 실행을 담당합니다.
"""

import sys
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class StopLossTakeProfitConfig:
    """손절/익절 설정"""
    stop_loss_pct: float = 0.5      # 손절률 (%)
    take_profit_pct: float = 1.5    # 익절률 (%)
    check_interval: int = 10        # 체크 주기 (초)
    enabled: bool = True            # 활성화 여부
    use_market_order: bool = True   # 시장가 주문 사용


@dataclass
class PositionAlert:
    """포지션 알림"""
    stock_code: str
    stock_name: str
    alert_type: str  # "stop_loss", "take_profit", "warning"
    avg_price: float
    current_price: int
    quantity: int
    pnl: int
    pnl_rate: float
    threshold: float
    action_taken: str  # "sold", "pending", "skipped"
    order_no: Optional[str] = None
    timestamp: str = ""


class PositionMonitorService:
    """
    포지션 모니터링 서비스

    보유 종목을 주기적으로 모니터링하여 손절/익절 조건 충족 시
    자동으로 매도 주문을 실행합니다.
    """

    def __init__(self):
        self._config = StopLossTakeProfitConfig()
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._order_executor = None
        self._ka = None
        self._trenv = None
        self._lock = asyncio.Lock()

        # 알림 히스토리
        self._alert_history: List[PositionAlert] = []

        # 콜백 함수들
        self._on_alert: Optional[Callable[[PositionAlert], Any]] = None
        self._on_sell_executed: Optional[Callable[[Dict], Any]] = None
        self._on_position_update: Optional[Callable[[List[Dict]], Any]] = None

        # 마지막으로 체크한 포지션들 (중복 알림 방지)
        self._last_alerted: Dict[str, str] = {}  # stock_code -> alert_type

    def set_callbacks(
        self,
        on_alert: Optional[Callable[[PositionAlert], Any]] = None,
        on_sell_executed: Optional[Callable[[Dict], Any]] = None,
        on_position_update: Optional[Callable[[List[Dict]], Any]] = None
    ):
        """콜백 함수 설정"""
        self._on_alert = on_alert
        self._on_sell_executed = on_sell_executed
        self._on_position_update = on_position_update

    async def _ensure_auth(self) -> bool:
        """인증 확인"""
        if self._ka is not None:
            return True

        try:
            loop = asyncio.get_event_loop()

            def _auth():
                import kis_auth as ka
                ka.auth(svr=settings.kis_env)
                return ka, ka.getTREnv()

            self._ka, self._trenv = await loop.run_in_executor(None, _auth)
            logger.info("포지션 모니터 인증 성공")
            return True
        except Exception as e:
            logger.error(f"포지션 모니터 인증 실패: {e}")
            return False

    async def _ensure_order_executor(self) -> bool:
        """주문 실행기 초기화"""
        if self._order_executor is not None:
            return True

        if not await self._ensure_auth():
            return False

        try:
            from modules.order_executor import OrderExecutor

            self._order_executor = OrderExecutor(
                env_dv="real" if settings.kis_env == "prod" else "demo",
                cano=self._trenv.my_acct,
                acnt_prdt_cd=self._trenv.my_prod,
                default_ord_dvsn="01" if self._config.use_market_order else "00"
            )
            logger.info("주문 실행기 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"주문 실행기 초기화 실패: {e}")
            return False

    def update_config(self, config: Dict[str, Any]):
        """설정 업데이트"""
        if "stop_loss_pct" in config:
            self._config.stop_loss_pct = config["stop_loss_pct"]
        if "take_profit_pct" in config:
            self._config.take_profit_pct = config["take_profit_pct"]
        if "check_interval" in config:
            self._config.check_interval = config["check_interval"]
        if "enabled" in config:
            self._config.enabled = config["enabled"]
        if "use_market_order" in config:
            self._config.use_market_order = config["use_market_order"]

        logger.info(f"포지션 모니터 설정 업데이트: 손절={self._config.stop_loss_pct}%, "
                   f"익절={self._config.take_profit_pct}%")

    def get_config(self) -> Dict[str, Any]:
        """현재 설정 조회"""
        return asdict(self._config)

    async def start(self) -> bool:
        """모니터링 시작"""
        if self._is_running:
            logger.warning("포지션 모니터가 이미 실행 중입니다")
            return False

        if not await self._ensure_order_executor():
            logger.error("포지션 모니터 시작 실패: 주문 실행기 초기화 실패")
            return False

        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"포지션 모니터 시작 (주기: {self._config.check_interval}초)")
        return True

    async def stop(self):
        """모니터링 중지"""
        self._is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("포지션 모니터 중지")

    def is_running(self) -> bool:
        """실행 상태 확인"""
        return self._is_running

    async def _get_holdings(self) -> List[Dict]:
        """보유 종목 조회"""
        if not await self._ensure_auth():
            return []

        loop = asyncio.get_event_loop()

        def _fetch_balance():
            from domestic_stock.domestic_stock_functions import inquire_balance

            df1, df2 = inquire_balance(
                env_dv="real" if settings.kis_env == "prod" else "demo",
                cano=self._trenv.my_acct,
                acnt_prdt_cd=self._trenv.my_prod,
                afhr_flpr_yn="N",
                inqr_dvsn="02",
                unpr_dvsn="01",
                fund_sttl_icld_yn="N",
                fncg_amt_auto_rdpt_yn="N",
                prcs_dvsn="00"
            )
            return df1

        try:
            df = await loop.run_in_executor(None, _fetch_balance)

            if df is None or df.empty:
                return []

            holdings = []
            for _, row in df.iterrows():
                qty = int(row.get("hldg_qty", 0))
                if qty <= 0:
                    continue

                holding = {
                    "code": str(row.get("pdno", "")),
                    "name": str(row.get("prdt_name", "")),
                    "quantity": qty,
                    "avg_price": float(row.get("pchs_avg_pric", 0)),
                    "current_price": int(row.get("prpr", 0)),
                    "eval_amount": int(row.get("evlu_amt", 0)),
                    "pnl": int(row.get("evlu_pfls_amt", 0)),
                    "pnl_rate": float(row.get("evlu_pfls_rt", 0))
                }
                holdings.append(holding)

            return holdings

        except Exception as e:
            logger.error(f"보유 종목 조회 실패: {e}")
            return []

    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self._is_running:
            try:
                if self._config.enabled:
                    await self._check_positions()

                await asyncio.sleep(self._config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(5)

    async def _check_positions(self):
        """포지션 체크 및 손절/익절 실행"""
        holdings = await self._get_holdings()

        if not holdings:
            return

        # 포지션 업데이트 콜백
        if self._on_position_update:
            try:
                await self._on_position_update(holdings)
            except Exception as e:
                logger.error(f"포지션 업데이트 콜백 오류: {e}")

        for holding in holdings:
            await self._check_single_position(holding)

    async def _check_single_position(self, holding: Dict):
        """단일 포지션 체크"""
        stock_code = holding["code"]
        stock_name = holding["name"]
        avg_price = holding["avg_price"]
        current_price = holding["current_price"]
        quantity = holding["quantity"]
        pnl = holding["pnl"]
        pnl_rate = holding["pnl_rate"]

        if avg_price <= 0 or current_price <= 0:
            return

        # 손익률 계산 (API 제공 값 또는 직접 계산)
        if pnl_rate == 0 and avg_price > 0:
            pnl_rate = ((current_price - avg_price) / avg_price) * 100

        timestamp = datetime.now().isoformat()

        # 손절 체크 (음수이므로 절대값 비교)
        if pnl_rate <= -self._config.stop_loss_pct:
            alert_key = f"{stock_code}_stop_loss"

            # 이미 알림을 보냈고 매도 주문을 냈는지 확인
            if self._last_alerted.get(stock_code) == "stop_loss":
                return

            logger.warning(f"[손절] {stock_name}({stock_code}): "
                          f"손익률 {pnl_rate:.2f}% <= -{self._config.stop_loss_pct}%")

            # 매도 실행
            alert = await self._execute_sell(
                holding=holding,
                alert_type="stop_loss",
                threshold=self._config.stop_loss_pct,
                timestamp=timestamp
            )

            self._last_alerted[stock_code] = "stop_loss"
            self._alert_history.append(alert)

            # 알림 콜백
            if self._on_alert:
                try:
                    await self._on_alert(alert)
                except Exception as e:
                    logger.error(f"알림 콜백 오류: {e}")

        # 익절 체크
        elif pnl_rate >= self._config.take_profit_pct:
            alert_key = f"{stock_code}_take_profit"

            # 이미 알림을 보냈고 매도 주문을 냈는지 확인
            if self._last_alerted.get(stock_code) == "take_profit":
                return

            logger.info(f"[익절] {stock_name}({stock_code}): "
                       f"손익률 {pnl_rate:.2f}% >= {self._config.take_profit_pct}%")

            # 매도 실행
            alert = await self._execute_sell(
                holding=holding,
                alert_type="take_profit",
                threshold=self._config.take_profit_pct,
                timestamp=timestamp
            )

            self._last_alerted[stock_code] = "take_profit"
            self._alert_history.append(alert)

            # 알림 콜백
            if self._on_alert:
                try:
                    await self._on_alert(alert)
                except Exception as e:
                    logger.error(f"알림 콜백 오류: {e}")

        else:
            # 포지션이 정상 범위면 알림 상태 초기화
            if stock_code in self._last_alerted:
                del self._last_alerted[stock_code]

    async def _execute_sell(
        self,
        holding: Dict,
        alert_type: str,
        threshold: float,
        timestamp: str
    ) -> PositionAlert:
        """매도 주문 실행"""
        stock_code = holding["code"]
        stock_name = holding["name"]
        quantity = holding["quantity"]
        current_price = holding["current_price"]

        alert = PositionAlert(
            stock_code=stock_code,
            stock_name=stock_name,
            alert_type=alert_type,
            avg_price=holding["avg_price"],
            current_price=current_price,
            quantity=quantity,
            pnl=holding["pnl"],
            pnl_rate=holding["pnl_rate"],
            threshold=threshold,
            action_taken="pending",
            timestamp=timestamp
        )

        try:
            loop = asyncio.get_event_loop()

            def _sell():
                return self._order_executor.execute_order(
                    stock_code=stock_code,
                    order_type="sell",
                    order_qty=quantity,
                    order_price=0 if self._config.use_market_order else current_price,
                    ord_dvsn="01" if self._config.use_market_order else "00"
                )

            result = await loop.run_in_executor(None, _sell)

            if result.success:
                alert.action_taken = "sold"
                alert.order_no = result.order_no
                logger.info(f"[{alert_type.upper()}] 매도 완료: {stock_name} {quantity}주, "
                           f"주문번호={result.order_no}")

                # 매도 실행 콜백
                if self._on_sell_executed:
                    try:
                        await self._on_sell_executed({
                            "stock_code": stock_code,
                            "stock_name": stock_name,
                            "alert_type": alert_type,
                            "quantity": quantity,
                            "price": current_price,
                            "order_no": result.order_no,
                            "pnl": holding["pnl"],
                            "pnl_rate": holding["pnl_rate"],
                            "timestamp": timestamp
                        })
                    except Exception as e:
                        logger.error(f"매도 콜백 오류: {e}")
            else:
                alert.action_taken = "failed"
                logger.error(f"[{alert_type.upper()}] 매도 실패: {stock_name}, {result.message}")

        except Exception as e:
            alert.action_taken = "error"
            logger.error(f"[{alert_type.upper()}] 매도 오류: {stock_name}, {e}")

        return alert

    async def manual_check(self) -> List[Dict]:
        """수동으로 포지션 체크 (테스트용)"""
        holdings = await self._get_holdings()

        results = []
        for holding in holdings:
            pnl_rate = holding["pnl_rate"]

            status = "normal"
            if pnl_rate <= -self._config.stop_loss_pct:
                status = "stop_loss_trigger"
            elif pnl_rate >= self._config.take_profit_pct:
                status = "take_profit_trigger"
            elif pnl_rate <= -self._config.stop_loss_pct * 0.8:
                status = "stop_loss_warning"
            elif pnl_rate >= self._config.take_profit_pct * 0.8:
                status = "take_profit_warning"

            results.append({
                **holding,
                "status": status,
                "stop_loss_threshold": -self._config.stop_loss_pct,
                "take_profit_threshold": self._config.take_profit_pct
            })

        return results

    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        """알림 히스토리 조회"""
        return [asdict(alert) for alert in self._alert_history[-limit:]]

    def clear_alert_state(self, stock_code: str = None):
        """알림 상태 초기화"""
        if stock_code:
            if stock_code in self._last_alerted:
                del self._last_alerted[stock_code]
        else:
            self._last_alerted.clear()


# 싱글톤 인스턴스
_position_monitor: Optional[PositionMonitorService] = None


def get_position_monitor() -> PositionMonitorService:
    """전역 PositionMonitorService 인스턴스"""
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitorService()
    return _position_monitor
