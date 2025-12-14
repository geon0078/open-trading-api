# -*- coding: utf-8 -*-
"""
리스크 관리 모듈

손절/익절 관리, 포지션 크기 제한, 일일 거래 한도 등
리스크 관리 기능을 제공합니다.

사용 예시:
    >>> rm = RiskManager(stop_loss_pct=3.0, take_profit_pct=5.0)
    >>> rm.add_position("005930", 100, 55000)
    >>> signal = rm.check_position("005930", 53000)  # 손절 체크
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RiskSignal(Enum):
    """리스크 시그널 타입"""
    HOLD = "hold"              # 유지
    STOP_LOSS = "stop_loss"    # 손절
    TAKE_PROFIT = "take_profit"  # 익절
    TRAILING_STOP = "trailing_stop"  # 트레일링 스탑
    TIME_STOP = "time_stop"    # 시간 기반 청산
    MAX_LOSS_DAILY = "max_loss_daily"  # 일일 최대 손실


@dataclass
class Position:
    """포지션 정보"""
    stock_code: str           # 종목코드
    quantity: int             # 수량
    entry_price: float        # 진입가
    entry_time: datetime      # 진입시간
    highest_price: float = 0  # 최고가 (트레일링 스탑용)
    lowest_price: float = 0   # 최저가
    current_price: float = 0  # 현재가
    stop_loss_price: float = 0  # 손절가
    take_profit_price: float = 0  # 익절가
    trailing_stop_pct: float = 0  # 트레일링 스탑 %
    note: str = ""            # 메모

    @property
    def pnl(self) -> float:
        """손익금액"""
        if self.current_price <= 0:
            return 0
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        """손익률"""
        if self.entry_price <= 0:
            return 0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    @property
    def market_value(self) -> float:
        """현재 평가금액"""
        return self.current_price * self.quantity

    @property
    def entry_value(self) -> float:
        """진입 금액"""
        return self.entry_price * self.quantity


@dataclass
class RiskCheckResult:
    """리스크 체크 결과"""
    signal: RiskSignal        # 시그널
    stock_code: str           # 종목코드
    current_price: float      # 현재가
    trigger_price: float      # 트리거 가격
    pnl_pct: float           # 손익률
    message: str             # 메시지
    timestamp: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    리스크 관리자

    손절/익절, 포지션 관리, 일일 거래 한도 등을 관리합니다.

    Attributes:
        stop_loss_pct: 기본 손절률 (%)
        take_profit_pct: 기본 익절률 (%)
        trailing_stop_pct: 트레일링 스탑률 (%)
        max_position_value: 단일 포지션 최대 금액
        max_daily_loss: 일일 최대 손실 금액
        max_holding_hours: 최대 보유 시간

    Example:
        >>> rm = RiskManager(stop_loss_pct=3.0, take_profit_pct=5.0)
        >>> rm.add_position("005930", 100, 55000)
        >>> result = rm.check_position("005930", 53000)
        >>> if result.signal == RiskSignal.STOP_LOSS:
        ...     print("손절 신호 발생!")
    """

    def __init__(
        self,
        stop_loss_pct: float = 3.0,
        take_profit_pct: float = 5.0,
        trailing_stop_pct: float = 0.0,
        max_position_value: float = 10_000_000,
        max_daily_loss: float = 500_000,
        max_holding_hours: int = 24,
    ):
        """
        Args:
            stop_loss_pct: 기본 손절률 (%, 기본값: 3%)
            take_profit_pct: 기본 익절률 (%, 기본값: 5%)
            trailing_stop_pct: 트레일링 스탑률 (%, 0이면 비활성)
            max_position_value: 단일 포지션 최대 금액 (기본값: 1000만원)
            max_daily_loss: 일일 최대 손실 금액 (기본값: 50만원)
            max_holding_hours: 최대 보유 시간 (시간, 기본값: 24시간)
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_position_value = max_position_value
        self.max_daily_loss = max_daily_loss
        self.max_holding_hours = max_holding_hours

        # 포지션 저장소
        self.positions: Dict[str, Position] = {}

        # 일일 실현손익 추적
        self.daily_pnl: float = 0.0
        self.daily_pnl_date: datetime = datetime.now().date()

        # 거래 히스토리
        self.trade_history: List[Dict] = []

    def add_position(
        self,
        stock_code: str,
        quantity: int,
        entry_price: float,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        note: str = ""
    ) -> Tuple[bool, str]:
        """
        포지션 추가

        Args:
            stock_code: 종목코드
            quantity: 수량
            entry_price: 진입가
            stop_loss_pct: 손절률 (None이면 기본값 사용)
            take_profit_pct: 익절률 (None이면 기본값 사용)
            trailing_stop_pct: 트레일링 스탑률 (None이면 기본값 사용)
            note: 메모

        Returns:
            Tuple[bool, str]: (성공여부, 메시지)

        Example:
            >>> success, msg = rm.add_position("005930", 100, 55000)
        """
        # 포지션 금액 체크
        position_value = entry_price * quantity
        if position_value > self.max_position_value:
            return False, f"포지션 금액 초과: {position_value:,.0f}원 > {self.max_position_value:,.0f}원"

        # 손절/익절 가격 계산
        sl_pct = stop_loss_pct if stop_loss_pct is not None else self.stop_loss_pct
        tp_pct = take_profit_pct if take_profit_pct is not None else self.take_profit_pct
        ts_pct = trailing_stop_pct if trailing_stop_pct is not None else self.trailing_stop_pct

        stop_loss_price = entry_price * (1 - sl_pct / 100)
        take_profit_price = entry_price * (1 + tp_pct / 100)

        # 기존 포지션 업데이트 또는 신규 추가
        if stock_code in self.positions:
            pos = self.positions[stock_code]
            # 평균 단가 계산
            total_qty = pos.quantity + quantity
            avg_price = (pos.entry_price * pos.quantity + entry_price * quantity) / total_qty
            pos.quantity = total_qty
            pos.entry_price = avg_price
            pos.stop_loss_price = avg_price * (1 - sl_pct / 100)
            pos.take_profit_price = avg_price * (1 + tp_pct / 100)
            pos.note = note
            logger.info(f"[{stock_code}] 포지션 추가: {quantity}주 @ {entry_price:,.0f}원 (총 {total_qty}주, 평균가 {avg_price:,.0f}원)")
        else:
            self.positions[stock_code] = Position(
                stock_code=stock_code,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=datetime.now(),
                highest_price=entry_price,
                lowest_price=entry_price,
                current_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                trailing_stop_pct=ts_pct,
                note=note
            )
            logger.info(f"[{stock_code}] 신규 포지션: {quantity}주 @ {entry_price:,.0f}원 (손절: {stop_loss_price:,.0f}원, 익절: {take_profit_price:,.0f}원)")

        return True, "포지션 추가 완료"

    def check_position(
        self,
        stock_code: str,
        current_price: float
    ) -> Optional[RiskCheckResult]:
        """
        포지션 리스크 체크

        Args:
            stock_code: 종목코드
            current_price: 현재가

        Returns:
            RiskCheckResult: 리스크 체크 결과

        Example:
            >>> result = rm.check_position("005930", 53000)
            >>> if result.signal == RiskSignal.STOP_LOSS:
            ...     execute_sell_order(stock_code, quantity)
        """
        if stock_code not in self.positions:
            return None

        pos = self.positions[stock_code]
        pos.current_price = current_price

        # 최고가/최저가 업데이트
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        if current_price < pos.lowest_price or pos.lowest_price == 0:
            pos.lowest_price = current_price

        # 손익률 계산
        pnl_pct = pos.pnl_pct

        # 1. 손절 체크
        if current_price <= pos.stop_loss_price:
            return RiskCheckResult(
                signal=RiskSignal.STOP_LOSS,
                stock_code=stock_code,
                current_price=current_price,
                trigger_price=pos.stop_loss_price,
                pnl_pct=pnl_pct,
                message=f"손절 트리거: {current_price:,.0f}원 <= {pos.stop_loss_price:,.0f}원"
            )

        # 2. 익절 체크
        if current_price >= pos.take_profit_price:
            return RiskCheckResult(
                signal=RiskSignal.TAKE_PROFIT,
                stock_code=stock_code,
                current_price=current_price,
                trigger_price=pos.take_profit_price,
                pnl_pct=pnl_pct,
                message=f"익절 트리거: {current_price:,.0f}원 >= {pos.take_profit_price:,.0f}원"
            )

        # 3. 트레일링 스탑 체크
        if pos.trailing_stop_pct > 0:
            trailing_stop_price = pos.highest_price * (1 - pos.trailing_stop_pct / 100)
            if current_price <= trailing_stop_price:
                return RiskCheckResult(
                    signal=RiskSignal.TRAILING_STOP,
                    stock_code=stock_code,
                    current_price=current_price,
                    trigger_price=trailing_stop_price,
                    pnl_pct=pnl_pct,
                    message=f"트레일링 스탑: {current_price:,.0f}원 <= {trailing_stop_price:,.0f}원 (최고가 {pos.highest_price:,.0f}원에서 {pos.trailing_stop_pct}% 하락)"
                )

        # 4. 시간 기반 청산 체크
        holding_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
        if holding_hours >= self.max_holding_hours:
            return RiskCheckResult(
                signal=RiskSignal.TIME_STOP,
                stock_code=stock_code,
                current_price=current_price,
                trigger_price=current_price,
                pnl_pct=pnl_pct,
                message=f"시간 기반 청산: {holding_hours:.1f}시간 >= {self.max_holding_hours}시간"
            )

        # 모든 체크 통과 -> 유지
        return RiskCheckResult(
            signal=RiskSignal.HOLD,
            stock_code=stock_code,
            current_price=current_price,
            trigger_price=0,
            pnl_pct=pnl_pct,
            message=f"유지: 손익률 {pnl_pct:+.2f}%"
        )

    def check_all_positions(
        self,
        prices: Dict[str, float]
    ) -> List[RiskCheckResult]:
        """
        모든 포지션 리스크 체크

        Args:
            prices: 종목별 현재가 딕셔너리

        Returns:
            List[RiskCheckResult]: 리스크 체크 결과 리스트
        """
        results = []
        for stock_code in self.positions:
            if stock_code in prices:
                result = self.check_position(stock_code, prices[stock_code])
                if result:
                    results.append(result)
        return results

    def close_position(
        self,
        stock_code: str,
        close_price: float,
        quantity: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        포지션 청산

        Args:
            stock_code: 종목코드
            close_price: 청산가
            quantity: 청산 수량 (None이면 전체)

        Returns:
            Tuple[bool, float]: (성공여부, 실현손익)
        """
        if stock_code not in self.positions:
            return False, 0

        pos = self.positions[stock_code]
        close_qty = quantity if quantity else pos.quantity

        # 실현손익 계산
        realized_pnl = (close_price - pos.entry_price) * close_qty

        # 일일 손익 업데이트
        self._update_daily_pnl(realized_pnl)

        # 거래 히스토리 추가
        self.trade_history.append({
            "stock_code": stock_code,
            "action": "close",
            "quantity": close_qty,
            "entry_price": pos.entry_price,
            "close_price": close_price,
            "pnl": realized_pnl,
            "pnl_pct": ((close_price - pos.entry_price) / pos.entry_price) * 100,
            "timestamp": datetime.now(),
        })

        # 포지션 업데이트 또는 삭제
        if close_qty >= pos.quantity:
            del self.positions[stock_code]
            logger.info(f"[{stock_code}] 포지션 전체 청산: {close_qty}주 @ {close_price:,.0f}원 (손익: {realized_pnl:+,.0f}원)")
        else:
            pos.quantity -= close_qty
            logger.info(f"[{stock_code}] 부분 청산: {close_qty}주 @ {close_price:,.0f}원 (잔여: {pos.quantity}주)")

        return True, realized_pnl

    def _update_daily_pnl(self, pnl: float):
        """일일 손익 업데이트"""
        today = datetime.now().date()
        if today != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today
        self.daily_pnl += pnl

    def can_trade(self) -> Tuple[bool, str]:
        """
        거래 가능 여부 확인

        Returns:
            Tuple[bool, str]: (거래가능여부, 사유)
        """
        # 일일 최대 손실 체크
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"일일 최대 손실 도달: {self.daily_pnl:,.0f}원 <= -{self.max_daily_loss:,.0f}원"
        return True, "거래 가능"

    def get_position(self, stock_code: str) -> Optional[Position]:
        """포지션 조회"""
        return self.positions.get(stock_code)

    def get_all_positions(self) -> Dict[str, Position]:
        """모든 포지션 조회"""
        return self.positions.copy()

    def get_total_exposure(self) -> float:
        """총 포지션 금액"""
        return sum(pos.market_value for pos in self.positions.values())

    def get_total_pnl(self) -> float:
        """총 미실현손익"""
        return sum(pos.pnl for pos in self.positions.values())

    def get_summary(self) -> Dict:
        """리스크 관리 요약"""
        total_exposure = self.get_total_exposure()
        total_pnl = self.get_total_pnl()

        return {
            "position_count": len(self.positions),
            "total_exposure": total_exposure,
            "total_unrealized_pnl": total_pnl,
            "daily_realized_pnl": self.daily_pnl,
            "settings": {
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "trailing_stop_pct": self.trailing_stop_pct,
                "max_position_value": self.max_position_value,
                "max_daily_loss": self.max_daily_loss,
                "max_holding_hours": self.max_holding_hours,
            }
        }

    def adjust_stop_loss(
        self,
        stock_code: str,
        new_stop_price: float
    ) -> bool:
        """손절가 조정 (보호적 스탑)"""
        if stock_code not in self.positions:
            return False

        pos = self.positions[stock_code]
        # 새 손절가는 기존보다 높아야 함 (매수 포지션 기준)
        if new_stop_price > pos.stop_loss_price:
            pos.stop_loss_price = new_stop_price
            logger.info(f"[{stock_code}] 손절가 상향 조정: {new_stop_price:,.0f}원")
            return True
        return False


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=== 리스크 관리 모듈 테스트 ===\n")

    # 리스크 관리자 생성
    rm = RiskManager(
        stop_loss_pct=3.0,      # 3% 손절
        take_profit_pct=5.0,   # 5% 익절
        trailing_stop_pct=2.0, # 2% 트레일링 스탑
        max_position_value=10_000_000,  # 1000만원 한도
        max_daily_loss=500_000,  # 일일 50만원 손실 한도
    )

    # 포지션 추가
    print("[1] 포지션 추가 테스트")
    rm.add_position("005930", 100, 55000, note="삼성전자 매수")
    rm.add_position("000660", 50, 180000, note="SK하이닉스 매수")

    # 현재가 기준 리스크 체크
    print("\n[2] 리스크 체크 테스트")

    # 삼성전자: 수익 상태
    result = rm.check_position("005930", 56000)
    print(f"삼성전자 (56,000원): {result.signal.value} - {result.message}")

    # 삼성전자: 손절 트리거
    result = rm.check_position("005930", 53000)
    print(f"삼성전자 (53,000원): {result.signal.value} - {result.message}")

    # SK하이닉스: 익절 트리거
    result = rm.check_position("000660", 190000)
    print(f"SK하이닉스 (190,000원): {result.signal.value} - {result.message}")

    # 요약 출력
    print("\n[3] 리스크 관리 요약")
    summary = rm.get_summary()
    print(f"포지션 수: {summary['position_count']}")
    print(f"총 포지션 금액: {summary['total_exposure']:,.0f}원")
    print(f"미실현 손익: {summary['total_unrealized_pnl']:+,.0f}원")

    # 포지션 청산
    print("\n[4] 포지션 청산 테스트")
    success, pnl = rm.close_position("005930", 53000)
    print(f"삼성전자 청산 - 실현손익: {pnl:+,.0f}원")

    print("\n=== 테스트 완료 ===")
