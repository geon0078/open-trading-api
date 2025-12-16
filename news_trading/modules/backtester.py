# -*- coding: utf-8 -*-
"""
백테스팅 모듈

과거 데이터를 기반으로 뉴스 기반 매매 전략의 성과를 시뮬레이션합니다.

사용 예시:
    >>> bt = Backtester(initial_capital=10_000_000)
    >>> bt.load_price_data("005930", start_date="2024-01-01")
    >>> results = bt.run(strategy=my_strategy)
    >>> print(results.summary())
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import sys

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """거래 행위"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Trade:
    """거래 기록"""
    timestamp: datetime
    stock_code: str
    action: TradeAction
    price: float
    quantity: int
    value: float
    commission: float = 0
    note: str = ""


@dataclass
class BacktestPosition:
    """백테스트 포지션"""
    stock_code: str
    quantity: int
    avg_price: float
    current_price: float = 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class BacktestResult:
    """백테스트 결과"""
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    daily_returns: pd.Series

    def summary(self) -> str:
        """결과 요약 문자열"""
        return f"""
========================================
백테스트 결과 요약
========================================
초기 자본:        {self.initial_capital:,.0f}원
최종 평가액:      {self.final_value:,.0f}원
총 수익:          {self.total_return:+,.0f}원 ({self.total_return_pct:+.2f}%)

총 거래 수:       {self.total_trades}회
승리 거래:        {self.winning_trades}회
패배 거래:        {self.losing_trades}회
승률:             {self.win_rate:.1f}%

평균 수익:        {self.avg_win:+,.0f}원
평균 손실:        {self.avg_loss:+,.0f}원
손익비:           {self.profit_factor:.2f}

최대 낙폭:        {self.max_drawdown:,.0f}원 ({self.max_drawdown_pct:.2f}%)
샤프 비율:        {self.sharpe_ratio:.2f}
========================================
"""


class Backtester:
    """
    백테스팅 엔진

    과거 가격 데이터를 기반으로 매매 전략을 시뮬레이션합니다.

    Attributes:
        initial_capital: 초기 자본금
        commission_rate: 수수료율 (%)
        slippage: 슬리피지 (%)

    Example:
        >>> bt = Backtester(initial_capital=10_000_000)
        >>> bt.load_price_data("005930", start_date="2024-01-01")
        >>> results = bt.run(strategy=my_strategy)
        >>> print(results.summary())
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        commission_rate: float = 0.015,  # 0.015% (증권사 수수료)
        slippage: float = 0.05,  # 0.05% 슬리피지
    ):
        """
        Args:
            initial_capital: 초기 자본금 (기본: 1000만원)
            commission_rate: 수수료율 (%, 기본: 0.015%)
            slippage: 슬리피지 (%, 기본: 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate / 100
        self.slippage = slippage / 100

        # 가격 데이터
        self.price_data: Dict[str, pd.DataFrame] = {}

        # 시뮬레이션 상태
        self.cash = initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Dict] = []

    def load_price_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        가격 데이터 로드

        Args:
            stock_code: 종목코드
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            data: 직접 제공할 DataFrame (없으면 API에서 조회)

        Returns:
            bool: 성공 여부

        DataFrame 컬럼:
            - date: 날짜 (datetime)
            - open: 시가
            - high: 고가
            - low: 저가
            - close: 종가
            - volume: 거래량
        """
        if data is not None:
            self.price_data[stock_code] = data
            logger.info(f"[{stock_code}] 가격 데이터 로드: {len(data)}일")
            return True

        # KIS API에서 일봉 데이터 조회
        try:
            sys.path.extend(['../..', '.', '../../..'])

            from domestic_stock.inquire_daily_itemchartprice.inquire_daily_itemchartprice import (
                inquire_daily_itemchartprice
            )

            # 날짜 설정
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            else:
                end_date = end_date.replace("-", "")

            if start_date is None:
                start_dt = datetime.now() - timedelta(days=365)
                start_date = start_dt.strftime("%Y%m%d")
            else:
                start_date = start_date.replace("-", "")

            # API 호출
            df = inquire_daily_itemchartprice(
                env_dv="real",
                fid_cond_mrkt_div_code="J",
                fid_input_iscd=stock_code,
                fid_input_date_1=start_date,
                fid_input_date_2=end_date,
                fid_period_div_code="D",
                fid_org_adj_prc="0"
            )

            if df is None or df.empty:
                logger.warning(f"[{stock_code}] 가격 데이터 없음")
                return False

            # 컬럼 변환
            df = df.rename(columns={
                "stck_bsop_date": "date",
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_clpr": "close",
                "acml_vol": "volume"
            })

            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("date").reset_index(drop=True)
            self.price_data[stock_code] = df

            logger.info(f"[{stock_code}] 가격 데이터 로드: {len(df)}일 ({df['date'].min()} ~ {df['date'].max()})")
            return True

        except ImportError as e:
            logger.error(f"API 모듈 로드 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"[{stock_code}] 가격 데이터 로드 실패: {e}")
            return False

    def _reset(self):
        """시뮬레이션 상태 초기화"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []

    def _get_price_with_slippage(self, price: float, is_buy: bool) -> float:
        """슬리피지 적용 가격"""
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def _calculate_commission(self, value: float) -> float:
        """수수료 계산"""
        return value * self.commission_rate

    def _execute_buy(
        self,
        timestamp: datetime,
        stock_code: str,
        price: float,
        quantity: int,
        note: str = ""
    ) -> bool:
        """매수 실행"""
        exec_price = self._get_price_with_slippage(price, is_buy=True)
        value = exec_price * quantity
        commission = self._calculate_commission(value)
        total_cost = value + commission

        if total_cost > self.cash:
            logger.warning(f"[{stock_code}] 매수 실패: 자금 부족 ({total_cost:,.0f} > {self.cash:,.0f})")
            return False

        # 포지션 업데이트
        if stock_code in self.positions:
            pos = self.positions[stock_code]
            new_qty = pos.quantity + quantity
            new_avg = (pos.avg_price * pos.quantity + exec_price * quantity) / new_qty
            pos.quantity = new_qty
            pos.avg_price = new_avg
        else:
            self.positions[stock_code] = BacktestPosition(
                stock_code=stock_code,
                quantity=quantity,
                avg_price=exec_price,
                current_price=exec_price
            )

        self.cash -= total_cost

        # 거래 기록
        self.trades.append(Trade(
            timestamp=timestamp,
            stock_code=stock_code,
            action=TradeAction.BUY,
            price=exec_price,
            quantity=quantity,
            value=value,
            commission=commission,
            note=note
        ))

        return True

    def _execute_sell(
        self,
        timestamp: datetime,
        stock_code: str,
        price: float,
        quantity: int,
        note: str = ""
    ) -> bool:
        """매도 실행"""
        if stock_code not in self.positions:
            logger.warning(f"[{stock_code}] 매도 실패: 포지션 없음")
            return False

        pos = self.positions[stock_code]
        if quantity > pos.quantity:
            quantity = pos.quantity

        exec_price = self._get_price_with_slippage(price, is_buy=False)
        value = exec_price * quantity
        commission = self._calculate_commission(value)
        net_proceeds = value - commission

        # 포지션 업데이트
        pos.quantity -= quantity
        if pos.quantity <= 0:
            del self.positions[stock_code]

        self.cash += net_proceeds

        # 거래 기록
        self.trades.append(Trade(
            timestamp=timestamp,
            stock_code=stock_code,
            action=TradeAction.SELL,
            price=exec_price,
            quantity=quantity,
            value=value,
            commission=commission,
            note=note
        ))

        return True

    def _calculate_equity(self) -> float:
        """현재 총 자산 계산"""
        equity = self.cash
        for pos in self.positions.values():
            equity += pos.market_value
        return equity

    def _update_positions(self, prices: Dict[str, float]):
        """포지션 현재가 업데이트"""
        for stock_code, price in prices.items():
            if stock_code in self.positions:
                self.positions[stock_code].current_price = price

    def run(
        self,
        strategy: Callable[[Dict, Dict[str, BacktestPosition], float], List[Dict]],
        stock_codes: Optional[List[str]] = None
    ) -> BacktestResult:
        """
        백테스트 실행

        Args:
            strategy: 전략 함수
                - 입력: (market_data, positions, cash)
                - 출력: [{"action": "buy/sell", "stock_code": "...", "quantity": ...}]
            stock_codes: 백테스트할 종목 (None이면 로드된 전체)

        Returns:
            BacktestResult: 백테스트 결과

        Example:
            >>> def simple_strategy(data, positions, cash):
            ...     signals = []
            ...     if data["005930"]["close"] > data["005930"]["ma20"]:
            ...         signals.append({"action": "buy", "stock_code": "005930", "quantity": 10})
            ...     return signals
            >>> results = bt.run(strategy=simple_strategy)
        """
        self._reset()

        if stock_codes is None:
            stock_codes = list(self.price_data.keys())

        if not stock_codes:
            raise ValueError("가격 데이터가 없습니다. load_price_data()를 먼저 호출하세요.")

        # 날짜 범위 결정
        all_dates = set()
        for code in stock_codes:
            if code in self.price_data:
                dates = self.price_data[code]["date"].tolist()
                all_dates.update(dates)

        sorted_dates = sorted(all_dates)

        logger.info(f"백테스트 시작: {sorted_dates[0]} ~ {sorted_dates[-1]} ({len(sorted_dates)}일)")

        # 날짜별 시뮬레이션
        for current_date in sorted_dates:
            # 당일 가격 데이터 수집
            market_data = {}
            for code in stock_codes:
                if code not in self.price_data:
                    continue

                df = self.price_data[code]
                day_data = df[df["date"] == current_date]

                if day_data.empty:
                    continue

                row = day_data.iloc[0]
                market_data[code] = {
                    "date": current_date,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }

            if not market_data:
                continue

            # 포지션 현재가 업데이트
            prices = {code: data["close"] for code, data in market_data.items()}
            self._update_positions(prices)

            # 전략 실행
            try:
                signals = strategy(market_data, self.positions.copy(), self.cash)
            except Exception as e:
                logger.error(f"전략 실행 오류 ({current_date}): {e}")
                signals = []

            # 신호 처리
            for signal in signals:
                action = signal.get("action", "hold")
                stock_code = signal.get("stock_code", "")
                quantity = signal.get("quantity", 0)
                note = signal.get("note", "")

                if stock_code not in market_data:
                    continue

                price = market_data[stock_code]["close"]

                if action == "buy" and quantity > 0:
                    self._execute_buy(current_date, stock_code, price, quantity, note)
                elif action == "sell" and quantity > 0:
                    self._execute_sell(current_date, stock_code, price, quantity, note)

            # 일일 자산 기록
            equity = self._calculate_equity()
            self.equity_history.append({
                "date": current_date,
                "equity": equity,
                "cash": self.cash,
                "positions_value": equity - self.cash,
            })

        # 결과 계산
        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        """백테스트 결과 계산"""
        # 자산 곡선
        equity_df = pd.DataFrame(self.equity_history)
        if equity_df.empty:
            equity_df = pd.DataFrame([{
                "date": datetime.now(),
                "equity": self.initial_capital,
                "cash": self.initial_capital,
                "positions_value": 0
            }])

        # 기본 통계
        final_value = equity_df["equity"].iloc[-1]
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # 최대 낙폭 계산
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = equity_df["peak"] - equity_df["equity"]
        max_drawdown = equity_df["drawdown"].max()
        max_drawdown_pct = (max_drawdown / equity_df["peak"].max()) * 100 if equity_df["peak"].max() > 0 else 0

        # 일일 수익률
        equity_df["daily_return"] = equity_df["equity"].pct_change()
        daily_returns = equity_df["daily_return"].dropna()

        # 샤프 비율 (연율화)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        else:
            sharpe_ratio = 0

        # 거래 통계
        buy_trades = [t for t in self.trades if t.action == TradeAction.BUY]
        sell_trades = [t for t in self.trades if t.action == TradeAction.SELL]

        # 매수-매도 쌍으로 손익 계산
        trade_pnls = []
        for sell in sell_trades:
            # 해당 종목의 직전 매수 찾기
            matching_buys = [b for b in buy_trades
                            if b.stock_code == sell.stock_code
                            and b.timestamp < sell.timestamp]
            if matching_buys:
                buy = matching_buys[-1]
                pnl = (sell.price - buy.price) * min(sell.quantity, buy.quantity)
                trade_pnls.append(pnl)

        winning_trades = len([p for p in trade_pnls if p > 0])
        losing_trades = len([p for p in trade_pnls if p < 0])
        total_trades = len(trade_pnls)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # 손익비
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=equity_df,
            daily_returns=daily_returns
        )


# =====================================================
# 예제 전략
# =====================================================

def simple_ma_strategy(
    market_data: Dict,
    positions: Dict[str, BacktestPosition],
    cash: float,
    ma_short: int = 5,
    ma_long: int = 20
) -> List[Dict]:
    """
    단순 이동평균 교차 전략 (예제)

    Args:
        market_data: 시장 데이터
        positions: 현재 포지션
        cash: 현금
        ma_short: 단기 이평선 기간
        ma_long: 장기 이평선 기간

    Returns:
        매매 신호 리스트
    """
    signals = []

    for stock_code, data in market_data.items():
        price = data["close"]

        # 간단한 예: 종가가 고가의 90% 이상이면 매수
        if price >= data["high"] * 0.9:
            if stock_code not in positions:
                # 총 자산의 10%로 매수
                buy_value = cash * 0.1
                quantity = int(buy_value // price)
                if quantity > 0:
                    signals.append({
                        "action": "buy",
                        "stock_code": stock_code,
                        "quantity": quantity,
                        "note": "상승 돌파"
                    })

        # 종가가 저가의 102% 이하이면 매도
        elif price <= data["low"] * 1.02:
            if stock_code in positions:
                signals.append({
                    "action": "sell",
                    "stock_code": stock_code,
                    "quantity": positions[stock_code].quantity,
                    "note": "하락 손절"
                })

    return signals


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=== 백테스팅 모듈 테스트 ===\n")

    # 설정 로드
    try:
        from config_loader import setup_kis_config
        setup_kis_config()

        import kis_auth as ka
        ka.auth()
    except Exception as e:
        print(f"인증 오류: {e}")
        print("테스트용 샘플 데이터로 진행합니다.\n")

    # 백테스터 생성
    bt = Backtester(
        initial_capital=10_000_000,
        commission_rate=0.015,
        slippage=0.05
    )

    # 샘플 데이터 생성 (API 연동 불가시)
    import numpy as np

    dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="B")
    np.random.seed(42)

    # 삼성전자 가상 데이터
    base_price = 70000
    prices = [base_price]
    for _ in range(len(dates) - 1):
        change = np.random.randn() * 1000
        prices.append(max(prices[-1] + change, 50000))

    sample_data = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000000] * len(dates)
    })

    # 데이터 로드
    bt.load_price_data("005930", data=sample_data)

    # 백테스트 실행
    print("\n전략: 단순 이동평균 교차 전략")
    results = bt.run(strategy=simple_ma_strategy, stock_codes=["005930"])

    # 결과 출력
    print(results.summary())

    # 거래 내역 출력
    print("\n거래 내역:")
    for trade in results.trades[:10]:  # 처음 10개만
        print(f"  {trade.timestamp.date()} {trade.action.value} "
              f"{trade.stock_code} {trade.quantity}주 @ {trade.price:,.0f}원")

    print("\n=== 테스트 완료 ===")
