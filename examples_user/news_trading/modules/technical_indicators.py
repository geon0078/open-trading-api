# -*- coding: utf-8 -*-
"""
기술적 보조지표 계산 모듈

스캘핑에 유용한 기술적 지표들을 계산합니다:
- 이동평균선 (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 볼린저 밴드 (Bollinger Bands)
- 스토캐스틱 (Stochastic)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- 체결강도, 매수/매도 비율

사용 예시:
    >>> from technical_indicators import TechnicalAnalyzer
    >>> analyzer = TechnicalAnalyzer()
    >>> indicators = analyzer.calculate_all(df)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScalpingSignal:
    """스캘핑 신호 데이터"""
    signal: str              # BUY, SELL, HOLD
    strength: float          # 신호 강도 (0.0 ~ 1.0)
    reason: str              # 신호 발생 이유
    entry_price: float       # 진입 추천가
    stop_loss: float         # 손절가
    take_profit: float       # 익절가
    risk_reward_ratio: float # 손익비


@dataclass
class TechnicalSummary:
    """기술적 분석 요약"""
    trend: str               # BULLISH, BEARISH, NEUTRAL
    momentum: str            # STRONG, WEAK, NEUTRAL
    volatility: str          # HIGH, MEDIUM, LOW
    volume_trend: str        # INCREASING, DECREASING, STABLE
    support_level: float     # 지지선
    resistance_level: float  # 저항선
    overall_score: float     # 종합 점수 (-100 ~ +100)


class TechnicalAnalyzer:
    """
    기술적 분석기

    OHLCV 데이터를 바탕으로 다양한 기술적 지표를 계산하고
    스캘핑에 유용한 신호를 생성합니다.
    """

    def __init__(self):
        self.indicators = {}

    # ==================== 이동평균선 ====================

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """단순이동평균 (Simple Moving Average)"""
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """지수이동평균 (Exponential Moving Average)"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """가중이동평균 (Weighted Moving Average)"""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    # ==================== RSI ====================

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index)

        - RSI > 70: 과매수 (매도 신호)
        - RSI < 30: 과매도 (매수 신호)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ==================== MACD ====================

    @staticmethod
    def macd(series: pd.Series,
             fast: int = 12,
             slow: int = 26,
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)

        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    # ==================== 볼린저 밴드 ====================

    @staticmethod
    def bollinger_bands(series: pd.Series,
                        period: int = 20,
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        볼린저 밴드 (Bollinger Bands)

        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def bollinger_bandwidth(upper: pd.Series,
                           lower: pd.Series,
                           middle: pd.Series) -> pd.Series:
        """볼린저 밴드폭 (변동성 지표)"""
        return (upper - lower) / middle * 100

    @staticmethod
    def bollinger_percent_b(close: pd.Series,
                            upper: pd.Series,
                            lower: pd.Series) -> pd.Series:
        """%B (현재가의 밴드 내 위치)"""
        return (close - lower) / (upper - lower)

    # ==================== 스토캐스틱 ====================

    @staticmethod
    def stochastic(high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   k_period: int = 14,
                   d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        스토캐스틱 (Stochastic Oscillator)

        - %K > 80: 과매수
        - %K < 20: 과매도

        Returns:
            (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return k, d

    # ==================== ATR ====================

    @staticmethod
    def atr(high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        ATR (Average True Range) - 변동성 지표

        스캘핑 시 손절/익절 폭 설정에 활용
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    # ==================== VWAP ====================

    @staticmethod
    def vwap(high: pd.Series,
             low: pd.Series,
             close: pd.Series,
             volume: pd.Series) -> pd.Series:
        """
        VWAP (Volume Weighted Average Price)

        - 가격 > VWAP: 상승 추세
        - 가격 < VWAP: 하락 추세
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    # ==================== 체결강도 ====================

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """거래량 비율 (현재 거래량 / 평균 거래량)"""
        avg_vol = volume.rolling(window=period).mean()
        return volume / avg_vol

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV (On Balance Volume) - 거래량 누적"""
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        obv = (direction * volume).cumsum()
        return obv

    # ==================== 피봇 포인트 ====================

    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """
        피봇 포인트 (지지/저항 레벨)

        스캘핑 시 진입/청산 레벨로 활용
        """
        pivot = (high + low + close) / 3

        return {
            "pivot": pivot,
            "r1": 2 * pivot - low,
            "r2": pivot + (high - low),
            "r3": high + 2 * (pivot - low),
            "s1": 2 * pivot - high,
            "s2": pivot - (high - low),
            "s3": low - 2 * (high - pivot),
        }

    # ==================== 종합 분석 ====================

    def calculate_all(self, df: pd.DataFrame) -> Dict:
        """
        모든 기술적 지표 계산

        Args:
            df: OHLCV DataFrame
                필수 컬럼: open, high, low, close, volume

        Returns:
            Dict: 모든 지표 값
        """
        # 컬럼명 소문자 변환
        df = df.copy()
        df.columns = df.columns.str.lower()

        # 기본 데이터 추출
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)

        result = {}

        # 이동평균선
        result['sma_5'] = self.sma(close, 5)
        result['sma_10'] = self.sma(close, 10)
        result['sma_20'] = self.sma(close, 20)
        result['ema_5'] = self.ema(close, 5)
        result['ema_10'] = self.ema(close, 10)
        result['ema_20'] = self.ema(close, 20)

        # RSI
        result['rsi_14'] = self.rsi(close, 14)
        result['rsi_7'] = self.rsi(close, 7)  # 단기 RSI (스캘핑용)

        # MACD
        macd_line, signal_line, histogram = self.macd(close)
        result['macd'] = macd_line
        result['macd_signal'] = signal_line
        result['macd_histogram'] = histogram

        # 볼린저 밴드
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        result['bb_bandwidth'] = self.bollinger_bandwidth(bb_upper, bb_lower, bb_middle)
        result['bb_percent_b'] = self.bollinger_percent_b(close, bb_upper, bb_lower)

        # 스토캐스틱
        stoch_k, stoch_d = self.stochastic(high, low, close)
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d

        # ATR
        result['atr_14'] = self.atr(high, low, close, 14)
        result['atr_7'] = self.atr(high, low, close, 7)  # 단기 ATR

        # VWAP
        result['vwap'] = self.vwap(high, low, close, volume)

        # 거래량 지표
        result['volume_ratio'] = self.volume_ratio(volume)
        result['obv'] = self.obv(close, volume)

        self.indicators = result
        return result

    def get_latest_indicators(self, df: pd.DataFrame) -> Dict:
        """최신 지표 값만 반환 (LLM 입력용)"""
        indicators = self.calculate_all(df)

        latest = {}
        for key, value in indicators.items():
            if isinstance(value, pd.Series):
                latest[key] = round(float(value.iloc[-1]), 4) if not pd.isna(value.iloc[-1]) else None
            else:
                latest[key] = value

        return latest

    def get_scalping_summary(self, df: pd.DataFrame) -> Dict:
        """
        스캘핑용 요약 정보 (LLM 입력에 최적화)

        Returns:
            Dict: 스캘핑에 필요한 핵심 지표들
        """
        indicators = self.calculate_all(df)

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)

        current_price = float(close.iloc[-1])

        # 최근 값 추출
        rsi = float(indicators['rsi_14'].iloc[-1])
        rsi_7 = float(indicators['rsi_7'].iloc[-1])
        stoch_k = float(indicators['stoch_k'].iloc[-1])
        stoch_d = float(indicators['stoch_d'].iloc[-1])

        bb_upper = float(indicators['bb_upper'].iloc[-1])
        bb_lower = float(indicators['bb_lower'].iloc[-1])
        bb_percent = float(indicators['bb_percent_b'].iloc[-1])

        macd = float(indicators['macd'].iloc[-1])
        macd_signal = float(indicators['macd_signal'].iloc[-1])
        macd_hist = float(indicators['macd_histogram'].iloc[-1])

        ema_5 = float(indicators['ema_5'].iloc[-1])
        ema_20 = float(indicators['ema_20'].iloc[-1])

        atr = float(indicators['atr_14'].iloc[-1])
        vwap = float(indicators['vwap'].iloc[-1])
        vol_ratio = float(indicators['volume_ratio'].iloc[-1])

        # 피봇 포인트
        pivot = self.pivot_points(
            float(high.iloc[-1]),
            float(low.iloc[-1]),
            current_price
        )

        # 신호 판단
        signals = []

        # RSI 신호
        if rsi < 30:
            signals.append(("RSI 과매도", "BUY", 0.7))
        elif rsi > 70:
            signals.append(("RSI 과매수", "SELL", 0.7))
        elif rsi < 40:
            signals.append(("RSI 저점 근접", "BUY", 0.4))
        elif rsi > 60:
            signals.append(("RSI 고점 근접", "SELL", 0.4))

        # 스토캐스틱 신호
        if stoch_k < 20 and stoch_k > stoch_d:
            signals.append(("스토캐스틱 골든크로스", "BUY", 0.6))
        elif stoch_k > 80 and stoch_k < stoch_d:
            signals.append(("스토캐스틱 데드크로스", "SELL", 0.6))

        # MACD 신호
        if macd > macd_signal and macd_hist > 0:
            signals.append(("MACD 상승", "BUY", 0.5))
        elif macd < macd_signal and macd_hist < 0:
            signals.append(("MACD 하락", "SELL", 0.5))

        # 볼린저 밴드 신호
        if bb_percent < 0:
            signals.append(("볼린저 하단 이탈", "BUY", 0.8))
        elif bb_percent > 1:
            signals.append(("볼린저 상단 이탈", "SELL", 0.8))
        elif bb_percent < 0.2:
            signals.append(("볼린저 하단 근접", "BUY", 0.5))
        elif bb_percent > 0.8:
            signals.append(("볼린저 상단 근접", "SELL", 0.5))

        # 이평선 배열
        if ema_5 > ema_20:
            signals.append(("단기>장기 이평선", "BUY", 0.4))
        else:
            signals.append(("단기<장기 이평선", "SELL", 0.4))

        # VWAP 신호
        if current_price > vwap * 1.01:
            signals.append(("VWAP 상회", "BUY", 0.3))
        elif current_price < vwap * 0.99:
            signals.append(("VWAP 하회", "SELL", 0.3))

        # 종합 점수 계산
        buy_score = sum(s[2] for s in signals if s[1] == "BUY")
        sell_score = sum(s[2] for s in signals if s[1] == "SELL")
        total_score = (buy_score - sell_score) / max(len(signals), 1) * 100

        # 추세 판단
        if total_score > 30:
            trend = "BULLISH"
        elif total_score < -30:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # 변동성 판단
        avg_atr_ratio = atr / current_price * 100
        if avg_atr_ratio > 3:
            volatility = "HIGH"
        elif avg_atr_ratio > 1.5:
            volatility = "MEDIUM"
        else:
            volatility = "LOW"

        return {
            "current_price": current_price,
            "trend": trend,
            "volatility": volatility,
            "total_score": round(total_score, 1),

            # 핵심 지표
            "rsi_14": round(rsi, 1),
            "rsi_7": round(rsi_7, 1),
            "stoch_k": round(stoch_k, 1),
            "stoch_d": round(stoch_d, 1),
            "macd": round(macd, 2),
            "macd_signal": round(macd_signal, 2),
            "macd_histogram": round(macd_hist, 2),

            # 밴드/레벨
            "bb_upper": round(bb_upper, 0),
            "bb_lower": round(bb_lower, 0),
            "bb_percent_b": round(bb_percent, 3),
            "vwap": round(vwap, 0),

            # 이평선
            "ema_5": round(ema_5, 0),
            "ema_20": round(ema_20, 0),

            # 변동성/거래량
            "atr_14": round(atr, 0),
            "atr_percent": round(avg_atr_ratio, 2),
            "volume_ratio": round(vol_ratio, 2),

            # 지지/저항
            "pivot": round(pivot['pivot'], 0),
            "resistance_1": round(pivot['r1'], 0),
            "resistance_2": round(pivot['r2'], 0),
            "support_1": round(pivot['s1'], 0),
            "support_2": round(pivot['s2'], 0),

            # 신호 목록
            "signals": [{"reason": s[0], "direction": s[1], "strength": s[2]} for s in signals],

            # 스캘핑 추천
            "recommended_stop_loss": round(current_price - atr * 1.5, 0),
            "recommended_take_profit": round(current_price + atr * 2, 0),
        }

    def format_for_llm(self, df: pd.DataFrame, stock_name: str = "") -> str:
        """
        LLM 입력용 포맷팅된 문자열 생성

        Args:
            df: OHLCV DataFrame
            stock_name: 종목명

        Returns:
            str: LLM에 입력할 기술적 분석 요약
        """
        summary = self.get_scalping_summary(df)

        # 최근 OHLCV
        df = df.copy()
        df.columns = df.columns.str.lower()
        recent = df.tail(5)

        ohlcv_text = "최근 5개 캔들 (시가/고가/저가/종가/거래량):\n"
        for idx, row in recent.iterrows():
            ohlcv_text += f"  {row.get('close', 0):,.0f} (H:{row.get('high', 0):,.0f} L:{row.get('low', 0):,.0f} V:{row.get('volume', 0):,.0f})\n"

        # 신호 텍스트
        signals_text = ""
        for sig in summary['signals']:
            direction_kr = "매수" if sig['direction'] == "BUY" else "매도"
            signals_text += f"  - {sig['reason']}: {direction_kr} (강도: {sig['strength']:.1f})\n"

        text = f"""
[기술적 분석 요약] {stock_name}
현재가: {summary['current_price']:,.0f}원
추세: {summary['trend']} | 변동성: {summary['volatility']} | 종합점수: {summary['total_score']:+.1f}

{ohlcv_text}
[모멘텀 지표]
- RSI(14): {summary['rsi_14']:.1f} | RSI(7): {summary['rsi_7']:.1f}
- 스토캐스틱 %K: {summary['stoch_k']:.1f} | %D: {summary['stoch_d']:.1f}
- MACD: {summary['macd']:.2f} | Signal: {summary['macd_signal']:.2f} | Hist: {summary['macd_histogram']:.2f}

[밴드/레벨]
- 볼린저: 상단 {summary['bb_upper']:,.0f} | 하단 {summary['bb_lower']:,.0f} | %B: {summary['bb_percent_b']:.2f}
- VWAP: {summary['vwap']:,.0f}원
- 피봇: {summary['pivot']:,.0f} | R1: {summary['resistance_1']:,.0f} | S1: {summary['support_1']:,.0f}

[이동평균]
- EMA5: {summary['ema_5']:,.0f} | EMA20: {summary['ema_20']:,.0f}
- 배열: {"정배열(상승)" if summary['ema_5'] > summary['ema_20'] else "역배열(하락)"}

[변동성/거래량]
- ATR(14): {summary['atr_14']:,.0f}원 ({summary['atr_percent']:.2f}%)
- 거래량비율: {summary['volume_ratio']:.2f}x (평균 대비)

[발생 신호]
{signals_text}
[스캘핑 추천]
- 손절가: {summary['recommended_stop_loss']:,.0f}원
- 익절가: {summary['recommended_take_profit']:,.0f}원
"""
        return text.strip()


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트용 더미 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')

    base_price = 50000
    prices = [base_price]
    for _ in range(99):
        change = np.random.randn() * 100
        prices.append(prices[-1] + change)

    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.uniform(50, 200) for p in prices],
        'low': [p - np.random.uniform(50, 200) for p in prices],
        'close': [p + np.random.randn() * 50 for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # 기술적 분석
    analyzer = TechnicalAnalyzer()

    # 전체 지표 계산
    indicators = analyzer.calculate_all(df)
    print("계산된 지표:", list(indicators.keys()))

    # 스캘핑 요약
    summary = analyzer.get_scalping_summary(df)
    print("\n스캘핑 요약:")
    for key, value in summary.items():
        if key != 'signals':
            print(f"  {key}: {value}")

    # LLM 입력용 포맷
    llm_text = analyzer.format_for_llm(df, "테스트종목")
    print("\nLLM 입력용 텍스트:")
    print(llm_text)
