# -*- coding: utf-8 -*-
"""
뉴스 기반 자동매매 모듈

구성:
- llm: 하이브리드 LLM 앙상블 (32GB VRAM 최적화)
- news_collector: KIS API 뉴스 수집기
- order_executor: KIS API 주문 실행기
- price_checker: KIS API 현재가 조회기
- realtime_monitor: KIS WebSocket 실시간 시세 모니터
- risk_manager: 리스크 관리 (손절/익절)
- backtester: 백테스팅 엔진
- config_loader: 환경 설정 로더
- ohlcv_fetcher: OHLCV 데이터 조회기
- technical_indicators: 기술적 보조지표 계산기
"""

from .llm import FinancialHybridLLM, AnalysisResult, EnsembleResult
from .news_collector import NewsCollector
from .order_executor import OrderExecutor, OrderResult
from .price_checker import PriceChecker, PriceInfo
from .realtime_monitor import RealtimeMonitor, RealtimePrice, RealtimeAskBid
from .risk_manager import RiskManager, RiskSignal, RiskCheckResult, Position
from .backtester import Backtester, BacktestResult, Trade, TradeAction
from .config_loader import setup_kis_config, load_env_file, get_api_credentials
from .ohlcv_fetcher import OHLCVFetcher
from .technical_indicators import TechnicalAnalyzer, ScalpingSignal, TechnicalSummary
from .surge_detector import SurgeDetector, SurgeCandidate

__all__ = [
    # LLM
    "FinancialHybridLLM",
    "AnalysisResult",
    "EnsembleResult",
    # News
    "NewsCollector",
    # Order
    "OrderExecutor",
    "OrderResult",
    # Price
    "PriceChecker",
    "PriceInfo",
    # Realtime
    "RealtimeMonitor",
    "RealtimePrice",
    "RealtimeAskBid",
    # Risk Management
    "RiskManager",
    "RiskSignal",
    "RiskCheckResult",
    "Position",
    # Backtesting
    "Backtester",
    "BacktestResult",
    "Trade",
    "TradeAction",
    # Config
    "setup_kis_config",
    "load_env_file",
    "get_api_credentials",
    # OHLCV
    "OHLCVFetcher",
    # Technical Analysis
    "TechnicalAnalyzer",
    "ScalpingSignal",
    "TechnicalSummary",
    # Surge Detection
    "SurgeDetector",
    "SurgeCandidate",
]
