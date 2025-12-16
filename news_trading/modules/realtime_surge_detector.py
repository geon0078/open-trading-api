# -*- coding: utf-8 -*-
"""
실시간 급등 감지 모듈 (Realtime Surge Detector)

하이브리드 방식으로 급등종목을 실시간 탐지합니다:
1. REST API로 급등 후보 종목 선별 (주기적 갱신)
2. WebSocket으로 후보 종목 실시간 체결가 구독
3. 실시간 데이터로 급등 점수 업데이트

사용 예시:
    >>> detector = RealtimeSurgeDetector()
    >>> detector.start()  # 실시간 모니터링 시작
"""

import sys
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pandas as pd

sys.path.extend(['..', '.', '../..'])

logger = logging.getLogger(__name__)


@dataclass
class RealtimeStock:
    """실시간 종목 데이터"""
    code: str                    # 종목코드
    name: str                    # 종목명
    price: int = 0               # 현재가
    prev_price: int = 0          # 직전가
    change: int = 0              # 전일대비
    change_rate: float = 0.0     # 전일대비율
    volume: int = 0              # 누적거래량
    volume_power: float = 100.0  # 체결강도
    buy_volume: int = 0          # 매수체결량
    sell_volume: int = 0         # 매도체결량
    bid_balance: int = 0         # 총매수호가잔량
    ask_balance: int = 0         # 총매도호가잔량
    balance_ratio: float = 1.0   # 호가잔량비 (매수/매도)
    surge_score: float = 0.0     # 종합 급등 점수
    signal: str = "NEUTRAL"      # 시그널
    rank: int = 0                # 순위
    last_update: str = ""        # 마지막 업데이트 시간
    tick_count: int = 0          # 체결 틱 카운트
    price_momentum: float = 0.0  # 가격 모멘텀 (최근 변화율)
    reasons: List[str] = field(default_factory=list)
    # LLM 분석 필드
    llm_sentiment: str = ""           # LLM 감성 분석 결과 (POSITIVE/NEGATIVE/NEUTRAL)
    llm_recommendation: str = ""      # LLM 추천 (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)
    llm_confidence: float = 0.0       # LLM 신뢰도 (0-1)
    llm_impact: str = ""              # LLM 영향도 (HIGH/MEDIUM/LOW)
    llm_analysis: str = ""            # LLM 분석 요약
    llm_score_bonus: float = 0.0      # LLM 점수 보너스
    llm_analyzed: bool = False        # LLM 분석 완료 여부
    llm_analyze_time: str = ""        # LLM 분석 시간


class RealtimeSurgeDetector:
    """
    실시간 급등 종목 탐지기 (하이브리드 방식)

    REST API + WebSocket을 결합하여 급등종목을 실시간으로 탐지합니다.
    """

    # 스캘핑 기준 상수
    MIN_VOLUME_POWER = 120      # 최소 체결강도
    MIN_CHANGE_RATE = 1.0       # 최소 등락률 (%)
    MAX_CHANGE_RATE = 15.0      # 최대 등락률 (과열 방지)
    MIN_BALANCE_RATIO = 1.2     # 최소 호가잔량비
    MIN_VOLUME = 100000         # 최소 거래량

    # 시장 구분 코드
    MARKET_CODES = ["0001", "1001"]

    # 대상 제외 필터 (우선주, ETF, ETN, SPAC 제외)
    EXCLUDE_FILTER_CODE = "0000101101"

    # 후보 종목 갱신 주기 (초)
    CANDIDATE_REFRESH_INTERVAL = 60

    # 최대 구독 종목 수 (WebSocket 제한 고려)
    MAX_SUBSCRIBE_COUNT = 40

    # LLM 분석 관련 상수
    LLM_ANALYZE_THRESHOLD = 50      # LLM 분석 대상 최소 점수
    LLM_ANALYZE_TOP_COUNT = 10      # LLM 분석 대상 상위 종목 수
    LLM_ANALYZE_INTERVAL = 30       # LLM 분석 주기 (초)

    def __init__(self, on_update: Optional[Callable] = None, enable_llm: bool = True, llm_preset: str = "deepseek"):
        """
        Args:
            on_update: 실시간 업데이트 콜백 함수
                       signature: on_update(stocks: List[RealtimeStock])
            enable_llm: LLM 분석 활성화 여부
            llm_preset: LLM 프리셋 ("deepseek", "default", "lightweight")
        """
        self._authenticated = False
        self._ws_authenticated = False
        self._running = False
        self._kws = None

        # 실시간 종목 데이터 저장소
        self._stocks: Dict[str, RealtimeStock] = {}
        self._subscribed_codes: set = set()

        # 콜백
        self._on_update = on_update

        # 스레드
        self._refresh_thread: Optional[threading.Thread] = None
        self._llm_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 통계
        self._last_refresh_time: Optional[datetime] = None
        self._tick_count = 0

        # LLM 분석기
        self._enable_llm = enable_llm
        self._llm_preset = llm_preset
        self._llm = None
        self._llm_analyze_queue: List[str] = []  # 분석 대기 종목 코드
        self._last_llm_analyze_time: Optional[datetime] = None

    def _ensure_auth(self):
        """REST API 인증"""
        if self._authenticated:
            return
        try:
            import kis_auth as ka
            ka.auth()
            self._authenticated = True
            logger.info("KIS REST API 인증 완료")
        except Exception as e:
            logger.error(f"KIS REST API 인증 실패: {e}")
            raise

    def _ensure_ws_auth(self):
        """WebSocket 인증"""
        if self._ws_authenticated:
            return
        try:
            import kis_auth as ka
            ka.auth_ws()
            self._ws_authenticated = True
            logger.info("KIS WebSocket 인증 완료")
        except Exception as e:
            logger.error(f"KIS WebSocket 인증 실패: {e}")
            raise

    def _init_llm(self):
        """LLM 분석기 초기화"""
        if not self._enable_llm:
            return

        try:
            from modules.llm.hybrid_llm_32gb import FinancialHybridLLM
            self._llm = FinancialHybridLLM()
            self._llm.set_preset(self._llm_preset)
            logger.info(f"LLM 분석기 초기화 완료 (프리셋: {self._llm_preset})")
        except ImportError as e:
            logger.warning(f"LLM 모듈 로드 실패: {e}")
            self._enable_llm = False
        except Exception as e:
            logger.warning(f"LLM 초기화 실패: {e}")
            self._enable_llm = False

    def _analyze_stock_with_llm(self, stock: RealtimeStock) -> bool:
        """
        LLM으로 개별 종목 분석

        Args:
            stock: 분석할 종목

        Returns:
            분석 성공 여부
        """
        if not self._llm or not self._enable_llm:
            return False

        try:
            # 급등 종목용 분석 프롬프트 생성
            analysis_prompt = self._build_surge_analysis_prompt(stock)

            # LLM 분석 실행
            result = self._llm.analyze(analysis_prompt, stock.code)

            if result:
                # 결과 적용
                stock.llm_sentiment = result.sentiment
                stock.llm_recommendation = result.recommendation
                stock.llm_confidence = result.confidence
                stock.llm_impact = result.impact
                stock.llm_analysis = result.analysis[:100] if result.analysis else ""
                stock.llm_analyzed = True
                stock.llm_analyze_time = datetime.now().strftime("%H:%M:%S")

                # LLM 점수 보너스 계산
                stock.llm_score_bonus = self._calculate_llm_bonus(result)

                logger.debug(f"LLM 분석 완료: {stock.name} - {result.recommendation} (신뢰도: {result.confidence:.2f})")
                return True

        except Exception as e:
            logger.error(f"LLM 분석 오류 ({stock.name}): {e}")

        return False

    def _build_surge_analysis_prompt(self, stock: RealtimeStock) -> str:
        """급등 종목 분석용 프롬프트 생성"""
        return (
            f"[급등종목 분석] {stock.name} ({stock.code})\n"
            f"현재가: {stock.price:,}원 (전일대비 {stock.change_rate:+.2f}%)\n"
            f"체결강도: {stock.volume_power:.1f} (100 기준)\n"
            f"호가잔량비: 매수/매도 = {stock.balance_ratio:.2f}배\n"
            f"누적거래량: {stock.volume:,}주\n"
            f"가격모멘텀: {stock.price_momentum:+.3f}%\n"
            f"매수체결량: {stock.buy_volume:,} / 매도체결량: {stock.sell_volume:,}\n"
            f"기술적 신호: {stock.signal}\n"
            f"급등점수: {stock.surge_score:.1f}점"
        )

    def _calculate_llm_bonus(self, result) -> float:
        """LLM 분석 결과 기반 점수 보너스 계산"""
        bonus = 0.0

        # 추천 등급별 보너스 (-20 ~ +20)
        recommendation_bonus = {
            "STRONG_BUY": 20,
            "BUY": 10,
            "HOLD": 0,
            "SELL": -10,
            "STRONG_SELL": -20
        }
        bonus += recommendation_bonus.get(result.recommendation, 0)

        # 신뢰도 가중치 적용 (0.5 ~ 1.0)
        confidence_weight = 0.5 + (result.confidence * 0.5)
        bonus *= confidence_weight

        # 영향도 보너스
        if result.impact == "HIGH":
            bonus *= 1.2
        elif result.impact == "LOW":
            bonus *= 0.8

        return bonus

    def _llm_analyze_thread(self):
        """LLM 분석 백그라운드 스레드"""
        logger.info("LLM 분석 스레드 시작")

        while self._running:
            try:
                # 분석 주기 체크
                now = datetime.now()
                if self._last_llm_analyze_time:
                    elapsed = (now - self._last_llm_analyze_time).total_seconds()
                    if elapsed < self.LLM_ANALYZE_INTERVAL:
                        time.sleep(1)
                        continue

                # 상위 종목 선별
                with self._lock:
                    sorted_stocks = sorted(
                        self._stocks.values(),
                        key=lambda x: x.surge_score,
                        reverse=True
                    )

                    # 분석 대상: 점수 임계값 이상 + 아직 분석 안 된 종목 우선
                    analyze_targets = [
                        s for s in sorted_stocks[:self.LLM_ANALYZE_TOP_COUNT]
                        if s.surge_score >= self.LLM_ANALYZE_THRESHOLD
                    ]

                if not analyze_targets:
                    time.sleep(5)
                    continue

                logger.info(f"LLM 분석 시작: {len(analyze_targets)}개 종목")

                # 각 종목 분석
                for stock in analyze_targets:
                    if not self._running:
                        break

                    with self._lock:
                        # 최신 종목 데이터로 분석
                        current_stock = self._stocks.get(stock.code)
                        if current_stock:
                            self._analyze_stock_with_llm(current_stock)

                            # 급등 점수 재계산 (LLM 보너스 포함)
                            score, signal, reasons = self._calculate_surge_score(current_stock)
                            current_stock.surge_score = score + current_stock.llm_score_bonus
                            current_stock.signal = signal
                            current_stock.reasons = reasons

                            # LLM 분석 이유 추가
                            if current_stock.llm_recommendation:
                                current_stock.reasons.append(
                                    f"LLM: {current_stock.llm_recommendation} (신뢰도 {current_stock.llm_confidence:.0%})"
                                )

                self._last_llm_analyze_time = now
                logger.info("LLM 분석 완료")

                # 콜백 호출
                self._notify_update()

            except Exception as e:
                logger.error(f"LLM 분석 스레드 오류: {e}")
                time.sleep(5)

        logger.info("LLM 분석 스레드 종료")

    def _get_initial_candidates(self) -> List[Dict]:
        """
        REST API로 초기 급등 후보 종목 조회

        Returns:
            후보 종목 리스트 [{code, name, price, change_rate, volume_power, ...}, ...]
        """
        self._ensure_auth()

        candidates = {}

        try:
            # 1. 체결강도 상위 종목
            from domestic_stock.volume_power.volume_power import volume_power

            for mkt_code in self.MARKET_CODES:
                df = volume_power(
                    fid_trgt_exls_cls_code="0",
                    fid_cond_mrkt_div_code="J",
                    fid_cond_scr_div_code="20168",
                    fid_input_iscd=mkt_code,
                    fid_div_cls_code="1",  # 보통주만
                    fid_input_price_1="",
                    fid_input_price_2="",
                    fid_vol_cnt=str(self.MIN_VOLUME),
                    fid_trgt_cls_code="0",
                    max_depth=1
                )
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        code = row.get('stck_shrn_iscd', '')
                        if code:
                            candidates[code] = {
                                'code': code,
                                'name': row.get('hts_kor_isnm', ''),
                                'price': int(row.get('stck_prpr', 0)),
                                'change': int(row.get('prdy_vrss', 0)),
                                'change_rate': float(row.get('prdy_ctrt', 0)),
                                'volume': int(row.get('acml_vol', 0)),
                                'volume_power': float(row.get('tday_rltv', 0)),
                                'buy_volume': int(row.get('shnu_cnqn_smtn', 0)),
                                'sell_volume': int(row.get('seln_cnqn_smtn', 0)),
                            }

            # 2. 등락률 상위 종목
            from domestic_stock.fluctuation.fluctuation import fluctuation

            for mkt_code in self.MARKET_CODES:
                df = fluctuation(
                    fid_cond_mrkt_div_code="J",
                    fid_cond_scr_div_code="20170",
                    fid_input_iscd=mkt_code,
                    fid_rank_sort_cls_code="0",
                    fid_input_cnt_1="0",
                    fid_prc_cls_code="0",
                    fid_input_price_1="",
                    fid_input_price_2="",
                    fid_vol_cnt=str(self.MIN_VOLUME),
                    fid_trgt_cls_code="0",
                    fid_trgt_exls_cls_code=self.EXCLUDE_FILTER_CODE,
                    fid_div_cls_code="0",
                    fid_rsfl_rate1=str(self.MIN_CHANGE_RATE),
                    fid_rsfl_rate2=str(self.MAX_CHANGE_RATE),
                )
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        code = row.get('stck_shrn_iscd', '')
                        if code and code not in candidates:
                            candidates[code] = {
                                'code': code,
                                'name': row.get('hts_kor_isnm', ''),
                                'price': int(row.get('stck_prpr', 0)),
                                'change': int(row.get('prdy_vrss', 0)),
                                'change_rate': float(row.get('prdy_ctrt', 0)),
                                'volume': int(row.get('acml_vol', 0)),
                                'volume_power': 100.0,
                                'buy_volume': 0,
                                'sell_volume': 0,
                            }

            logger.info(f"초기 후보 종목 {len(candidates)}개 조회 완료")
            return list(candidates.values())

        except Exception as e:
            logger.error(f"초기 후보 종목 조회 오류: {e}")
            return []

    def _calculate_surge_score(self, stock: RealtimeStock) -> tuple:
        """
        급등 점수 계산

        Returns:
            (score, signal, reasons)
        """
        score = 0.0
        reasons = []

        # 체결강도 점수 (0-40점)
        vp = stock.volume_power
        if vp >= 200:
            score += 40
            reasons.append(f"체결강도 매우 강함 ({vp:.1f})")
        elif vp >= 150:
            score += 30
            reasons.append(f"체결강도 강함 ({vp:.1f})")
        elif vp >= self.MIN_VOLUME_POWER:
            score += 20
            reasons.append(f"체결강도 양호 ({vp:.1f})")
        elif vp >= 100:
            score += 10
        else:
            score -= 10

        # 등락률 점수 (0-30점)
        cr = stock.change_rate
        if 3.0 <= cr <= 8.0:
            score += 30
            reasons.append(f"적정 상승률 ({cr:.2f}%)")
        elif 1.5 <= cr < 3.0:
            score += 25
            reasons.append(f"상승 초기 ({cr:.2f}%)")
        elif 8.0 < cr <= 12.0:
            score += 20
            reasons.append(f"급등 진행중 ({cr:.2f}%)")
        elif cr > 12.0:
            score += 10
            reasons.append(f"과열 주의 ({cr:.2f}%)")
        elif cr > 0:
            score += 15

        # 호가잔량비 점수 (0-30점)
        br = stock.balance_ratio
        if br >= 2.0:
            score += 30
            reasons.append(f"매수잔량 압도적 ({br:.2f}배)")
        elif br >= 1.5:
            score += 25
            reasons.append(f"매수잔량 우세 ({br:.2f}배)")
        elif br >= self.MIN_BALANCE_RATIO:
            score += 20
            reasons.append(f"매수잔량 양호 ({br:.2f}배)")
        elif br >= 1.0:
            score += 10
        else:
            score -= 5

        # 실시간 모멘텀 보너스 (0-10점)
        if stock.price_momentum > 0.5:
            score += 10
            reasons.append(f"가격 상승 모멘텀 (+{stock.price_momentum:.2f}%)")
        elif stock.price_momentum > 0.2:
            score += 5

        # 시그널 결정
        if score >= 70:
            signal = "STRONG_BUY"
        elif score >= 50:
            signal = "BUY"
        elif score >= 30:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        return score, signal, reasons

    def _on_realtime_data(self, ws, tr_id: str, result: pd.DataFrame, data_map: dict):
        """
        실시간 체결 데이터 수신 콜백
        """
        try:
            self._tick_count += 1

            for _, row in result.iterrows():
                code = str(row.get('MKSC_SHRN_ISCD', ''))
                if not code or code not in self._stocks:
                    continue

                with self._lock:
                    stock = self._stocks[code]

                    # 이전 가격 저장
                    prev_price = stock.price

                    # 실시간 데이터 업데이트
                    stock.price = int(row.get('STCK_PRPR', stock.price))
                    stock.change = int(row.get('PRDY_VRSS', stock.change))
                    stock.change_rate = float(row.get('PRDY_CTRT', stock.change_rate))
                    stock.volume = int(row.get('ACML_VOL', stock.volume))
                    stock.volume_power = float(row.get('CTTR', stock.volume_power))
                    stock.buy_volume = int(row.get('SHNU_CNTG_SMTN', stock.buy_volume))
                    stock.sell_volume = int(row.get('SELN_CNTG_SMTN', stock.sell_volume))
                    stock.bid_balance = int(row.get('TOTAL_BIDP_RSQN', stock.bid_balance))
                    stock.ask_balance = int(row.get('TOTAL_ASKP_RSQN', stock.ask_balance))

                    # 호가잔량비 계산
                    if stock.ask_balance > 0:
                        stock.balance_ratio = stock.bid_balance / stock.ask_balance

                    # 가격 모멘텀 계산 (직전가 대비 변화율)
                    if prev_price > 0:
                        stock.price_momentum = (stock.price - prev_price) / prev_price * 100

                    stock.prev_price = prev_price
                    stock.tick_count += 1
                    stock.last_update = datetime.now().strftime("%H:%M:%S")

                    # 급등 점수 재계산
                    score, signal, reasons = self._calculate_surge_score(stock)
                    stock.surge_score = score
                    stock.signal = signal
                    stock.reasons = reasons

            # 콜백 호출
            if self._on_update and self._tick_count % 5 == 0:  # 5틱마다 콜백
                self._notify_update()

        except Exception as e:
            logger.error(f"실시간 데이터 처리 오류: {e}")

    def _notify_update(self):
        """업데이트 콜백 호출"""
        if not self._on_update:
            return

        with self._lock:
            # 점수순 정렬
            sorted_stocks = sorted(
                self._stocks.values(),
                key=lambda x: x.surge_score,
                reverse=True
            )

            # 순위 부여
            for i, stock in enumerate(sorted_stocks, 1):
                stock.rank = i

            self._on_update(sorted_stocks)

    def _subscribe_stocks(self, codes: List[str]):
        """
        종목 WebSocket 구독
        """
        if not self._kws:
            return

        try:
            # 기존 모든 구독 해제는 하지 않음 (WebSocket 재연결 부담)
            # 새로운 종목만 추가 구독

            from domestic_stock.ccnl_krx.ccnl_krx import ccnl_krx

            new_codes = [c for c in codes if c not in self._subscribed_codes]

            if new_codes:
                # 최대 구독 수 제한
                codes_to_subscribe = new_codes[:self.MAX_SUBSCRIBE_COUNT - len(self._subscribed_codes)]

                if codes_to_subscribe:
                    self._kws.subscribe(request=ccnl_krx, data=codes_to_subscribe)
                    self._subscribed_codes.update(codes_to_subscribe)
                    logger.info(f"종목 {len(codes_to_subscribe)}개 구독 추가 (총 {len(self._subscribed_codes)}개)")

        except Exception as e:
            logger.error(f"종목 구독 오류: {e}")

    def _refresh_candidates(self):
        """
        후보 종목 주기적 갱신 스레드
        """
        while self._running:
            try:
                logger.info("후보 종목 갱신 시작...")

                # REST API로 새 후보 조회
                candidates = self._get_initial_candidates()

                if candidates:
                    with self._lock:
                        # 새 후보 종목 초기화/업데이트
                        for c in candidates[:self.MAX_SUBSCRIBE_COUNT]:
                            code = c['code']
                            if code not in self._stocks:
                                self._stocks[code] = RealtimeStock(
                                    code=code,
                                    name=c['name'],
                                    price=c['price'],
                                    change=c['change'],
                                    change_rate=c['change_rate'],
                                    volume=c['volume'],
                                    volume_power=c.get('volume_power', 100.0),
                                    buy_volume=c.get('buy_volume', 0),
                                    sell_volume=c.get('sell_volume', 0),
                                )
                            else:
                                # 기존 종목 REST 데이터로 보강
                                stock = self._stocks[code]
                                if stock.volume_power == 100.0:
                                    stock.volume_power = c.get('volume_power', 100.0)

                    # 새 종목 구독
                    new_codes = [c['code'] for c in candidates[:self.MAX_SUBSCRIBE_COUNT]]
                    self._subscribe_stocks(new_codes)

                self._last_refresh_time = datetime.now()
                logger.info(f"후보 종목 갱신 완료: {len(self._stocks)}개 모니터링 중")

            except Exception as e:
                logger.error(f"후보 종목 갱신 오류: {e}")

            # 다음 갱신까지 대기
            for _ in range(self.CANDIDATE_REFRESH_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

    def start(self):
        """
        실시간 모니터링 시작
        """
        if self._running:
            logger.warning("이미 실행 중입니다.")
            return

        logger.info("실시간 급등종목 탐지 시작...")

        # 인증
        self._ensure_auth()
        self._ensure_ws_auth()

        # 초기 후보 종목 조회
        candidates = self._get_initial_candidates()

        if not candidates:
            logger.error("초기 후보 종목이 없습니다.")
            return

        # 초기 종목 데이터 설정
        for c in candidates[:self.MAX_SUBSCRIBE_COUNT]:
            code = c['code']
            self._stocks[code] = RealtimeStock(
                code=code,
                name=c['name'],
                price=c['price'],
                change=c['change'],
                change_rate=c['change_rate'],
                volume=c['volume'],
                volume_power=c.get('volume_power', 100.0),
                buy_volume=c.get('buy_volume', 0),
                sell_volume=c.get('sell_volume', 0),
            )

            # 초기 점수 계산
            stock = self._stocks[code]
            score, signal, reasons = self._calculate_surge_score(stock)
            stock.surge_score = score
            stock.signal = signal
            stock.reasons = reasons

        logger.info(f"초기 후보 {len(self._stocks)}개 종목 설정 완료")

        # WebSocket 설정
        import kis_auth as ka
        self._kws = ka.KISWebSocket(api_url="/tryitout")

        # 초기 종목 구독
        initial_codes = list(self._stocks.keys())
        from domestic_stock.ccnl_krx.ccnl_krx import ccnl_krx
        self._kws.subscribe(request=ccnl_krx, data=initial_codes)
        self._subscribed_codes = set(initial_codes)

        logger.info(f"종목 {len(initial_codes)}개 WebSocket 구독 완료")

        # LLM 초기화
        if self._enable_llm:
            self._init_llm()

        self._running = True

        # 후보 갱신 스레드 시작
        self._refresh_thread = threading.Thread(target=self._refresh_candidates, daemon=True)
        self._refresh_thread.start()

        # LLM 분석 스레드 시작
        if self._enable_llm and self._llm:
            self._llm_thread = threading.Thread(target=self._llm_analyze_thread, daemon=True)
            self._llm_thread.start()

        # WebSocket 시작 (블로킹)
        logger.info("WebSocket 실시간 수신 시작...")
        self._kws.start(on_result=self._on_realtime_data)

    def stop(self):
        """
        실시간 모니터링 중지
        """
        logger.info("실시간 모니터링 중지 중...")
        self._running = False

        if self._kws:
            try:
                self._kws.close()
            except:
                pass
            self._kws = None

        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None

        if self._llm_thread:
            self._llm_thread.join(timeout=5)
            self._llm_thread = None

        logger.info("실시간 모니터링 중지 완료")

    def get_top_stocks(self, count: int = 20) -> List[RealtimeStock]:
        """
        상위 급등 종목 조회

        Args:
            count: 조회할 종목 수

        Returns:
            상위 급등 종목 리스트
        """
        with self._lock:
            sorted_stocks = sorted(
                self._stocks.values(),
                key=lambda x: x.surge_score,
                reverse=True
            )

            for i, stock in enumerate(sorted_stocks, 1):
                stock.rank = i

            return sorted_stocks[:count]

    def get_stock(self, code: str) -> Optional[RealtimeStock]:
        """특정 종목 조회"""
        with self._lock:
            return self._stocks.get(code)

    def get_statistics(self) -> dict:
        """통계 정보 조회"""
        with self._lock:
            llm_analyzed_count = sum(1 for s in self._stocks.values() if s.llm_analyzed)

        return {
            'total_stocks': len(self._stocks),
            'subscribed_count': len(self._subscribed_codes),
            'tick_count': self._tick_count,
            'last_refresh': self._last_refresh_time.strftime("%H:%M:%S") if self._last_refresh_time else None,
            'is_running': self._running,
            'llm_enabled': self._enable_llm,
            'llm_preset': self._llm_preset if self._enable_llm else None,
            'llm_analyzed_count': llm_analyzed_count,
            'last_llm_analyze': self._last_llm_analyze_time.strftime("%H:%M:%S") if self._last_llm_analyze_time else None,
        }


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 설정 로드
    try:
        from config_loader import setup_kis_config, load_env_file
        load_env_file()
        setup_kis_config()
    except ImportError:
        pass

    # 콜백 함수
    def on_update(stocks: List[RealtimeStock]):
        print("\n" + "=" * 80)
        print(f"      실시간 급등 종목 ({datetime.now().strftime('%H:%M:%S')})")
        print("=" * 80)

        for stock in stocks[:10]:
            signal_icon = {
                "STRONG_BUY": "!!",
                "BUY": "! ",
                "WATCH": "? ",
                "NEUTRAL": "  "
            }.get(stock.signal, "  ")

            print(f"[{stock.rank:2d}] {signal_icon} {stock.name:12s} ({stock.code})")
            print(f"     {stock.price:>8,}원 ({stock.change_rate:+.2f}%) | "
                  f"체결강도: {stock.volume_power:6.1f} | "
                  f"점수: {stock.surge_score:5.1f}")

        print("=" * 80)

    # 실시간 탐지기 시작
    detector = RealtimeSurgeDetector(on_update=on_update)

    try:
        detector.start()
    except KeyboardInterrupt:
        print("\n종료 중...")
        detector.stop()
