# -*- coding: utf-8 -*-
"""
급등 감지 모듈 (Surge Detector)

3분 스캘핑을 위한 갑작스런 매수세 유입 종목을 탐지합니다.

핵심 지표:
- 체결강도: 매수체결량 / 매도체결량 * 100 (100 이상이면 매수우세)
- 호가잔량비: 총매수호가잔량 / 총매도호가잔량 (1 이상이면 매수우세)
- 등락률: 전일대비 가격 변동률
- 거래량 급증: 평균 대비 거래량 증가율

사용 예시:
    >>> detector = SurgeDetector()
    >>> candidates = detector.scan_surge_stocks()
    >>> for stock in candidates:
    ...     print(f"{stock['name']}: 체결강도 {stock['volume_power']}, 등락률 {stock['change_rate']}%")
"""

import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

sys.path.extend(['..', '.', '../..'])

logger = logging.getLogger(__name__)


@dataclass
class SurgeCandidate:
    """급등 후보 종목 데이터"""
    code: str                    # 종목코드
    name: str                    # 종목명
    price: int                   # 현재가
    change: int                  # 전일대비
    change_rate: float          # 전일대비율
    volume: int                  # 누적거래량
    volume_power: float         # 체결강도
    buy_volume: int             # 매수체결량
    sell_volume: int            # 매도체결량
    bid_balance: int            # 총매수호가잔량
    ask_balance: int            # 총매도호가잔량
    balance_ratio: float        # 호가잔량비 (매수/매도)
    surge_score: float          # 종합 급등 점수
    rank: int                   # 순위
    detected_at: str            # 탐지 시간
    signal: str                 # 시그널 (STRONG_BUY, BUY, WATCH, NEUTRAL)
    reasons: List[str] = field(default_factory=list)  # 매수 사유


class SurgeDetector:
    """
    실시간 급등 종목 탐지기

    체결강도, 호가잔량, 등락률을 종합 분석하여
    스캘핑에 적합한 급등 종목을 탐지합니다.
    """

    # 스캘핑 기준 상수
    MIN_VOLUME_POWER = 120      # 최소 체결강도 (매수우세)
    MIN_CHANGE_RATE = 1.0       # 최소 등락률 (%)
    MAX_CHANGE_RATE = 15.0      # 최대 등락률 (과열 방지)
    MIN_BALANCE_RATIO = 1.2     # 최소 호가잔량비 (매수/매도)
    MIN_VOLUME = 100000         # 최소 거래량

    # 시장 구분 코드 (KIS API)
    # 0001: 코스피, 1001: 코스닥
    # fid_div_cls_code="1": 보통주만 (ETF/ETN/우선주 제외)
    MARKET_CODES = ["0001", "1001"]

    # fid_trgt_exls_cls_code: 대상 제외 구분 코드 (10자리, 각 자리별 "1"이면 제외)
    # 순서: 투자위험/경고/주의, 관리종목, 정리매매, 불성실공시, 우선주, 거래정지, ETF, ETN, 신용주문불가, SPAC
    # "0000101101" = 우선주(5번째), ETF(7번째), ETN(8번째), SPAC(10번째) 제외
    EXCLUDE_FILTER_CODE = "0000101101"

    def __init__(self):
        self._authenticated = False

    def _ensure_auth(self):
        """인증 확인"""
        if self._authenticated:
            return
        try:
            import kis_auth as ka
            ka.auth()
            self._authenticated = True
            logger.info("KIS API 인증 완료")
        except Exception as e:
            logger.error(f"KIS API 인증 실패: {e}")
            raise

    def get_volume_power_ranking(self, market: str = "0000", max_count: int = 30) -> pd.DataFrame:
        """
        체결강도 상위 종목 조회

        Args:
            market: 시장 구분 (0000:전체, 0001:코스피, 1001:코스닥)
            max_count: 조회 종목 수

        Returns:
            DataFrame: 체결강도 상위 종목 데이터
        """
        self._ensure_auth()

        try:
            from domestic_stock.volume_power.volume_power import volume_power

            # 코스피(0001)와 코스닥(1001)만 조회하여 합침 (ETF/ETN 제외)
            all_dfs = []

            for mkt_code in self.MARKET_CODES:  # 코스피, 코스닥만
                df = volume_power(
                    fid_trgt_exls_cls_code="0",     # 전체
                    fid_cond_mrkt_div_code="J",     # KRX
                    fid_cond_scr_div_code="20168",  # 체결강도
                    fid_input_iscd=mkt_code,        # 코스피 또는 코스닥
                    fid_div_cls_code="1",           # 보통주만 (1: 보통주, 2: 우선주)
                    fid_input_price_1="",           # 가격 하한
                    fid_input_price_2="",           # 가격 상한
                    fid_vol_cnt=str(self.MIN_VOLUME),  # 최소 거래량
                    fid_trgt_cls_code="0",          # 전체
                    max_depth=1                      # 1페이지만
                )
                if df is not None and not df.empty:
                    df['market'] = 'KOSPI' if mkt_code == '0001' else 'KOSDAQ'
                    all_dfs.append(df)

            if not all_dfs:
                logger.warning("체결강도 데이터 없음")
                return pd.DataFrame()

            # 합치고 체결강도 기준 정렬
            combined_df = pd.concat(all_dfs, ignore_index=True)
            if 'tday_rltv' in combined_df.columns:
                combined_df['tday_rltv'] = pd.to_numeric(combined_df['tday_rltv'], errors='coerce')
                combined_df = combined_df.sort_values('tday_rltv', ascending=False)

            # 상위 N개만 선택
            combined_df = combined_df.head(max_count)

            logger.info(f"체결강도 상위 {len(combined_df)}개 종목 조회 (코스피/코스닥 보통주)")
            return combined_df

        except Exception as e:
            logger.error(f"체결강도 조회 오류: {e}")
            return pd.DataFrame()

    def get_fluctuation_ranking(self, max_count: int = 30) -> pd.DataFrame:
        """
        등락률 상위 종목 조회 (상승 종목)

        Args:
            max_count: 조회 종목 수

        Returns:
            DataFrame: 등락률 상위 종목 데이터
        """
        self._ensure_auth()

        try:
            from domestic_stock.fluctuation.fluctuation import fluctuation

            # 코스피(0001)와 코스닥(1001)만 조회하여 합침 (ETF/ETN 제외)
            all_dfs = []

            for mkt_code in self.MARKET_CODES:  # 코스피, 코스닥만
                df = fluctuation(
                    fid_cond_mrkt_div_code="J",      # KRX
                    fid_cond_scr_div_code="20170",   # 등락률
                    fid_input_iscd=mkt_code,         # 코스피 또는 코스닥
                    fid_rank_sort_cls_code="0",      # 등락률순
                    fid_input_cnt_1="0",             # 조회 수
                    fid_prc_cls_code="0",            # 전체
                    fid_input_price_1="",            # 가격 하한
                    fid_input_price_2="",            # 가격 상한
                    fid_vol_cnt=str(self.MIN_VOLUME),  # 최소 거래량
                    fid_trgt_cls_code="0",           # 전체
                    fid_trgt_exls_cls_code=self.EXCLUDE_FILTER_CODE,  # 우선주/ETF/ETN/SPAC 제외
                    fid_div_cls_code="0",            # 전체 (fid_trgt_exls_cls_code로 필터링)
                    fid_rsfl_rate1=str(self.MIN_CHANGE_RATE),  # 최소 등락률
                    fid_rsfl_rate2=str(self.MAX_CHANGE_RATE),  # 최대 등락률
                )
                if df is not None and not df.empty:
                    df['market'] = 'KOSPI' if mkt_code == '0001' else 'KOSDAQ'
                    all_dfs.append(df)

            if not all_dfs:
                logger.warning("등락률 데이터 없음")
                return pd.DataFrame()

            # 합치고 등락률 기준 정렬
            combined_df = pd.concat(all_dfs, ignore_index=True)
            if 'prdy_ctrt' in combined_df.columns:
                combined_df['prdy_ctrt'] = pd.to_numeric(combined_df['prdy_ctrt'], errors='coerce')
                combined_df = combined_df.sort_values('prdy_ctrt', ascending=False)

            combined_df = combined_df.head(max_count)

            logger.info(f"등락률 상위 {len(combined_df)}개 종목 조회 (코스피/코스닥 보통주)")
            return combined_df

        except Exception as e:
            logger.error(f"등락률 조회 오류: {e}")
            return pd.DataFrame()

    def get_quote_balance_ranking(self, sort_type: str = "0", max_count: int = 30) -> pd.DataFrame:
        """
        호가잔량 순위 조회

        Args:
            sort_type: 정렬 기준 (0:순매수잔량순, 2:매수비율순)
            max_count: 조회 종목 수

        Returns:
            DataFrame: 호가잔량 순위 데이터
        """
        self._ensure_auth()

        try:
            from domestic_stock.quote_balance.quote_balance import quote_balance

            # 코스피(0001)와 코스닥(1001)만 조회하여 합침 (ETF/ETN 제외)
            all_dfs = []

            for mkt_code in self.MARKET_CODES:  # 코스피, 코스닥만
                df = quote_balance(
                    fid_vol_cnt=str(self.MIN_VOLUME),  # 최소 거래량
                    fid_cond_mrkt_div_code="J",        # KRX
                    fid_cond_scr_div_code="20172",     # 호가잔량
                    fid_input_iscd=mkt_code,           # 코스피 또는 코스닥
                    fid_rank_sort_cls_code=sort_type,  # 정렬 기준
                    fid_div_cls_code="1",              # 보통주만 (1: 보통주, 2: 우선주)
                    fid_trgt_cls_code="0",             # 전체
                    fid_trgt_exls_cls_code="0",        # 전체
                    fid_input_price_1="",              # 가격 하한
                    fid_input_price_2="",              # 가격 상한
                )
                if df is not None and not df.empty:
                    df['market'] = 'KOSPI' if mkt_code == '0001' else 'KOSDAQ'
                    all_dfs.append(df)

            if not all_dfs:
                logger.warning("호가잔량 데이터 없음")
                return pd.DataFrame()

            # 합치고 순매수잔량 기준 정렬
            combined_df = pd.concat(all_dfs, ignore_index=True)
            if 'ntby_qty' in combined_df.columns:
                combined_df['ntby_qty'] = pd.to_numeric(combined_df['ntby_qty'], errors='coerce')
                combined_df = combined_df.sort_values('ntby_qty', ascending=False)

            combined_df = combined_df.head(max_count)

            logger.info(f"호가잔량 순위 {len(combined_df)}개 종목 조회 (코스피/코스닥 보통주)")
            return combined_df

        except Exception as e:
            logger.error(f"호가잔량 조회 오류: {e}")
            return pd.DataFrame()

    def calculate_surge_score(
        self,
        volume_power: float,
        change_rate: float,
        balance_ratio: float
    ) -> Tuple[float, str, List[str]]:
        """
        급등 점수 계산

        Args:
            volume_power: 체결강도
            change_rate: 등락률
            balance_ratio: 호가잔량비

        Returns:
            Tuple[score, signal, reasons]
        """
        score = 0.0
        reasons = []

        # 체결강도 점수 (0-40점)
        if volume_power >= 200:
            score += 40
            reasons.append(f"체결강도 매우 강함 ({volume_power:.1f})")
        elif volume_power >= 150:
            score += 30
            reasons.append(f"체결강도 강함 ({volume_power:.1f})")
        elif volume_power >= self.MIN_VOLUME_POWER:
            score += 20
            reasons.append(f"체결강도 양호 ({volume_power:.1f})")
        elif volume_power >= 100:
            score += 10
        else:
            score -= 10  # 매도 우세

        # 등락률 점수 (0-30점)
        if 3.0 <= change_rate <= 8.0:
            score += 30
            reasons.append(f"적정 상승률 ({change_rate:.2f}%)")
        elif 1.5 <= change_rate < 3.0:
            score += 25
            reasons.append(f"상승 초기 ({change_rate:.2f}%)")
        elif 8.0 < change_rate <= 12.0:
            score += 20
            reasons.append(f"급등 진행중 ({change_rate:.2f}%)")
        elif change_rate > 12.0:
            score += 10
            reasons.append(f"과열 주의 ({change_rate:.2f}%)")
        elif change_rate > 0:
            score += 15

        # 호가잔량비 점수 (0-30점)
        if balance_ratio >= 2.0:
            score += 30
            reasons.append(f"매수잔량 압도적 ({balance_ratio:.2f}배)")
        elif balance_ratio >= 1.5:
            score += 25
            reasons.append(f"매수잔량 우세 ({balance_ratio:.2f}배)")
        elif balance_ratio >= self.MIN_BALANCE_RATIO:
            score += 20
            reasons.append(f"매수잔량 양호 ({balance_ratio:.2f}배)")
        elif balance_ratio >= 1.0:
            score += 10
        else:
            score -= 5  # 매도 잔량 우세

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

    def scan_surge_stocks(self, min_score: float = 50.0) -> List[SurgeCandidate]:
        """
        급등 종목 스캔

        체결강도, 등락률, 호가잔량을 종합 분석하여
        스캘핑에 적합한 급등 종목을 탐지합니다.

        Args:
            min_score: 최소 급등 점수 (0-100)

        Returns:
            List[SurgeCandidate]: 급등 후보 종목 리스트
        """
        logger.info("급등 종목 스캔 시작...")

        candidates = []
        stock_data = {}

        # 1. 체결강도 상위 종목 조회
        logger.info("[1/3] 체결강도 상위 종목 조회...")
        vp_df = self.get_volume_power_ranking(max_count=50)

        if not vp_df.empty:
            for _, row in vp_df.iterrows():
                code = row.get('stck_shrn_iscd', '')
                name = row.get('hts_kor_isnm', '')
                # API에서 fid_div_cls_code="1"로 보통주만 조회됨
                if code:
                    stock_data[code] = {
                        'code': code,
                        'name': name,
                        'price': int(row.get('stck_prpr', 0)),
                        'change': int(row.get('prdy_vrss', 0)),
                        'change_rate': float(row.get('prdy_ctrt', 0)),
                        'volume': int(row.get('acml_vol', 0)),
                        'volume_power': float(row.get('tday_rltv', 0)),
                        'buy_volume': int(row.get('shnu_cnqn_smtn', 0)),
                        'sell_volume': int(row.get('seln_cnqn_smtn', 0)),
                        'bid_balance': 0,
                        'ask_balance': 0,
                        'balance_ratio': 1.0,
                    }

        # 2. 등락률 상위 종목으로 보강
        logger.info("[2/3] 등락률 상위 종목 조회...")
        fl_df = self.get_fluctuation_ranking(max_count=50)

        if not fl_df.empty:
            for _, row in fl_df.iterrows():
                code = row.get('stck_shrn_iscd', '')
                name = row.get('hts_kor_isnm', '')
                # API에서 fid_trgt_exls_cls_code로 우선주/ETF/ETN/SPAC 제외됨
                if code and code not in stock_data:
                    stock_data[code] = {
                        'code': code,
                        'name': name,
                        'price': int(row.get('stck_prpr', 0)),
                        'change': int(row.get('prdy_vrss', 0)),
                        'change_rate': float(row.get('prdy_ctrt', 0)),
                        'volume': int(row.get('acml_vol', 0)),
                        'volume_power': 100.0,  # 기본값
                        'buy_volume': 0,
                        'sell_volume': 0,
                        'bid_balance': 0,
                        'ask_balance': 0,
                        'balance_ratio': 1.0,
                    }
                elif code in stock_data:
                    # 등락률 정보 업데이트
                    stock_data[code]['change_rate'] = float(row.get('prdy_ctrt', stock_data[code]['change_rate']))

        # 3. 호가잔량 정보로 보강
        logger.info("[3/3] 호가잔량 순위 조회...")
        qb_df = self.get_quote_balance_ranking(sort_type="0", max_count=50)

        if not qb_df.empty:
            for _, row in qb_df.iterrows():
                code = row.get('mksc_shrn_iscd', '')
                name = row.get('hts_kor_isnm', '')
                bid_balance = int(row.get('total_bidp_rsqn', 0))
                ask_balance = int(row.get('total_askp_rsqn', 0))

                balance_ratio = bid_balance / ask_balance if ask_balance > 0 else 1.0

                if code in stock_data:
                    stock_data[code]['bid_balance'] = bid_balance
                    stock_data[code]['ask_balance'] = ask_balance
                    stock_data[code]['balance_ratio'] = balance_ratio
                # 호가잔량 API는 fid_trgt_exls_cls_code가 "0"만 지원하므로
                # 체결강도/등락률에서 이미 필터링된 종목만 추가
                elif code:
                    # 호가잔량에서만 나온 종목은 추가하지 않음 (필터링 불가)
                    pass

        # 4. 종합 점수 계산 및 필터링
        # API 파라미터로 필터링 완료: fid_div_cls_code="1" (보통주), fid_trgt_exls_cls_code로 ETF/ETN/SPAC 제외
        logger.info(f"총 {len(stock_data)}개 종목 분석 중... (코스피/코스닥 보통주만)")

        for code, data in stock_data.items():
            score, signal, reasons = self.calculate_surge_score(
                volume_power=data['volume_power'],
                change_rate=data['change_rate'],
                balance_ratio=data['balance_ratio']
            )

            if score >= min_score:
                candidate = SurgeCandidate(
                    code=data['code'],
                    name=data['name'],
                    price=data['price'],
                    change=data['change'],
                    change_rate=data['change_rate'],
                    volume=data['volume'],
                    volume_power=data['volume_power'],
                    buy_volume=data['buy_volume'],
                    sell_volume=data['sell_volume'],
                    bid_balance=data['bid_balance'],
                    ask_balance=data['ask_balance'],
                    balance_ratio=data['balance_ratio'],
                    surge_score=score,
                    rank=0,
                    detected_at=datetime.now().strftime("%H:%M:%S"),
                    signal=signal,
                    reasons=reasons
                )
                candidates.append(candidate)

        # 5. 점수순 정렬 및 순위 부여
        candidates.sort(key=lambda x: x.surge_score, reverse=True)

        # 순위 부여
        for i, c in enumerate(candidates, 1):
            c.rank = i

        logger.info(f"급등 후보 {len(candidates)}개 종목 탐지 완료")

        return candidates

    def format_for_display(self, candidates: List[SurgeCandidate]) -> str:
        """디스플레이용 포맷팅"""
        if not candidates:
            return "급등 후보 종목이 없습니다."

        lines = []
        lines.append("=" * 80)
        lines.append("                    급등 종목 스캔 결과")
        lines.append(f"                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        for c in candidates[:20]:  # 상위 20개
            signal_icon = {
                "STRONG_BUY": "🔥",
                "BUY": "✅",
                "WATCH": "👀",
                "NEUTRAL": "⚪"
            }.get(c.signal, "")

            lines.append(f"\n[{c.rank}] {c.name} ({c.code}) {signal_icon} {c.signal}")
            lines.append(f"    현재가: {c.price:,}원 ({c.change:+,}, {c.change_rate:+.2f}%)")
            lines.append(f"    체결강도: {c.volume_power:.1f} | 호가비: {c.balance_ratio:.2f} | 점수: {c.surge_score:.1f}")
            lines.append(f"    거래량: {c.volume:,}주")
            if c.reasons:
                lines.append(f"    사유: {', '.join(c.reasons[:3])}")

        lines.append("\n" + "=" * 80)
        lines.append("  ⚠️ 스캘핑 주의사항")
        lines.append("  - 3분 내 청산 목표")
        lines.append("  - 손절가: 진입가 -0.5%")
        lines.append("  - 익절가: 진입가 +1.0~1.5%")
        lines.append("=" * 80)

        return "\n".join(lines)

    def format_for_llm(self, candidates: List[SurgeCandidate]) -> str:
        """LLM 분석용 포맷팅"""
        if not candidates:
            return "급등 후보 종목 없음"

        lines = []
        lines.append("[급등 종목 분석 데이터]")
        lines.append(f"탐지시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"총 후보: {len(candidates)}개\n")

        for c in candidates[:10]:  # LLM에는 상위 10개만
            lines.append(f"종목: {c.name} ({c.code})")
            lines.append(f"  - 시그널: {c.signal} (점수: {c.surge_score:.1f}/100)")
            lines.append(f"  - 현재가: {c.price:,}원, 등락률: {c.change_rate:+.2f}%")
            lines.append(f"  - 체결강도: {c.volume_power:.1f} (100 이상: 매수우세)")
            lines.append(f"  - 호가잔량비: {c.balance_ratio:.2f} (1 이상: 매수우세)")
            lines.append(f"  - 매수체결량: {c.buy_volume:,}, 매도체결량: {c.sell_volume:,}")
            lines.append(f"  - 매수사유: {', '.join(c.reasons)}")
            lines.append("")

        return "\n".join(lines)


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

    # 급등 탐지기 실행
    detector = SurgeDetector()

    print("\n" + "=" * 80)
    print("         급등 종목 스캔 (3분 스캘핑용)")
    print("=" * 80)

    candidates = detector.scan_surge_stocks(min_score=40)

    # 결과 출력
    print(detector.format_for_display(candidates))

    # LLM 포맷도 출력
    print("\n\n[LLM 분석용 데이터]")
    print(detector.format_for_llm(candidates))
