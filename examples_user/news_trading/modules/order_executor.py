# -*- coding: utf-8 -*-
"""
KIS Open API 주문 실행 모듈

한국투자증권 Open API의 order_cash API를 사용하여
LLM 분석 결과를 기반으로 주식 매수/매도 주문을 실행합니다.

사용 예시:
    >>> executor = OrderExecutor(env_dv="demo", cano="12345678", acnt_prdt_cd="01")
    >>> result = executor.execute_from_llm_result(llm_result, order_qty=10, order_price=70000)
    >>> print(result)
"""

import sys
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import pandas as pd

# KIS API 모듈 경로 추가
sys.path.extend(['../..', '.'])

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """주문 결과 데이터 클래스"""
    success: bool
    order_no: Optional[str]  # 주문번호
    order_time: Optional[str]  # 주문시각
    stock_code: str
    order_type: str  # "buy" or "sell"
    order_qty: int
    order_price: int
    message: str
    raw_response: Optional[pd.DataFrame] = None


class OrderExecutor:
    """
    KIS Open API 주문 실행기

    LLM 분석 결과를 기반으로 주식 매수/매도 주문을 실행합니다.

    Attributes:
        env_dv: 실전/모의 구분 ("real" 또는 "demo")
        cano: 계좌번호 앞 8자리
        acnt_prdt_cd: 계좌상품코드 (뒤 2자리)
        default_ord_dvsn: 기본 주문구분 ("00": 지정가, "01": 시장가)

    Example:
        >>> executor = OrderExecutor(
        ...     env_dv="demo",
        ...     cano="12345678",
        ...     acnt_prdt_cd="01"
        ... )
        >>> # LLM 결과로부터 주문
        >>> result = executor.execute_from_llm_result(
        ...     llm_result={"recommendation": "BUY", "stock_code": "005930"},
        ...     order_qty=10,
        ...     order_price=70000
        ... )
    """

    # 주문구분 코드
    ORD_DVSN_CODES = {
        "지정가": "00",
        "시장가": "01",
        "조건부지정가": "02",
        "최유리지정가": "03",
        "최우선지정가": "04",
    }

    def __init__(
        self,
        env_dv: str = "demo",
        cano: str = "",
        acnt_prdt_cd: str = "",
        default_ord_dvsn: str = "00"
    ):
        """
        Args:
            env_dv: 실전/모의 구분 ("real": 실전, "demo": 모의)
            cano: 계좌번호 앞 8자리
            acnt_prdt_cd: 계좌상품코드 (뒤 2자리)
            default_ord_dvsn: 기본 주문구분 (기본값: "00" 지정가)
        """
        self.env_dv = env_dv
        self.cano = cano
        self.acnt_prdt_cd = acnt_prdt_cd
        self.default_ord_dvsn = default_ord_dvsn

        # 파라미터 검증
        if env_dv not in ["real", "demo"]:
            raise ValueError("env_dv는 'real' 또는 'demo'여야 합니다.")

        if not cano or not acnt_prdt_cd:
            logger.warning("계좌 정보가 설정되지 않았습니다. execute 전에 설정하세요.")

    def execute_order(
        self,
        stock_code: str,
        order_type: str,
        order_qty: int,
        order_price: int,
        ord_dvsn: Optional[str] = None
    ) -> OrderResult:
        """
        주문 실행

        Args:
            stock_code: 종목코드 (6자리)
            order_type: 주문유형 ("buy" 또는 "sell")
            order_qty: 주문수량
            order_price: 주문단가 (시장가 주문 시 0)
            ord_dvsn: 주문구분 (None이면 default_ord_dvsn 사용)

        Returns:
            OrderResult: 주문 결과

        Example:
            >>> result = executor.execute_order(
            ...     stock_code="005930",
            ...     order_type="buy",
            ...     order_qty=10,
            ...     order_price=70000
            ... )
            >>> if result.success:
            ...     print(f"주문 성공: {result.order_no}")
        """
        try:
            from domestic_stock.order_cash.order_cash import order_cash
        except ImportError:
            logger.error("domestic_stock.order_cash 모듈을 찾을 수 없습니다.")
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type=order_type,
                order_qty=order_qty,
                order_price=order_price,
                message="order_cash 모듈 임포트 실패"
            )

        # 파라미터 검증
        if not self.cano or not self.acnt_prdt_cd:
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type=order_type,
                order_qty=order_qty,
                order_price=order_price,
                message="계좌 정보가 설정되지 않았습니다."
            )

        if order_type not in ["buy", "sell"]:
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type=order_type,
                order_qty=order_qty,
                order_price=order_price,
                message=f"잘못된 주문유형: {order_type}"
            )

        # 주문구분 설정
        if ord_dvsn is None:
            ord_dvsn = self.default_ord_dvsn

        # 시장가 주문 시 가격은 0
        if ord_dvsn == "01":
            order_price = 0

        try:
            logger.info(f"주문 실행: [{stock_code}] {order_type.upper()} {order_qty}주 @ {order_price}원")

            # KIS API 호출
            result_df = order_cash(
                env_dv=self.env_dv,
                ord_dv=order_type,
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                pdno=stock_code,
                ord_dvsn=ord_dvsn,
                ord_qty=str(order_qty),
                ord_unpr=str(order_price),
                excg_id_dvsn_cd="KRX"
            )

            if result_df is not None and not result_df.empty:
                order_no = result_df.iloc[0].get('ODNO', '')
                order_time = result_df.iloc[0].get('ORD_TMD', '')

                logger.info(f"주문 성공: 주문번호={order_no}, 시각={order_time}")

                return OrderResult(
                    success=True,
                    order_no=order_no,
                    order_time=order_time,
                    stock_code=stock_code,
                    order_type=order_type,
                    order_qty=order_qty,
                    order_price=order_price,
                    message="주문 성공",
                    raw_response=result_df
                )
            else:
                return OrderResult(
                    success=False,
                    order_no=None,
                    order_time=None,
                    stock_code=stock_code,
                    order_type=order_type,
                    order_qty=order_qty,
                    order_price=order_price,
                    message="API 응답 없음"
                )

        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type=order_type,
                order_qty=order_qty,
                order_price=order_price,
                message=str(e)
            )

    def execute_from_llm_result(
        self,
        llm_result: Dict[str, Any],
        order_qty: int,
        order_price: int,
        min_confidence: float = 0.7,
        allowed_recommendations: Optional[List[str]] = None
    ) -> Optional[OrderResult]:
        """
        LLM 분석 결과를 기반으로 주문 실행

        Args:
            llm_result: LLM 분석 결과 딕셔너리
                {
                    "stock_code": str,
                    "recommendation": str,  # BUY, SELL, HOLD 등
                    "confidence": float,
                    ...
                }
            order_qty: 주문수량
            order_price: 주문단가
            min_confidence: 주문 실행 최소 신뢰도 (기본값: 0.7)
            allowed_recommendations: 주문 허용 추천 리스트 (기본값: ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"])

        Returns:
            OrderResult 또는 None (HOLD인 경우)

        Example:
            >>> llm_result = {
            ...     "stock_code": "005930",
            ...     "recommendation": "BUY",
            ...     "confidence": 0.85
            ... }
            >>> result = executor.execute_from_llm_result(llm_result, 10, 70000)
        """
        if allowed_recommendations is None:
            allowed_recommendations = ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]

        recommendation = llm_result.get("recommendation", "HOLD")
        confidence = llm_result.get("confidence", 0.0)
        stock_code = llm_result.get("stock_code", "")

        # HOLD인 경우 주문 없음
        if recommendation == "HOLD":
            logger.info(f"[{stock_code}] HOLD - 주문 없음")
            return None

        # 신뢰도 체크
        if confidence < min_confidence:
            logger.info(f"[{stock_code}] 신뢰도 부족 ({confidence:.1%} < {min_confidence:.1%}) - 주문 스킵")
            return None

        # 허용된 추천인지 체크
        if recommendation not in allowed_recommendations:
            logger.info(f"[{stock_code}] {recommendation}은 허용된 추천이 아님 - 주문 스킵")
            return None

        # 주문 유형 결정
        if recommendation in ["BUY", "STRONG_BUY", "WEAK_BUY"]:
            order_type = "buy"
        elif recommendation in ["SELL", "STRONG_SELL", "WEAK_SELL"]:
            order_type = "sell"
        else:
            logger.warning(f"[{stock_code}] 알 수 없는 추천: {recommendation}")
            return None

        # 주문 실행
        return self.execute_order(
            stock_code=stock_code,
            order_type=order_type,
            order_qty=order_qty,
            order_price=order_price
        )

    def get_balance(self) -> Optional[pd.DataFrame]:
        """
        잔고 조회

        Returns:
            pd.DataFrame: 잔고 데이터 (output1)
        """
        try:
            from domestic_stock.inquire_balance import inquire_balance
        except ImportError:
            logger.error("domestic_stock.inquire_balance 모듈을 찾을 수 없습니다.")
            return None

        try:
            df1, df2 = inquire_balance(
                env_dv=self.env_dv,
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                afhr_flpr_yn="N",
                inqr_dvsn="02",  # 종목별
                unpr_dvsn="01",
                fund_sttl_icld_yn="N",
                fncg_amt_auto_rdpt_yn="N",
                prcs_dvsn="00"
            )
            return df1
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return None

    def get_holding_qty(self, stock_code: str) -> int:
        """
        특정 종목 보유수량 조회

        Args:
            stock_code: 종목코드

        Returns:
            int: 보유수량 (없으면 0)
        """
        balance_df = self.get_balance()
        if balance_df is None or balance_df.empty:
            return 0

        # 종목코드로 필터링
        stock_row = balance_df[balance_df['pdno'] == stock_code]
        if stock_row.empty:
            return 0

        return int(stock_row.iloc[0].get('hldg_qty', 0))

    def calculate_order_quantity(
        self,
        current_price: int,
        max_order_amount: int = 100000
    ) -> int:
        """
        주문 한도 내에서 주문 수량 계산

        Args:
            current_price: 현재가
            max_order_amount: 최대 주문 금액 (기본: 10만원)

        Returns:
            int: 주문 가능 수량 (최소 1주)
        """
        if current_price <= 0:
            return 0

        quantity = max_order_amount // current_price
        return max(1, quantity)  # 최소 1주

    def execute_auto_trade(
        self,
        ensemble_result: Any,
        max_order_amount: int = 100000,
        min_confidence: float = 0.7,
        min_consensus: float = 0.67,
        allowed_buy_signals: Optional[List[str]] = None,
        allowed_sell_signals: Optional[List[str]] = None,
        use_entry_price: bool = True,
        ord_dvsn: str = "00"
    ) -> OrderResult:
        """
        앙상블 분석 결과 기반 자동 매매 실행

        Args:
            ensemble_result: EnsembleAnalysis 객체 또는 dict
            max_order_amount: 1회 최대 주문 금액 (기본: 10만원)
            min_confidence: 최소 신뢰도 (기본: 0.7 = 70%)
            min_consensus: 최소 합의도 (기본: 0.67 = 67%)
            allowed_buy_signals: 매수 허용 시그널 리스트
            allowed_sell_signals: 매도 허용 시그널 리스트
            use_entry_price: True면 분석 결과의 진입가 사용, False면 현재가 사용
            ord_dvsn: 주문구분 ("00": 지정가, "01": 시장가)

        Returns:
            OrderResult: 주문 결과

        Example:
            >>> from modules.ensemble_analyzer import get_ensemble_analyzer
            >>> analyzer = get_ensemble_analyzer()
            >>> result = analyzer.analyze_with_technical_data("005930", "삼성전자", 70000)
            >>> order = executor.execute_auto_trade(result, max_order_amount=100000)
        """
        if allowed_buy_signals is None:
            allowed_buy_signals = ["STRONG_BUY", "BUY"]
        if allowed_sell_signals is None:
            allowed_sell_signals = ["STRONG_SELL", "SELL"]

        # EnsembleAnalysis 객체 또는 dict 처리
        if hasattr(ensemble_result, 'ensemble_signal'):
            signal = ensemble_result.ensemble_signal
            confidence = ensemble_result.ensemble_confidence
            consensus = ensemble_result.consensus_score
            stock_code = ensemble_result.stock_code
            stock_name = ensemble_result.stock_name
            entry_price = int(ensemble_result.avg_entry_price) if ensemble_result.avg_entry_price else 0
            current_price = ensemble_result.input_data.get('stock', {}).get('price', entry_price)
        else:
            # dict 형태
            signal = ensemble_result.get('ensemble_signal', 'HOLD')
            confidence = ensemble_result.get('ensemble_confidence', 0)
            consensus = ensemble_result.get('consensus_score', 0)
            stock_code = ensemble_result.get('stock_code', '')
            stock_name = ensemble_result.get('stock_name', '')
            entry_price = int(ensemble_result.get('avg_entry_price', 0))
            current_price = ensemble_result.get('input_data', {}).get('stock', {}).get('price', entry_price)

        # 가격 결정
        order_price = entry_price if use_entry_price and entry_price > 0 else current_price
        if order_price <= 0:
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type="none",
                order_qty=0,
                order_price=0,
                message=f"유효하지 않은 가격: {order_price}"
            )

        # 신뢰도 체크
        if confidence < min_confidence:
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type="skip",
                order_qty=0,
                order_price=order_price,
                message=f"신뢰도 부족: {confidence:.1%} < {min_confidence:.1%}"
            )

        # 합의도 체크
        if consensus < min_consensus:
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type="skip",
                order_qty=0,
                order_price=order_price,
                message=f"합의도 부족: {consensus:.1%} < {min_consensus:.1%}"
            )

        # 시그널에 따른 주문 유형 결정
        if signal in allowed_buy_signals:
            order_type = "buy"
        elif signal in allowed_sell_signals:
            order_type = "sell"
        elif signal == "HOLD":
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type="hold",
                order_qty=0,
                order_price=order_price,
                message="HOLD 시그널 - 주문 없음"
            )
        else:
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type="skip",
                order_qty=0,
                order_price=order_price,
                message=f"허용되지 않은 시그널: {signal}"
            )

        # 매도 시 보유 수량 확인
        if order_type == "sell":
            holding_qty = self.get_holding_qty(stock_code)
            if holding_qty <= 0:
                return OrderResult(
                    success=False,
                    order_no=None,
                    order_time=None,
                    stock_code=stock_code,
                    order_type="sell",
                    order_qty=0,
                    order_price=order_price,
                    message=f"보유 수량 없음 - 매도 불가"
                )
            order_qty = holding_qty  # 전량 매도
        else:
            # 매수 시 주문 수량 계산
            order_qty = self.calculate_order_quantity(order_price, max_order_amount)

        if order_qty <= 0:
            return OrderResult(
                success=False,
                order_no=None,
                order_time=None,
                stock_code=stock_code,
                order_type=order_type,
                order_qty=0,
                order_price=order_price,
                message="주문 수량 계산 실패"
            )

        # 주문 금액 한도 확인
        order_amount = order_qty * order_price
        if order_type == "buy" and order_amount > max_order_amount:
            # 한도 초과 시 수량 조정
            order_qty = max_order_amount // order_price
            if order_qty <= 0:
                return OrderResult(
                    success=False,
                    order_no=None,
                    order_time=None,
                    stock_code=stock_code,
                    order_type=order_type,
                    order_qty=0,
                    order_price=order_price,
                    message=f"주문 한도 초과: 가격 {order_price:,}원이 한도 {max_order_amount:,}원을 초과"
                )

        logger.info(f"[자동매매] {stock_name}({stock_code}) {order_type.upper()} "
                   f"{order_qty}주 @ {order_price:,}원 (시그널: {signal}, 신뢰도: {confidence:.0%})")

        # 주문 실행
        return self.execute_order(
            stock_code=stock_code,
            order_type=order_type,
            order_qty=order_qty,
            order_price=order_price,
            ord_dvsn=ord_dvsn
        )


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # KIS API 인증
    try:
        import kis_auth as ka
        ka.auth()
    except Exception as e:
        print(f"KIS API 인증 실패: {e}")
        sys.exit(1)

    # 주문 실행기 초기화 (모의투자)
    executor = OrderExecutor(
        env_dv="demo",
        cano=ka.trenv.my_acct,
        acnt_prdt_cd=ka.trenv.my_prod
    )

    print("\n=== 잔고 조회 테스트 ===")
    balance = executor.get_balance()
    if balance is not None and not balance.empty:
        print(balance[['pdno', 'prdt_name', 'hldg_qty', 'pchs_avg_pric']].head())
    else:
        print("잔고가 없습니다.")

    print("\n=== LLM 결과 기반 주문 테스트 ===")
    # 테스트용 LLM 결과 (실제로는 LLM 분석 결과)
    test_llm_result = {
        "stock_code": "005930",
        "recommendation": "BUY",
        "confidence": 0.85,
        "sentiment": "positive"
    }

    print(f"테스트 LLM 결과: {test_llm_result}")
    print("실제 주문은 실행하지 않습니다. (테스트 모드)")

    # 실제 주문 실행 시:
    # result = executor.execute_from_llm_result(
    #     llm_result=test_llm_result,
    #     order_qty=1,
    #     order_price=70000
    # )
    # if result:
    #     print(f"주문 결과: {result}")
