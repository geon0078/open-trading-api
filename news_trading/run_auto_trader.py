#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2단계 앙상블 LLM 자동 매매 실행기

EXAONE 4.0을 메인 모델로, 3개 하위 모델(deepseek, qwen3, solar)이
데이터를 탐색하고 EXAONE이 최종 판단하여 자동 매매를 실행합니다.

사용법:
    # 기본 실행 (실전투자, 10만원 한도)
    python run_auto_trader.py

    # 모의투자 모드
    python run_auto_trader.py --demo

    # 주문 한도 변경
    python run_auto_trader.py --max-amount 50000

    # 단일 종목 분석
    python run_auto_trader.py --stock 005930 --name 삼성전자 --price 55000

    # 폴링 모드 (60초마다 스캔)
    python run_auto_trader.py --polling --interval 60

옵션:
    --demo              모의투자 모드 (기본: 실전투자)
    --max-amount        1회 주문 한도 (기본: 100000원)
    --min-confidence    최소 신뢰도 (기본: 0.7)
    --min-consensus     최소 합의도 (기본: 0.67)
    --min-score         최소 급등 점수 (기본: 50)
    --max-stocks        분석할 최대 종목 수 (기본: 5)
    --stock             단일 종목 코드
    --name              단일 종목명
    --price             단일 종목 현재가
    --polling           폴링 모드 활성화
    --interval          폴링 주기 (초, 기본: 60)
    --skip-market-check 장 시간 체크 비활성화

주의:
    - 실전투자 모드에서는 실제 돈이 움직입니다!
    - 반드시 테스트 후 사용하세요.
    - 장 시간 외에는 자동으로 거래가 중단됩니다.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '..'))
sys.path.insert(0, os.path.join(script_dir, '../..'))

# examples_llm 경로 추가 (API 모듈용)
examples_llm_path = os.path.join(script_dir, '../../..', 'examples_llm')
if os.path.exists(examples_llm_path):
    sys.path.insert(0, examples_llm_path)


def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def print_banner():
    """배너 출력"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       2단계 앙상블 LLM 자동 매매 시스템                       ║
║                                                              ║
║   [1단계] deepseek-r1:8b, qwen3:8b, solar:10.7b              ║
║   [2단계] ingu627/exaone4.0:32b (최종 판단)                   ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_result(result):
    """결과 출력"""
    status_icon = "✅" if result.success else "❌" if result.action == "ERROR" else "⏭️"
    signal_color = {
        "STRONG_BUY": "\033[92m",   # 초록
        "BUY": "\033[92m",
        "STRONG_SELL": "\033[91m",  # 빨강
        "SELL": "\033[91m",
        "HOLD": "\033[93m",         # 노랑
    }.get(result.ensemble_signal, "\033[0m")
    reset = "\033[0m"

    print(f"""
{status_icon} {result.stock_name} ({result.stock_code})
   시그널: {signal_color}{result.ensemble_signal}{reset}
   신뢰도: {result.confidence:.0%} | 합의도: {result.consensus:.0%}
   기술점수: {result.technical_score:+.1f} | 추세: {result.trend}
   행동: {result.action}
   결과: {result.reason}
""")

    if result.success:
        print(f"   📦 주문: {result.order_qty}주 @ {result.order_price:,}원")
        if result.order_no:
            print(f"   📝 주문번호: {result.order_no}")


def run_single_stock(trader, stock_code: str, stock_name: str, price: int, skip_market_check: bool):
    """단일 종목 분석 및 매매"""
    print(f"\n{'='*60}")
    print(f"📊 {stock_name} ({stock_code}) 분석 시작...")
    print(f"   현재가: {price:,}원")
    print(f"{'='*60}")

    result = trader.analyze_and_trade(
        stock_code=stock_code,
        stock_name=stock_name,
        current_price=price,
        check_market_hours=not skip_market_check
    )

    print_result(result)
    return result


def run_scan_once(trader, skip_market_check: bool):
    """급등 종목 스캔 후 매매 (1회)"""
    print(f"\n{'='*60}")
    print(f"🔍 급등 종목 스캔 시작...")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    results = trader.run_scan_and_trade(check_market_hours=not skip_market_check)

    if not results:
        print("\n📭 분석 결과 없음 (급등 종목 없거나 거래 불가 시간)")
        return results

    print(f"\n📋 분석 결과: {len(results)}개 종목")
    print("-" * 60)

    for result in results:
        print_result(result)

    # 요약
    success_count = sum(1 for r in results if r.success)
    print(f"\n📈 요약: {success_count}/{len(results)}개 매매 실행")

    return results


def run_scalping(trader, skip_market_check: bool):
    """스캘핑 모드 실행 (09:00 ~ 09:30)"""
    print(f"\n{'='*60}")
    print(f"⚡ 스캘핑 모드 시작 (09:00 ~ 09:30)")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"📰 야간 뉴스 수집 중 (전일 장 마감 ~ 금일 장 시작)...")

    results = trader.run_scalping_trade()

    if not results:
        print("\n📭 스캘핑 결과 없음 (스캘핑 시간 아니거나 대상 종목 없음)")
        return results

    print(f"\n📋 스캘핑 결과: {len(results)}개 종목")
    print("-" * 60)

    for result in results:
        print_result(result)

    # 요약
    success_count = sum(1 for r in results if r.success)
    print(f"\n⚡ 스캘핑 요약: {success_count}/{len(results)}개 매매 실행")

    return results


def run_polling(trader, interval: int, skip_market_check: bool):
    """폴링 모드 (지속적 스캔, 스캘핑 시간 자동 감지)"""
    print(f"\n🔄 폴링 모드 시작 (주기: {interval}초)")
    print("   스캘핑 시간(09:00~09:30)에는 자동으로 스캘핑 모드 실행")
    print("   Ctrl+C로 종료\n")

    try:
        while True:
            now = datetime.now()

            # 스캘핑 시간 확인 (09:00 ~ 09:30)
            if trader._is_scalping_time():
                print(f"\n⚡ 스캘핑 시간 감지 ({now.strftime('%H:%M')})")
                run_scalping(trader, skip_market_check)
            else:
                run_scan_once(trader, skip_market_check)

            print(f"\n⏳ 다음 스캔까지 {interval}초 대기...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n🛑 폴링 모드 종료")


def main():
    parser = argparse.ArgumentParser(
        description="2단계 앙상블 LLM 자동 매매 실행기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_auto_trader.py                    # 기본 실행 (실전투자)
  python run_auto_trader.py --demo             # 모의투자
  python run_auto_trader.py --max-amount 50000 # 5만원 한도
  python run_auto_trader.py --stock 005930 --name 삼성전자 --price 55000
  python run_auto_trader.py --polling --interval 60
        """
    )

    # 환경 설정
    parser.add_argument('--demo', action='store_true', help='모의투자 모드')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그')

    # 주문 설정
    parser.add_argument('--max-amount', type=int, default=100000, help='1회 주문 한도 (기본: 100000)')
    parser.add_argument('--min-confidence', type=float, default=0.7, help='최소 신뢰도 (기본: 0.7)')
    parser.add_argument('--min-consensus', type=float, default=0.67, help='최소 합의도 (기본: 0.67)')

    # 스캔 설정
    parser.add_argument('--min-score', type=float, default=50.0, help='최소 급등 점수 (기본: 50)')
    parser.add_argument('--max-stocks', type=int, default=5, help='분석할 최대 종목 수 (기본: 5)')

    # 단일 종목 모드
    parser.add_argument('--stock', type=str, help='단일 종목 코드')
    parser.add_argument('--name', type=str, help='단일 종목명')
    parser.add_argument('--price', type=int, help='단일 종목 현재가')

    # 폴링 모드
    parser.add_argument('--polling', action='store_true', help='폴링 모드')
    parser.add_argument('--interval', type=int, default=60, help='폴링 주기 (초, 기본: 60)')

    # 스캘핑 모드
    parser.add_argument('--scalping', action='store_true', help='스캘핑 모드 (09:00~09:30)')

    # 기타
    parser.add_argument('--skip-market-check', action='store_true', help='장 시간 체크 비활성화')

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.verbose)

    # 배너 출력
    print_banner()

    # 환경 확인
    env_dv = "demo" if args.demo else "real"
    print(f"⚙️  환경: {'모의투자' if args.demo else '🚨 실전투자 🚨'}")
    print(f"💰 주문 한도: {args.max_amount:,}원")
    print(f"📊 최소 신뢰도: {args.min_confidence:.0%}")
    print(f"🤝 최소 합의도: {args.min_consensus:.0%}")

    if not args.demo:
        print("\n⚠️  경고: 실전투자 모드입니다. 실제 돈이 움직입니다!")
        confirm = input("계속하시겠습니까? (y/N): ")
        if confirm.lower() != 'y':
            print("취소되었습니다.")
            return

    # AutoTrader 생성
    from modules.auto_trader import AutoTrader, AutoTradeConfig

    config = AutoTradeConfig(
        env_dv=env_dv,
        max_order_amount=args.max_amount,
        min_confidence=args.min_confidence,
        min_consensus=args.min_consensus,
        min_surge_score=args.min_score,
        max_stocks_per_scan=args.max_stocks,
    )

    trader = AutoTrader(config)

    # 상태 확인
    status = trader.get_status()
    print(f"\n🔧 앙상블 모델: {', '.join(status['ensemble_models'])}")
    print(f"🎯 메인 모델: {status['main_model']}")

    if status['can_trade']:
        print("✅ 거래 가능 상태")
    else:
        print(f"⚠️  거래 제한: {status['market_status']['reason'] or status['risk_status']['reason']}")

    # 실행 모드 선택
    if args.stock and args.name and args.price:
        # 단일 종목 모드
        run_single_stock(trader, args.stock, args.name, args.price, args.skip_market_check)
    elif args.scalping:
        # 스캘핑 모드 (09:00~09:30)
        run_scalping(trader, args.skip_market_check)
    elif args.polling:
        # 폴링 모드 (스캘핑 시간 자동 감지)
        run_polling(trader, args.interval, args.skip_market_check)
    else:
        # 단일 스캔 모드
        run_scan_once(trader, args.skip_market_check)

    print("\n✨ 완료")


if __name__ == "__main__":
    main()
