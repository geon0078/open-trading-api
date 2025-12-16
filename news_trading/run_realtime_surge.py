# -*- coding: utf-8 -*-
"""
실시간 급등종목 탐지 실행 스크립트

하이브리드 방식 (REST API + WebSocket)으로
급등종목을 실시간으로 탐지하고 모니터링합니다.

실행:
    python run_realtime_surge.py
"""

import os
import sys
import logging
from datetime import datetime

# 경로 설정
sys.path.extend(['.', '..', '../..', 'modules'])
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_screen():
    """화면 클리어"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 105)
    print("                    실시간 급등종목 모니터링 (하이브리드 + LLM 분석)")
    print("                    REST API + WebSocket + Hybrid Ensemble LLM")
    print("=" * 105)


def print_stock_table(stocks, stats):
    """종목 테이블 출력"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n  갱신시간: {now}")
    print(f"  모니터링: {stats['total_stocks']}개 | "
          f"구독: {stats['subscribed_count']}개 | "
          f"틱수: {stats['tick_count']:,}")

    # LLM 상태 표시
    if stats.get('llm_enabled'):
        llm_status = f"  LLM: {stats.get('llm_preset', 'default')} | "
        llm_status += f"분석: {stats.get('llm_analyzed_count', 0)}개"
        if stats.get('last_llm_analyze'):
            llm_status += f" (마지막: {stats['last_llm_analyze']})"
        print(llm_status)

    print("-" * 105)
    print(f"{'순위':^4} {'시그널':^10} {'종목명':^14} {'코드':^8} "
          f"{'현재가':>10} {'등락률':>8} {'체결강도':>8} {'호가비':>6} {'점수':>6} {'LLM':^10}")
    print("-" * 105)

    signal_colors = {
        "STRONG_BUY": "\033[91m",  # 빨강
        "BUY": "\033[93m",          # 노랑
        "WATCH": "\033[94m",        # 파랑
        "NEUTRAL": "\033[90m",      # 회색
    }
    reset = "\033[0m"

    for stock in stocks[:20]:
        color = signal_colors.get(stock.signal, "")

        # 시그널 아이콘
        signal_icon = {
            "STRONG_BUY": "[!! 강력매수]",
            "BUY": "[!  매수   ]",
            "WATCH": "[?  관망   ]",
            "NEUTRAL": "[   중립   ]"
        }.get(stock.signal, "[         ]")

        # LLM 분석 결과 표시
        if stock.llm_analyzed and stock.llm_recommendation:
            llm_icon = {
                "STRONG_BUY": "\033[92m[++]\033[0m",  # 초록
                "BUY": "\033[92m[+ ]\033[0m",
                "HOLD": "\033[90m[= ]\033[0m",
                "SELL": "\033[91m[- ]\033[0m",
                "STRONG_SELL": "\033[91m[--]\033[0m"
            }.get(stock.llm_recommendation, "[  ]")
            llm_display = f"{llm_icon} {stock.llm_confidence:.0%}"
        else:
            llm_display = "    -     "

        print(f"{color}"
              f"{stock.rank:>3}  "
              f"{signal_icon:^10} "
              f"{stock.name:^14} "
              f"{stock.code:^8} "
              f"{stock.price:>10,} "
              f"{stock.change_rate:>+7.2f}% "
              f"{stock.volume_power:>7.1f} "
              f"{stock.balance_ratio:>5.2f}x "
              f"{stock.surge_score:>5.1f} "
              f"{llm_display}"
              f"{reset}")

    print("-" * 105)
    print("  [Ctrl+C] 종료 | LLM: [++]강력매수 [+]매수 [=]홀드 [-]매도 [--]강력매도")
    print("=" * 105)


def main():
    """메인 함수"""
    import argparse

    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="실시간 급등종목 모니터링 (하이브리드 + LLM)")
    parser.add_argument("--no-llm", action="store_true", help="LLM 분석 비활성화")
    parser.add_argument("--llm-preset", type=str, default="deepseek",
                        choices=["deepseek", "default", "lightweight"],
                        help="LLM 프리셋 선택 (default: deepseek)")
    args = parser.parse_args()

    # 설정 로드
    try:
        from config_loader import setup_kis_config, load_env_file
        load_env_file()
        setup_kis_config()
        logger.info("설정 로드 완료")
    except ImportError as e:
        logger.warning(f"config_loader 없음: {e}")

    # 실시간 탐지기 임포트
    from modules.realtime_surge_detector import RealtimeSurgeDetector

    # 업데이트 콜백
    def on_update(stocks):
        try:
            clear_screen()
            print_header()
            stats = detector.get_statistics()
            print_stock_table(stocks, stats)
        except Exception as e:
            logger.error(f"출력 오류: {e}")

    # 탐지기 생성 (LLM 옵션 적용)
    enable_llm = not args.no_llm
    detector = RealtimeSurgeDetector(
        on_update=on_update,
        enable_llm=enable_llm,
        llm_preset=args.llm_preset
    )

    print_header()
    print("\n  초기화 중...")
    print("  - REST API로 급등 후보 종목 조회")
    print("  - WebSocket 실시간 체결가 구독")
    if enable_llm:
        print(f"  - LLM 앙상블 분석 활성화 (프리셋: {args.llm_preset})")
    else:
        print("  - LLM 분석 비활성화")
    print("\n  잠시만 기다려주세요...\n")

    try:
        # 실시간 모니터링 시작
        detector.start()
    except KeyboardInterrupt:
        print("\n\n종료 요청...")
    except Exception as e:
        logger.error(f"실행 오류: {e}")
    finally:
        detector.stop()
        print("\n실시간 모니터링이 종료되었습니다.")


if __name__ == "__main__":
    main()
