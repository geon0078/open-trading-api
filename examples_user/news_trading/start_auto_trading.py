#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
자동 트레이딩 원클릭 시작 스크립트 (실전투자 전용)

config/auto_trade.yaml 설정을 읽어 바로 자동 매매를 시작합니다.

사용법:
    python start_auto_trading.py              # 설정 파일 기반 실행
    python start_auto_trading.py --dry-run    # 주문 없이 분석만 실행
    python start_auto_trading.py --once       # 1회만 실행

주의:
    - 실전투자 모드입니다. 실제 돈이 움직입니다!
    - config/auto_trade.yaml 파일에서 설정을 조정하세요.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# 경로 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

# YAML 파싱
try:
    import yaml
except ImportError:
    print("PyYAML이 필요합니다: pip install pyyaml")
    sys.exit(1)


def load_config(config_path: str = None) -> dict:
    """설정 파일 로드"""
    if config_path is None:
        config_path = SCRIPT_DIR / "config" / "auto_trade.yaml"

    if not os.path.exists(config_path):
        print(f"설정 파일이 없습니다: {config_path}")
        print("기본 설정으로 실행합니다.")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """로깅 설정"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file', '')

    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = SCRIPT_DIR / log_file
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def print_banner(config: dict, env_dv: str, dry_run: bool):
    """시작 배너 출력"""
    print("""
 ===============================================================
    KIS Open Trading - 자동 트레이딩 시스템 (실전투자)
 ===============================================================
    """)

    print("    [실전투자 모드] - 실제 돈이 움직입니다!")

    if dry_run:
        print("    [DRY-RUN: 주문 없이 분석만 실행]")

    # 설정 요약
    order_cfg = config.get('order', {})
    threshold_cfg = config.get('threshold', {})
    risk_cfg = config.get('risk', {})

    print(f"""
    주문 한도: {order_cfg.get('max_order_amount', 100000):,}원
    최소 신뢰도: {threshold_cfg.get('min_confidence', 0.7):.0%}
    최소 합의도: {threshold_cfg.get('min_consensus', 0.67):.0%}
    일일 최대 거래: {risk_cfg.get('max_daily_trades', 10)}회
    일일 최대 손실: {risk_cfg.get('max_daily_loss', 50000):,}원
 ===============================================================
    """)


def confirm_real_trading() -> bool:
    """실전투자 확인"""
    print("\n" + "=" * 50)
    print("  [경고] 실전투자 모드입니다!")
    print("  실제 돈이 움직입니다. 계속하시겠습니까?")
    print("=" * 50)

    response = input("\n계속하려면 'yes'를 입력하세요: ").strip().lower()
    return response == 'yes'


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="자동 트레이딩 원클릭 시작 (실전투자 전용)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dry-run', action='store_true', help='주문 없이 분석만 실행')
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--no-confirm', action='store_true', help='실전투자 확인 생략')
    parser.add_argument('--once', action='store_true', help='1회만 실행 (폴링 없이)')

    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config)

    # 로깅 설정
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # 환경: 실전투자 전용
    env_dv = "real"

    # 배너 출력
    print_banner(config, env_dv, args.dry_run)

    # 실전투자 확인
    if env_dv == "real" and not args.no_confirm and not args.dry_run:
        if not confirm_real_trading():
            print("취소되었습니다.")
            return

    # KIS API 인증
    print("\nKIS API 인증 중...")
    try:
        import kis_auth as ka
        ka.auth()
        print("KIS API 인증 성공")
    except Exception as e:
        print(f"KIS API 인증 실패: {e}")
        print("kis_devlp.yaml 파일을 확인하세요.")
        return

    # AutoTrader 설정 생성
    from modules.auto_trader import AutoTrader, AutoTradeConfig

    order_cfg = config.get('order', {})
    threshold_cfg = config.get('threshold', {})
    risk_cfg = config.get('risk', {})
    scan_cfg = config.get('scan', {})
    market_cfg = config.get('market', {})
    scalping_cfg = config.get('scalping', {})
    ensemble_cfg = config.get('ensemble', {})

    trader_config = AutoTradeConfig(
        env_dv=env_dv,
        cano=ka.getTREnv().my_acct,
        acnt_prdt_cd=ka.getTREnv().my_prod,
        max_order_amount=order_cfg.get('max_order_amount', 100000),
        ord_dvsn=order_cfg.get('ord_dvsn', '00'),
        min_confidence=threshold_cfg.get('min_confidence', 0.7),
        min_consensus=threshold_cfg.get('min_consensus', 0.67),
        stop_loss_pct=risk_cfg.get('stop_loss_pct', 0.5),
        take_profit_pct=risk_cfg.get('take_profit_pct', 1.5),
        max_daily_trades=risk_cfg.get('max_daily_trades', 10),
        max_daily_loss=risk_cfg.get('max_daily_loss', 50000),
        min_surge_score=scan_cfg.get('min_surge_score', 50.0),
        max_stocks_per_scan=scan_cfg.get('max_stocks_per_scan', 5),
        market_start=market_cfg.get('start', '09:00'),
        market_end=market_cfg.get('end', '15:20'),
        scalping_enabled=scalping_cfg.get('enabled', True),
        scalping_start=scalping_cfg.get('start', '09:00'),
        scalping_end=scalping_cfg.get('end', '09:30'),
        scalping_min_confidence=scalping_cfg.get('min_confidence', 0.65),
        scalping_max_order_amount=scalping_cfg.get('max_order_amount', 50000),
        scalping_stop_loss_pct=scalping_cfg.get('stop_loss_pct', 0.3),
        scalping_take_profit_pct=scalping_cfg.get('take_profit_pct', 0.8),
        use_parallel=ensemble_cfg.get('use_parallel', False),
        auto_unload=ensemble_cfg.get('auto_unload', True),
    )

    # DRY-RUN 모드에서는 실제 주문 비활성화
    if args.dry_run:
        # dry-run 플래그를 설정할 수 없으므로 일일 거래 횟수를 0으로 설정
        trader_config.max_daily_trades = 0
        print("\n[DRY-RUN] 분석만 실행됩니다. 실제 주문은 없습니다.")

    # AutoTrader 생성
    print("\nAutoTrader 초기화 중...")
    trader = AutoTrader(trader_config)

    # 상태 확인
    status = trader.get_status()
    print(f"\n앙상블 모델: {', '.join(status.get('ensemble_models', []))}")
    print(f"메인 모델: {status.get('main_model', 'N/A')}")

    if status.get('can_trade'):
        print("거래 가능 상태입니다.")
    else:
        market_reason = status.get('market_status', {}).get('reason', '')
        risk_reason = status.get('risk_status', {}).get('reason', '')
        print(f"거래 제한: {market_reason or risk_reason}")

    # 실행
    scan_interval = scan_cfg.get('scan_interval', 60)

    if args.once:
        # 1회 실행
        print(f"\n{'='*50}")
        print("1회 스캔 실행...")
        print(f"{'='*50}")

        results = trader.run_scan_and_trade(check_market_hours=True)
        print_results(results)
    else:
        # 폴링 모드
        print(f"\n{'='*50}")
        print(f"폴링 모드 시작 (주기: {scan_interval}초)")
        print("Ctrl+C로 종료")
        print(f"{'='*50}")

        import time

        try:
            while True:
                now = datetime.now()
                print(f"\n[{now.strftime('%H:%M:%S')}] 스캔 시작...")

                # 스캘핑 시간 확인
                if trader._is_scalping_time():
                    print("스캘핑 모드 활성화 (09:00~09:30)")
                    results = trader.run_scalping_trade()
                else:
                    results = trader.run_scan_and_trade(check_market_hours=True)

                print_results(results)

                print(f"\n다음 스캔까지 {scan_interval}초 대기...")
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\n\n사용자에 의해 종료됨")

    print("\n자동 트레이딩 종료")


def print_results(results: list):
    """결과 출력"""
    if not results:
        print("분석 결과 없음 (급등 종목 없거나 거래 불가 시간)")
        return

    print(f"\n분석 결과: {len(results)}개 종목")
    print("-" * 50)

    for r in results:
        status = "O" if r.success else "X"
        signal_color = {
            "STRONG_BUY": "\033[92m",
            "BUY": "\033[92m",
            "STRONG_SELL": "\033[91m",
            "SELL": "\033[91m",
            "HOLD": "\033[93m",
        }.get(r.ensemble_signal, "")
        reset = "\033[0m"

        print(f"[{status}] {r.stock_name}({r.stock_code})")
        print(f"    시그널: {signal_color}{r.ensemble_signal}{reset}")
        print(f"    신뢰도: {r.confidence:.0%} | 합의도: {r.consensus:.0%}")
        print(f"    행동: {r.action} - {r.reason}")

        if r.success and r.order_qty > 0:
            print(f"    주문: {r.order_qty}주 @ {r.order_price:,}원")

    success_count = sum(1 for r in results if r.success)
    print(f"\n매매 실행: {success_count}/{len(results)}건")


if __name__ == "__main__":
    main()
