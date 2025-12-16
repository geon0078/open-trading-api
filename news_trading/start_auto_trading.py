#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
자동 트레이딩 원클릭 시작 스크립트 (실전투자 전용)

config/auto_trade.yaml 설정을 읽어 바로 자동 매매를 시작합니다.
웹 대시보드: http://localhost:5002

사용법:
    python start_auto_trading.py              # 설정 파일 기반 실행
    python start_auto_trading.py --dry-run    # 주문 없이 분석만 실행
    python start_auto_trading.py --once       # 1회만 실행
    python start_auto_trading.py --no-web     # 웹 대시보드 없이 실행

주의:
    - 실전투자 모드입니다. 실제 돈이 움직입니다!
    - config/auto_trade.yaml 파일에서 설정을 조정하세요.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
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

# 데이터 저장 디렉토리
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

class TradingDataSaver:
    """거래 데이터 저장 클래스"""

    def __init__(self):
        self.today = datetime.now().strftime("%Y%m%d")
        self.trades_file = DATA_DIR / f"trades_{self.today}.json"
        self.llm_log_file = DATA_DIR / f"llm_logs_{self.today}.json"
        self.trades = self._load_trades()
        self.llm_logs = self._load_llm_logs()

    def _load_trades(self):
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _load_llm_logs(self):
        if self.llm_log_file.exists():
            try:
                with open(self.llm_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_trade(self, result):
        trade_data = {
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "action": result.action,
            "stock_code": result.stock_code,
            "stock_name": result.stock_name,
            "current_price": result.current_price,
            "ensemble_signal": result.ensemble_signal,
            "confidence": result.confidence,
            "consensus": result.consensus,
            "order_qty": result.order_qty,
            "order_price": result.order_price,
            "order_no": result.order_no,
            "reason": result.reason,
            "technical_score": result.technical_score,
            "trend": result.trend,
        }
        self.trades.append(trade_data)
        with open(self.trades_file, 'w', encoding='utf-8') as f:
            json.dump(self.trades, f, ensure_ascii=False, indent=2, default=str)

    def save_llm_io(self, stock_code: str, stock_name: str, ensemble_result):
        """LLM 입출력 데이터 저장"""
        try:
            # 모델별 결과 변환
            model_results = []
            for mr in getattr(ensemble_result, 'model_results', []):
                model_results.append({
                    "model_name": getattr(mr, 'model_name', ''),
                    "signal": getattr(mr, 'signal', ''),
                    "confidence": getattr(mr, 'confidence', 0),
                    "reasoning": getattr(mr, 'reasoning', ''),
                    "raw_output": getattr(mr, 'raw_output', ''),
                    "processing_time": getattr(mr, 'processing_time', 0),
                    "success": getattr(mr, 'success', False),
                })

            llm_data = {
                "timestamp": datetime.now().isoformat(),
                "stock_code": stock_code,
                "stock_name": stock_name,
                "input_prompt": getattr(ensemble_result, 'input_prompt', ''),
                "input_data": getattr(ensemble_result, 'input_data', {}),
                "ensemble_signal": getattr(ensemble_result, 'ensemble_signal', ''),
                "ensemble_confidence": getattr(ensemble_result, 'ensemble_confidence', 0),
                "consensus_score": getattr(ensemble_result, 'consensus_score', 0),
                "signal_votes": getattr(ensemble_result, 'signal_votes', {}),
                "models_used": getattr(ensemble_result, 'models_used', []),
                "model_results": model_results,
                "total_processing_time": getattr(ensemble_result, 'total_processing_time', 0),
            }

            self.llm_logs.append(llm_data)
            with open(self.llm_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.llm_logs, f, ensure_ascii=False, indent=2, default=str)

            logging.info(f"LLM 로그 저장: {stock_name} ({stock_code})")
        except Exception as e:
            logging.error(f"LLM 로그 저장 실패: {e}")

    def get_trade_count(self):
        return len([t for t in self.trades if t.get('success')])


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
    parser.add_argument('--no-web', action='store_true', help='웹 대시보드 비활성화')

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

    # 데이터 저장 초기화
    data_saver = TradingDataSaver()
    print(f"\n데이터 저장 경로:")
    print(f"  거래 기록: {data_saver.trades_file}")
    print(f"  LLM 로그: {data_saver.llm_log_file}")

    # 웹 대시보드 시작
    web_dashboard = None
    if not args.no_web:
        try:
            from modules.web_dashboard import (
                start_dashboard_thread, update_state, set_news_analysis_result,
                set_trading_result, set_system_status, set_llm_status, DASHBOARD_PORT
            )
            start_dashboard_thread()
            print(f"\n웹 대시보드: http://localhost:{DASHBOARD_PORT}")
            web_dashboard = True

            # 초기 상태 설정
            set_system_status(
                is_running=True,
                llm_model=', '.join(status.get('ensemble_models', [])),
                scan_interval=scan_cfg.get('scan_interval', 60)
            )
        except Exception as e:
            print(f"\n웹 대시보드 시작 실패: {e}")
            web_dashboard = False
    else:
        print("\n웹 대시보드: 비활성화됨")

    # 실행
    scan_interval = scan_cfg.get('scan_interval', 60)

    if args.once:
        # 1회 실행
        print(f"\n{'='*50}")
        print("1회 스캔 실행...")
        print(f"{'='*50}")

        # 웹 대시보드: 분석 시작 상태 업데이트
        if web_dashboard:
            set_llm_status("분석 중...")

        results = trader.run_scan_and_trade(check_market_hours=True)
        print_results(results, data_saver, trader)

        # 웹 대시보드: 매매 결과 업데이트
        if web_dashboard:
            if results:
                set_trading_result(results, trader)
            set_llm_status("완료")
            set_system_status(is_running=False, next_scan="1회 실행 완료")
    else:
        # 폴링 모드
        print(f"\n{'='*50}")
        print(f"폴링 모드 시작 (주기: {scan_interval}초)")
        print("- 장 시작 전: 분석 전용 모드 (주문 없음)")
        print("- 장 시작 후: 분석 + 매매 모드")
        print("Ctrl+C로 종료")
        print(f"{'='*50}")

        import time
        from datetime import time as dt_time

        # 시장 시간 설정
        market_start = dt_time(*map(int, market_cfg.get('start', '09:00').split(":")))
        market_end = dt_time(*map(int, market_cfg.get('end', '15:20').split(":")))

        # 뉴스 분석 결과 저장용
        last_news_analysis = None
        last_news_analysis_time = None

        try:
            while True:
                now = datetime.now()
                current_time = now.time()
                is_weekend = now.weekday() >= 5

                print(f"\n[{now.strftime('%H:%M:%S')}] 스캔 시작...")

                # 웹 대시보드: LLM 분석 시작 상태 업데이트
                if web_dashboard:
                    set_llm_status("분석 중...")

                if is_weekend:
                    print("주말 - 뉴스 분석 모드")
                    news_result = trader.run_news_analysis(max_news=30)
                    print_news_analysis(news_result, data_saver)
                    results = None
                    mode = "NEWS"
                    # 웹 대시보드: 뉴스 분석 결과 업데이트
                    if web_dashboard and news_result:
                        set_news_analysis_result(news_result)
                elif current_time < market_start:
                    # 장 시작 전: 뉴스 분석 모드
                    print(f"장 시작 전 ({market_cfg.get('start', '09:00')}) - 뉴스 분석 모드")
                    news_result = trader.run_news_analysis(max_news=30)
                    print_news_analysis(news_result, data_saver)
                    last_news_analysis = news_result
                    last_news_analysis_time = now
                    results = None
                    mode = "NEWS"
                    # 웹 대시보드: 뉴스 분석 결과 업데이트
                    if web_dashboard and news_result:
                        set_news_analysis_result(news_result)
                elif current_time > market_end:
                    # 장 마감 후: 뉴스 분석 모드
                    print(f"장 마감 후 ({market_cfg.get('end', '15:20')}) - 뉴스 분석 모드")
                    news_result = trader.run_news_analysis(max_news=30)
                    print_news_analysis(news_result, data_saver)
                    results = None
                    mode = "NEWS"
                    # 웹 대시보드: 뉴스 분석 결과 업데이트
                    if web_dashboard and news_result:
                        set_news_analysis_result(news_result)
                elif trader._is_scalping_time():
                    # 스캘핑 시간: 매매 + 분석 모드
                    print("스캘핑 모드 활성화 (09:00~09:30) - 매매 + 분석")
                    results = trader.run_scalping_trade()
                    mode = "TRADING"
                    # 웹 대시보드: 매매 결과 업데이트
                    if web_dashboard and results:
                        set_trading_result(results, trader)
                else:
                    # 정규 장 시간: 매매 + 분석 모드
                    print("정규 장 시간 - 매매 + 분석 모드")
                    results = trader.run_scan_and_trade(check_market_hours=False)
                    mode = "TRADING"
                    # 웹 대시보드: 매매 결과 업데이트
                    if web_dashboard and results:
                        set_trading_result(results, trader)

                if results is not None:
                    print_results(results, data_saver, trader, mode)

                # 다음 스캔 시간 계산
                next_scan_time = datetime.now() + timedelta(seconds=scan_interval)
                next_scan_str = next_scan_time.strftime("%H:%M:%S")

                # 웹 대시보드: 다음 스캔 시간 업데이트
                if web_dashboard:
                    set_system_status(
                        is_running=True,
                        llm_model=', '.join(status.get('ensemble_models', [])),
                        scan_interval=scan_interval,
                        next_scan=next_scan_str
                    )
                    set_llm_status("대기")

                print(f"\n다음 스캔까지 {scan_interval}초 대기... (다음: {next_scan_str})")
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\n\n사용자에 의해 종료됨")
            # 웹 대시보드: 종료 상태 업데이트
            if web_dashboard:
                set_system_status(is_running=False, next_scan="종료됨")
                set_llm_status("종료")

    # 일일 요약 저장
    summary = {
        "date": datetime.now().strftime("%Y%m%d"),
        "total_trades": len(data_saver.trades),
        "successful_trades": data_saver.get_trade_count(),
        "llm_logs_count": len(data_saver.llm_logs),
        "end_time": datetime.now().isoformat()
    }
    summary_file = DATA_DIR / f"daily_summary_{summary['date']}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("일일 매매 요약")
    print(f"{'='*50}")
    print(f"총 분석: {summary['total_trades']}건")
    print(f"매매 성공: {summary['successful_trades']}건")
    print(f"LLM 로그: {summary['llm_logs_count']}건")
    print(f"거래 데이터: {data_saver.trades_file}")
    print(f"LLM 로그: {data_saver.llm_log_file}")
    print("\n자동 트레이딩 종료")


def print_news_analysis(news_result: dict, data_saver=None):
    """뉴스 분석 결과 출력 및 저장"""
    if not news_result:
        print("뉴스 분석 결과 없음")
        return

    print(f"\n[뉴스 분석] 수집 뉴스: {news_result.get('news_count', 0)}건")
    print("-" * 50)

    # 시장 심리
    sentiment = news_result.get('market_sentiment', 'NEUTRAL')
    sentiment_emoji = {"BULLISH": "++", "BEARISH": "--", "NEUTRAL": "=="}
    print(f"시장 심리: [{sentiment_emoji.get(sentiment, '==')}] {sentiment}")

    # 주요 테마
    themes = news_result.get('key_themes', [])
    if themes:
        print(f"주요 테마: {', '.join(themes)}")

    # 주목 종목
    attention_stocks = news_result.get('attention_stocks', [])
    if attention_stocks:
        print(f"\n주목 종목 ({len(attention_stocks)}개):")
        for stock in attention_stocks[:5]:
            name = stock.get('name', '알 수 없음')
            reason = stock.get('reason', '')
            print(f"  - {name}: {reason[:50]}...")

    # LLM 분석 결과
    llm_analysis = news_result.get('llm_analysis', {})
    if llm_analysis:
        outlook = llm_analysis.get('market_outlook', '')
        if outlook:
            print(f"\n장 전망: {outlook}")

    # 주요 뉴스 헤드라인 (최대 5개)
    news_list = news_result.get('news_list', [])
    if news_list:
        print(f"\n주요 뉴스 ({len(news_list)}건 중 5건):")
        for i, title in enumerate(news_list[:5], 1):
            print(f"  {i}. {title[:60]}...")

    # 데이터 저장
    if data_saver:
        news_file = DATA_DIR / f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(news_file, 'w', encoding='utf-8') as f:
            json.dump(news_result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n뉴스 분석 저장: {news_file}")


def print_results(results: list, data_saver=None, trader=None, mode="TRADING"):
    """결과 출력 및 저장 (LLM 로그 포함)"""
    if not results:
        print("분석 결과 없음 (급등 종목 없음)")
        return

    mode_str = "[분석]" if mode == "ANALYSIS" else "[매매]"
    print(f"\n{mode_str} 분석 결과: {len(results)}개 종목")
    print("-" * 50)

    for r in results:
        if mode == "ANALYSIS":
            # 분석 모드: 시그널 기반 표시
            if r.ensemble_signal in ["STRONG_BUY", "BUY"]:
                status = "BUY"
            elif r.ensemble_signal in ["STRONG_SELL", "SELL"]:
                status = "SELL"
            else:
                status = "HOLD"
        else:
            # 매매 모드: 주문 성공 여부
            status = "O" if r.success else "X"

        print(f"[{status}] {r.stock_name}({r.stock_code})")
        print(f"    시그널: {r.ensemble_signal}")
        print(f"    신뢰도: {r.confidence:.0%} | 합의도: {r.consensus:.0%}")
        print(f"    기술점수: {r.technical_score:+.1f} | 추세: {r.trend}")
        print(f"    행동: {r.action} - {r.reason}")

        if r.success and r.order_qty > 0:
            print(f"    주문: {r.order_qty}주 @ {r.order_price:,}원")

        # 거래 데이터 저장
        if data_saver:
            data_saver.save_trade(r)

            # LLM 입출력 데이터 저장
            if trader:
                ensemble_result = trader.get_ensemble_result(r.stock_code)
                if ensemble_result:
                    data_saver.save_llm_io(r.stock_code, r.stock_name, ensemble_result)

    if mode == "ANALYSIS":
        buy_count = sum(1 for r in results if r.ensemble_signal in ["STRONG_BUY", "BUY"])
        sell_count = sum(1 for r in results if r.ensemble_signal in ["STRONG_SELL", "SELL"])
        print(f"\n분석 요약: 매수 시그널 {buy_count}개 | 매도 시그널 {sell_count}개")
    else:
        success_count = sum(1 for r in results if r.success)
        print(f"\n매매 실행: {success_count}/{len(results)}건")

    if data_saver:
        print(f"데이터 저장: {data_saver.trades_file}")
        print(f"LLM 로그: {data_saver.llm_log_file}")


if __name__ == "__main__":
    main()
