#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실전 자동 매매 실행기 (시황분석 + 데이터 저장)

아침 시황분석을 수행하고 09:00부터 실전 매매를 시작합니다.
모든 거래 데이터는 JSON 파일로 저장됩니다.

사용법:
    python run_real_trading.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

# 경로 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

# 로그 디렉토리
LOG_DIR = SCRIPT_DIR / "logs"
DATA_DIR = SCRIPT_DIR / "data"
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# 오늘 날짜
TODAY = datetime.now().strftime("%Y%m%d")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"trading_{TODAY}.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class TradingDataSaver:
    """거래 데이터 저장 클래스"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.today = datetime.now().strftime("%Y%m%d")
        self.trades_file = data_dir / f"trades_{self.today}.json"
        self.analysis_file = data_dir / f"market_analysis_{self.today}.json"
        self.summary_file = data_dir / f"daily_summary_{self.today}.json"

        # 기존 데이터 로드
        self.trades = self._load_json(self.trades_file) or []
        self.market_analysis = self._load_json(self.analysis_file) or {}

    def _load_json(self, file_path: Path) -> Optional[Dict]:
        """JSON 파일 로드"""
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_json(self, data: Dict, file_path: Path):
        """JSON 파일 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def save_market_analysis(self, analysis: Dict):
        """시황분석 결과 저장"""
        self.market_analysis = {
            "timestamp": datetime.now().isoformat(),
            "date": self.today,
            **analysis
        }
        self._save_json(self.market_analysis, self.analysis_file)
        logger.info(f"시황분석 저장: {self.analysis_file}")

    def save_trade(self, trade_result):
        """거래 결과 저장"""
        trade_data = {
            "timestamp": datetime.now().isoformat(),
            "success": trade_result.success,
            "action": trade_result.action,
            "stock_code": trade_result.stock_code,
            "stock_name": trade_result.stock_name,
            "current_price": trade_result.current_price,
            "ensemble_signal": trade_result.ensemble_signal,
            "confidence": trade_result.confidence,
            "consensus": trade_result.consensus,
            "order_qty": trade_result.order_qty,
            "order_price": trade_result.order_price,
            "order_no": trade_result.order_no,
            "reason": trade_result.reason,
            "technical_score": trade_result.technical_score,
            "trend": trade_result.trend,
        }
        self.trades.append(trade_data)
        self._save_json(self.trades, self.trades_file)
        logger.info(f"거래 저장: {trade_result.stock_name} - {trade_result.action}")

    def save_daily_summary(self, summary: Dict):
        """일일 요약 저장"""
        self._save_json(summary, self.summary_file)
        logger.info(f"일일 요약 저장: {self.summary_file}")

    def get_trade_count(self) -> int:
        """오늘 거래 횟수"""
        return len([t for t in self.trades if t.get('success')])


class MarketAnalyzer:
    """시황분석 클래스"""

    def __init__(self):
        self.news_list = []
        self.analysis_result = {}

    def collect_overnight_news(self) -> List[str]:
        """야간 뉴스 수집 (전일 15:30 ~ 금일 09:00)"""
        logger.info("야간 뉴스 수집 중...")

        try:
            from modules.news_collector import NewsCollector
            collector = NewsCollector()

            # 뉴스 수집 (깊이 5페이지)
            news_df = collector.collect(
                stock_codes=None,  # 전체 뉴스
                filter_duplicates=False,
                max_depth=5
            )

            if news_df.empty:
                logger.info("수집된 뉴스 없음")
                return []

            # 야간 시간대 필터링
            overnight_news = []
            now = datetime.now()
            today_str = now.strftime("%Y%m%d")
            yesterday = now - timedelta(days=1)
            yesterday_str = yesterday.strftime("%Y%m%d")

            for _, row in news_df.iterrows():
                news_date = str(row.get('data_dt', ''))
                news_time = str(row.get('data_tm', ''))[:4]
                title = row.get('titl', '')

                # 전일 15:30 이후 또는 금일 09:00 이전
                if (news_date == yesterday_str and news_time >= "1530") or \
                   (news_date == today_str and news_time < "0900"):
                    overnight_news.append({
                        "title": title,
                        "date": news_date,
                        "time": news_time
                    })

            self.news_list = overnight_news
            logger.info(f"야간 뉴스 {len(overnight_news)}건 수집")
            return [n['title'] for n in overnight_news]

        except Exception as e:
            logger.error(f"뉴스 수집 오류: {e}")
            return []

    def analyze_market_sentiment(self) -> Dict:
        """LLM을 이용한 시황분석"""
        logger.info("시황분석 시작...")

        if not self.news_list:
            return {
                "sentiment": "NEUTRAL",
                "confidence": 0.5,
                "summary": "분석할 야간 뉴스가 없습니다.",
                "key_factors": [],
                "recommendation": "신중하게 접근하세요."
            }

        try:
            from modules.ensemble_analyzer import EnsembleLLMAnalyzer

            analyzer = EnsembleLLMAnalyzer(keep_alive="5m", auto_unload=True)
            analyzer.discover_models()
            analyzer.setup_ensemble(use_financial_ensemble=True)

            # 뉴스 제목 리스트
            news_titles = "\n".join([f"- {n['title']}" for n in self.news_list[:30]])

            prompt = f"""
오늘의 한국 주식시장 시황을 분석해주세요.

=== 야간 뉴스 (전일 장 마감 ~ 금일 장 시작) ===
수집된 뉴스: {len(self.news_list)}건

주요 헤드라인:
{news_titles}

=== 분석 요청 ===
1. 전반적인 시장 분위기 (BULLISH/NEUTRAL/BEARISH)
2. 신뢰도 (0.0 ~ 1.0)
3. 핵심 요약 (2-3문장)
4. 주요 영향 요인 3가지
5. 오늘의 투자 전략 권고

JSON 형식으로 답변:
{{
    "sentiment": "BULLISH/NEUTRAL/BEARISH",
    "confidence": 0.0~1.0,
    "summary": "요약",
    "key_factors": ["요인1", "요인2", "요인3"],
    "recommendation": "투자 전략"
}}
"""

            # 메인 모델로 분석
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "ingu627/exaone4.0:32b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=180
            )

            if response.status_code == 200:
                result_text = response.json().get('response', '')

                # JSON 파싱 시도
                try:
                    import re
                    json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                    if json_match:
                        self.analysis_result = json.loads(json_match.group())
                    else:
                        self.analysis_result = {
                            "sentiment": "NEUTRAL",
                            "confidence": 0.5,
                            "summary": result_text[:500],
                            "key_factors": [],
                            "recommendation": "분석 결과를 확인하세요."
                        }
                except:
                    self.analysis_result = {
                        "sentiment": "NEUTRAL",
                        "confidence": 0.5,
                        "summary": result_text[:500],
                        "key_factors": [],
                        "recommendation": "분석 결과를 확인하세요."
                    }
            else:
                self.analysis_result = {
                    "sentiment": "NEUTRAL",
                    "confidence": 0.5,
                    "summary": "LLM 분석 실패",
                    "key_factors": [],
                    "recommendation": "신중하게 접근하세요."
                }

        except Exception as e:
            logger.error(f"시황분석 오류: {e}")
            self.analysis_result = {
                "sentiment": "NEUTRAL",
                "confidence": 0.5,
                "summary": f"분석 오류: {str(e)}",
                "key_factors": [],
                "recommendation": "신중하게 접근하세요."
            }

        logger.info(f"시황분석 완료: {self.analysis_result.get('sentiment', 'N/A')}")
        return self.analysis_result


def print_banner():
    """배너 출력"""
    print("""
================================================================
       KIS Real Auto Trading System
================================================================
   [Market Analysis] -> [Scalping(09:00~09:30)] -> [Trading(~15:20)]

   All trading data will be saved automatically
================================================================
    """)


def print_market_analysis(analysis: Dict):
    """시황분석 결과 출력"""
    sentiment = analysis.get('sentiment', 'NEUTRAL')

    print("\n" + "=" * 60)
    print("              Market Analysis Result")
    print("=" * 60)
    print(f"  Sentiment: {sentiment}")
    print(f"  Confidence: {analysis.get('confidence', 0):.0%}")
    print(f"\n  Summary: {analysis.get('summary', 'N/A')}")
    print(f"\n  Key Factors:")
    for factor in analysis.get('key_factors', []):
        print(f"    - {factor}")
    print(f"\n  Strategy: {analysis.get('recommendation', 'N/A')}")
    print("=" * 60)


def run_real_trading():
    """실전 매매 실행"""
    import yaml

    # 설정 파일 로드
    config_path = SCRIPT_DIR / "config" / "auto_trade.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 배너 출력
    print_banner()

    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"데이터 저장 경로: {DATA_DIR}")
    print(f"로그 저장 경로: {LOG_DIR}")

    # KIS API 인증
    print("\nKIS API 인증 중...")
    import kis_auth as ka
    ka.auth()
    print("KIS API 인증 성공")

    # 데이터 저장 초기화
    data_saver = TradingDataSaver(DATA_DIR)

    # ========== 시황분석 ==========
    print("\n" + "=" * 60)
    print("         아침 시황분석 시작")
    print("=" * 60)

    market_analyzer = MarketAnalyzer()

    # 야간 뉴스 수집
    overnight_news = market_analyzer.collect_overnight_news()
    print(f"수집된 야간 뉴스: {len(overnight_news)}건")

    # 시황분석 수행
    market_analysis = market_analyzer.analyze_market_sentiment()

    # 시황분석 결과 저장
    data_saver.save_market_analysis({
        "news_count": len(overnight_news),
        "news_list": market_analyzer.news_list[:50],  # 최대 50개 저장
        "analysis": market_analysis
    })

    # 시황분석 결과 출력
    print_market_analysis(market_analysis)

    # ========== 실전 매매 확인 ==========
    print("\n" + "=" * 60)
    print("         실전 매매 확인")
    print("=" * 60)

    order_cfg = config.get('order', {})
    risk_cfg = config.get('risk', {})

    print(f"  환경: 실전투자")
    print(f"  주문 한도: {order_cfg.get('max_order_amount', 100000):,}원")
    print(f"  일일 최대 거래: {risk_cfg.get('max_daily_trades', 10)}회")
    print(f"  일일 최대 손실: {risk_cfg.get('max_daily_loss', 50000):,}원")

    # 자동 시작 모드 (사용자가 이미 승인함)
    print("\n실전 매매를 시작합니다...")
    # confirm = input("\n실전 매매를 시작하시겠습니까? (yes/no): ").strip().lower()
    # if confirm != 'yes':
    #     print("취소되었습니다.")
    #     return

    # ========== AutoTrader 초기화 ==========
    from modules.auto_trader import AutoTrader, AutoTradeConfig

    threshold_cfg = config.get('threshold', {})
    scan_cfg = config.get('scan', {})
    market_cfg = config.get('market', {})
    scalping_cfg = config.get('scalping', {})
    ensemble_cfg = config.get('ensemble', {})

    trader_config = AutoTradeConfig(
        env_dv="real",
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

    print("\nAutoTrader 초기화 중...")
    trader = AutoTrader(trader_config)

    # 상태 확인
    status = trader.get_status()
    print(f"앙상블 모델: {', '.join(status.get('ensemble_models', []))}")
    print(f"메인 모델: {status.get('main_model', 'N/A')}")

    # ========== 폴링 매매 시작 ==========
    scan_interval = scan_cfg.get('scan_interval', 60)

    print(f"\n{'=' * 60}")
    print(f"실전 매매 시작 (주기: {scan_interval}초)")
    print("Ctrl+C로 종료")
    print(f"{'=' * 60}")

    total_trades = 0
    successful_trades = 0

    try:
        while True:
            now = datetime.now()

            # 장 시간 체크
            if now.hour < 9 or (now.hour >= 15 and now.minute > 30):
                if now.hour >= 15 and now.minute > 30:
                    print(f"\n[{now.strftime('%H:%M:%S')}] 장 마감 - 매매 종료")
                    break
                else:
                    wait_seconds = (datetime(now.year, now.month, now.day, 9, 0) - now).seconds
                    print(f"\n[{now.strftime('%H:%M:%S')}] 장 시작 대기 중... ({wait_seconds}초 후)")
                    time.sleep(min(wait_seconds, 60))
                    continue

            print(f"\n[{now.strftime('%H:%M:%S')}] 스캔 시작...")

            # 스캘핑 시간 확인 (09:00 ~ 09:30)
            if trader._is_scalping_time():
                print("스캘핑 모드 활성화 (09:00~09:30)")
                results = trader.run_scalping_trade()
            else:
                results = trader.run_scan_and_trade(check_market_hours=True)

            # 결과 저장 및 출력
            if results:
                for result in results:
                    data_saver.save_trade(result)
                    total_trades += 1
                    if result.success:
                        successful_trades += 1

                    # 결과 출력
                    status_icon = "O" if result.success else "X"
                    print(f"  [{status_icon}] {result.stock_name}({result.stock_code})")
                    print(f"      시그널: {result.ensemble_signal} | 신뢰도: {result.confidence:.0%}")
                    print(f"      행동: {result.action} - {result.reason}")
                    if result.success:
                        print(f"      주문: {result.order_qty}주 @ {result.order_price:,}원")

                print(f"\n  스캔 결과: {len(results)}개 분석, {sum(1 for r in results if r.success)}개 매매")
            else:
                print("  분석 결과 없음")

            print(f"\n  [누적] 총 분석: {total_trades}개, 매매 성공: {successful_trades}개")
            print(f"  다음 스캔까지 {scan_interval}초 대기...")
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 종료됨")

    # ========== 일일 요약 저장 ==========
    daily_summary = {
        "date": TODAY,
        "market_analysis": market_analysis,
        "total_scans": total_trades,
        "successful_trades": successful_trades,
        "trades": data_saver.trades,
        "end_time": datetime.now().isoformat()
    }
    data_saver.save_daily_summary(daily_summary)

    print("\n" + "=" * 60)
    print("              일일 매매 요약")
    print("=" * 60)
    print(f"  날짜: {TODAY}")
    print(f"  시장 분위기: {market_analysis.get('sentiment', 'N/A')}")
    print(f"  총 분석: {total_trades}건")
    print(f"  매매 성공: {successful_trades}건")
    print(f"  데이터 저장 위치:")
    print(f"    - 거래: {data_saver.trades_file}")
    print(f"    - 시황분석: {data_saver.analysis_file}")
    print(f"    - 일일요약: {data_saver.summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_real_trading()
