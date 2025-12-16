# -*- coding: utf-8 -*-
"""
핫한 주식 보고서 생성기

KIS Open API 뉴스 + 현재가 + LLM 분석을 통합하여
투자 관심 종목에 대한 보고서를 생성합니다.
"""

import os
import sys
import io
import logging
from datetime import datetime

# UTF-8 출력 설정 (Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples_llm'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_ollama_server():
    """Ollama 서버 상태 확인"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, [m["name"] for m in models]
    except Exception as e:
        return False, str(e)
    return False, "Unknown error"


def generate_report():
    """핫한 주식 보고서 생성"""

    print("\n" + "=" * 70)
    print("      HOT STOCK REPORT - KIS API + LLM Analysis")
    print(f"      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. 환경 설정 로드
    print("\n[1/5] Environment Setup...")
    try:
        from modules.config_loader import setup_kis_config, load_env_file
        load_env_file()
        setup_kis_config()
        print("      [OK] Config loaded")
    except Exception as e:
        print(f"      [FAIL] Config error: {e}")
        return

    # 2. KIS API 인증
    print("\n[2/5] KIS API Authentication...")
    try:
        import kis_auth as ka
        ka.auth(svr="prod")
        print("      [OK] Authenticated")
    except Exception as e:
        print(f"      [FAIL] Auth error: {e}")
        return

    # 3. 주요 종목 현재가 조회
    print("\n[3/5] Fetching Stock Prices...")
    watch_list = {
        "005930": "Samsung Electronics",
        "000660": "SK Hynix",
        "035720": "Kakao",
        "035420": "NAVER",
        "051910": "LG Chem",
        "006400": "Samsung SDI",
        "003670": "Posco Future M",
        "207940": "Samsung Biologics",
        "068270": "Celltrion",
        "005380": "Hyundai Motor"
    }

    try:
        from modules.price_checker import PriceChecker
        checker = PriceChecker(env_dv="prod")

        stock_data = {}
        for code, name in watch_list.items():
            try:
                price = checker.get_price(code)
                if price:
                    stock_data[code] = {
                        "name": price.stock_name or name,
                        "price": price.current_price,
                        "change_rate": price.change_rate,
                        "volume": price.volume,
                        "per": price.per,
                        "pbr": price.pbr,
                        "week52_high": price.week52_high,
                        "week52_low": price.week52_low,
                    }
                    print(f"      {code} {price.stock_name or name}: {price.current_price:,}won ({price.change_rate:+.2f}%)")
            except Exception as e:
                logger.debug(f"Price fetch error for {code}: {e}")
                continue

        print(f"      [OK] Fetched {len(stock_data)} stocks")
    except Exception as e:
        print(f"      [FAIL] Price fetch error: {e}")
        stock_data = {}

    # 4. 뉴스 수집
    print("\n[4/5] Collecting News...")
    try:
        from domestic_stock.news_title.news_title import news_title

        news_df = news_title(
            fid_news_ofer_entp_code="",
            fid_cond_mrkt_cls_code="",
            fid_input_iscd="",
            fid_titl_cntt="",
            fid_input_date_1="",
            fid_input_hour_1="",
            fid_rank_sort_cls_code="",
            fid_input_srno="",
            max_depth=2
        )

        if news_df is not None and not news_df.empty:
            print(f"      [OK] Collected {len(news_df)} news articles")

            # 뉴스 미리보기
            print("\n      Recent Headlines:")
            for idx, row in news_df.head(10).iterrows():
                title = row.get('titl', '')[:60]
                time_str = row.get('data_tm', '')[:6]
                if time_str:
                    time_str = f"{time_str[:2]}:{time_str[2:4]}"
                print(f"        [{time_str}] {title}...")
        else:
            print("      [WARN] No news collected")
            news_df = None
    except Exception as e:
        print(f"      [FAIL] News collection error: {e}")
        news_df = None

    # 5. LLM 분석
    print("\n[5/5] LLM Analysis...")

    # Ollama 서버 확인
    is_ollama_running, ollama_info = check_ollama_server()

    if not is_ollama_running:
        print(f"      [WARN] Ollama server not running: {ollama_info}")
        print("      Skipping LLM analysis. Start Ollama with: ollama serve")
        llm_results = None
    else:
        print(f"      Available models: {', '.join(ollama_info[:5])}...")

        try:
            from modules.llm import FinancialHybridLLM

            llm = FinancialHybridLLM(
                api_url="http://localhost:11434",
                enable_parallel=True,
                timeout=180
            )

            # 뉴스 분석 (최대 10개)
            if news_df is not None and not news_df.empty:
                sample_news = news_df.head(10).copy()
                llm_results = llm.analyze_kis_news(sample_news)
                print(f"      [OK] Analyzed {len(llm_results)} news articles")
            else:
                # 뉴스가 없으면 주요 종목에 대한 가상 분석
                llm_results = []
                print("      [WARN] No news to analyze")

        except ConnectionError as e:
            print(f"      [WARN] LLM connection error: {e}")
            llm_results = None
        except Exception as e:
            print(f"      [FAIL] LLM analysis error: {e}")
            llm_results = None

    # ===== 보고서 생성 =====
    print("\n")
    print("=" * 70)
    print("                        REPORT SUMMARY")
    print("=" * 70)

    # 상승률 상위 종목
    print("\n[Top Gainers Today]")
    print("-" * 50)
    if stock_data:
        sorted_by_gain = sorted(
            stock_data.items(),
            key=lambda x: x[1].get('change_rate', 0),
            reverse=True
        )

        for i, (code, data) in enumerate(sorted_by_gain[:5], 1):
            name = data.get('name', code)
            price = data.get('price', 0)
            change = data.get('change_rate', 0)
            volume = data.get('volume', 0)

            trend = "UP" if change > 0 else ("DOWN" if change < 0 else "FLAT")
            print(f"  {i}. {name} ({code})")
            print(f"     Price: {price:,}won  Change: {change:+.2f}%  [{trend}]")
            print(f"     Volume: {volume:,}")
            print()
    else:
        print("  No stock data available")

    # 하락률 상위 종목
    print("\n[Top Losers Today]")
    print("-" * 50)
    if stock_data:
        sorted_by_loss = sorted(
            stock_data.items(),
            key=lambda x: x[1].get('change_rate', 0)
        )

        for i, (code, data) in enumerate(sorted_by_loss[:3], 1):
            if data.get('change_rate', 0) >= 0:
                continue
            name = data.get('name', code)
            price = data.get('price', 0)
            change = data.get('change_rate', 0)

            print(f"  {i}. {name} ({code})")
            print(f"     Price: {price:,}won  Change: {change:+.2f}%")
            print()

    # LLM 분석 결과
    if llm_results:
        print("\n[LLM News Analysis Results]")
        print("-" * 50)

        # 추천별 분류
        buy_signals = [r for r in llm_results if 'BUY' in r.get('recommendation', '')]
        sell_signals = [r for r in llm_results if 'SELL' in r.get('recommendation', '')]

        if buy_signals:
            print("\n  ** BUY Signals **")
            for r in buy_signals[:5]:
                title = r.get('news_title', '')[:50]
                rec = r.get('recommendation', '')
                conf = r.get('confidence', 0)
                sentiment = r.get('sentiment', '')
                print(f"    [{rec}] {title}...")
                print(f"          Sentiment: {sentiment}, Confidence: {conf:.1%}")
                print(f"          Reason: {r.get('reasoning', '')[:80]}...")
                print()

        if sell_signals:
            print("\n  ** SELL Signals **")
            for r in sell_signals[:3]:
                title = r.get('news_title', '')[:50]
                rec = r.get('recommendation', '')
                conf = r.get('confidence', 0)
                print(f"    [{rec}] {title}...")
                print(f"          Confidence: {conf:.1%}")
                print()

        # 감성 분포
        sentiments = [r.get('sentiment', 'neutral') for r in llm_results]
        print(f"\n  Sentiment Distribution:")
        print(f"    Positive: {sentiments.count('positive')}")
        print(f"    Negative: {sentiments.count('negative')}")
        print(f"    Neutral: {sentiments.count('neutral')}")

    # 밸류에이션 분석
    print("\n[Valuation Analysis]")
    print("-" * 50)
    if stock_data:
        print(f"  {'Stock':<20} {'PER':>8} {'PBR':>8} {'52W Position':>15}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*15}")

        for code, data in stock_data.items():
            name = data.get('name', code)[:18]
            per = data.get('per', 0)
            pbr = data.get('pbr', 0)

            # 52주 위치 계산
            current = data.get('price', 0)
            high52 = data.get('week52_high', 0)
            low52 = data.get('week52_low', 0)

            if high52 > low52:
                position = (current - low52) / (high52 - low52) * 100
                pos_str = f"{position:.0f}%"
            else:
                pos_str = "N/A"

            print(f"  {name:<20} {per:>8.1f} {pbr:>8.2f} {pos_str:>15}")

    # 결론
    print("\n" + "=" * 70)
    print("                          CONCLUSION")
    print("=" * 70)

    if stock_data:
        # 가장 상승한 종목
        top_gainer = sorted_by_gain[0] if sorted_by_gain else None
        if top_gainer:
            code, data = top_gainer
            print(f"\n  Today's Hot Stock: {data['name']} ({code})")
            print(f"  - Change: {data['change_rate']:+.2f}%")
            print(f"  - Current Price: {data['price']:,}won")

    if llm_results:
        buy_count = len([r for r in llm_results if 'BUY' in r.get('recommendation', '')])
        sell_count = len([r for r in llm_results if 'SELL' in r.get('recommendation', '')])

        print(f"\n  Market Sentiment from News:")
        if buy_count > sell_count:
            print(f"  - Overall: BULLISH (Buy signals: {buy_count}, Sell signals: {sell_count})")
        elif sell_count > buy_count:
            print(f"  - Overall: BEARISH (Buy signals: {buy_count}, Sell signals: {sell_count})")
        else:
            print(f"  - Overall: NEUTRAL (Buy signals: {buy_count}, Sell signals: {sell_count})")

    print("\n" + "=" * 70)
    print("  DISCLAIMER: This report is for informational purposes only.")
    print("  Not financial advice. Always do your own research.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    generate_report()
