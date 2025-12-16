# -*- coding: utf-8 -*-
"""
핫한 주식 보고서 생성기 (Final Version)

KIS Open API 뉴스 + 현재가 + LLM 분석 통합
"""

import os
import sys
import io
import json
import logging
import requests
import time
from datetime import datetime

# UTF-8 출력 설정 (Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples_llm'))

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)


def get_available_model():
    """사용 가능한 LLM 모델 선택"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            # 선호 모델 순서 (DeepSeek-R1 금융 추론 최우선)
            preferred = ["deepseek-r1:8b", "deepseek-r1", "qwen3:8b", "qwen3", "fin-r1", "qwen2.5", "llama"]

            for pref in preferred:
                for name in model_names:
                    if pref in name.lower():
                        return name
            if model_names:
                return model_names[0]
    except Exception:
        pass
    return None


def analyze_news_with_llm(news_title: str, model_name: str) -> dict:
    """단일 LLM으로 뉴스 분석"""
    prompt = f"""당신은 한국 증권시장 전문 애널리스트입니다.
다음 뉴스 제목을 읽고 주가에 미칠 영향을 분석해주세요.

[뉴스 제목]
{news_title}

다음 JSON 형식으로만 응답하세요:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0~1.0,
    "impact": "high|medium|low",
    "reasoning": "판단 근거를 한글로 1문장으로 설명",
    "recommendation": "BUY|SELL|HOLD"
}}

JSON만 출력하세요. 다른 텍스트 없이 JSON만."""

    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 250,
                "num_ctx": 1024
            }
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=90
        )

        if response.status_code == 200:
            result_text = response.json().get("response", "").strip()

            # JSON 추출
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result_text = result_text[start:end]

            parsed = json.loads(result_text.strip())

            # 유효성 검증
            if parsed.get("sentiment") not in ["positive", "negative", "neutral"]:
                parsed["sentiment"] = "neutral"
            if parsed.get("recommendation") not in ["BUY", "SELL", "HOLD"]:
                parsed["recommendation"] = "HOLD"
            if parsed.get("impact") not in ["high", "medium", "low"]:
                parsed["impact"] = "low"

            return parsed
    except json.JSONDecodeError:
        pass
    except Exception as e:
        logger.debug(f"LLM error: {e}")
    return None


def format_time(time_str: str) -> str:
    """시간 문자열 포맷팅"""
    if len(time_str) >= 4:
        return f"{time_str[:2]}:{time_str[2:4]}"
    return time_str


def generate_report():
    """핫한 주식 보고서 생성"""

    print("\n" + "=" * 70)
    print("          HOT STOCK REPORT - KIS API + LLM Analysis")
    print(f"          Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
                    sign = "+" if price.change_rate >= 0 else ""
                    display_name = (price.stock_name or name)[:15]
                    print(f"      {code} {display_name}: {price.current_price:,}won ({sign}{price.change_rate:.2f}%)")
            except Exception:
                continue

        print(f"      [OK] Fetched {len(stock_data)} stocks")
    except Exception as e:
        print(f"      [FAIL] Price fetch error: {e}")
        stock_data = {}

    # 4. 뉴스 수집
    print("\n[4/5] Collecting News...")
    news_list = []
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

            # 뉴스 리스트 추출 (실제 컬럼명 사용)
            for idx, row in news_df.head(20).iterrows():
                # hts_pbnt_titl_cntt가 실제 뉴스 제목 컬럼
                title = row.get('hts_pbnt_titl_cntt', '') or row.get('titl', '')
                time_str = str(row.get('data_tm', ''))[:6]
                stock_code = row.get('iscd1', '')
                stock_name = row.get('kor_isnm1', '')
                source = row.get('dorg', '')

                if title and len(title.strip()) > 5:
                    news_list.append({
                        "title": title.strip(),
                        "time": format_time(time_str),
                        "date": row.get('data_dt', ''),
                        "stock_code": stock_code,
                        "stock_name": stock_name,
                        "source": source,
                    })

            # 뉴스 미리보기
            print("\n      Recent Headlines:")
            for news in news_list[:10]:
                src = f"[{news['source']}]" if news['source'] else ""
                print(f"        [{news['time']}] {news['title'][:50]}... {src}")
        else:
            print("      [WARN] No news collected")
    except Exception as e:
        print(f"      [FAIL] News collection error: {e}")

    # 5. LLM 분석
    print("\n[5/5] LLM Analysis...")
    model_name = get_available_model()

    llm_results = []
    if not model_name:
        print("      [WARN] No LLM model available. Skipping analysis.")
        print("      To enable LLM analysis, run: ollama serve")
    elif not news_list:
        print("      [WARN] No news to analyze.")
    else:
        print(f"      Using model: {model_name}")
        print()

        # 증권 관련 뉴스만 필터링 (일반 뉴스 제외)
        finance_keywords = ['코스피', '코스닥', '주가', '상승', '하락', '급등', '급락',
                           '실적', '영업이익', '매출', '투자', '수주', '계약',
                           '반도체', '배터리', '전기차', '바이오', 'HBM', 'AI',
                           '외국인', '기관', '개인', '수급', '시총']

        filtered_news = []
        for news in news_list:
            title = news['title']
            if any(kw in title for kw in finance_keywords) or news.get('stock_code'):
                filtered_news.append(news)

        analyze_count = min(8, len(filtered_news))
        print(f"      Analyzing {analyze_count} finance-related news...")
        print()

        for i, news in enumerate(filtered_news[:analyze_count], 1):
            title = news['title']
            short_title = title[:35] + "..." if len(title) > 35 else title
            print(f"      ({i}/{analyze_count}) {short_title}")

            result = analyze_news_with_llm(title, model_name)
            if result:
                result['news_title'] = title
                result['news_time'] = news['time']
                result['stock_code'] = news.get('stock_code', '')
                result['stock_name'] = news.get('stock_name', '')
                llm_results.append(result)

                # 결과 미리보기
                rec = result.get('recommendation', 'HOLD')
                sent = result.get('sentiment', 'neutral')
                print(f"               -> {rec} ({sent})")
            else:
                print(f"               -> Analysis failed")

            time.sleep(0.3)  # API 부하 방지

        print(f"\n      [OK] Successfully analyzed {len(llm_results)} articles")

    # ===== 보고서 생성 =====
    print("\n")
    print("=" * 70)
    print("                        REPORT SUMMARY")
    print("=" * 70)

    # 상승률 상위 종목
    print("\n[Top Gainers Today]")
    print("-" * 60)
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

            trend = "[UP]" if change > 0 else ("[DOWN]" if change < 0 else "[FLAT]")
            print(f"  {i}. {name} ({code})")
            print(f"     Price: {price:,}won  Change: {change:+.2f}%  {trend}")
            print(f"     Volume: {volume:,}")
            print()
    else:
        print("  No stock data available")

    # 하락 종목
    if stock_data:
        losers = [(c, d) for c, d in stock_data.items() if d.get('change_rate', 0) < 0]
        if losers:
            print("\n[Stocks in Decline]")
            print("-" * 60)
            losers.sort(key=lambda x: x[1].get('change_rate', 0))
            for i, (code, data) in enumerate(losers[:3], 1):
                name = data.get('name', code)
                price = data.get('price', 0)
                change = data.get('change_rate', 0)
                print(f"  {i}. {name} ({code})")
                print(f"     Price: {price:,}won  Change: {change:+.2f}%")
                print()

    # LLM 분석 결과
    if llm_results:
        print("\n[LLM News Analysis Results]")
        print("-" * 60)

        # 추천별 분류
        buy_signals = [r for r in llm_results if r.get('recommendation') == 'BUY']
        sell_signals = [r for r in llm_results if r.get('recommendation') == 'SELL']
        hold_signals = [r for r in llm_results if r.get('recommendation') == 'HOLD']

        if buy_signals:
            print("\n  ** BUY Signals **")
            for r in buy_signals[:5]:
                title = r.get('news_title', '')[:45]
                conf = r.get('confidence', 0)
                impact = r.get('impact', 'low')
                reasoning = r.get('reasoning', '')[:60]
                stock = f"({r.get('stock_name', '')})" if r.get('stock_name') else ""
                print(f"    [BUY] {title}... {stock}")
                print(f"          Confidence: {conf:.0%}, Impact: {impact}")
                print(f"          Reason: {reasoning}...")
                print()

        if sell_signals:
            print("\n  ** SELL Signals **")
            for r in sell_signals[:4]:
                title = r.get('news_title', '')[:45]
                conf = r.get('confidence', 0)
                reasoning = r.get('reasoning', '')[:60]
                print(f"    [SELL] {title}...")
                print(f"          Confidence: {conf:.0%}")
                print(f"          Reason: {reasoning}...")
                print()

        if hold_signals and not buy_signals and not sell_signals:
            print("\n  ** HOLD Signals (Neutral Market) **")
            for r in hold_signals[:3]:
                title = r.get('news_title', '')[:45]
                print(f"    [HOLD] {title}...")

        # 감성 분포
        sentiments = [r.get('sentiment', 'neutral') for r in llm_results]
        print(f"\n  Analysis Summary:")
        print(f"    Sentiment: Positive({sentiments.count('positive')}) / Negative({sentiments.count('negative')}) / Neutral({sentiments.count('neutral')})")
        print(f"    Signals:   BUY({len(buy_signals)}) / SELL({len(sell_signals)}) / HOLD({len(hold_signals)})")

    # 밸류에이션 분석
    print("\n[Valuation Analysis]")
    print("-" * 60)
    if stock_data:
        print(f"  {'Stock':<18} {'Price':>12} {'Change':>8} {'PER':>8} {'52W':>8}")
        print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

        for code, data in sorted(stock_data.items(), key=lambda x: x[1].get('change_rate', 0), reverse=True):
            name = data.get('name', code)[:16]
            price = data.get('price', 0)
            change = data.get('change_rate', 0)
            per = data.get('per', 0)

            # 52주 위치 계산
            high52 = data.get('week52_high', 0)
            low52 = data.get('week52_low', 0)
            if high52 > low52:
                position = (price - low52) / (high52 - low52) * 100
                pos_str = f"{position:.0f}%"
            else:
                pos_str = "N/A"

            print(f"  {name:<18} {price:>12,} {change:>+7.2f}% {per:>8.1f} {pos_str:>8}")

    # 결론
    print("\n" + "=" * 70)
    print("                          CONCLUSION")
    print("=" * 70)

    if stock_data:
        sorted_by_gain = sorted(
            stock_data.items(),
            key=lambda x: x[1].get('change_rate', 0),
            reverse=True
        )

        top_gainer = sorted_by_gain[0] if sorted_by_gain else None
        if top_gainer:
            code, data = top_gainer
            print(f"\n  Today's Hot Stock: {data['name']} ({code})")
            print(f"  - Change: {data['change_rate']:+.2f}%")
            print(f"  - Current Price: {data['price']:,}won")
            print(f"  - Trading Volume: {data['volume']:,}")

            # 52주 위치
            high52 = data.get('week52_high', 0)
            low52 = data.get('week52_low', 0)
            if high52 > low52:
                position = (data['price'] - low52) / (high52 - low52) * 100
                print(f"  - 52-Week Position: {position:.0f}%")

    if llm_results:
        buy_count = len([r for r in llm_results if r.get('recommendation') == 'BUY'])
        sell_count = len([r for r in llm_results if r.get('recommendation') == 'SELL'])

        print(f"\n  Market Sentiment from News Analysis:")
        total = len(llm_results)
        if total > 0:
            if buy_count > sell_count * 1.5:
                sentiment_str = "BULLISH (Positive bias in news)"
                emoji = "[+]"
            elif sell_count > buy_count * 1.5:
                sentiment_str = "BEARISH (Negative bias in news)"
                emoji = "[-]"
            else:
                sentiment_str = "NEUTRAL/MIXED"
                emoji = "[=]"

            print(f"  {emoji} {sentiment_str}")
            print(f"      (Based on {total} analyzed articles)")

    print("\n" + "=" * 70)
    print("  DISCLAIMER: This report is for informational purposes only.")
    print("  This is NOT financial advice. Always do your own research (DYOR).")
    print("  Past performance does not guarantee future results.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    generate_report()
