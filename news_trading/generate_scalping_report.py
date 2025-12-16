# -*- coding: utf-8 -*-
"""
스캘핑 분석 보고서 생성기

OHLCV 시계열 데이터 + 기술적 지표 + 뉴스 + LLM 분석을 통합하여
스캘핑에 최적화된 분석 보고서를 생성합니다.
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


def analyze_with_llm(prompt: str, model_name: str) -> dict:
    """LLM으로 스캘핑 분석"""
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 500,
                "num_ctx": 4096
            }
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result_text = response.json().get("response", "").strip()

            # JSON 추출
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                parts = result_text.split("```")
                if len(parts) >= 2:
                    result_text = parts[1]

            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result_text = result_text[start:end]

            return json.loads(result_text.strip())
    except json.JSONDecodeError:
        pass
    except Exception as e:
        logger.debug(f"LLM error: {e}")
    return None


def generate_report():
    """스캘핑 분석 보고서 생성"""

    print("\n" + "=" * 75)
    print("         SCALPING ANALYSIS REPORT - Technical + News + LLM")
    print(f"         Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # 1. 환경 설정
    print("\n[1/6] Environment Setup...")
    try:
        from modules.config_loader import setup_kis_config, load_env_file
        load_env_file()
        setup_kis_config()
        print("      [OK] Config loaded")
    except Exception as e:
        print(f"      [FAIL] Config error: {e}")
        return

    # 2. KIS API 인증
    print("\n[2/6] KIS API Authentication...")
    try:
        import kis_auth as ka
        ka.auth(svr="prod")
        print("      [OK] Authenticated")
    except Exception as e:
        print(f"      [FAIL] Auth error: {e}")
        return

    # 분석할 종목 (스캘핑용 주요 대형주)
    target_stocks = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "003670": "포스코퓨처엠",
    }

    # 3. OHLCV 데이터 조회
    print("\n[3/6] Fetching OHLCV Data...")
    try:
        from modules.ohlcv_fetcher import OHLCVFetcher
        fetcher = OHLCVFetcher(env_dv="prod")

        ohlcv_data = {}
        for code, name in target_stocks.items():
            print(f"      Fetching {name} ({code})...")
            daily_df = fetcher.get_daily_data(code, days=30)
            minute_df = fetcher.get_minute_data(code)

            if not daily_df.empty:
                ohlcv_data[code] = {
                    'name': name,
                    'daily': daily_df,
                    'minute': minute_df
                }
                print(f"        - Daily: {len(daily_df)} candles, Minute: {len(minute_df)} candles")

        print(f"      [OK] Fetched data for {len(ohlcv_data)} stocks")
    except Exception as e:
        print(f"      [FAIL] OHLCV fetch error: {e}")
        import traceback
        traceback.print_exc()
        ohlcv_data = {}

    # 4. 기술적 지표 계산
    print("\n[4/6] Calculating Technical Indicators...")
    try:
        from modules.technical_indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()

        technical_data = {}
        for code, data in ohlcv_data.items():
            name = data['name']
            df = data['daily']

            if df.empty or len(df) < 20:
                print(f"      {name}: 데이터 부족 (최소 20개 필요)")
                continue

            print(f"      Analyzing {name}...")
            try:
                summary = analyzer.get_scalping_summary(df)
                llm_text = analyzer.format_for_llm(df, name)

                technical_data[code] = {
                    'name': name,
                    'summary': summary,
                    'llm_text': llm_text
                }

                # 간단한 결과 출력
                trend = summary.get('trend', 'NEUTRAL')
                score = summary.get('total_score', 0)
                rsi = summary.get('rsi_14', 50)
                print(f"        - Trend: {trend}, Score: {score:+.1f}, RSI: {rsi:.1f}")
            except Exception as e:
                print(f"        - Error: {e}")
                continue

        print(f"      [OK] Analyzed {len(technical_data)} stocks")
    except Exception as e:
        print(f"      [FAIL] Technical analysis error: {e}")
        import traceback
        traceback.print_exc()
        technical_data = {}

    # 5. 뉴스 수집
    print("\n[5/6] Collecting Related News...")
    news_data = {}
    try:
        from domestic_stock.news_title.news_title import news_title

        for code, name in target_stocks.items():
            news_df = news_title(
                fid_news_ofer_entp_code="",
                fid_cond_mrkt_cls_code="",
                fid_input_iscd=code,
                fid_titl_cntt="",
                fid_input_date_1="",
                fid_input_hour_1="",
                fid_rank_sort_cls_code="",
                fid_input_srno="",
                max_depth=1
            )

            if news_df is not None and not news_df.empty:
                titles = news_df['hts_pbnt_titl_cntt'].head(5).tolist()
                news_data[code] = titles
                print(f"      {name}: {len(titles)} news")

        print(f"      [OK] Collected news for {len(news_data)} stocks")
    except Exception as e:
        print(f"      [WARN] News collection error: {e}")

    # 6. LLM 종합 분석
    print("\n[6/6] LLM Comprehensive Analysis...")
    model_name = get_available_model()
    llm_results = {}

    if not model_name:
        print("      [WARN] No LLM available. Skipping.")
    elif not technical_data:
        print("      [WARN] No technical data to analyze.")
    else:
        print(f"      Using model: {model_name}")

        for code, tech_data in technical_data.items():
            name = tech_data['name']
            summary = tech_data['summary']
            llm_text = tech_data['llm_text']

            # 뉴스 텍스트
            news_text = ""
            if code in news_data:
                news_text = "\n[관련 뉴스]\n" + "\n".join([f"- {t}" for t in news_data[code][:3]])

            prompt = f"""당신은 전문 스캘핑 트레이더입니다.
아래의 기술적 분석 데이터와 뉴스를 바탕으로 스캘핑 매매 전략을 제시하세요.

{llm_text}
{news_text}

다음 JSON 형식으로만 응답하세요:
{{
    "signal": "BUY|SELL|HOLD",
    "confidence": 0.0~1.0,
    "entry_price": 진입가 (숫자),
    "stop_loss": 손절가 (숫자),
    "take_profit": 익절가 (숫자),
    "risk_reward": 손익비 (숫자),
    "timeframe": "추천 보유 시간 (예: 5분, 30분, 1시간)",
    "key_levels": "주요 지지/저항 레벨 설명",
    "entry_condition": "진입 조건 설명",
    "exit_condition": "청산 조건 설명",
    "risk_warning": "리스크 경고"
}}

JSON만 출력하세요."""

            print(f"      Analyzing {name}...")
            result = analyze_with_llm(prompt, model_name)

            if result:
                llm_results[code] = result
                signal = result.get('signal', 'HOLD')
                conf = result.get('confidence', 0)
                print(f"        -> {signal} (confidence: {conf:.0%})")
            else:
                print(f"        -> Analysis failed")

            time.sleep(0.5)

    # ===== 보고서 출력 =====
    print("\n")
    print("=" * 75)
    print("                     SCALPING ANALYSIS RESULTS")
    print("=" * 75)

    for code, tech_data in technical_data.items():
        name = tech_data['name']
        summary = tech_data['summary']

        print(f"\n{'='*75}")
        print(f"  {name} ({code})")
        print(f"{'='*75}")

        print(f"\n  [Current Status]")
        print(f"  - Price: {summary['current_price']:,.0f}원")
        print(f"  - Trend: {summary['trend']} | Volatility: {summary['volatility']}")
        print(f"  - Score: {summary['total_score']:+.1f} / 100")

        print(f"\n  [Momentum Indicators]")
        print(f"  - RSI(14): {summary['rsi_14']:.1f} | RSI(7): {summary['rsi_7']:.1f}")
        print(f"  - Stochastic %K: {summary['stoch_k']:.1f} | %D: {summary['stoch_d']:.1f}")
        print(f"  - MACD: {summary['macd']:.2f} | Signal: {summary['macd_signal']:.2f}")

        print(f"\n  [Price Levels]")
        print(f"  - Bollinger: Upper {summary['bb_upper']:,.0f} | Lower {summary['bb_lower']:,.0f}")
        print(f"  - %B: {summary['bb_percent_b']:.2f} (0=하단, 1=상단)")
        print(f"  - VWAP: {summary['vwap']:,.0f}원")
        print(f"  - Pivot: {summary['pivot']:,.0f} | R1: {summary['resistance_1']:,.0f} | S1: {summary['support_1']:,.0f}")

        print(f"\n  [Volatility & Volume]")
        print(f"  - ATR(14): {summary['atr_14']:,.0f}원 ({summary['atr_percent']:.2f}%)")
        print(f"  - Volume Ratio: {summary['volume_ratio']:.2f}x")

        print(f"\n  [Technical Signals]")
        for sig in summary['signals']:
            direction_kr = "매수" if sig['direction'] == "BUY" else "매도"
            bar = "+" * int(sig['strength'] * 10)
            print(f"    [{direction_kr}] {sig['reason']} ({sig['strength']:.1f}) {bar}")

        print(f"\n  [Recommended Levels]")
        print(f"  - Stop Loss: {summary['recommended_stop_loss']:,.0f}원")
        print(f"  - Take Profit: {summary['recommended_take_profit']:,.0f}원")

        # LLM 분석 결과
        if code in llm_results:
            llm = llm_results[code]
            print(f"\n  [LLM Analysis]")
            print(f"  - Signal: {llm.get('signal', 'N/A')} (Confidence: {llm.get('confidence', 0):.0%})")
            print(f"  - Entry: {llm.get('entry_price', 0):,.0f}원")
            print(f"  - Stop Loss: {llm.get('stop_loss', 0):,.0f}원")
            print(f"  - Take Profit: {llm.get('take_profit', 0):,.0f}원")
            print(f"  - Risk/Reward: {llm.get('risk_reward', 0):.2f}")
            print(f"  - Timeframe: {llm.get('timeframe', 'N/A')}")
            print(f"  - Entry Condition: {llm.get('entry_condition', 'N/A')[:60]}...")
            print(f"  - Risk Warning: {llm.get('risk_warning', 'N/A')[:60]}...")

        # 관련 뉴스
        if code in news_data:
            print(f"\n  [Related News]")
            for i, title in enumerate(news_data[code][:3], 1):
                print(f"    {i}. {title[:60]}...")

    # 종합 결론
    print("\n" + "=" * 75)
    print("                          TRADING SUMMARY")
    print("=" * 75)

    # 매수/매도 추천 종목
    buy_candidates = []
    sell_candidates = []

    for code, tech_data in technical_data.items():
        name = tech_data['name']
        summary = tech_data['summary']
        score = summary['total_score']

        if score > 20:
            buy_candidates.append((name, code, score, summary))
        elif score < -20:
            sell_candidates.append((name, code, score, summary))

    if buy_candidates:
        print("\n  [BUY Candidates]")
        buy_candidates.sort(key=lambda x: x[2], reverse=True)
        for name, code, score, summary in buy_candidates:
            print(f"    - {name} ({code}): Score {score:+.1f}, RSI {summary['rsi_14']:.1f}")

    if sell_candidates:
        print("\n  [SELL Candidates]")
        sell_candidates.sort(key=lambda x: x[2])
        for name, code, score, summary in sell_candidates:
            print(f"    - {name} ({code}): Score {score:+.1f}, RSI {summary['rsi_14']:.1f}")

    if not buy_candidates and not sell_candidates:
        print("\n  [Market Status] NEUTRAL - No clear trading signals")

    print("\n" + "=" * 75)
    print("  DISCLAIMER: This is for educational purposes only.")
    print("  Scalping involves high risk. Trade responsibly.")
    print("  Past performance does not guarantee future results.")
    print("=" * 75)
    print()


if __name__ == "__main__":
    generate_report()
