# -*- coding: utf-8 -*-
"""
3분 스캘핑 실시간 스캐너

급등 종목을 실시간으로 탐지하고 LLM 분석을 통해
스캘핑 진입/청산 신호를 생성합니다.

특징:
- 체결강도 급등 종목 실시간 모니터링
- 호가잔량 불균형 탐지
- 등락률 급등 감지
- LLM 기반 진입 타이밍 분석
- 3분 내 청산 목표 알림

사용법:
    python run_scalping_scanner.py
"""

import os
import sys
import io
import json
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# UTF-8 출력 설정 (Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'modules'))
sys.path.insert(0, os.path.join(current_dir, '..'))  # examples_user (kis_auth 위치)
sys.path.insert(0, os.path.join(current_dir, '..', '..'))  # 루트
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'examples_llm'))  # examples_llm

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
                "num_predict": 400,
                "num_ctx": 4096
            }
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
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


def print_header():
    """헤더 출력"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 80)
    print("              3분 스캘핑 실시간 스캐너")
    print(f"              Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


def run_scanner():
    """스캘핑 스캐너 실행"""

    print_header()

    # 1. 환경 설정
    print("[1/4] Environment Setup...")
    try:
        from modules.config_loader import setup_kis_config, load_env_file
        load_env_file()
        setup_kis_config()
        print("      [OK] Config loaded")
    except Exception as e:
        print(f"      [FAIL] Config error: {e}")
        return

    # 2. KIS API 인증
    print("\n[2/4] KIS API Authentication...")
    try:
        import kis_auth as ka
        ka.auth(svr="prod")
        print("      [OK] Authenticated")
    except Exception as e:
        print(f"      [FAIL] Auth error: {e}")
        return

    # 3. 급등 탐지기 초기화
    print("\n[3/4] Initializing Surge Detector...")
    try:
        from modules.surge_detector import SurgeDetector
        detector = SurgeDetector()
        detector._authenticated = True  # 이미 인증됨
        print("      [OK] Surge Detector ready")
    except Exception as e:
        print(f"      [FAIL] Detector error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. LLM 모델 확인
    print("\n[4/4] Checking LLM Model...")
    model_name = get_available_model()
    if model_name:
        print(f"      [OK] Model: {model_name}")
    else:
        print("      [WARN] No LLM available. Running without LLM analysis.")

    print("\n" + "=" * 80)
    print("                    SCANNING FOR SURGE STOCKS...")
    print("=" * 80)

    # 급등 종목 스캔
    print("\n[Scanning] Detecting surge stocks...")
    candidates = detector.scan_surge_stocks(min_score=40)

    if not candidates:
        print("\n  No surge candidates detected at this moment.")
        print("  Try again during market hours (09:00-15:30 KST)")
        return

    # 결과 출력
    print("\n" + "=" * 80)
    print("                    SURGE CANDIDATES DETECTED")
    print("=" * 80)

    for c in candidates[:15]:
        signal_icon = {
            "STRONG_BUY": "🔥",
            "BUY": "✅",
            "WATCH": "👀",
            "NEUTRAL": "⚪"
        }.get(c.signal, "")

        print(f"\n[{c.rank}] {c.name} ({c.code}) {signal_icon} {c.signal}")
        print(f"    Price: {c.price:,}원 ({c.change:+,}, {c.change_rate:+.2f}%)")
        print(f"    Volume Power: {c.volume_power:.1f} | Balance Ratio: {c.balance_ratio:.2f}")
        print(f"    Surge Score: {c.surge_score:.1f}/100")
        print(f"    Volume: {c.volume:,} shares")
        if c.reasons:
            print(f"    Reasons: {', '.join(c.reasons[:3])}")

    # LLM 분석 (상위 종목만)
    if model_name and candidates:
        print("\n" + "=" * 80)
        print("                    LLM SCALPING ANALYSIS")
        print("=" * 80)

        top_candidates = [c for c in candidates if c.signal in ["STRONG_BUY", "BUY"]][:5]

        if not top_candidates:
            top_candidates = candidates[:3]

        for c in top_candidates:
            prompt = f"""당신은 3분 스캘핑 전문 트레이더입니다.
아래 급등 종목에 대해 즉시 진입 여부를 판단하세요.

종목: {c.name} ({c.code})
현재가: {c.price:,}원
등락률: {c.change_rate:+.2f}%
체결강도: {c.volume_power:.1f} (100 이상 매수우세)
호가잔량비: {c.balance_ratio:.2f} (1 이상 매수잔량 우세)
매수사유: {', '.join(c.reasons)}
급등점수: {c.surge_score:.1f}/100

3분 내 청산 전략으로 분석하세요.

다음 JSON 형식으로만 응답:
{{
    "action": "ENTER|WAIT|SKIP",
    "confidence": 0.0~1.0,
    "entry_price": 진입가 (숫자),
    "stop_loss": 손절가 (현재가 -0.5% 내외),
    "take_profit": 익절가 (현재가 +1.0~1.5%),
    "hold_time": "추천 보유시간 (예: 2분, 3분)",
    "entry_timing": "즉시|호가창 확인후|다음 눌림목",
    "risk_level": "LOW|MEDIUM|HIGH",
    "key_signal": "주요 진입 시그널 설명 (30자 이내)"
}}

JSON만 출력:"""

            print(f"\n  Analyzing {c.name}...", end=" ", flush=True)
            result = analyze_with_llm(prompt, model_name)

            if result:
                action = result.get('action', 'WAIT')
                action_icon = {"ENTER": "🟢", "WAIT": "🟡", "SKIP": "🔴"}.get(action, "")

                print(f"{action_icon} {action}")
                print(f"    - Entry: {result.get('entry_price', 0):,.0f}원")
                print(f"    - Stop Loss: {result.get('stop_loss', 0):,.0f}원")
                print(f"    - Take Profit: {result.get('take_profit', 0):,.0f}원")
                print(f"    - Hold Time: {result.get('hold_time', '3분')}")
                print(f"    - Timing: {result.get('entry_timing', 'N/A')}")
                print(f"    - Risk: {result.get('risk_level', 'MEDIUM')}")
                print(f"    - Signal: {result.get('key_signal', 'N/A')}")
            else:
                print("Analysis failed")

            time.sleep(0.3)

    # 스캘핑 요약
    print("\n" + "=" * 80)
    print("                    SCALPING SUMMARY")
    print("=" * 80)

    strong_buy = [c for c in candidates if c.signal == "STRONG_BUY"]
    buy = [c for c in candidates if c.signal == "BUY"]
    watch = [c for c in candidates if c.signal == "WATCH"]

    print(f"\n  🔥 STRONG_BUY: {len(strong_buy)} stocks")
    for c in strong_buy[:5]:
        print(f"     - {c.name} ({c.code}): Score {c.surge_score:.0f}, VP {c.volume_power:.0f}")

    print(f"\n  ✅ BUY: {len(buy)} stocks")
    for c in buy[:5]:
        print(f"     - {c.name} ({c.code}): Score {c.surge_score:.0f}, VP {c.volume_power:.0f}")

    print(f"\n  👀 WATCH: {len(watch)} stocks")
    for c in watch[:3]:
        print(f"     - {c.name} ({c.code}): Score {c.surge_score:.0f}")

    print("\n" + "=" * 80)
    print("  ⚠️  SCALPING RULES")
    print("=" * 80)
    print("  1. Entry: 체결강도 150 이상 + 호가잔량비 1.5 이상일 때")
    print("  2. Stop Loss: 진입가 -0.5% (즉시 손절)")
    print("  3. Take Profit: 진입가 +1.0~1.5%")
    print("  4. Max Hold: 3분 (시간 초과시 청산)")
    print("  5. Volume: 거래량 급증 확인 후 진입")
    print("=" * 80)
    print()


if __name__ == "__main__":
    run_scanner()
