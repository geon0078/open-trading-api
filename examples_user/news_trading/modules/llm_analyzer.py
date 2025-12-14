# -*- coding: utf-8 -*-
"""
LLM 지속 분석 엔진

급등 종목 + 뉴스를 종합하여 추세를 예측하고
모든 Input/Output을 로깅합니다.

기능:
- 급등 종목 실시간 분석
- 관련 뉴스 수집 및 통합
- 추세 예측 및 매매 신호 생성
- Input/Output 전체 로깅
"""

import os
import sys
import json
import logging
import time
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
import queue

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysis:
    """LLM 분석 결과"""
    timestamp: str
    stock_code: str
    stock_name: str

    # Input 데이터
    input_prompt: str
    input_data: Dict

    # Output 데이터
    output_raw: str
    output_parsed: Dict

    # 분석 결과
    signal: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence: float
    trend_prediction: str  # UP, DOWN, SIDEWAYS
    entry_price: float
    stop_loss: float
    take_profit: float
    hold_time: str
    reasoning: str
    risk_factors: List[str]

    # 메타
    model_name: str
    processing_time: float
    success: bool
    error_message: str = ""


class LLMAnalyzer:
    """
    LLM 지속 분석 엔진

    급등 종목과 뉴스를 종합하여 실시간 분석을 수행합니다.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = None
        self.is_running = False
        self.analysis_queue = queue.Queue()
        self.analysis_history = deque(maxlen=100)  # 최근 100개 분석 보관
        self.callbacks: List[Callable] = []
        self._worker_thread = None

    def get_available_model(self) -> Optional[str]:
        """사용 가능한 LLM 모델 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                # 선호 모델 순서 (DeepSeek-R1 금융 추론 최우선)
                preferred = ["deepseek-r1:8b", "deepseek-r1", "qwen3:8b", "qwen3", "fin-r1", "qwen2.5", "llama3", "mistral"]
                for pref in preferred:
                    for name in model_names:
                        if pref in name.lower():
                            self.model_name = name
                            return name

                if model_names:
                    self.model_name = model_names[0]
                    return model_names[0]
        except Exception as e:
            logger.error(f"모델 확인 실패: {e}")
        return None

    def add_callback(self, callback: Callable):
        """분석 완료 콜백 추가"""
        self.callbacks.append(callback)

    def _notify_callbacks(self, analysis: LLMAnalysis):
        """콜백 알림"""
        for callback in self.callbacks:
            try:
                callback(analysis)
            except Exception as e:
                logger.error(f"콜백 오류: {e}")

    def build_analysis_prompt(
        self,
        stock_data: Dict,
        news_list: List[str],
        technical_summary: Dict = None
    ) -> str:
        """분석 프롬프트 생성"""

        # 기본 종목 정보
        prompt_parts = [
            "당신은 전문 스캘핑 트레이더입니다. 아래 데이터를 분석하여 3분 스캘핑 전략을 제시하세요.",
            "",
            f"## 종목 정보",
            f"- 종목명: {stock_data.get('name', 'N/A')}",
            f"- 종목코드: {stock_data.get('code', 'N/A')}",
            f"- 현재가: {stock_data.get('price', 0):,}원",
            f"- 등락률: {stock_data.get('change_rate', 0):+.2f}%",
            f"- 체결강도: {stock_data.get('volume_power', 100):.1f} (100 이상 매수우세)",
            f"- 호가잔량비: {stock_data.get('balance_ratio', 1.0):.2f} (1 이상 매수우세)",
            f"- 거래량: {stock_data.get('volume', 0):,}주",
            f"- 급등점수: {stock_data.get('surge_score', 0):.1f}/100",
        ]

        # 기술적 지표
        if technical_summary:
            prompt_parts.extend([
                "",
                "## 기술적 지표",
                f"- RSI(14): {technical_summary.get('rsi_14', 50):.1f}",
                f"- RSI(7): {technical_summary.get('rsi_7', 50):.1f}",
                f"- MACD: {technical_summary.get('macd', 0):.2f}",
                f"- 스토캐스틱 %K: {technical_summary.get('stoch_k', 50):.1f}",
                f"- 볼린저 %B: {technical_summary.get('bb_percent_b', 0.5):.2f}",
                f"- ATR: {technical_summary.get('atr_14', 0):,.0f}원 ({technical_summary.get('atr_percent', 0):.2f}%)",
                f"- 추세: {technical_summary.get('trend', 'NEUTRAL')}",
            ])

        # 뉴스
        if news_list:
            prompt_parts.extend([
                "",
                "## 관련 뉴스",
            ])
            for i, news in enumerate(news_list[:5], 1):
                prompt_parts.append(f"{i}. {news}")

        # 분석 요청
        prompt_parts.extend([
            "",
            "## 분석 요청",
            "위 데이터를 종합하여 다음을 분석하세요:",
            "1. 현재 추세와 모멘텀 상태",
            "2. 뉴스가 주가에 미치는 영향",
            "3. 3분 내 스캘핑 진입 여부",
            "4. 구체적인 진입가, 손절가, 익절가",
            "",
            "다음 JSON 형식으로만 응답하세요:",
            "```json",
            "{",
            '    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",',
            '    "confidence": 0.0~1.0,',
            '    "trend_prediction": "UP|DOWN|SIDEWAYS",',
            '    "entry_price": 진입가(숫자),',
            '    "stop_loss": 손절가(숫자),',
            '    "take_profit": 익절가(숫자),',
            '    "hold_time": "추천 보유시간",',
            '    "reasoning": "분석 근거 (50자 이내)",',
            '    "news_impact": "POSITIVE|NEGATIVE|NEUTRAL",',
            '    "news_summary": "뉴스 영향 요약 (30자 이내)",',
            '    "risk_factors": ["리스크1", "리스크2"],',
            '    "entry_timing": "즉시|눌림목|돌파시"',
            "}",
            "```",
            "",
            "JSON만 출력하세요."
        ])

        return "\n".join(prompt_parts)

    def call_llm(self, prompt: str, max_retries: int = 2) -> tuple[str, Dict]:
        """LLM 호출 (재시도 로직 포함)"""
        if not self.model_name:
            self.get_available_model()

        if not self.model_name:
            raise ValueError("사용 가능한 LLM 모델이 없습니다")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 600,
                "num_ctx": 4096
            }
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:
                    raw_output = response.json().get("response", "").strip()
                    parsed = self._parse_json_response(raw_output)
                    return raw_output, parsed
                elif response.status_code == 500:
                    # 서버 오류 - 재시도
                    last_error = f"서버 오류 (500): {response.text[:100] if response.text else 'No response'}"
                    logger.warning(f"LLM 호출 실패 (시도 {attempt + 1}/{max_retries + 1}): {last_error}")
                    if attempt < max_retries:
                        time.sleep(1)  # 재시도 전 대기
                        continue
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:100] if response.text else 'No response'}"
                    break

            except requests.exceptions.Timeout:
                last_error = "요청 타임아웃 (120초 초과)"
                logger.warning(f"LLM 타임아웃 (시도 {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    continue
            except requests.exceptions.ConnectionError:
                last_error = "Ollama 서버 연결 실패 (localhost:11434)"
                break
            except Exception as e:
                last_error = str(e)
                break

        raise Exception(f"LLM 호출 실패: {last_error}")

    def _parse_json_response(self, text: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 블록 추출
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]

            # 중괄호 찾기
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    def analyze_stock(
        self,
        stock_data: Dict,
        news_list: List[str] = None,
        technical_summary: Dict = None
    ) -> LLMAnalysis:
        """단일 종목 분석"""
        start_time = time.time()

        # 프롬프트 생성
        prompt = self.build_analysis_prompt(
            stock_data,
            news_list or [],
            technical_summary
        )

        # 입력 데이터 기록
        input_data = {
            "stock": stock_data,
            "news": news_list or [],
            "technical": technical_summary or {}
        }

        try:
            # LLM 호출
            raw_output, parsed_output = self.call_llm(prompt)

            processing_time = time.time() - start_time

            # 분석 결과 생성
            analysis = LLMAnalysis(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                stock_code=stock_data.get('code', ''),
                stock_name=stock_data.get('name', ''),
                input_prompt=prompt,
                input_data=input_data,
                output_raw=raw_output,
                output_parsed=parsed_output,
                signal=parsed_output.get('signal', 'HOLD'),
                confidence=float(parsed_output.get('confidence', 0)),
                trend_prediction=parsed_output.get('trend_prediction', 'SIDEWAYS'),
                entry_price=float(parsed_output.get('entry_price', 0)),
                stop_loss=float(parsed_output.get('stop_loss', 0)),
                take_profit=float(parsed_output.get('take_profit', 0)),
                hold_time=parsed_output.get('hold_time', '3분'),
                reasoning=parsed_output.get('reasoning', ''),
                risk_factors=parsed_output.get('risk_factors', []),
                model_name=self.model_name,
                processing_time=processing_time,
                success=bool(parsed_output)
            )

        except Exception as e:
            processing_time = time.time() - start_time
            analysis = LLMAnalysis(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                stock_code=stock_data.get('code', ''),
                stock_name=stock_data.get('name', ''),
                input_prompt=prompt,
                input_data=input_data,
                output_raw="",
                output_parsed={},
                signal="HOLD",
                confidence=0,
                trend_prediction="SIDEWAYS",
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                hold_time="",
                reasoning="",
                risk_factors=[],
                model_name=self.model_name or "N/A",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

        # 히스토리에 추가
        self.analysis_history.append(analysis)

        # 콜백 호출
        self._notify_callbacks(analysis)

        return analysis

    def get_news_for_stock(self, stock_code: str, stock_name: str) -> List[str]:
        """종목 관련 뉴스 조회"""
        try:
            from domestic_stock.news_title.news_title import news_title

            news_df = news_title(
                fid_news_ofer_entp_code="",
                fid_cond_mrkt_cls_code="",
                fid_input_iscd=stock_code,
                fid_titl_cntt="",
                fid_input_date_1="",
                fid_input_hour_1="",
                fid_rank_sort_cls_code="",
                fid_input_srno="",
                max_depth=1
            )

            if news_df is not None and not news_df.empty:
                return news_df['hts_pbnt_titl_cntt'].head(5).tolist()

        except Exception as e:
            logger.debug(f"뉴스 조회 실패 ({stock_name}): {e}")

        return []

    def analyze_surge_candidates(
        self,
        candidates: List,
        max_analyze: int = 10,
        include_news: bool = True
    ) -> List[LLMAnalysis]:
        """급등 후보 종목 일괄 분석"""
        results = []

        # STRONG_BUY, BUY 우선 분석
        priority_candidates = [c for c in candidates if c.signal in ["STRONG_BUY", "BUY"]]
        other_candidates = [c for c in candidates if c.signal not in ["STRONG_BUY", "BUY"]]

        all_candidates = (priority_candidates + other_candidates)[:max_analyze]

        for candidate in all_candidates:
            # 종목 데이터 변환
            stock_data = {
                'code': candidate.code,
                'name': candidate.name,
                'price': candidate.price,
                'change': candidate.change,
                'change_rate': candidate.change_rate,
                'volume': candidate.volume,
                'volume_power': candidate.volume_power,
                'balance_ratio': candidate.balance_ratio,
                'surge_score': candidate.surge_score,
            }

            # 뉴스 조회
            news_list = []
            if include_news:
                news_list = self.get_news_for_stock(candidate.code, candidate.name)

            # 분석
            analysis = self.analyze_stock(stock_data, news_list)
            results.append(analysis)

            time.sleep(0.5)  # API 부하 방지

        return results

    def get_history(self, limit: int = 50) -> List[Dict]:
        """분석 히스토리 조회"""
        history = list(self.analysis_history)[-limit:]
        return [asdict(a) for a in history]

    def get_latest_analyses(self, count: int = 10) -> List[Dict]:
        """최근 분석 결과 조회"""
        latest = list(self.analysis_history)[-count:]
        return [asdict(a) for a in reversed(latest)]


# 전역 분석기 인스턴스
_analyzer_instance = None

def get_analyzer() -> LLMAnalyzer:
    """전역 분석기 인스턴스 반환"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = LLMAnalyzer()
    return _analyzer_instance


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 테스트
    analyzer = LLMAnalyzer()

    # 모델 확인
    model = analyzer.get_available_model()
    print(f"사용 모델: {model}")

    # 테스트 분석
    test_stock = {
        'code': '005930',
        'name': '삼성전자',
        'price': 55000,
        'change_rate': 2.5,
        'volume_power': 180,
        'balance_ratio': 1.8,
        'surge_score': 75,
        'volume': 5000000
    }

    test_news = [
        "삼성전자, AI 반도체 수주 급증",
        "외국인 삼성전자 3거래일 연속 순매수"
    ]

    result = analyzer.analyze_stock(test_stock, test_news)

    print("\n=== 분석 결과 ===")
    print(f"종목: {result.stock_name}")
    print(f"시그널: {result.signal}")
    print(f"신뢰도: {result.confidence:.0%}")
    print(f"추세: {result.trend_prediction}")
    print(f"진입가: {result.entry_price:,.0f}원")
    print(f"손절가: {result.stop_loss:,.0f}원")
    print(f"익절가: {result.take_profit:,.0f}원")
    print(f"근거: {result.reasoning}")
    print(f"\n처리시간: {result.processing_time:.1f}초")

    print("\n=== Input Prompt ===")
    print(result.input_prompt[:500] + "...")

    print("\n=== Output Raw ===")
    print(result.output_raw[:500])
