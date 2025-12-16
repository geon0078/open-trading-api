# -*- coding: utf-8 -*-
"""
JSON 기반 LLM 상세 로그 저장

저장 구조:
data/llm_logs/
├── YYYY-MM-DD/
│   ├── {stock_code}_{HHmmss}.json      # 개별 분석 상세
│   └── {stock_code}_{HHmmss}_news.json # 뉴스 분석 상세
└── index/
    └── YYYY-MM-DD.json                  # 일별 인덱스

각 로그 파일 구조:
{
    "analysis_id": "005930_20241215_093015",
    "timestamp": "2024-12-15 09:30:15",
    "stock_code": "005930",
    "stock_name": "삼성전자",
    "analysis_type": "ensemble" | "news" | "scalping",
    "input": {
        "prompt": "전체 프롬프트...",
        "stock_data": {...},
        "news_list": [...],
        "technical_summary": {...}
    },
    "model_outputs": [
        {
            "model_name": "deepseek-r1:8b",
            "raw_output": "LLM 원본 응답...",
            "parsed_result": {...},
            "processing_time": 5.2,
            "success": true
        },
        ...
    ],
    "ensemble_result": {
        "signal": "STRONG_BUY",
        "confidence": 0.85,
        "trend": "UP",
        "consensus": 0.75,
        ...
    },
    "metadata": {
        "total_processing_time": 25.5,
        "models_used": [...],
        "version": "1.0"
    }
}
"""

import os
import json
import gzip
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


class LLMLogStorage:
    """JSON 기반 LLM 상세 로그 저장소"""

    VERSION = "1.0"

    def __init__(self, base_dir: str = None, compress: bool = False):
        """
        Args:
            base_dir: 로그 저장 기본 디렉토리 (None이면 기본 경로 사용)
            compress: True이면 gzip 압축 사용 (.json.gz)
        """
        if base_dir is None:
            # 기본 경로: examples_user/news_trading/data/llm_logs
            base_dir = str(Path(__file__).parent.parent.parent / "data" / "llm_logs")

        self.base_dir = base_dir
        self.compress = compress
        self.index_dir = os.path.join(base_dir, "index")

        # 디렉토리 생성
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        logger.info(f"LLMLogStorage 초기화: {base_dir}, compress={compress}")

    def _get_date_dir(self, dt: datetime = None) -> str:
        """날짜별 디렉토리 경로 반환"""
        if dt is None:
            dt = datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        date_dir = os.path.join(self.base_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        return date_dir

    def _generate_log_id(self, stock_code: str, dt: datetime = None) -> str:
        """로그 ID 생성"""
        if dt is None:
            dt = datetime.now()
        time_str = dt.strftime("%H%M%S")
        return f"{stock_code}_{time_str}"

    def _serialize(self, obj: Any) -> Any:
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        # 알 수 없는 타입은 문자열로 변환
        return str(obj)

    def _write_json(self, filepath: str, data: Dict):
        """JSON 파일 쓰기 (압축 옵션 지원)"""
        serialized = self._serialize(data)

        if self.compress:
            filepath = filepath + ".gz" if not filepath.endswith(".gz") else filepath
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)

    def _read_json(self, filepath: str) -> Dict:
        """JSON 파일 읽기 (압축 파일 자동 감지)"""
        if filepath.endswith(".gz") or os.path.exists(filepath + ".gz"):
            actual_path = filepath if filepath.endswith(".gz") else filepath + ".gz"
            with gzip.open(actual_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)

    # =========================================================================
    # 로그 저장
    # =========================================================================

    def save_ensemble_log(
        self,
        stock_code: str,
        stock_name: str,
        input_prompt: str,
        stock_data: Dict,
        news_list: List[str],
        technical_summary: Dict,
        model_outputs: List[Dict],
        ensemble_result: Dict,
        analysis_type: str = "ensemble",
        additional_context: str = "",
        timestamp: datetime = None
    ) -> str:
        """
        앙상블 분석 로그 저장

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            input_prompt: 입력 프롬프트
            stock_data: 종목 데이터
            news_list: 뉴스 리스트
            technical_summary: 기술적 지표 요약
            model_outputs: 모델별 출력 리스트
            ensemble_result: 앙상블 결과
            analysis_type: 분석 유형 (ensemble, scalping, news)
            additional_context: 추가 컨텍스트
            timestamp: 타임스탬프

        Returns:
            str: 저장된 로그 파일 경로
        """
        if timestamp is None:
            timestamp = datetime.now()

        log_id = self._generate_log_id(stock_code, timestamp)
        date_dir = self._get_date_dir(timestamp)
        log_file = os.path.join(date_dir, f"{log_id}.json")

        log_data = {
            "analysis_id": log_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "stock_code": stock_code,
            "stock_name": stock_name,
            "analysis_type": analysis_type,
            "input": {
                "prompt": input_prompt,
                "stock_data": stock_data,
                "news_list": news_list or [],
                "technical_summary": technical_summary or {},
                "additional_context": additional_context
            },
            "model_outputs": [
                {
                    "model_name": output.get("model_name", "unknown"),
                    "raw_output": output.get("raw_output", ""),
                    "parsed_result": {
                        "signal": output.get("signal", "HOLD"),
                        "confidence": output.get("confidence", 0),
                        "trend_prediction": output.get("trend_prediction", "SIDEWAYS"),
                        "entry_price": output.get("entry_price", 0),
                        "stop_loss": output.get("stop_loss", 0),
                        "take_profit": output.get("take_profit", 0),
                        "reasoning": output.get("reasoning", ""),
                        "news_impact": output.get("news_impact", "NEUTRAL"),
                        "risk_factors": output.get("risk_factors", [])
                    },
                    "processing_time": output.get("processing_time", 0),
                    "success": output.get("success", False),
                    "error_message": output.get("error_message", "")
                }
                for output in model_outputs
            ],
            "ensemble_result": ensemble_result,
            "metadata": {
                "total_processing_time": ensemble_result.get("total_processing_time", 0),
                "models_used": ensemble_result.get("models_used", []),
                "version": self.VERSION
            }
        }

        self._write_json(log_file, log_data)
        logger.info(f"LLM 로그 저장: {log_file}")

        # 인덱스 업데이트
        self._update_index(timestamp, log_id, stock_code, stock_name, analysis_type, ensemble_result)

        return log_file

    def save_news_analysis_log(
        self,
        news_list: List[str],
        llm_analysis: Dict,
        market_sentiment: str,
        key_themes: List[str],
        attention_stocks: List[Dict],
        model_used: str,
        raw_output: str,
        processing_time: float,
        timestamp: datetime = None
    ) -> str:
        """
        뉴스 분석 로그 저장

        Args:
            news_list: 뉴스 리스트
            llm_analysis: LLM 분석 결과
            market_sentiment: 시장 심리
            key_themes: 주요 테마
            attention_stocks: 주목 종목
            model_used: 사용된 모델
            raw_output: LLM 원본 응답
            processing_time: 처리 시간
            timestamp: 타임스탬프

        Returns:
            str: 저장된 로그 파일 경로
        """
        if timestamp is None:
            timestamp = datetime.now()

        log_id = f"news_{timestamp.strftime('%H%M%S')}"
        date_dir = self._get_date_dir(timestamp)
        log_file = os.path.join(date_dir, f"{log_id}.json")

        log_data = {
            "analysis_id": log_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_type": "news",
            "input": {
                "news_list": news_list,
                "news_count": len(news_list)
            },
            "model_outputs": [
                {
                    "model_name": model_used,
                    "raw_output": raw_output,
                    "parsed_result": llm_analysis,
                    "processing_time": processing_time,
                    "success": True
                }
            ],
            "result": {
                "market_sentiment": market_sentiment,
                "key_themes": key_themes,
                "attention_stocks": attention_stocks,
                "market_outlook": llm_analysis.get("market_outlook", "")
            },
            "metadata": {
                "total_processing_time": processing_time,
                "models_used": [model_used],
                "version": self.VERSION
            }
        }

        self._write_json(log_file, log_data)
        logger.info(f"뉴스 분석 로그 저장: {log_file}")

        # 인덱스 업데이트
        self._update_index(
            timestamp, log_id, "NEWS", "뉴스분석", "news",
            {"market_sentiment": market_sentiment}
        )

        return log_file

    def _update_index(
        self,
        timestamp: datetime,
        log_id: str,
        stock_code: str,
        stock_name: str,
        analysis_type: str,
        result: Dict
    ):
        """일별 인덱스 업데이트"""
        date_str = timestamp.strftime("%Y-%m-%d")
        index_file = os.path.join(self.index_dir, f"{date_str}.json")

        # 기존 인덱스 로드 또는 새로 생성
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {
                "date": date_str,
                "entries": [],
                "summary": {
                    "total_analyses": 0,
                    "by_type": {},
                    "by_signal": {}
                }
            }

        # 새 엔트리 추가
        entry = {
            "log_id": log_id,
            "timestamp": timestamp.strftime("%H:%M:%S"),
            "stock_code": stock_code,
            "stock_name": stock_name,
            "analysis_type": analysis_type,
            "signal": result.get("ensemble_signal") or result.get("market_sentiment", "N/A"),
            "confidence": result.get("ensemble_confidence", 0)
        }
        index["entries"].append(entry)

        # 요약 업데이트
        index["summary"]["total_analyses"] += 1
        index["summary"]["by_type"][analysis_type] = index["summary"]["by_type"].get(analysis_type, 0) + 1
        if entry["signal"]:
            index["summary"]["by_signal"][entry["signal"]] = index["summary"]["by_signal"].get(entry["signal"], 0) + 1

        # 저장
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # 로그 조회
    # =========================================================================

    def get_log(self, log_path: str) -> Dict:
        """
        단일 로그 파일 조회

        Args:
            log_path: 로그 파일 경로

        Returns:
            Dict: 로그 데이터
        """
        if not os.path.exists(log_path):
            # .gz 확장자 시도
            if os.path.exists(log_path + ".gz"):
                log_path = log_path + ".gz"
            else:
                raise FileNotFoundError(f"로그 파일 없음: {log_path}")

        return self._read_json(log_path)

    def get_logs_by_date(self, date_str: str) -> List[Dict]:
        """
        특정 날짜의 모든 로그 조회

        Args:
            date_str: 날짜 (YYYY-MM-DD)

        Returns:
            List[Dict]: 로그 리스트
        """
        date_dir = os.path.join(self.base_dir, date_str)
        if not os.path.exists(date_dir):
            return []

        logs = []
        for filename in sorted(os.listdir(date_dir)):
            if filename.endswith('.json') or filename.endswith('.json.gz'):
                filepath = os.path.join(date_dir, filename)
                try:
                    logs.append(self._read_json(filepath))
                except Exception as e:
                    logger.warning(f"로그 읽기 실패 ({filepath}): {e}")

        return logs

    def get_logs_by_stock(self, stock_code: str, days: int = 7) -> List[Dict]:
        """
        특정 종목의 최근 로그 조회

        Args:
            stock_code: 종목코드
            days: 조회 일수

        Returns:
            List[Dict]: 로그 리스트
        """
        from datetime import timedelta

        logs = []
        today = datetime.now()

        for i in range(days):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            date_logs = self.get_logs_by_date(date)

            for log in date_logs:
                if log.get("stock_code") == stock_code:
                    logs.append(log)

        return logs

    def get_index(self, date_str: str) -> Dict:
        """
        일별 인덱스 조회

        Args:
            date_str: 날짜 (YYYY-MM-DD)

        Returns:
            Dict: 인덱스 데이터
        """
        index_file = os.path.join(self.index_dir, f"{date_str}.json")
        if not os.path.exists(index_file):
            return {
                "date": date_str,
                "entries": [],
                "summary": {
                    "total_analyses": 0,
                    "by_type": {},
                    "by_signal": {}
                }
            }

        with open(index_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_today_summary(self) -> Dict:
        """오늘의 분석 요약"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_index(today).get("summary", {})

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_model_performance(self, days: int = 7) -> Dict:
        """
        모델별 성능 분석

        Args:
            days: 분석 기간 (일)

        Returns:
            Dict: 모델별 성능 통계
        """
        from datetime import timedelta

        stats = {}  # model_name -> {total, success, avg_time, signals}
        today = datetime.now()

        for i in range(days):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            logs = self.get_logs_by_date(date)

            for log in logs:
                for output in log.get("model_outputs", []):
                    model = output.get("model_name", "unknown")
                    if model not in stats:
                        stats[model] = {
                            "total": 0,
                            "success": 0,
                            "total_time": 0,
                            "signals": {}
                        }

                    stats[model]["total"] += 1
                    if output.get("success"):
                        stats[model]["success"] += 1
                    stats[model]["total_time"] += output.get("processing_time", 0)

                    signal = output.get("parsed_result", {}).get("signal", "UNKNOWN")
                    stats[model]["signals"][signal] = stats[model]["signals"].get(signal, 0) + 1

        # 평균 계산
        result = {}
        for model, data in stats.items():
            result[model] = {
                "total_calls": data["total"],
                "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
                "avg_processing_time": data["total_time"] / data["total"] if data["total"] > 0 else 0,
                "signal_distribution": data["signals"]
            }

        return result

    def cleanup_old_logs(self, keep_days: int = 30) -> int:
        """
        오래된 로그 삭제

        Args:
            keep_days: 보관 일수

        Returns:
            int: 삭제된 파일 수
        """
        from datetime import timedelta
        import shutil

        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted = 0

        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)

            # index 폴더는 스킵
            if item == "index":
                continue

            # 날짜 형식 폴더만 처리 (YYYY-MM-DD)
            try:
                folder_date = datetime.strptime(item, "%Y-%m-%d")
                if folder_date < cutoff_date:
                    shutil.rmtree(item_path)
                    deleted += 1
                    logger.info(f"오래된 로그 삭제: {item_path}")
            except ValueError:
                continue

        # 인덱스 파일도 정리
        for item in os.listdir(self.index_dir):
            if item.endswith(".json"):
                try:
                    file_date = datetime.strptime(item.replace(".json", ""), "%Y-%m-%d")
                    if file_date < cutoff_date:
                        os.remove(os.path.join(self.index_dir, item))
                        logger.info(f"오래된 인덱스 삭제: {item}")
                except ValueError:
                    continue

        return deleted

    def get_storage_stats(self) -> Dict:
        """
        저장소 통계 조회

        Returns:
            Dict: 저장소 통계
        """
        total_size = 0
        total_files = 0
        dates = []

        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path) and item != "index":
                dates.append(item)
                for f in os.listdir(item_path):
                    filepath = os.path.join(item_path, f)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
                        total_files += 1

        return {
            "base_dir": self.base_dir,
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "date_range": {
                "oldest": min(dates) if dates else None,
                "newest": max(dates) if dates else None,
                "total_days": len(dates)
            },
            "compression": self.compress
        }


# 전역 인스턴스
_llm_log_storage: Optional[LLMLogStorage] = None


def get_llm_log_storage(base_dir: str = None, compress: bool = False) -> LLMLogStorage:
    """전역 LLMLogStorage 인스턴스 반환"""
    global _llm_log_storage
    if _llm_log_storage is None:
        _llm_log_storage = LLMLogStorage(base_dir, compress)
    return _llm_log_storage


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    storage = LLMLogStorage()

    # 테스트 앙상블 로그 저장
    log_path = storage.save_ensemble_log(
        stock_code="005930",
        stock_name="삼성전자",
        input_prompt="테스트 프롬프트...",
        stock_data={"code": "005930", "name": "삼성전자", "price": 55000},
        news_list=["삼성전자 AI 반도체 수주 급증", "외국인 순매수 지속"],
        technical_summary={"rsi_14": 65, "trend": "UP"},
        model_outputs=[
            {
                "model_name": "deepseek-r1:8b",
                "raw_output": '{"signal": "STRONG_BUY", "confidence": 0.85}',
                "signal": "STRONG_BUY",
                "confidence": 0.85,
                "reasoning": "기술적 강세 + 뉴스 호재",
                "processing_time": 5.2,
                "success": True
            },
            {
                "model_name": "qwen3:8b",
                "raw_output": '{"signal": "BUY", "confidence": 0.75}',
                "signal": "BUY",
                "confidence": 0.75,
                "reasoning": "상승 추세 지속",
                "processing_time": 4.8,
                "success": True
            }
        ],
        ensemble_result={
            "ensemble_signal": "STRONG_BUY",
            "ensemble_confidence": 0.80,
            "ensemble_trend": "UP",
            "consensus_score": 0.75,
            "total_processing_time": 15.5,
            "models_used": ["deepseek-r1:8b", "qwen3:8b"]
        }
    )
    print(f"앙상블 로그 저장: {log_path}")

    # 테스트 뉴스 분석 로그 저장
    news_log_path = storage.save_news_analysis_log(
        news_list=["뉴스1", "뉴스2", "뉴스3"],
        llm_analysis={"market_outlook": "긍정적 전망"},
        market_sentiment="BULLISH",
        key_themes=["AI", "반도체"],
        attention_stocks=[{"name": "삼성전자", "reason": "AI 수혜"}],
        model_used="qwen3:8b",
        raw_output="LLM 원본 응답...",
        processing_time=3.5
    )
    print(f"뉴스 분석 로그 저장: {news_log_path}")

    # 조회 테스트
    print("\n=== 저장된 로그 읽기 ===")
    log_data = storage.get_log(log_path)
    print(f"종목: {log_data['stock_name']}")
    print(f"분석 유형: {log_data['analysis_type']}")
    print(f"모델 수: {len(log_data['model_outputs'])}")

    print("\n=== 오늘 분석 요약 ===")
    summary = storage.get_today_summary()
    print(f"총 분석: {summary.get('total_analyses', 0)}")
    print(f"타입별: {summary.get('by_type', {})}")
    print(f"시그널별: {summary.get('by_signal', {})}")

    print("\n=== 저장소 통계 ===")
    stats = storage.get_storage_stats()
    print(f"총 파일: {stats['total_files']}")
    print(f"총 크기: {stats['total_size_mb']} MB")
