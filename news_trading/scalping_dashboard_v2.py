# -*- coding: utf-8 -*-
"""
스캘핑 대시보드 v2 - 앙상블 LLM 분석 패널 포함

기능:
- 실시간 급등 종목 탐지 및 표시
- 앙상블 LLM 지속 분석 (다중 모델 투표)
- 모델별 결과 비교 및 합의도 표시
- 뉴스 + 기술적 지표 종합 분석
- 계좌 잔고 및 수익률 조회
- 주문 체결 내역 조회

실행:
    python scalping_dashboard_v2.py

접속:
    http://localhost:5001
"""

# 고정 포트 설정
DASHBOARD_PORT = 5001

import os
import sys
import io
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict
from collections import deque

# UTF-8 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'modules'))
sys.path.insert(0, os.path.join(current_dir, '..'))
sys.path.insert(0, os.path.join(current_dir, '..', '..'))
sys.path.insert(0, os.path.join(current_dir, '..', '..', 'examples_llm'))

from flask import Flask, render_template_string, jsonify, request, Response
from flask_cors import CORS
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# SSE 클라이언트 관리
sse_clients: List[queue.Queue] = []
sse_lock = threading.Lock()


# 전역 데이터 저장소
class DataStore:
    def __init__(self):
        self.surge_candidates = []
        self.llm_analyses = deque(maxlen=100)  # LLM 분석 히스토리
        self.ensemble_analyses = deque(maxlen=50)  # 앙상블 분석 히스토리
        self.account_balance = {}
        self.holdings = []
        self.orders = []
        self.last_scan_time = None
        self.last_llm_time = None
        self.is_scanning = False
        self.is_llm_running = False
        self.llm_auto_enabled = True  # 자동 분석 기본 활성화
        self.use_ensemble = True  # 앙상블 LLM 사용 여부
        self.kis_authenticated = False
        self.llm_model = None
        self.ensemble_models = []  # 앙상블에 사용된 모델 목록
        self.error_message = None
        self.scan_interval = 60  # 스캔 주기 (초)
        self.llm_interval = 120  # LLM 분석 주기 (초)
        self.background_thread = None
        self.stop_background = False

data_store = DataStore()


# =============================================================================
# SSE (Server-Sent Events) 실시간 업데이트
# =============================================================================
def broadcast_sse(event_type: str, data: dict):
    """모든 SSE 클라이언트에게 이벤트 브로드캐스트"""
    message = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with sse_lock:
        client_count = len(sse_clients)
        if client_count == 0:
            logger.debug(f"SSE 브로드캐스트: {event_type} - 연결된 클라이언트 없음")
            return

        dead_clients = []
        sent_count = 0
        for client_queue in sse_clients:
            try:
                client_queue.put_nowait(message)
                sent_count += 1
            except queue.Full:
                dead_clients.append(client_queue)
        # 죽은 클라이언트 제거
        for dead in dead_clients:
            sse_clients.remove(dead)

        logger.info(f"📡 SSE 브로드캐스트: {event_type} -> {sent_count}/{client_count} 클라이언트")


def broadcast_surge_update():
    """급등 종목 업데이트 브로드캐스트"""
    candidates = []
    for c in data_store.surge_candidates:
        candidates.append({
            "rank": c.rank, "code": c.code, "name": c.name,
            "price": c.price, "change": c.change, "change_rate": c.change_rate,
            "volume": c.volume, "volume_power": c.volume_power,
            "balance_ratio": c.balance_ratio, "surge_score": c.surge_score,
            "signal": c.signal, "detected_at": c.detected_at, "reasons": c.reasons,
        })
    broadcast_sse("surge_update", {
        "candidates": candidates,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "count": len(candidates)
    })


def broadcast_llm_update(analysis_result):
    """LLM 분석 결과 브로드캐스트"""
    try:
        # EnsembleAnalysis 객체를 딕셔너리로 변환
        result_dict = {
            "timestamp": analysis_result.timestamp,
            "stock_code": analysis_result.stock_code,
            "stock_name": analysis_result.stock_name,
            "ensemble_signal": analysis_result.ensemble_signal,
            "ensemble_confidence": analysis_result.ensemble_confidence,
            "ensemble_trend": analysis_result.ensemble_trend,
            "avg_entry_price": analysis_result.avg_entry_price,
            "avg_stop_loss": analysis_result.avg_stop_loss,
            "avg_take_profit": analysis_result.avg_take_profit,
            "signal_votes": analysis_result.signal_votes,
            "trend_votes": analysis_result.trend_votes,
            "models_used": analysis_result.models_used,
            "models_agreed": analysis_result.models_agreed,
            "total_models": analysis_result.total_models,
            "consensus_score": analysis_result.consensus_score,
            "total_processing_time": analysis_result.total_processing_time,
            "success": analysis_result.success,
            "input_prompt": analysis_result.input_prompt,
            "model_results": [
                {
                    "model_name": r.model_name,
                    "signal": r.signal,
                    "confidence": r.confidence,
                    "trend_prediction": r.trend_prediction,
                    "entry_price": r.entry_price,
                    "stop_loss": r.stop_loss,
                    "take_profit": r.take_profit,
                    "reasoning": r.reasoning,
                    "processing_time": r.processing_time,
                    "success": r.success,
                    "raw_output": r.raw_output,
                    "error_message": getattr(r, 'error_message', '')
                }
                for r in analysis_result.model_results
            ]
        }
        broadcast_sse("llm_update", result_dict)
    except Exception as e:
        logger.error(f"LLM 브로드캐스트 오류: {e}")


def broadcast_status_update():
    """상태 업데이트 브로드캐스트"""
    broadcast_sse("status_update", {
        "is_scanning": data_store.is_scanning,
        "is_llm_running": data_store.is_llm_running,
        "last_scan_time": data_store.last_scan_time.strftime("%H:%M:%S") if data_store.last_scan_time else None,
        "last_llm_time": data_store.last_llm_time.strftime("%H:%M:%S") if data_store.last_llm_time else None,
        "llm_model": data_store.llm_model,
        "ensemble_models": data_store.ensemble_models,
        "candidates_count": len(data_store.surge_candidates),
        "llm_history_count": len(data_store.ensemble_analyses)
    })


# =============================================================================
# 백그라운드 자동 분석 스레드
# =============================================================================
def background_analysis_loop():
    """백그라운드에서 지속적으로 급등 종목 스캔 + LLM 분석 수행"""
    logger.info("백그라운드 분석 스레드 시작")

    last_scan = 0
    last_llm = 0

    while not data_store.stop_background:
        try:
            current_time = time.time()

            # 급등 종목 스캔 (주기적)
            if current_time - last_scan >= data_store.scan_interval:
                if not data_store.is_scanning:
                    try:
                        data_store.is_scanning = True
                        from modules.surge_detector import SurgeDetector

                        detector = SurgeDetector()
                        detector._authenticated = True
                        candidates = detector.scan_surge_stocks(min_score=40)

                        data_store.surge_candidates = candidates
                        data_store.last_scan_time = datetime.now()
                        logger.info(f"[백그라운드] 급등 종목 스캔 완료: {len(candidates)}개")

                        # 실시간 업데이트 브로드캐스트
                        broadcast_surge_update()
                        broadcast_status_update()

                    except Exception as e:
                        logger.error(f"[백그라운드] 스캔 오류: {e}")
                    finally:
                        data_store.is_scanning = False
                        last_scan = current_time
                        broadcast_status_update()

            # LLM 분석 (자동 활성화 시) - 앙상블 또는 단일 모델
            if data_store.llm_auto_enabled and current_time - last_llm >= data_store.llm_interval:
                if not data_store.is_llm_running and data_store.surge_candidates:
                    try:
                        data_store.is_llm_running = True
                        broadcast_status_update()  # LLM 시작 알림

                        # 상위 STRONG_BUY/BUY 종목만 분석
                        priority = [c for c in data_store.surge_candidates
                                   if c.signal in ["STRONG_BUY", "BUY"]][:3]

                        if not priority:
                            priority = data_store.surge_candidates[:2]

                        if priority and data_store.use_ensemble:
                            # 앙상블 LLM 분석
                            from modules.ensemble_analyzer import get_ensemble_analyzer
                            from modules.llm_analyzer import LLMAnalyzer

                            ensemble = get_ensemble_analyzer()
                            if not ensemble.ensemble_models:
                                ensemble.setup_ensemble(use_financial_ensemble=True)

                            data_store.ensemble_models = ensemble.ensemble_models
                            data_store.llm_model = f"앙상블 ({len(ensemble.ensemble_models)}모델)"

                            # 뉴스 조회 함수
                            single_analyzer = LLMAnalyzer()

                            analyzed_count = 0
                            for candidate in priority:
                                try:
                                    stock_data = {
                                        'code': candidate.code,
                                        'name': candidate.name,
                                        'price': candidate.price,
                                        'change_rate': candidate.change_rate,
                                        'volume_power': candidate.volume_power,
                                        'balance_ratio': candidate.balance_ratio,
                                        'surge_score': candidate.surge_score,
                                        'volume': candidate.volume
                                    }

                                    # 뉴스 조회
                                    news_list = single_analyzer.get_news_for_stock(candidate.code, candidate.name)

                                    # 앙상블 분석 (순차 실행으로 GPU 충돌 방지)
                                    result = ensemble.analyze_stock(stock_data, news_list, parallel=False)
                                    data_store.ensemble_analyses.append(result)
                                    analyzed_count += 1
                                    logger.info(f"[백그라운드] {candidate.name} 분석 완료: {result.ensemble_signal}")

                                    # 실시간 LLM 결과 브로드캐스트
                                    broadcast_llm_update(result)
                                except Exception as stock_err:
                                    logger.error(f"[백그라운드] {candidate.name} 분석 실패: {stock_err}")

                            data_store.last_llm_time = datetime.now()
                            logger.info(f"[백그라운드] 앙상블 LLM 분석 완료: {analyzed_count}/{len(priority)}개 종목, 모델: {ensemble.ensemble_models}")
                            broadcast_status_update()

                        elif priority:
                            # 단일 모델 분석 (폴백)
                            from modules.llm_analyzer import LLMAnalyzer

                            analyzer = LLMAnalyzer()
                            model = analyzer.get_available_model()

                            if model:
                                data_store.llm_model = model
                                analyses = analyzer.analyze_surge_candidates(
                                    priority,
                                    max_analyze=3,
                                    include_news=True
                                )

                                for a in analyses:
                                    data_store.llm_analyses.append(a)

                                data_store.last_llm_time = datetime.now()
                                logger.info(f"[백그라운드] 단일 LLM 분석 완료: {len(analyses)}개 종목")
                            else:
                                logger.warning("[백그라운드] LLM 모델 없음")

                    except Exception as e:
                        logger.error(f"[백그라운드] LLM 분석 오류: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        data_store.is_llm_running = False
                        last_llm = current_time

            time.sleep(5)  # 5초마다 체크

        except Exception as e:
            logger.error(f"[백그라운드] 루프 오류: {e}")
            time.sleep(10)

    logger.info("백그라운드 분석 스레드 종료")


def start_background_thread():
    """백그라운드 스레드 시작"""
    if data_store.background_thread is None or not data_store.background_thread.is_alive():
        data_store.stop_background = False
        data_store.background_thread = threading.Thread(target=background_analysis_loop, daemon=True)
        data_store.background_thread.start()
        logger.info("백그라운드 분석 스레드 시작됨")


def stop_background_thread():
    """백그라운드 스레드 중지"""
    data_store.stop_background = True
    logger.info("백그라운드 분석 스레드 중지 요청")


# HTML 템플릿 (LLM 패널 포함)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>스캘핑 대시보드 v2 - 앙상블 LLM 분석</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 15px 20px;
            border-bottom: 2px solid #0f4c75;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header h1 { font-size: 22px; color: #00d9ff; }
        .header-info { display: flex; gap: 15px; align-items: center; }
        .status-badge {
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 11px;
            font-weight: bold;
        }
        .status-active { background: #00c853; color: #000; }
        .status-inactive { background: #ff5252; color: #fff; }
        .status-running { background: #ffc107; color: #000; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            padding: 15px;
            max-width: 1900px;
            margin: 0 auto;
        }
        .panel {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #2a2a4e;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a2a4e;
        }
        .panel-title { font-size: 14px; color: #00d9ff; font-weight: bold; }
        .full-width { grid-column: 1 / -1; }

        /* 계좌 정보 */
        .account-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        .account-item {
            background: #16213e;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        .account-label { font-size: 11px; color: #888; margin-bottom: 4px; }
        .account-value { font-size: 18px; font-weight: bold; }
        .positive { color: #00c853; }
        .negative { color: #ff5252; }

        /* 테이블 공통 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .data-table th {
            background: #16213e;
            padding: 10px 6px;
            text-align: left;
            color: #00d9ff;
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        .data-table td {
            padding: 8px 6px;
            border-bottom: 1px solid #2a2a4e;
        }
        .data-table tr:hover { background: #16213e; }
        .signal-badge {
            padding: 3px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
        }
        .signal-strong-buy { background: #ff5722; color: #fff; }
        .signal-buy { background: #4caf50; color: #fff; }
        .signal-hold { background: #2196f3; color: #fff; }
        .signal-sell { background: #9c27b0; color: #fff; }
        .signal-watch { background: #ff9800; color: #000; }
        .signal-neutral { background: #607d8b; color: #fff; }

        /* 버튼 */
        .btn {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-primary { background: #0f4c75; color: #fff; }
        .btn-primary:hover { background: #1b6fa3; }
        .btn-danger { background: #c62828; color: #fff; }
        .btn-success { background: #2e7d32; color: #fff; }
        .btn-warning { background: #f57c00; color: #fff; }
        .btn-sm { padding: 4px 8px; font-size: 10px; }

        /* LLM 패널 */
        .llm-panel { max-height: 600px; overflow-y: auto; }
        .llm-card {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            margin-bottom: 12px;
            overflow: hidden;
        }
        .llm-card-header {
            background: #161b22;
            padding: 10px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .llm-card-header:hover { background: #1c2128; }
        .llm-card-title { font-weight: bold; color: #58a6ff; }
        .llm-card-meta { font-size: 11px; color: #8b949e; }
        .llm-card-body {
            padding: 12px;
            display: none;
        }
        .llm-card.expanded .llm-card-body { display: block; }

        .llm-section { margin-bottom: 12px; }
        .llm-section-title {
            font-size: 11px;
            color: #8b949e;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .llm-code {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 200px;
            overflow-y: auto;
            color: #c9d1d9;
        }
        .llm-result {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .llm-result-item {
            background: #21262d;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
        }
        .llm-result-label { font-size: 10px; color: #8b949e; }
        .llm-result-value { font-size: 14px; font-weight: bold; margin-top: 2px; }

        /* 컨트롤 패널 */
        .control-panel { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        .control-panel select, .control-panel input {
            padding: 6px 10px;
            border-radius: 4px;
            border: 1px solid #2a2a4e;
            background: #16213e;
            color: #e0e0e0;
            font-size: 12px;
        }

        /* 로그 */
        .log-container {
            background: #0a0a0a;
            border-radius: 6px;
            padding: 8px;
            max-height: 150px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 11px;
        }
        .log-entry { padding: 3px 0; border-bottom: 1px solid #1a1a2e; }
        .log-time { color: #666; }
        .log-info { color: #00d9ff; }
        .log-warn { color: #ffc107; }
        .log-error { color: #ff5252; }
        .log-llm { color: #a855f7; }

        .clock { font-size: 13px; color: #00d9ff; font-family: monospace; }

        /* 스크롤바 */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #1a1a2e; }
        ::-webkit-scrollbar-thumb { background: #0f4c75; border-radius: 3px; }

        @media (max-width: 1400px) {
            .container { grid-template-columns: 1fr; }
            .account-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 스캘핑 대시보드 v2 - 앙상블 LLM</h1>
        <div class="header-info">
            <div class="clock" id="clock"></div>
            <span id="llm-status" class="status-badge status-inactive">LLM 대기</span>
            <span id="scan-status" class="status-badge status-inactive">스캔 대기</span>
            <span id="auth-status" class="status-badge status-inactive">미인증</span>
        </div>
    </div>

    <div class="container">
        <!-- 계좌 정보 -->
        <div class="panel full-width">
            <div class="panel-header">
                <span class="panel-title">💰 계좌 정보</span>
                <button class="btn btn-primary btn-sm" onclick="refreshAccount()">새로고침</button>
            </div>
            <div class="account-grid">
                <div class="account-item">
                    <div class="account-label">예수금</div>
                    <div class="account-value" id="deposit">-</div>
                </div>
                <div class="account-item">
                    <div class="account-label">총평가금액</div>
                    <div class="account-value" id="total-eval">-</div>
                </div>
                <div class="account-item">
                    <div class="account-label">총손익</div>
                    <div class="account-value" id="total-pl">-</div>
                </div>
                <div class="account-item">
                    <div class="account-label">수익률</div>
                    <div class="account-value" id="pl-rate">-</div>
                </div>
            </div>
        </div>

        <!-- 급등 종목 -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">🔥 급등 종목</span>
                <div class="control-panel">
                    <select id="scan-interval">
                        <option value="30">30초</option>
                        <option value="60" selected>1분</option>
                        <option value="120">2분</option>
                    </select>
                    <button class="btn btn-success btn-sm" onclick="startAutoScan()">자동</button>
                    <button class="btn btn-danger btn-sm" onclick="stopAutoScan()">중지</button>
                    <button class="btn btn-primary btn-sm" onclick="manualScan()">스캔</button>
                </div>
            </div>
            <div style="max-height: 400px; overflow-y: auto;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>종목</th>
                            <th>현재가</th>
                            <th>등락</th>
                            <th>체결강도</th>
                            <th>점수</th>
                            <th>시그널</th>
                        </tr>
                    </thead>
                    <tbody id="surge-table-body">
                        <tr><td colspan="7" style="text-align:center; color:#666;">스캔 대기중...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 앙상블 LLM 분석 -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">🤖 앙상블 LLM 분석 (다중 모델 투표)</span>
                <div class="control-panel">
                    <select id="llm-interval">
                        <option value="60">1분</option>
                        <option value="120" selected>2분</option>
                        <option value="300">5분</option>
                    </select>
                    <button class="btn btn-warning btn-sm" onclick="startAutoLLM()">자동분석</button>
                    <button class="btn btn-danger btn-sm" onclick="stopAutoLLM()">중지</button>
                    <button class="btn btn-primary btn-sm" onclick="runEnsembleAnalysis()">앙상블분석</button>
                </div>
            </div>
            <div id="ensemble-models" style="padding:8px; background:#16213e; border-radius:6px; margin-bottom:10px; font-size:11px;">
                <strong style="color:#ffc107;">사용 모델:</strong> <span id="model-list">로딩중...</span>
            </div>
            <div class="llm-panel" id="llm-panel">
                <div style="text-align:center; color:#666; padding:20px;">앙상블 LLM 분석 대기중...</div>
            </div>
        </div>

        <!-- 보유 종목 + 체결 내역 -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">📊 보유 종목</span>
            </div>
            <div style="max-height: 200px; overflow-y: auto;">
                <table class="data-table">
                    <thead>
                        <tr><th>종목</th><th>수량</th><th>손익</th><th>수익률</th></tr>
                    </thead>
                    <tbody id="holdings-body">
                        <tr><td colspan="4" style="text-align:center; color:#666;">-</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">📝 오늘 체결</span>
                <button class="btn btn-primary btn-sm" onclick="refreshOrders()">새로고침</button>
            </div>
            <div style="max-height: 200px; overflow-y: auto;">
                <table class="data-table">
                    <thead>
                        <tr><th>시간</th><th>종목</th><th>구분</th><th>수량</th><th>가격</th></tr>
                    </thead>
                    <tbody id="orders-body">
                        <tr><td colspan="5" style="text-align:center; color:#666;">-</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 로그 -->
        <div class="panel full-width">
            <div class="panel-header">
                <span class="panel-title">📋 시스템 로그</span>
                <button class="btn btn-danger btn-sm" onclick="clearLogs()">지우기</button>
            </div>
            <div class="log-container" id="log-container"></div>
        </div>
    </div>

    <script>
        // 시계
        function updateClock() {
            document.getElementById('clock').textContent = new Date().toLocaleString('ko-KR');
        }
        setInterval(updateClock, 1000);
        updateClock();

        // 로그
        function addLog(message, type = 'info') {
            const container = document.getElementById('log-container');
            const time = new Date().toLocaleTimeString('ko-KR');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-${type}">${message}</span>`;
            container.insertBefore(entry, container.firstChild);
            if (container.children.length > 100) container.removeChild(container.lastChild);
        }

        function clearLogs() {
            document.getElementById('log-container').innerHTML = '';
            addLog('로그 초기화');
        }

        // 포맷
        function formatNumber(num) {
            if (num === null || num === undefined || num === '-') return '-';
            return Number(num).toLocaleString('ko-KR');
        }

        function formatPercent(num) {
            if (num === null || num === undefined || num === '-') return '-';
            const val = Number(num);
            return (val >= 0 ? '+' : '') + val.toFixed(2) + '%';
        }

        // 계좌
        async function refreshAccount() {
            try {
                const res = await fetch('/api/account');
                const data = await res.json();
                if (data.error) { addLog(data.error, 'error'); return; }

                document.getElementById('deposit').textContent = formatNumber(data.deposit) + '원';
                document.getElementById('total-eval').textContent = formatNumber(data.total_eval) + '원';

                const plEl = document.getElementById('total-pl');
                plEl.textContent = formatNumber(data.total_pl) + '원';
                plEl.className = 'account-value ' + (data.total_pl >= 0 ? 'positive' : 'negative');

                const rateEl = document.getElementById('pl-rate');
                rateEl.textContent = formatPercent(data.pl_rate);
                rateEl.className = 'account-value ' + (data.pl_rate >= 0 ? 'positive' : 'negative');

                updateHoldings(data.holdings || []);

                document.getElementById('auth-status').className = 'status-badge status-active';
                document.getElementById('auth-status').textContent = '인증됨';
            } catch (e) {
                addLog('계좌 조회 실패: ' + e.message, 'error');
            }
        }

        function updateHoldings(holdings) {
            const tbody = document.getElementById('holdings-body');
            if (!holdings.length) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:#666;">보유 종목 없음</td></tr>';
                return;
            }
            tbody.innerHTML = holdings.map(h => `
                <tr>
                    <td>${h.name}</td>
                    <td>${formatNumber(h.qty)}</td>
                    <td class="${h.pl >= 0 ? 'positive' : 'negative'}">${formatNumber(h.pl)}원</td>
                    <td class="${h.pl_rate >= 0 ? 'positive' : 'negative'}">${formatPercent(h.pl_rate)}</td>
                </tr>
            `).join('');
        }

        // 체결 내역
        async function refreshOrders() {
            try {
                const res = await fetch('/api/orders');
                const data = await res.json();
                const tbody = document.getElementById('orders-body');
                if (!data.orders?.length) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#666;">-</td></tr>';
                    return;
                }
                tbody.innerHTML = data.orders.slice(0, 10).map(o => `
                    <tr>
                        <td>${o.time}</td>
                        <td>${o.name}</td>
                        <td class="${o.side === '매수' ? 'positive' : 'negative'}">${o.side}</td>
                        <td>${formatNumber(o.qty)}</td>
                        <td>${formatNumber(o.price)}원</td>
                    </tr>
                `).join('');
            } catch (e) {
                addLog('체결 조회 실패', 'error');
            }
        }

        // 급등 종목 스캔
        let autoScanTimer = null;

        async function manualScan() {
            addLog('급등 종목 스캔...');
            document.getElementById('scan-status').className = 'status-badge status-running';
            document.getElementById('scan-status').textContent = '스캔중';

            try {
                const res = await fetch('/api/scan');
                const data = await res.json();
                if (data.error) { addLog(data.error, 'error'); return; }
                updateSurgeTable(data.candidates || []);
                addLog(`스캔 완료: ${data.candidates?.length || 0}개`, 'info');
            } catch (e) {
                addLog('스캔 실패', 'error');
            } finally {
                document.getElementById('scan-status').className = 'status-badge status-active';
                document.getElementById('scan-status').textContent = '활성';
            }
        }

        function updateSurgeTable(candidates) {
            const tbody = document.getElementById('surge-table-body');
            if (!candidates.length) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:#666;">-</td></tr>';
                return;
            }
            const signalClass = {
                'STRONG_BUY': 'signal-strong-buy', 'BUY': 'signal-buy',
                'WATCH': 'signal-watch', 'NEUTRAL': 'signal-neutral'
            };
            tbody.innerHTML = candidates.slice(0, 15).map(c => `
                <tr onclick="selectStock('${c.code}', '${c.name}')" style="cursor:pointer">
                    <td>${c.rank}</td>
                    <td><strong>${c.name}</strong><br><small style="color:#666">${c.code}</small></td>
                    <td>${formatNumber(c.price)}원</td>
                    <td class="${c.change_rate >= 0 ? 'positive' : 'negative'}">${formatPercent(c.change_rate)}</td>
                    <td><strong>${c.volume_power?.toFixed(0) || '-'}</strong></td>
                    <td>${c.surge_score?.toFixed(0) || '-'}</td>
                    <td><span class="signal-badge ${signalClass[c.signal] || ''}">${c.signal}</span></td>
                </tr>
            `).join('');
        }

        function startAutoScan() {
            const interval = parseInt(document.getElementById('scan-interval').value) * 1000;
            stopAutoScan();
            addLog(`자동 스캔 시작 (${interval/1000}초)`);
            manualScan();
            autoScanTimer = setInterval(manualScan, interval);
        }

        function stopAutoScan() {
            if (autoScanTimer) { clearInterval(autoScanTimer); autoScanTimer = null; }
            document.getElementById('scan-status').className = 'status-badge status-inactive';
            document.getElementById('scan-status').textContent = '대기';
        }

        // LLM 분석
        let autoLLMTimer = null;

        async function runLLMAnalysis() {
            addLog('LLM 분석 시작...', 'llm');
            document.getElementById('llm-status').className = 'status-badge status-running';
            document.getElementById('llm-status').textContent = 'LLM 분석중';

            try {
                const res = await fetch('/api/llm/analyze');
                const data = await res.json();

                if (data.error) {
                    addLog('LLM 오류: ' + data.error, 'error');
                    return;
                }

                updateLLMPanel(data.analyses || []);
                addLog(`LLM 분석 완료: ${data.analyses?.length || 0}개 종목`, 'llm');

            } catch (e) {
                addLog('LLM 분석 실패: ' + e.message, 'error');
            } finally {
                document.getElementById('llm-status').className = 'status-badge status-active';
                document.getElementById('llm-status').textContent = 'LLM 활성';
            }
        }

        async function runEnsembleAnalysis() {
            addLog('앙상블 LLM 분석 시작... (다중 모델)', 'llm');
            document.getElementById('llm-status').className = 'status-badge status-running';
            document.getElementById('llm-status').textContent = '앙상블 분석중';

            try {
                const res = await fetch('/api/ensemble/analyze');
                const data = await res.json();

                if (data.error) {
                    addLog('앙상블 오류: ' + data.error, 'error');
                    return;
                }

                // 모델 목록 업데이트
                if (data.models_used) {
                    document.getElementById('model-list').textContent = data.models_used.join(', ');
                }

                updateLLMPanel(data.analyses || []);
                addLog(`앙상블 분석 완료: ${data.analyses?.length || 0}개 종목, ${data.models_used?.length || 0}모델`, 'llm');

            } catch (e) {
                addLog('앙상블 분석 실패: ' + e.message, 'error');
            } finally {
                document.getElementById('llm-status').className = 'status-badge status-active';
                document.getElementById('llm-status').textContent = '앙상블 활성';
            }
        }

        function updateLLMPanel(analyses) {
            const panel = document.getElementById('llm-panel');
            if (!analyses.length) {
                panel.innerHTML = '<div style="text-align:center; color:#666; padding:20px;">분석 결과 없음</div>';
                return;
            }

            const signalClass = {
                'STRONG_BUY': 'signal-strong-buy', 'BUY': 'signal-buy', 'HOLD': 'signal-hold',
                'SELL': 'signal-sell', 'STRONG_SELL': 'signal-strong-buy'
            };
            const trendIcon = { 'UP': '📈', 'DOWN': '📉', 'SIDEWAYS': '➡️' };

            // 앙상블 결과인지 확인 (ensemble_signal 필드 존재 여부)
            const isEnsemble = analyses[0] && analyses[0].ensemble_signal !== undefined;

            if (isEnsemble) {
                panel.innerHTML = analyses.map((a, i) => {
                    const isFailed = !a.success || a.consensus_score === 0;
                    const failedModels = (a.model_results || []).filter(r => !r.success).length;
                    const statusBadge = isFailed
                        ? '<span style="margin-left:8px; background:#f44336; padding:2px 6px; border-radius:3px; font-size:10px; color:#fff;">FAIL</span>'
                        : (failedModels > 0
                            ? `<span style="margin-left:8px; background:#ff9800; padding:2px 6px; border-radius:3px; font-size:10px; color:#000;">PARTIAL (${failedModels} failed)</span>`
                            : '<span style="margin-left:8px; background:#4caf50; padding:2px 6px; border-radius:3px; font-size:10px; color:#fff;">OK</span>');

                    return `
                    <div class="llm-card ${i === 0 ? 'expanded' : ''}" style="${isFailed ? 'border-color: #f44336;' : ''}">
                        <div class="llm-card-header" onclick="toggleCard(this.parentElement)">
                            <div>
                                <span class="llm-card-title">${a.stock_name} (${a.stock_code})</span>
                                ${statusBadge}
                                <span class="signal-badge ${signalClass[a.ensemble_signal] || 'signal-neutral'}" style="margin-left:8px">${a.ensemble_signal}</span>
                                <span style="margin-left:8px">${trendIcon[a.ensemble_trend] || ''} ${a.ensemble_trend || ''}</span>
                                <span style="margin-left:8px; background:#9c27b0; padding:2px 6px; border-radius:3px; font-size:10px; color:#fff;">
                                    앙상블 ${a.total_models || 0}모델
                                </span>
                            </div>
                            <div class="llm-card-meta">
                                신뢰도: ${((a.ensemble_confidence || 0) * 100).toFixed(0)}% |
                                합의도: ${((a.consensus_score || 0) * 100).toFixed(0)}% (${a.models_agreed || 0}/${a.total_models || 0}) |
                                ${a.timestamp || ''} | 클릭하여 펼치기 ▼
                            </div>
                        </div>
                        <div class="llm-card-body">
                            <!-- 앙상블 요약 -->
                            <div style="background:linear-gradient(135deg, #4a148c 0%, #7b1fa2 100%); padding:12px; border-radius:8px; margin-bottom:12px;">
                                <div style="color:#e1bee7; font-weight:bold; margin-bottom:8px;">🗳️ 앙상블 투표 결과</div>
                                <div style="display:flex; gap:10px; flex-wrap:wrap;">
                                    ${Object.entries(a.signal_votes || {}).map(([sig, cnt]) => `
                                        <span style="background:${sig.includes('BUY') ? '#4caf50' : sig.includes('SELL') ? '#f44336' : '#607d8b'};
                                              padding:4px 10px; border-radius:4px; font-size:12px;">
                                            ${sig}: ${cnt}표
                                        </span>
                                    `).join('')}
                                </div>
                                <div style="margin-top:8px; color:#ce93d8; font-size:11px;">
                                    추세 투표: ${Object.entries(a.trend_votes || {}).map(([t, c]) => t + ':' + c).join(', ')}
                                </div>
                            </div>

                            <div class="llm-result">
                                <div class="llm-result-item">
                                    <div class="llm-result-label">평균 진입가</div>
                                    <div class="llm-result-value">${formatNumber(a.avg_entry_price)}원</div>
                                </div>
                                <div class="llm-result-item">
                                    <div class="llm-result-label">평균 손절가</div>
                                    <div class="llm-result-value negative">${formatNumber(a.avg_stop_loss)}원</div>
                                </div>
                                <div class="llm-result-item">
                                    <div class="llm-result-label">평균 익절가</div>
                                    <div class="llm-result-value positive">${formatNumber(a.avg_take_profit)}원</div>
                                </div>
                            </div>

                            <!-- 모델별 상세 결과 -->
                            <div style="margin-top:15px; padding:10px; background:#1a1a2e; border-radius:8px; border:1px solid #9c27b0;">
                                <div style="color:#ce93d8; font-weight:bold; margin-bottom:10px;">🤖 모델별 분석 결과</div>
                                ${(a.model_results || []).map(r => `
                                    <div style="background:#16213e; padding:10px; border-radius:6px; margin-bottom:8px; border-left:3px solid ${r.success ? '#4caf50' : '#f44336'};">
                                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                            <span style="font-weight:bold; color:#64b5f6;">${r.model_name}</span>
                                            <div>
                                                ${!r.success ? '<span style="background:#f44336; padding:2px 6px; border-radius:3px; font-size:9px; color:#fff; margin-right:4px;">FAIL</span>' : ''}
                                                <span class="signal-badge ${signalClass[r.signal] || 'signal-neutral'}">${r.signal}</span>
                                            </div>
                                        </div>
                                        ${r.error_message ? `<div style="font-size:10px; color:#f44336; background:#2d1b1b; padding:4px 8px; border-radius:4px; margin-bottom:6px;">⚠️ ${r.error_message}</div>` : ''}
                                        <div style="font-size:11px; color:#aaa;">
                                            신뢰도: ${(r.confidence * 100).toFixed(0)}% |
                                            추세: ${r.trend_prediction} |
                                            진입: ${formatNumber(r.entry_price)}원 |
                                            처리: ${r.processing_time?.toFixed(1)}초
                                        </div>
                                        ${r.reasoning ? `<div style="font-size:11px; color:#81c784; margin-top:4px;">💡 ${r.reasoning}</div>` : ''}
                                        <div style="margin-top:8px; padding:6px; background:#0d1117; border-radius:4px; font-family:monospace; font-size:10px; max-height:100px; overflow-y:auto; color:#8b949e;">
                                            ${escapeHtml(r.raw_output || '').substring(0, 300)}${(r.raw_output || '').length > 300 ? '...' : ''}
                                        </div>
                                    </div>
                                `).join('')}
                            </div>

                            <!-- LLM Input (공통 프롬프트) -->
                            <div style="margin-top:15px; padding:10px; background:linear-gradient(135deg, #1a237e 0%, #0d47a1 100%); border-radius:8px; border:2px solid #2196f3;">
                                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                    <span style="color:#64b5f6; font-weight:bold; font-size:14px;">📥 공통 INPUT (프롬프트)</span>
                                    <span style="color:#90caf9; font-size:11px;">${(a.input_prompt || '').length} 글자</span>
                                </div>
                                <div class="llm-code" style="background:#0a1929; border:1px solid #1565c0; max-height:200px;">${escapeHtml(a.input_prompt || '')}</div>
                            </div>

                            <div class="llm-card-meta" style="margin-top:12px; padding:8px; background:#263238; border-radius:4px;">
                                🎯 모델: <strong>${(a.models_used || []).join(', ')}</strong> |
                                ⏱️ 총 처리시간: <strong>${a.total_processing_time?.toFixed(1) || 0}초</strong>
                            </div>
                        </div>
                    </div>
                `).join('');
            } else {
                // 기존 단일 모델 결과 표시
                panel.innerHTML = analyses.map((a, i) => {
                    const isFailed = a.success === false || (a.confidence === 0 && a.signal === 'HOLD');
                    const statusBadge = isFailed
                        ? '<span style="margin-left:8px; background:#f44336; padding:2px 6px; border-radius:3px; font-size:10px; color:#fff;">FAIL</span>'
                        : '<span style="margin-left:8px; background:#4caf50; padding:2px 6px; border-radius:3px; font-size:10px; color:#fff;">OK</span>';

                    return `
                    <div class="llm-card ${i === 0 ? 'expanded' : ''}" style="${isFailed ? 'border-color: #f44336;' : ''}">
                        <div class="llm-card-header" onclick="toggleCard(this.parentElement)">
                            <div>
                                <span class="llm-card-title">${a.stock_name} (${a.stock_code})</span>
                                ${statusBadge}
                                <span class="signal-badge ${signalClass[a.signal] || 'signal-neutral'}" style="margin-left:8px">${a.signal}</span>
                                <span style="margin-left:8px">${trendIcon[a.trend_prediction] || ''} ${a.trend_prediction}</span>
                            </div>
                            <div class="llm-card-meta">
                                신뢰도: ${(a.confidence * 100).toFixed(0)}% | ${a.timestamp} | 클릭하여 펼치기 ▼
                            </div>
                        </div>
                        <div class="llm-card-body">
                            <div class="llm-result">
                                <div class="llm-result-item">
                                    <div class="llm-result-label">진입가</div>
                                    <div class="llm-result-value">${formatNumber(a.entry_price)}원</div>
                                </div>
                                <div class="llm-result-item">
                                    <div class="llm-result-label">손절가</div>
                                    <div class="llm-result-value negative">${formatNumber(a.stop_loss)}원</div>
                                </div>
                                <div class="llm-result-item">
                                    <div class="llm-result-label">익절가</div>
                                    <div class="llm-result-value positive">${formatNumber(a.take_profit)}원</div>
                                </div>
                            </div>
                            <div class="llm-section" style="margin-top:12px">
                                <div class="llm-section-title">📝 분석 근거</div>
                                <div style="padding:8px; background:#21262d; border-radius:4px">${a.reasoning || '-'}</div>
                            </div>

                            <div style="margin-top:15px; padding:10px; background:linear-gradient(135deg, #1a237e 0%, #0d47a1 100%); border-radius:8px; border:2px solid #2196f3;">
                                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                    <span style="color:#64b5f6; font-weight:bold; font-size:14px;">📥 LLM INPUT</span>
                                    <span style="color:#90caf9; font-size:11px;">${(a.input_prompt || '').length} 글자</span>
                                </div>
                                <div class="llm-code" style="background:#0a1929; border:1px solid #1565c0; max-height:200px;">${escapeHtml(a.input_prompt || '')}</div>
                            </div>

                            <div style="margin-top:15px; padding:10px; background:linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); border-radius:8px; border:2px solid #4caf50;">
                                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                    <span style="color:#81c784; font-weight:bold; font-size:14px;">📤 LLM OUTPUT</span>
                                    <span style="color:#a5d6a7; font-size:11px;">${(a.output_raw || '').length} 글자</span>
                                </div>
                                <div class="llm-code" style="background:#0a1f0a; border:1px solid #388e3c; max-height:200px;">${escapeHtml(a.output_raw || '')}</div>
                            </div>

                            <div class="llm-card-meta" style="margin-top:12px; padding:8px; background:#263238; border-radius:4px;">
                                🤖 모델: <strong>${a.model_name}</strong> | ⏱️ 처리시간: <strong>${a.processing_time?.toFixed(1) || 0}초</strong>
                            </div>
                        </div>
                    </div>
                `;
                }).join('');
            }
        }

        function toggleCard(el) {
            el.classList.toggle('expanded');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function startAutoLLM() {
            const interval = parseInt(document.getElementById('llm-interval').value) * 1000;
            stopAutoLLM();
            addLog(`자동 LLM 분석 시작 (${interval/1000}초)`, 'llm');
            runLLMAnalysis();
            autoLLMTimer = setInterval(runLLMAnalysis, interval);
        }

        function stopAutoLLM() {
            if (autoLLMTimer) { clearInterval(autoLLMTimer); autoLLMTimer = null; }
            document.getElementById('llm-status').className = 'status-badge status-inactive';
            document.getElementById('llm-status').textContent = 'LLM 대기';
            addLog('자동 LLM 분석 중지', 'llm');
        }

        function selectStock(code, name) {
            addLog(`종목 선택: ${name} (${code})`);
        }

        // 백그라운드 데이터 자동 새로고침
        async function refreshBackgroundData() {
            try {
                // 급등 종목 새로고침
                const scanRes = await fetch('/api/scan');
                const scanData = await scanRes.json();
                if (!scanData.error && scanData.candidates) {
                    updateSurgeTable(scanData.candidates);
                }

                // 앙상블 LLM 히스토리 먼저 시도
                const ensembleRes = await fetch('/api/ensemble/history');
                const ensembleData = await ensembleRes.json();

                if (ensembleData.error) {
                    console.log('앙상블 히스토리 오류:', ensembleData.error);
                }

                if (!ensembleData.error && ensembleData.history && ensembleData.history.length > 0) {
                    console.log(`앙상블 히스토리 로드: ${ensembleData.history.length}개`);
                    updateLLMPanel(ensembleData.history);
                    document.getElementById('llm-status').className = 'status-badge status-active';
                    document.getElementById('llm-status').textContent = `앙상블 (${ensembleData.history.length})`;
                    if (ensembleData.models_used && ensembleData.models_used.length > 0) {
                        document.getElementById('model-list').textContent = ensembleData.models_used.join(', ');
                    } else {
                        document.getElementById('model-list').textContent = '모델 미설정';
                    }
                } else {
                    // 단일 LLM 히스토리로 폴백
                    const llmRes = await fetch('/api/llm/history');
                    const llmData = await llmRes.json();
                    if (!llmData.error && llmData.history && llmData.history.length > 0) {
                        console.log(`단일 LLM 히스토리 로드: ${llmData.history.length}개`);
                        updateLLMPanel(llmData.history);
                        document.getElementById('llm-status').className = 'status-badge status-active';
                        document.getElementById('llm-status').textContent = `LLM (${llmData.history.length})`;
                    } else {
                        console.log('LLM 히스토리 없음');
                    }
                }

                // 상태 정보 업데이트
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                if (statusData.background_running) {
                    document.getElementById('scan-status').className = 'status-badge status-active';
                    document.getElementById('scan-status').textContent = '자동실행중';
                }
                if (statusData.is_llm_running) {
                    document.getElementById('llm-status').className = 'status-badge status-running';
                    document.getElementById('llm-status').textContent = 'LLM 분석중...';
                }

            } catch (e) {
                console.error('백그라운드 새로고침 오류:', e);
            }
        }

        // =============================================================================
        // SSE (Server-Sent Events) 실시간 업데이트
        // =============================================================================
        let eventSource = null;
        let llmAnalyses = [];  // LLM 분석 결과 저장
        let sseReconnectTimer = null;
        let sseReconnectAttempts = 0;
        const MAX_RECONNECT_ATTEMPTS = 10;

        function connectSSE() {
            // 기존 연결 정리
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }

            // 재연결 타이머 정리
            if (sseReconnectTimer) {
                clearTimeout(sseReconnectTimer);
                sseReconnectTimer = null;
            }

            console.log('SSE 연결 시도...');
            addLog('🔄 실시간 연결 시도 중...', 'info');

            try {
                eventSource = new EventSource('/api/stream');

                eventSource.onopen = function() {
                    console.log('SSE 연결 성공');
                    sseReconnectAttempts = 0;  // 재연결 시도 횟수 리셋
                    addLog('🔗 실시간 연결 성공!', 'info');
                    document.getElementById('scan-status').className = 'status-badge status-active';
                    document.getElementById('scan-status').textContent = '🟢 실시간';
                };

                eventSource.onerror = function(e) {
                    console.error('SSE 연결 오류:', e);

                    if (eventSource.readyState === EventSource.CLOSED) {
                        addLog('⚠️ 실시간 연결 끊김', 'warn');
                        document.getElementById('scan-status').className = 'status-badge status-inactive';
                        document.getElementById('scan-status').textContent = '🔴 끊김';

                        // 재연결 시도
                        if (sseReconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                            sseReconnectAttempts++;
                            const delay = Math.min(1000 * sseReconnectAttempts, 10000);  // 최대 10초
                            addLog(`🔄 ${delay/1000}초 후 재연결 시도... (${sseReconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`, 'warn');
                            sseReconnectTimer = setTimeout(connectSSE, delay);
                        } else {
                            addLog('❌ 재연결 실패. 페이지를 새로고침 해주세요.', 'error');
                        }
                    }
                };

                // 연결 확인
                eventSource.addEventListener('connected', function(e) {
                    const data = JSON.parse(e.data);
                    console.log('SSE 연결됨:', data);
                    addLog(`✅ SSE 연결 완료 (클라이언트 ${data.client_count}개)`, 'info');
                });

                // 하트비트 - 연결 상태 표시 업데이트
                eventSource.addEventListener('heartbeat', function(e) {
                    const data = JSON.parse(e.data);
                    console.log('하트비트:', data.time);
                    // 연결 상태 표시 갱신
                    document.getElementById('scan-status').className = 'status-badge status-active';
                    document.getElementById('scan-status').textContent = '🟢 실시간';
                });

                // 급등 종목 업데이트
                eventSource.addEventListener('surge_update', function(e) {
                    try {
                        const data = JSON.parse(e.data);
                        console.log('📈 급등 종목 업데이트:', data.count + '개');
                        updateSurgeTable(data.candidates);
                        addLog(`📈 급등 종목 업데이트: ${data.count}개 (${data.timestamp})`, 'info');

                        // 스캔 상태 배지 갱신
                        document.getElementById('scan-status').className = 'status-badge status-active';
                        document.getElementById('scan-status').textContent = `🟢 ${data.count}종목`;
                    } catch (err) {
                        console.error('surge_update 파싱 오류:', err);
                    }
                });

                // LLM 분석 결과 업데이트
                eventSource.addEventListener('llm_update', function(e) {
                    try {
                        const data = JSON.parse(e.data);
                        console.log('🤖 LLM 분석 결과:', data.stock_name, data.ensemble_signal);

                        // 기존 결과에 추가 (최대 20개 유지)
                        llmAnalyses.unshift(data);
                        if (llmAnalyses.length > 20) {
                            llmAnalyses = llmAnalyses.slice(0, 20);
                        }

                        // UI 업데이트
                        updateLLMPanel(llmAnalyses);
                        addLog(`🤖 ${data.stock_name} 분석 완료: ${data.ensemble_signal} (신뢰도: ${(data.ensemble_confidence * 100).toFixed(0)}%)`, 'llm');

                        // LLM 상태 배지 갱신
                        document.getElementById('llm-status').className = 'status-badge status-active';
                        document.getElementById('llm-status').textContent = `앙상블 (${llmAnalyses.length})`;

                        // 모델 목록 업데이트
                        if (data.models_used && data.models_used.length > 0) {
                            document.getElementById('model-list').textContent = data.models_used.join(', ');
                        }
                    } catch (err) {
                        console.error('llm_update 파싱 오류:', err);
                    }
                });

                // 상태 업데이트
                eventSource.addEventListener('status_update', function(e) {
                    try {
                        const data = JSON.parse(e.data);
                        console.log('📊 상태 업데이트:', data);

                        // 스캔 상태
                        if (data.is_scanning) {
                            document.getElementById('scan-status').className = 'status-badge status-running';
                            document.getElementById('scan-status').textContent = '🔄 스캔중...';
                        } else if (data.candidates_count > 0) {
                            document.getElementById('scan-status').className = 'status-badge status-active';
                            document.getElementById('scan-status').textContent = `🟢 ${data.candidates_count}종목`;
                        }

                        // LLM 상태
                        if (data.is_llm_running) {
                            document.getElementById('llm-status').className = 'status-badge status-running';
                            document.getElementById('llm-status').textContent = '🔄 LLM 분석중...';
                        } else if (data.llm_history_count > 0) {
                            document.getElementById('llm-status').className = 'status-badge status-active';
                            document.getElementById('llm-status').textContent = `앙상블 (${data.llm_history_count})`;
                        }

                        // 모델 목록
                        if (data.ensemble_models && data.ensemble_models.length > 0) {
                            document.getElementById('model-list').textContent = data.ensemble_models.join(', ');
                        }
                    } catch (err) {
                        console.error('status_update 파싱 오류:', err);
                    }
                });

            } catch (err) {
                console.error('SSE EventSource 생성 오류:', err);
                addLog('❌ SSE 연결 생성 실패: ' + err.message, 'error');
            }
        }

        // SSE 연결 상태 확인
        function checkSSEConnection() {
            if (!eventSource || eventSource.readyState === EventSource.CLOSED) {
                console.log('SSE 연결 끊김 감지, 재연결...');
                connectSSE();
            }
        }

        // 초기화
        window.onload = function() {
            addLog('🚀 대시보드 v2 시작 - SSE 실시간 업데이트 활성화');
            refreshAccount();
            refreshOrders();
            refreshBackgroundData();  // 초기 데이터 로드

            // SSE 연결 시작
            connectSSE();

            // SSE 연결 상태 주기적 확인 (30초마다)
            setInterval(checkSSEConnection, 30000);
        };

        // 자동 새로고침: 계좌 5분마다
        setInterval(refreshAccount, 300000);

        // 페이지 visibility 변경 시 SSE 재연결
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'visible') {
                console.log('페이지 활성화됨, SSE 연결 확인');
                checkSSEConnection();
            }
        });
    </script>
</body>
</html>
"""


def init_kis():
    """KIS API 초기화"""
    try:
        from modules.config_loader import setup_kis_config, load_env_file
        load_env_file()
        setup_kis_config()

        import kis_auth as ka
        ka.auth(svr="prod")

        data_store.kis_authenticated = True
        logger.info("KIS API 인증 완료")
        return True
    except Exception as e:
        data_store.error_message = str(e)
        logger.error(f"KIS API 초기화 실패: {e}")
        return False


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/stream')
def api_stream():
    """SSE 실시간 스트림 엔드포인트"""
    def event_stream():
        client_queue = queue.Queue(maxsize=100)
        with sse_lock:
            sse_clients.append(client_queue)
            logger.info(f"🔗 SSE 클라이언트 연결됨 (총 {len(sse_clients)}개)")

        try:
            # 연결 시 초기 데이터 전송
            yield f"event: connected\ndata: {json.dumps({'message': 'SSE connected', 'client_count': len(sse_clients)})}\n\n"

            # 초기 데이터도 함께 전송
            if data_store.surge_candidates:
                candidates = []
                for c in data_store.surge_candidates[:15]:
                    candidates.append({
                        "rank": c.rank, "code": c.code, "name": c.name,
                        "price": c.price, "change": c.change, "change_rate": c.change_rate,
                        "volume": c.volume, "volume_power": c.volume_power,
                        "balance_ratio": c.balance_ratio, "surge_score": c.surge_score,
                        "signal": c.signal, "detected_at": c.detected_at, "reasons": c.reasons,
                    })
                yield f"event: surge_update\ndata: {json.dumps({'candidates': candidates, 'timestamp': datetime.now().strftime('%H:%M:%S'), 'count': len(candidates)}, ensure_ascii=False)}\n\n"

            while True:
                try:
                    message = client_queue.get(timeout=15)  # 15초 타임아웃으로 단축
                    yield message
                except queue.Empty:
                    # 하트비트 전송 (연결 유지)
                    yield f"event: heartbeat\ndata: {json.dumps({'time': datetime.now().strftime('%H:%M:%S'), 'clients': len(sse_clients)})}\n\n"
        except GeneratorExit:
            logger.info("SSE GeneratorExit")
        except Exception as e:
            logger.error(f"SSE 스트림 오류: {e}")
        finally:
            with sse_lock:
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)
                    logger.info(f"🔌 SSE 클라이언트 연결 해제 (남은 {len(sse_clients)}개)")

    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )


@app.route('/api/account')
def api_account():
    if not data_store.kis_authenticated:
        if not init_kis():
            return jsonify({"error": "KIS API 인증 실패"})

    try:
        import kis_auth as ka
        from domestic_stock.inquire_balance.inquire_balance import inquire_balance

        trenv = ka.getTREnv()
        df1, df2 = inquire_balance(
            env_dv="real", cano=trenv.my_acct, acnt_prdt_cd=trenv.my_prod,
            afhr_flpr_yn="N", inqr_dvsn="02", unpr_dvsn="01",
            fund_sttl_icld_yn="N", fncg_amt_auto_rdpt_yn="N", prcs_dvsn="00"
        )

        summary = {}
        if not df2.empty:
            row = df2.iloc[0]
            summary = {
                "deposit": int(row.get('dnca_tot_amt', 0)),
                "total_eval": int(row.get('tot_evlu_amt', 0)),
                "total_pl": int(row.get('evlu_pfls_smtl_amt', 0)),
                "pl_rate": float(row.get('asst_icdc_erng_rt', 0)),
            }

        holdings = []
        if not df1.empty:
            for _, row in df1.iterrows():
                if int(row.get('hldg_qty', 0)) > 0:
                    holdings.append({
                        "code": row.get('pdno', ''),
                        "name": row.get('prdt_name', ''),
                        "qty": int(row.get('hldg_qty', 0)),
                        "avg_price": float(row.get('pchs_avg_pric', 0)),
                        "current_price": int(row.get('prpr', 0)),
                        "pl": int(row.get('evlu_pfls_amt', 0)),
                        "pl_rate": float(row.get('evlu_pfls_rt', 0)),
                    })

        summary["holdings"] = holdings
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/orders')
def api_orders():
    if not data_store.kis_authenticated:
        if not init_kis():
            return jsonify({"error": "KIS API 인증 실패"})

    try:
        import kis_auth as ka
        from domestic_stock.inquire_daily_ccld.inquire_daily_ccld import inquire_daily_ccld

        trenv = ka.getTREnv()
        today = datetime.now().strftime("%Y%m%d")

        df1, df2 = inquire_daily_ccld(
            env_dv="real", pd_dv="inner", cano=trenv.my_acct, acnt_prdt_cd=trenv.my_prod,
            inqr_strt_dt=today, inqr_end_dt=today, sll_buy_dvsn_cd="00",
            ccld_dvsn="01", inqr_dvsn="00", inqr_dvsn_3="00"
        )

        orders = []
        if not df1.empty:
            for _, row in df1.iterrows():
                orders.append({
                    "time": row.get('ord_tmd', '')[:6],
                    "name": row.get('prdt_name', ''),
                    "side": "매수" if row.get('sll_buy_dvsn_cd') == "02" else "매도",
                    "qty": int(row.get('tot_ccld_qty', 0)),
                    "price": int(row.get('avg_prvs', 0)),
                })

        return jsonify({"orders": orders})
    except Exception as e:
        return jsonify({"error": str(e), "orders": []})


@app.route('/api/scan')
def api_scan():
    if not data_store.kis_authenticated:
        if not init_kis():
            return jsonify({"error": "KIS API 인증 실패"})

    try:
        from modules.surge_detector import SurgeDetector

        detector = SurgeDetector()
        detector._authenticated = True
        candidates = detector.scan_surge_stocks(min_score=40)

        data_store.surge_candidates = candidates
        data_store.last_scan_time = datetime.now()

        result = []
        for c in candidates:
            result.append({
                "rank": c.rank, "code": c.code, "name": c.name,
                "price": c.price, "change": c.change, "change_rate": c.change_rate,
                "volume": c.volume, "volume_power": c.volume_power,
                "balance_ratio": c.balance_ratio, "surge_score": c.surge_score,
                "signal": c.signal, "detected_at": c.detected_at, "reasons": c.reasons,
            })

        return jsonify({"candidates": result})
    except Exception as e:
        return jsonify({"error": str(e), "candidates": []})


@app.route('/api/llm/analyze')
def api_llm_analyze():
    """LLM 분석 API - 급등종목 + 뉴스 종합 분석"""
    if not data_store.kis_authenticated:
        if not init_kis():
            return jsonify({"error": "KIS API 인증 실패"})

    try:
        from modules.llm_analyzer import LLMAnalyzer
        from modules.surge_detector import SurgeDetector

        # 급등 종목 스캔 (캐시 또는 새로 스캔)
        if not data_store.surge_candidates:
            detector = SurgeDetector()
            detector._authenticated = True
            data_store.surge_candidates = detector.scan_surge_stocks(min_score=40)

        # LLM 분석기
        analyzer = LLMAnalyzer()

        # 모델 확인
        model = analyzer.get_available_model()
        if not model:
            return jsonify({"error": "LLM 모델 없음 (Ollama 실행 확인)", "analyses": []})

        data_store.llm_model = model

        # 상위 종목 분석 (뉴스 포함)
        analyses = analyzer.analyze_surge_candidates(
            data_store.surge_candidates,
            max_analyze=5,
            include_news=True
        )

        # 결과 저장
        for a in analyses:
            data_store.llm_analyses.append(a)

        data_store.last_llm_time = datetime.now()

        # 직렬화
        result = []
        for a in analyses:
            result.append(asdict(a))

        return jsonify({"analyses": result, "model": model})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "analyses": []})


@app.route('/api/llm/history')
def api_llm_history():
    """LLM 분석 히스토리"""
    try:
        from dataclasses import asdict
        history = [asdict(a) for a in list(data_store.llm_analyses)[-20:]]
        return jsonify({"history": list(reversed(history))})
    except Exception as e:
        return jsonify({"error": str(e), "history": []})


@app.route('/api/ensemble/analyze')
def api_ensemble_analyze():
    """앙상블 LLM 분석 API - 다중 모델 투표"""
    if not data_store.kis_authenticated:
        if not init_kis():
            return jsonify({"error": "KIS API 인증 실패"})

    try:
        from modules.ensemble_analyzer import get_ensemble_analyzer
        from modules.llm_analyzer import LLMAnalyzer
        from modules.surge_detector import SurgeDetector

        # 급등 종목 스캔 (캐시 또는 새로 스캔)
        if not data_store.surge_candidates:
            detector = SurgeDetector()
            detector._authenticated = True
            data_store.surge_candidates = detector.scan_surge_stocks(min_score=40)

        # 앙상블 분석기 (금융 특화 앙상블 사용)
        ensemble = get_ensemble_analyzer()
        if not ensemble.ensemble_models:
            ensemble.setup_ensemble(use_financial_ensemble=True)

        if not ensemble.ensemble_models:
            return jsonify({"error": "앙상블 모델 없음 (Ollama 실행 확인)", "analyses": []})

        data_store.ensemble_models = ensemble.ensemble_models
        data_store.llm_model = f"앙상블 ({len(ensemble.ensemble_models)}모델)"

        # 상위 종목 분석
        priority = [c for c in data_store.surge_candidates
                   if c.signal in ["STRONG_BUY", "BUY"]][:3]

        if not priority:
            priority = data_store.surge_candidates[:2]

        if not priority:
            return jsonify({"error": "분석할 종목 없음", "analyses": []})

        # 뉴스 조회용
        single_analyzer = LLMAnalyzer()

        analyses = []
        for candidate in priority:
            stock_data = {
                'code': candidate.code,
                'name': candidate.name,
                'price': candidate.price,
                'change_rate': candidate.change_rate,
                'volume_power': candidate.volume_power,
                'balance_ratio': candidate.balance_ratio,
                'surge_score': candidate.surge_score,
                'volume': candidate.volume
            }

            # 뉴스 조회
            news_list = single_analyzer.get_news_for_stock(candidate.code, candidate.name)

            # 앙상블 분석 (순차 실행으로 GPU 충돌 방지)
            result = ensemble.analyze_stock(stock_data, news_list, parallel=False)
            data_store.ensemble_analyses.append(result)
            analyses.append(result)

        data_store.last_llm_time = datetime.now()

        # 결과 직렬화
        result_list = ensemble.get_history(len(analyses))

        return jsonify({
            "analyses": result_list,
            "models_used": ensemble.ensemble_models,
            "model_weights": ensemble.model_weights
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "analyses": []})


@app.route('/api/ensemble/history')
def api_ensemble_history():
    """앙상블 분석 히스토리"""
    try:
        from modules.ensemble_analyzer import get_ensemble_analyzer
        from dataclasses import asdict

        ensemble = get_ensemble_analyzer()

        # 싱글톤 앙상블 분석기의 히스토리 가져오기
        history = ensemble.get_history(20)

        # data_store에 저장된 결과도 포함 (백그라운드 분석 결과)
        if data_store.ensemble_analyses:
            store_history = []
            for h in list(data_store.ensemble_analyses)[-20:]:
                try:
                    # EnsembleAnalysis 객체를 딕셔너리로 변환
                    h_dict = {
                        "timestamp": h.timestamp,
                        "stock_code": h.stock_code,
                        "stock_name": h.stock_name,
                        "ensemble_signal": h.ensemble_signal,
                        "ensemble_confidence": h.ensemble_confidence,
                        "ensemble_trend": h.ensemble_trend,
                        "avg_entry_price": h.avg_entry_price,
                        "avg_stop_loss": h.avg_stop_loss,
                        "avg_take_profit": h.avg_take_profit,
                        "signal_votes": h.signal_votes,
                        "trend_votes": h.trend_votes,
                        "models_used": h.models_used,
                        "models_agreed": h.models_agreed,
                        "total_models": h.total_models,
                        "consensus_score": h.consensus_score,
                        "total_processing_time": h.total_processing_time,
                        "success": h.success,
                        "input_prompt": h.input_prompt,
                        "model_results": [
                            {
                                "model_name": r.model_name,
                                "signal": r.signal,
                                "confidence": r.confidence,
                                "trend_prediction": r.trend_prediction,
                                "entry_price": r.entry_price,
                                "stop_loss": r.stop_loss,
                                "take_profit": r.take_profit,
                                "reasoning": r.reasoning,
                                "processing_time": r.processing_time,
                                "success": r.success,
                                "raw_output": r.raw_output,
                                "error_message": getattr(r, 'error_message', '')
                            }
                            for r in h.model_results
                        ]
                    }
                    store_history.append(h_dict)
                except Exception as conv_err:
                    logger.warning(f"히스토리 변환 오류: {conv_err}")

            # 기존 히스토리와 병합 (중복 제거)
            existing_keys = {(h.get("stock_code"), h.get("timestamp")) for h in history}
            for h in store_history:
                key = (h.get("stock_code"), h.get("timestamp"))
                if key not in existing_keys:
                    history.append(h)

            # 시간순 정렬 (최신순)
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            history = history[:20]

        return jsonify({
            "history": history,
            "models_used": ensemble.ensemble_models,
            "model_weights": ensemble.model_weights
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "history": []})


@app.route('/api/status')
def api_status():
    return jsonify({
        "authenticated": data_store.kis_authenticated,
        "llm_model": data_store.llm_model,
        "last_scan": data_store.last_scan_time.isoformat() if data_store.last_scan_time else None,
        "last_llm": data_store.last_llm_time.isoformat() if data_store.last_llm_time else None,
        "candidates_count": len(data_store.surge_candidates),
        "llm_history_count": len(data_store.llm_analyses),
        "llm_auto_enabled": data_store.llm_auto_enabled,
        "is_scanning": data_store.is_scanning,
        "is_llm_running": data_store.is_llm_running,
        "scan_interval": data_store.scan_interval,
        "llm_interval": data_store.llm_interval,
        "background_running": data_store.background_thread is not None and data_store.background_thread.is_alive(),
    })


@app.route('/api/settings', methods=['POST'])
def api_settings():
    """설정 변경 API"""
    try:
        data = request.get_json() or {}

        if 'llm_auto_enabled' in data:
            data_store.llm_auto_enabled = bool(data['llm_auto_enabled'])
            logger.info(f"LLM 자동 분석: {'활성화' if data_store.llm_auto_enabled else '비활성화'}")

        if 'scan_interval' in data:
            data_store.scan_interval = max(30, int(data['scan_interval']))  # 최소 30초

        if 'llm_interval' in data:
            data_store.llm_interval = max(60, int(data['llm_interval']))  # 최소 60초

        return jsonify({
            "success": True,
            "llm_auto_enabled": data_store.llm_auto_enabled,
            "scan_interval": data_store.scan_interval,
            "llm_interval": data_store.llm_interval,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/background/start', methods=['POST'])
def api_background_start():
    """백그라운드 분석 시작"""
    start_background_thread()
    return jsonify({"success": True, "message": "백그라운드 분석 시작됨"})


@app.route('/api/background/stop', methods=['POST'])
def api_background_stop():
    """백그라운드 분석 중지"""
    stop_background_thread()
    return jsonify({"success": True, "message": "백그라운드 분석 중지 요청됨"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=DASHBOARD_PORT, help='Server port (default: 5001)')
    parser.add_argument('--no-auto', action='store_true', help='Disable auto analysis')
    args = parser.parse_args()

    # 포트 고정 (항상 DASHBOARD_PORT 사용)
    port = DASHBOARD_PORT

    logger.info(f"=" * 60)
    logger.info(f"스캘핑 대시보드 v2 시작")
    logger.info(f"접속 URL: http://localhost:{port}")
    logger.info(f"=" * 60)

    init_kis()

    # 자동 분석 비활성화 옵션
    if args.no_auto:
        data_store.llm_auto_enabled = False
        logger.info("자동 LLM 분석 비활성화됨")

    # 백그라운드 분석 스레드 시작
    start_background_thread()

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
