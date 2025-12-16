# -*- coding: utf-8 -*-
"""
자동 매매 웹 대시보드

실시간으로 자동 매매 상태를 모니터링할 수 있는 웹 인터페이스입니다.

접속: http://localhost:5002
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path

# Flask 임포트
try:
    from flask import Flask, jsonify, render_template_string
    from flask_cors import CORS
except ImportError:
    print("Flask가 필요합니다: pip install flask flask-cors")
    sys.exit(1)

logger = logging.getLogger(__name__)

# 대시보드 포트
DASHBOARD_PORT = 5002


@dataclass
class DashboardState:
    """대시보드 상태"""
    mode: str = "INIT"  # INIT, NEWS, TRADING, IDLE
    mode_description: str = "초기화 중..."
    last_update: str = ""
    market_status: str = "장 시작 전"

    # 뉴스 분석
    news_count: int = 0
    market_sentiment: str = "NEUTRAL"
    key_themes: List[str] = None
    attention_stocks: List[Dict] = None
    market_outlook: str = ""

    # 매매 결과
    trade_results: List[Dict] = None
    total_trades: int = 0
    successful_trades: int = 0

    # LLM 상태
    llm_model: str = ""
    llm_status: str = "대기"

    # 시스템
    is_running: bool = False
    scan_interval: int = 60
    next_scan: str = ""

    def __post_init__(self):
        if self.key_themes is None:
            self.key_themes = []
        if self.attention_stocks is None:
            self.attention_stocks = []
        if self.trade_results is None:
            self.trade_results = []


# 전역 상태
dashboard_state = DashboardState()
state_lock = threading.Lock()


def update_state(**kwargs):
    """대시보드 상태 업데이트"""
    global dashboard_state
    with state_lock:
        for key, value in kwargs.items():
            if hasattr(dashboard_state, key):
                setattr(dashboard_state, key, value)
        dashboard_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_state() -> Dict:
    """현재 상태 가져오기"""
    with state_lock:
        return {
            "mode": dashboard_state.mode,
            "mode_description": dashboard_state.mode_description,
            "last_update": dashboard_state.last_update,
            "market_status": dashboard_state.market_status,
            "news_count": dashboard_state.news_count,
            "market_sentiment": dashboard_state.market_sentiment,
            "key_themes": dashboard_state.key_themes,
            "attention_stocks": dashboard_state.attention_stocks[:5] if dashboard_state.attention_stocks else [],
            "market_outlook": dashboard_state.market_outlook,
            "trade_results": dashboard_state.trade_results[-10:] if dashboard_state.trade_results else [],
            "total_trades": dashboard_state.total_trades,
            "successful_trades": dashboard_state.successful_trades,
            "llm_model": dashboard_state.llm_model,
            "llm_status": dashboard_state.llm_status,
            "is_running": dashboard_state.is_running,
            "scan_interval": dashboard_state.scan_interval,
            "next_scan": dashboard_state.next_scan,
        }


# Flask 앱
app = Flask(__name__)
CORS(app)

# HTML 템플릿
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>자동 매매 대시보드</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { color: #00d4ff; font-size: 24px; }
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        .status-running { background: #00c853; color: #000; }
        .status-stopped { background: #ff5252; color: #fff; }
        .status-news { background: #ffd600; color: #000; }
        .status-trading { background: #00e676; color: #000; }

        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        .card-title {
            color: #00d4ff;
            font-size: 16px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #0f3460;
        }

        .sentiment { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .sentiment.BULLISH { color: #00e676; }
        .sentiment.BEARISH { color: #ff5252; }
        .sentiment.NEUTRAL { color: #ffd600; }

        .themes { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
        .theme-tag {
            background: #0f3460;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 13px;
        }

        .stock-list { list-style: none; }
        .stock-item {
            padding: 10px;
            margin: 5px 0;
            background: #0f3460;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stock-name { font-weight: bold; color: #00d4ff; }
        .stock-reason { font-size: 12px; color: #aaa; margin-top: 5px; }

        .trade-item {
            padding: 10px;
            margin: 5px 0;
            background: #0f3460;
            border-radius: 5px;
        }
        .trade-item.success { border-left: 3px solid #00e676; }
        .trade-item.failed { border-left: 3px solid #ff5252; }
        .trade-header { display: flex; justify-content: space-between; }
        .trade-signal { font-weight: bold; }
        .trade-signal.BUY, .trade-signal.STRONG_BUY { color: #ff5252; }
        .trade-signal.SELL, .trade-signal.STRONG_SELL { color: #448aff; }
        .trade-signal.HOLD { color: #ffd600; }

        .info-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #0f3460; }
        .info-label { color: #888; }
        .info-value { font-weight: bold; }

        .outlook {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-style: italic;
            line-height: 1.6;
        }

        .refresh-time { color: #666; font-size: 12px; text-align: right; margin-top: 10px; }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .loading { animation: pulse 1.5s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>KIS 자동 매매 대시보드</h1>
                <p style="color: #888; margin-top: 5px;" id="mode-desc">초기화 중...</p>
            </div>
            <div>
                <span class="status-badge status-stopped" id="status-badge">STOPPED</span>
            </div>
        </header>

        <div class="grid">
            <!-- 시장 분석 카드 -->
            <div class="card">
                <div class="card-title">시장 분석</div>
                <div class="sentiment" id="sentiment">--</div>
                <div class="themes" id="themes"></div>
                <div class="info-row">
                    <span class="info-label">수집 뉴스</span>
                    <span class="info-value" id="news-count">0건</span>
                </div>
                <div class="outlook" id="outlook">분석 대기 중...</div>
            </div>

            <!-- 주목 종목 카드 -->
            <div class="card">
                <div class="card-title">주목 종목</div>
                <ul class="stock-list" id="attention-stocks">
                    <li class="stock-item">분석 대기 중...</li>
                </ul>
            </div>

            <!-- 매매 현황 카드 -->
            <div class="card">
                <div class="card-title">매매 현황</div>
                <div class="info-row">
                    <span class="info-label">총 분석</span>
                    <span class="info-value" id="total-trades">0건</span>
                </div>
                <div class="info-row">
                    <span class="info-label">매매 실행</span>
                    <span class="info-value" id="success-trades">0건</span>
                </div>
                <div id="trade-list" style="margin-top: 15px; max-height: 300px; overflow-y: auto;"></div>
            </div>

            <!-- 시스템 상태 카드 -->
            <div class="card">
                <div class="card-title">시스템 상태</div>
                <div class="info-row">
                    <span class="info-label">LLM 모델</span>
                    <span class="info-value" id="llm-model">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">LLM 상태</span>
                    <span class="info-value" id="llm-status">대기</span>
                </div>
                <div class="info-row">
                    <span class="info-label">스캔 주기</span>
                    <span class="info-value" id="scan-interval">60초</span>
                </div>
                <div class="info-row">
                    <span class="info-label">다음 스캔</span>
                    <span class="info-value" id="next-scan">-</span>
                </div>
                <div class="refresh-time" id="last-update">마지막 업데이트: -</div>
            </div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            fetch('/api/state')
                .then(res => res.json())
                .then(data => {
                    // 상태 배지
                    const badge = document.getElementById('status-badge');
                    badge.textContent = data.mode;
                    badge.className = 'status-badge';
                    if (data.mode === 'NEWS') badge.classList.add('status-news');
                    else if (data.mode === 'TRADING') badge.classList.add('status-trading');
                    else if (data.is_running) badge.classList.add('status-running');
                    else badge.classList.add('status-stopped');

                    document.getElementById('mode-desc').textContent = data.mode_description;

                    // 시장 분석
                    const sentiment = document.getElementById('sentiment');
                    sentiment.textContent = data.market_sentiment;
                    sentiment.className = 'sentiment ' + data.market_sentiment;

                    document.getElementById('news-count').textContent = data.news_count + '건';
                    document.getElementById('outlook').textContent = data.market_outlook || '분석 대기 중...';

                    // 테마
                    const themes = document.getElementById('themes');
                    themes.innerHTML = (data.key_themes || []).map(t =>
                        `<span class="theme-tag">${t}</span>`
                    ).join('');

                    // 주목 종목
                    const stocks = document.getElementById('attention-stocks');
                    if (data.attention_stocks && data.attention_stocks.length > 0) {
                        stocks.innerHTML = data.attention_stocks.map(s => `
                            <li class="stock-item">
                                <div>
                                    <div class="stock-name">${s.name || '알 수 없음'}</div>
                                    <div class="stock-reason">${(s.reason || '').substring(0, 50)}...</div>
                                </div>
                            </li>
                        `).join('');
                    } else {
                        stocks.innerHTML = '<li class="stock-item">분석 대기 중...</li>';
                    }

                    // 매매 현황
                    document.getElementById('total-trades').textContent = data.total_trades + '건';
                    document.getElementById('success-trades').textContent = data.successful_trades + '건';

                    const tradeList = document.getElementById('trade-list');
                    if (data.trade_results && data.trade_results.length > 0) {
                        tradeList.innerHTML = data.trade_results.slice().reverse().map(t => `
                            <div class="trade-item ${t.success ? 'success' : 'failed'}">
                                <div class="trade-header">
                                    <span>${t.stock_name || t.stock_code}</span>
                                    <span class="trade-signal ${t.ensemble_signal}">${t.ensemble_signal}</span>
                                </div>
                                <div style="font-size: 12px; color: #888; margin-top: 5px;">
                                    신뢰도: ${(t.confidence * 100).toFixed(0)}% | ${t.action}
                                </div>
                            </div>
                        `).join('');
                    } else {
                        tradeList.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">매매 내역 없음</div>';
                    }

                    // 시스템 상태
                    document.getElementById('llm-model').textContent = data.llm_model || '-';
                    document.getElementById('llm-status').textContent = data.llm_status || '대기';
                    document.getElementById('scan-interval').textContent = data.scan_interval + '초';
                    document.getElementById('next-scan').textContent = data.next_scan || '-';
                    document.getElementById('last-update').textContent = '마지막 업데이트: ' + data.last_update;
                })
                .catch(err => console.error('Update error:', err));
        }

        // 3초마다 업데이트
        setInterval(updateDashboard, 3000);
        updateDashboard();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/state')
def api_state():
    return jsonify(get_state())


def run_dashboard():
    """대시보드 서버 시작 (별도 스레드)"""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False, use_reloader=False, threaded=True)


def start_dashboard_thread():
    """대시보드를 별도 스레드로 시작"""
    thread = threading.Thread(target=run_dashboard, daemon=True)
    thread.start()
    logger.info(f"웹 대시보드 시작: http://localhost:{DASHBOARD_PORT}")
    return thread


# 상태 업데이트 헬퍼 함수들
def set_news_analysis_result(news_result: Dict):
    """뉴스 분석 결과 업데이트"""
    update_state(
        mode="NEWS",
        mode_description="뉴스 분석 모드",
        news_count=news_result.get('news_count', 0),
        market_sentiment=news_result.get('market_sentiment', 'NEUTRAL'),
        key_themes=news_result.get('key_themes', []),
        attention_stocks=news_result.get('attention_stocks', []),
        market_outlook=news_result.get('llm_analysis', {}).get('market_outlook', ''),
        llm_status="분석 완료"
    )


def set_trading_result(results: List, trader=None):
    """매매 결과 업데이트"""
    trade_data = []
    for r in results:
        trade_data.append({
            "stock_code": r.stock_code,
            "stock_name": r.stock_name,
            "ensemble_signal": r.ensemble_signal,
            "confidence": r.confidence,
            "consensus": r.consensus,
            "action": r.action,
            "success": r.success,
            "reason": r.reason,
        })

    with state_lock:
        dashboard_state.trade_results.extend(trade_data)
        dashboard_state.total_trades = len(dashboard_state.trade_results)
        dashboard_state.successful_trades = sum(1 for t in dashboard_state.trade_results if t['success'])

    update_state(
        mode="TRADING",
        mode_description="매매 + 분석 모드",
        llm_status="분석 완료"
    )


def set_system_status(is_running: bool, llm_model: str = "", scan_interval: int = 60, next_scan: str = ""):
    """시스템 상태 업데이트"""
    update_state(
        is_running=is_running,
        llm_model=llm_model,
        scan_interval=scan_interval,
        next_scan=next_scan
    )


def set_llm_status(status: str):
    """LLM 상태 업데이트"""
    update_state(llm_status=status)
