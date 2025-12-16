# -*- coding: utf-8 -*-
"""
스캘핑 대시보드 - 24/365 웹 UI 기반 모니터링 시스템

기능:
- 실시간 급등 종목 탐지 및 표시
- 계좌 잔고 및 수익률 조회
- 주문 체결 내역 조회
- 자동 스캔 백그라운드 실행
- WebSocket을 통한 실시간 업데이트

실행:
    python scalping_dashboard.py

접속:
    http://localhost:5000
"""

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

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 전역 데이터 저장소
class DataStore:
    def __init__(self):
        self.surge_candidates = []
        self.account_balance = {}
        self.holdings = []
        self.orders = []
        self.last_scan_time = None
        self.is_scanning = False
        self.scan_interval = 60  # 초
        self.auto_scan_enabled = True
        self.kis_authenticated = False
        self.error_message = None

data_store = DataStore()


# HTML 템플릿
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>스캘핑 대시보드 - 24/365</title>
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
            padding: 20px;
            border-bottom: 2px solid #0f4c75;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 24px;
            color: #00d9ff;
        }
        .header-info {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-active { background: #00c853; color: #000; }
        .status-inactive { background: #ff5252; color: #fff; }
        .status-scanning { background: #ffc107; color: #000; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .panel {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2a2a4e;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2a2a4e;
        }
        .panel-title {
            font-size: 16px;
            color: #00d9ff;
            font-weight: bold;
        }
        .full-width { grid-column: 1 / -1; }

        /* 계좌 정보 */
        .account-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }
        .account-item {
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .account-label { font-size: 12px; color: #888; margin-bottom: 5px; }
        .account-value { font-size: 20px; font-weight: bold; }
        .positive { color: #00c853; }
        .negative { color: #ff5252; }

        /* 급등 종목 테이블 */
        .surge-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .surge-table th {
            background: #16213e;
            padding: 12px 8px;
            text-align: left;
            color: #00d9ff;
            font-weight: 600;
        }
        .surge-table td {
            padding: 10px 8px;
            border-bottom: 1px solid #2a2a4e;
        }
        .surge-table tr:hover { background: #16213e; }
        .signal-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }
        .signal-strong-buy { background: #ff5722; color: #fff; }
        .signal-buy { background: #4caf50; color: #fff; }
        .signal-watch { background: #ff9800; color: #000; }
        .signal-neutral { background: #607d8b; color: #fff; }

        /* 보유 종목 */
        .holdings-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .holdings-table th {
            background: #16213e;
            padding: 10px 8px;
            text-align: left;
            color: #00d9ff;
        }
        .holdings-table td {
            padding: 8px;
            border-bottom: 1px solid #2a2a4e;
        }

        /* 버튼 */
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn-primary { background: #0f4c75; color: #fff; }
        .btn-primary:hover { background: #1b6fa3; }
        .btn-danger { background: #c62828; color: #fff; }
        .btn-danger:hover { background: #e53935; }
        .btn-success { background: #2e7d32; color: #fff; }
        .btn-success:hover { background: #388e3c; }

        /* 로그 */
        .log-container {
            background: #0a0a0a;
            border-radius: 8px;
            padding: 10px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .log-entry { padding: 4px 0; border-bottom: 1px solid #1a1a2e; }
        .log-time { color: #666; }
        .log-info { color: #00d9ff; }
        .log-warn { color: #ffc107; }
        .log-error { color: #ff5252; }

        /* 컨트롤 패널 */
        .control-panel {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .control-panel select, .control-panel input {
            padding: 8px;
            border-radius: 6px;
            border: 1px solid #2a2a4e;
            background: #16213e;
            color: #e0e0e0;
        }

        /* 실시간 시계 */
        .clock {
            font-size: 14px;
            color: #00d9ff;
            font-family: monospace;
        }

        /* 반응형 */
        @media (max-width: 1200px) {
            .container { grid-template-columns: 1fr; }
            .account-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 스캘핑 대시보드</h1>
        <div class="header-info">
            <div class="clock" id="clock"></div>
            <span id="scan-status" class="status-badge status-inactive">대기중</span>
            <span id="auth-status" class="status-badge status-inactive">미인증</span>
        </div>
    </div>

    <div class="container">
        <!-- 계좌 정보 -->
        <div class="panel full-width">
            <div class="panel-header">
                <span class="panel-title">💰 계좌 정보</span>
                <button class="btn btn-primary" onclick="refreshAccount()">새로고침</button>
            </div>
            <div class="account-grid" id="account-info">
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
        <div class="panel full-width">
            <div class="panel-header">
                <span class="panel-title">🔥 급등 종목 (체결강도 상위)</span>
                <div class="control-panel">
                    <select id="scan-interval" onchange="updateInterval()">
                        <option value="30">30초</option>
                        <option value="60" selected>1분</option>
                        <option value="120">2분</option>
                        <option value="300">5분</option>
                    </select>
                    <button class="btn btn-success" onclick="startAutoScan()">자동스캔 시작</button>
                    <button class="btn btn-danger" onclick="stopAutoScan()">중지</button>
                    <button class="btn btn-primary" onclick="manualScan()">수동스캔</button>
                </div>
            </div>
            <div style="overflow-x: auto;">
                <table class="surge-table">
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>종목명</th>
                            <th>현재가</th>
                            <th>등락률</th>
                            <th>체결강도</th>
                            <th>호가비</th>
                            <th>점수</th>
                            <th>시그널</th>
                            <th>탐지시간</th>
                        </tr>
                    </thead>
                    <tbody id="surge-table-body">
                        <tr><td colspan="9" style="text-align:center; color:#666;">스캔 대기중...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 보유 종목 -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">📊 보유 종목</span>
            </div>
            <table class="holdings-table">
                <thead>
                    <tr>
                        <th>종목명</th>
                        <th>수량</th>
                        <th>평균가</th>
                        <th>현재가</th>
                        <th>손익</th>
                        <th>수익률</th>
                    </tr>
                </thead>
                <tbody id="holdings-body">
                    <tr><td colspan="6" style="text-align:center; color:#666;">데이터 없음</td></tr>
                </tbody>
            </table>
        </div>

        <!-- 체결 내역 -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">📝 오늘 체결 내역</span>
                <button class="btn btn-primary" onclick="refreshOrders()">새로고침</button>
            </div>
            <table class="holdings-table">
                <thead>
                    <tr>
                        <th>시간</th>
                        <th>종목</th>
                        <th>구분</th>
                        <th>수량</th>
                        <th>가격</th>
                    </tr>
                </thead>
                <tbody id="orders-body">
                    <tr><td colspan="5" style="text-align:center; color:#666;">데이터 없음</td></tr>
                </tbody>
            </table>
        </div>

        <!-- 로그 -->
        <div class="panel full-width">
            <div class="panel-header">
                <span class="panel-title">📋 시스템 로그</span>
                <button class="btn btn-danger" onclick="clearLogs()">로그 지우기</button>
            </div>
            <div class="log-container" id="log-container">
                <div class="log-entry"><span class="log-time">[시작]</span> <span class="log-info">대시보드가 로드되었습니다.</span></div>
            </div>
        </div>
    </div>

    <script>
        // 시계 업데이트
        function updateClock() {
            const now = new Date();
            document.getElementById('clock').textContent = now.toLocaleString('ko-KR');
        }
        setInterval(updateClock, 1000);
        updateClock();

        // 로그 추가
        function addLog(message, type = 'info') {
            const container = document.getElementById('log-container');
            const time = new Date().toLocaleTimeString('ko-KR');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-${type}">${message}</span>`;
            container.insertBefore(entry, container.firstChild);
            if (container.children.length > 100) {
                container.removeChild(container.lastChild);
            }
        }

        function clearLogs() {
            document.getElementById('log-container').innerHTML = '';
            addLog('로그가 지워졌습니다.');
        }

        // 숫자 포맷
        function formatNumber(num) {
            if (num === null || num === undefined || num === '-') return '-';
            return Number(num).toLocaleString('ko-KR');
        }

        function formatPercent(num) {
            if (num === null || num === undefined || num === '-') return '-';
            const val = Number(num);
            const prefix = val >= 0 ? '+' : '';
            return prefix + val.toFixed(2) + '%';
        }

        // 계좌 정보 새로고침
        async function refreshAccount() {
            addLog('계좌 정보 조회 중...');
            try {
                const res = await fetch('/api/account');
                const data = await res.json();

                if (data.error) {
                    addLog(data.error, 'error');
                    return;
                }

                document.getElementById('deposit').textContent = formatNumber(data.deposit) + '원';
                document.getElementById('total-eval').textContent = formatNumber(data.total_eval) + '원';

                const plEl = document.getElementById('total-pl');
                plEl.textContent = formatNumber(data.total_pl) + '원';
                plEl.className = 'account-value ' + (data.total_pl >= 0 ? 'positive' : 'negative');

                const rateEl = document.getElementById('pl-rate');
                rateEl.textContent = formatPercent(data.pl_rate);
                rateEl.className = 'account-value ' + (data.pl_rate >= 0 ? 'positive' : 'negative');

                // 보유 종목
                updateHoldings(data.holdings || []);

                document.getElementById('auth-status').className = 'status-badge status-active';
                document.getElementById('auth-status').textContent = '인증됨';

                addLog('계좌 정보 업데이트 완료', 'info');
            } catch (e) {
                addLog('계좌 조회 실패: ' + e.message, 'error');
            }
        }

        function updateHoldings(holdings) {
            const tbody = document.getElementById('holdings-body');
            if (!holdings || holdings.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:#666;">보유 종목 없음</td></tr>';
                return;
            }

            tbody.innerHTML = holdings.map(h => `
                <tr>
                    <td>${h.name}</td>
                    <td>${formatNumber(h.qty)}</td>
                    <td>${formatNumber(h.avg_price)}원</td>
                    <td>${formatNumber(h.current_price)}원</td>
                    <td class="${h.pl >= 0 ? 'positive' : 'negative'}">${formatNumber(h.pl)}원</td>
                    <td class="${h.pl_rate >= 0 ? 'positive' : 'negative'}">${formatPercent(h.pl_rate)}</td>
                </tr>
            `).join('');
        }

        // 체결 내역 새로고침
        async function refreshOrders() {
            addLog('체결 내역 조회 중...');
            try {
                const res = await fetch('/api/orders');
                const data = await res.json();

                const tbody = document.getElementById('orders-body');
                if (!data.orders || data.orders.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#666;">오늘 체결 내역 없음</td></tr>';
                    return;
                }

                tbody.innerHTML = data.orders.map(o => `
                    <tr>
                        <td>${o.time}</td>
                        <td>${o.name}</td>
                        <td class="${o.side === '매수' ? 'positive' : 'negative'}">${o.side}</td>
                        <td>${formatNumber(o.qty)}</td>
                        <td>${formatNumber(o.price)}원</td>
                    </tr>
                `).join('');

                addLog('체결 내역 업데이트 완료');
            } catch (e) {
                addLog('체결 내역 조회 실패: ' + e.message, 'error');
            }
        }

        // 급등 종목 스캔
        async function manualScan() {
            addLog('급등 종목 스캔 시작...');
            document.getElementById('scan-status').className = 'status-badge status-scanning';
            document.getElementById('scan-status').textContent = '스캔중';

            try {
                const res = await fetch('/api/scan');
                const data = await res.json();

                if (data.error) {
                    addLog(data.error, 'error');
                    return;
                }

                updateSurgeTable(data.candidates || []);
                addLog(`스캔 완료: ${data.candidates?.length || 0}개 종목 탐지`);
            } catch (e) {
                addLog('스캔 실패: ' + e.message, 'error');
            } finally {
                document.getElementById('scan-status').className = 'status-badge status-active';
                document.getElementById('scan-status').textContent = '활성';
            }
        }

        function updateSurgeTable(candidates) {
            const tbody = document.getElementById('surge-table-body');
            if (!candidates || candidates.length === 0) {
                tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:#666;">급등 종목 없음</td></tr>';
                return;
            }

            const signalClass = {
                'STRONG_BUY': 'signal-strong-buy',
                'BUY': 'signal-buy',
                'WATCH': 'signal-watch',
                'NEUTRAL': 'signal-neutral'
            };

            tbody.innerHTML = candidates.slice(0, 20).map(c => `
                <tr>
                    <td>${c.rank}</td>
                    <td><strong>${c.name}</strong><br><small style="color:#666">${c.code}</small></td>
                    <td>${formatNumber(c.price)}원</td>
                    <td class="${c.change_rate >= 0 ? 'positive' : 'negative'}">${formatPercent(c.change_rate)}</td>
                    <td><strong>${c.volume_power?.toFixed(1) || '-'}</strong></td>
                    <td>${c.balance_ratio?.toFixed(2) || '-'}</td>
                    <td><strong>${c.surge_score?.toFixed(0) || '-'}</strong></td>
                    <td><span class="signal-badge ${signalClass[c.signal] || ''}">${c.signal}</span></td>
                    <td>${c.detected_at || '-'}</td>
                </tr>
            `).join('');
        }

        // 자동 스캔
        let autoScanTimer = null;

        function startAutoScan() {
            const interval = parseInt(document.getElementById('scan-interval').value) * 1000;
            stopAutoScan();

            addLog(`자동 스캔 시작 (${interval/1000}초 간격)`);
            manualScan();
            autoScanTimer = setInterval(manualScan, interval);

            document.getElementById('scan-status').className = 'status-badge status-active';
            document.getElementById('scan-status').textContent = '자동스캔';
        }

        function stopAutoScan() {
            if (autoScanTimer) {
                clearInterval(autoScanTimer);
                autoScanTimer = null;
                addLog('자동 스캔 중지');
            }
            document.getElementById('scan-status').className = 'status-badge status-inactive';
            document.getElementById('scan-status').textContent = '대기중';
        }

        function updateInterval() {
            if (autoScanTimer) {
                startAutoScan();
            }
        }

        // 초기화
        window.onload = function() {
            refreshAccount();
            refreshOrders();
            addLog('시스템 초기화 완료');
        };

        // 주기적 계좌 업데이트 (5분)
        setInterval(refreshAccount, 300000);
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
    """메인 대시보드"""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/account')
def api_account():
    """계좌 정보 API"""
    if not data_store.kis_authenticated:
        if not init_kis():
            return jsonify({"error": "KIS API 인증 실패"})

    try:
        import kis_auth as ka
        from domestic_stock.inquire_balance.inquire_balance import inquire_balance

        trenv = ka.getTREnv()

        df1, df2 = inquire_balance(
            env_dv="real",
            cano=trenv.my_acct,
            acnt_prdt_cd=trenv.my_prod,
            afhr_flpr_yn="N",
            inqr_dvsn="02",  # 종목별
            unpr_dvsn="01",
            fund_sttl_icld_yn="N",
            fncg_amt_auto_rdpt_yn="N",
            prcs_dvsn="00"
        )

        # 요약 정보
        summary = {}
        if not df2.empty:
            row = df2.iloc[0]
            summary = {
                "deposit": int(row.get('dnca_tot_amt', 0)),
                "total_eval": int(row.get('tot_evlu_amt', 0)),
                "total_pl": int(row.get('evlu_pfls_smtl_amt', 0)),
                "pl_rate": float(row.get('asst_icdc_erng_rt', 0)),
            }

        # 보유 종목
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
        logger.error(f"계좌 조회 오류: {e}")
        return jsonify({"error": str(e)})


@app.route('/api/orders')
def api_orders():
    """체결 내역 API"""
    if not data_store.kis_authenticated:
        if not init_kis():
            return jsonify({"error": "KIS API 인증 실패"})

    try:
        import kis_auth as ka
        from domestic_stock.inquire_daily_ccld.inquire_daily_ccld import inquire_daily_ccld

        trenv = ka.getTREnv()
        today = datetime.now().strftime("%Y%m%d")

        df1, df2 = inquire_daily_ccld(
            env_dv="real",
            pd_dv="inner",
            cano=trenv.my_acct,
            acnt_prdt_cd=trenv.my_prod,
            inqr_strt_dt=today,
            inqr_end_dt=today,
            sll_buy_dvsn_cd="00",
            ccld_dvsn="01",  # 체결만
            inqr_dvsn="00",
            inqr_dvsn_3="00"
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
        logger.error(f"체결 내역 조회 오류: {e}")
        return jsonify({"error": str(e), "orders": []})


@app.route('/api/scan')
def api_scan():
    """급등 종목 스캔 API"""
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

        # 직렬화
        result = []
        for c in candidates:
            result.append({
                "rank": c.rank,
                "code": c.code,
                "name": c.name,
                "price": c.price,
                "change": c.change,
                "change_rate": c.change_rate,
                "volume": c.volume,
                "volume_power": c.volume_power,
                "balance_ratio": c.balance_ratio,
                "surge_score": c.surge_score,
                "signal": c.signal,
                "detected_at": c.detected_at,
                "reasons": c.reasons,
            })

        return jsonify({"candidates": result})

    except Exception as e:
        logger.error(f"스캔 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "candidates": []})


@app.route('/api/status')
def api_status():
    """시스템 상태 API"""
    return jsonify({
        "authenticated": data_store.kis_authenticated,
        "is_scanning": data_store.is_scanning,
        "last_scan": data_store.last_scan_time.isoformat() if data_store.last_scan_time else None,
        "candidates_count": len(data_store.surge_candidates),
        "error": data_store.error_message,
    })


def run_server(host='0.0.0.0', port=5000, debug=False):
    """서버 실행"""
    logger.info(f"스캘핑 대시보드 시작: http://{host}:{port}")

    # 초기 인증
    init_kis()

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_server(debug=True)
