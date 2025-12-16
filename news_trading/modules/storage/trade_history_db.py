# -*- coding: utf-8 -*-
"""
SQLite 기반 거래 내역 및 분석 요약 저장

테이블 구조:
- trades: 거래 내역 (매수/매도)
- analysis_summary: LLM 앙상블 분석 요약
- daily_stats: 일별 거래 통계
"""

import os
import sqlite3
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeHistoryDB:
    """SQLite 기반 거래 내역 및 분석 요약 저장소"""

    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: SQLite DB 파일 경로 (None이면 기본 경로 사용)
        """
        if db_path is None:
            # 기본 경로: examples_user/news_trading/data/trading.db
            base_dir = Path(__file__).parent.parent.parent / "data"
            base_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(base_dir / "trading.db")

        self.db_path = db_path
        self._init_db()
        logger.info(f"TradeHistoryDB 초기화: {db_path}")

    def _init_db(self):
        """데이터베이스 테이블 초기화"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 거래 내역 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    stock_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    order_qty INTEGER DEFAULT 0,
                    order_price INTEGER DEFAULT 0,
                    order_amount INTEGER DEFAULT 0,
                    order_no TEXT,
                    success INTEGER DEFAULT 0,

                    -- 분석 결과
                    ensemble_signal TEXT,
                    confidence REAL DEFAULT 0,
                    consensus REAL DEFAULT 0,
                    technical_score REAL DEFAULT 0,
                    trend TEXT,

                    -- 메타데이터
                    reason TEXT,
                    analysis_id TEXT,
                    llm_log_path TEXT,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 분석 요약 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    stock_name TEXT NOT NULL,

                    -- 앙상블 결과
                    ensemble_signal TEXT,
                    ensemble_confidence REAL DEFAULT 0,
                    ensemble_trend TEXT,
                    consensus_score REAL DEFAULT 0,

                    -- 가격 정보
                    current_price INTEGER DEFAULT 0,
                    avg_entry_price REAL DEFAULT 0,
                    avg_stop_loss REAL DEFAULT 0,
                    avg_take_profit REAL DEFAULT 0,

                    -- 모델 정보
                    models_used TEXT,
                    models_agreed INTEGER DEFAULT 0,
                    total_models INTEGER DEFAULT 0,
                    signal_votes TEXT,
                    trend_votes TEXT,

                    -- 기술적 지표
                    technical_summary TEXT,

                    -- 처리 시간
                    total_processing_time REAL DEFAULT 0,

                    -- LLM 로그 참조
                    llm_log_path TEXT,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 일별 통계 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,

                    -- 거래 통계
                    total_trades INTEGER DEFAULT 0,
                    buy_count INTEGER DEFAULT 0,
                    sell_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,

                    -- 금액
                    total_buy_amount INTEGER DEFAULT 0,
                    total_sell_amount INTEGER DEFAULT 0,
                    realized_pnl INTEGER DEFAULT 0,

                    -- 분석 통계
                    total_analyses INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    avg_consensus REAL DEFAULT 0,

                    -- 시그널 분포
                    signal_distribution TEXT,

                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_stock ON trades(stock_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_date ON analysis_summary(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_stock ON analysis_summary(stock_code)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """SQLite 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # =========================================================================
    # 거래 내역 저장/조회
    # =========================================================================

    def save_trade(
        self,
        stock_code: str,
        stock_name: str,
        action: str,
        order_qty: int = 0,
        order_price: int = 0,
        order_no: str = None,
        success: bool = False,
        ensemble_signal: str = None,
        confidence: float = 0,
        consensus: float = 0,
        technical_score: float = 0,
        trend: str = None,
        reason: str = None,
        analysis_id: str = None,
        llm_log_path: str = None,
        timestamp: str = None
    ) -> int:
        """
        거래 내역 저장

        Returns:
            int: 저장된 레코드 ID
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        trade_date = timestamp[:10]
        order_amount = order_qty * order_price

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, date, stock_code, stock_name, action,
                    order_qty, order_price, order_amount, order_no, success,
                    ensemble_signal, confidence, consensus, technical_score, trend,
                    reason, analysis_id, llm_log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, trade_date, stock_code, stock_name, action,
                order_qty, order_price, order_amount, order_no, 1 if success else 0,
                ensemble_signal, confidence, consensus, technical_score, trend,
                reason, analysis_id, llm_log_path
            ))
            conn.commit()

            trade_id = cursor.lastrowid
            logger.info(f"거래 저장: ID={trade_id}, {stock_name}({stock_code}) {action} {order_qty}주 @ {order_price:,}원")

            # 일별 통계 업데이트
            self._update_daily_stats(trade_date, conn)

            return trade_id

    def get_trades(
        self,
        date: str = None,
        stock_code: str = None,
        action: str = None,
        success_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        거래 내역 조회

        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            stock_code: 종목코드
            action: 거래 유형 (BUY, SELL)
            success_only: 성공한 거래만
            limit: 조회 개수
            offset: 오프셋

        Returns:
            List[Dict]: 거래 내역 리스트
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if date:
            query += " AND date = ?"
            params.append(date)
        if stock_code:
            query += " AND stock_code = ?"
            params.append(stock_code)
        if action:
            query += " AND action = ?"
            params.append(action)
        if success_only:
            query += " AND success = 1"

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_today_trades(self, success_only: bool = False) -> List[Dict]:
        """오늘 거래 내역 조회"""
        today = date.today().strftime("%Y-%m-%d")
        return self.get_trades(date=today, success_only=success_only)

    # =========================================================================
    # 분석 요약 저장/조회
    # =========================================================================

    def save_analysis(
        self,
        stock_code: str,
        stock_name: str,
        ensemble_signal: str,
        ensemble_confidence: float,
        ensemble_trend: str,
        consensus_score: float,
        current_price: int = 0,
        avg_entry_price: float = 0,
        avg_stop_loss: float = 0,
        avg_take_profit: float = 0,
        models_used: List[str] = None,
        models_agreed: int = 0,
        total_models: int = 0,
        signal_votes: Dict = None,
        trend_votes: Dict = None,
        technical_summary: Dict = None,
        total_processing_time: float = 0,
        llm_log_path: str = None,
        timestamp: str = None
    ) -> str:
        """
        분석 요약 저장

        Returns:
            str: 분석 ID
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        analysis_date = timestamp[:10]
        analysis_id = f"{stock_code}_{timestamp.replace(' ', '_').replace(':', '')}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO analysis_summary (
                    analysis_id, timestamp, date, stock_code, stock_name,
                    ensemble_signal, ensemble_confidence, ensemble_trend, consensus_score,
                    current_price, avg_entry_price, avg_stop_loss, avg_take_profit,
                    models_used, models_agreed, total_models, signal_votes, trend_votes,
                    technical_summary, total_processing_time, llm_log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id, timestamp, analysis_date, stock_code, stock_name,
                ensemble_signal, ensemble_confidence, ensemble_trend, consensus_score,
                current_price, avg_entry_price, avg_stop_loss, avg_take_profit,
                json.dumps(models_used or [], ensure_ascii=False),
                models_agreed, total_models,
                json.dumps(signal_votes or {}, ensure_ascii=False),
                json.dumps(trend_votes or {}, ensure_ascii=False),
                json.dumps(technical_summary or {}, ensure_ascii=False),
                total_processing_time, llm_log_path
            ))
            conn.commit()

            logger.info(f"분석 저장: {analysis_id}, {stock_name}({stock_code}) -> {ensemble_signal}")
            return analysis_id

    def get_analyses(
        self,
        date: str = None,
        stock_code: str = None,
        signal: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        분석 요약 조회

        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            stock_code: 종목코드
            signal: 앙상블 시그널 필터
            limit: 조회 개수
            offset: 오프셋

        Returns:
            List[Dict]: 분석 요약 리스트
        """
        query = "SELECT * FROM analysis_summary WHERE 1=1"
        params = []

        if date:
            query += " AND date = ?"
            params.append(date)
        if stock_code:
            query += " AND stock_code = ?"
            params.append(stock_code)
        if signal:
            query += " AND ensemble_signal = ?"
            params.append(signal)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                d = dict(row)
                # JSON 필드 파싱
                for field in ['models_used', 'signal_votes', 'trend_votes', 'technical_summary']:
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except json.JSONDecodeError:
                            pass
                results.append(d)
            return results

    def get_today_analyses(self) -> List[Dict]:
        """오늘 분석 내역 조회"""
        today = date.today().strftime("%Y-%m-%d")
        return self.get_analyses(date=today)

    # =========================================================================
    # 일별 통계
    # =========================================================================

    def _update_daily_stats(self, trade_date: str, conn: sqlite3.Connection = None):
        """일별 통계 업데이트 (내부용)"""
        should_close = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            should_close = True

        try:
            cursor = conn.cursor()

            # 해당 날짜 거래 통계 계산
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                    SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                    SUM(success) as success_count,
                    SUM(CASE WHEN action = 'BUY' THEN order_amount ELSE 0 END) as total_buy_amount,
                    SUM(CASE WHEN action = 'SELL' THEN order_amount ELSE 0 END) as total_sell_amount
                FROM trades WHERE date = ?
            """, (trade_date,))
            trade_stats = cursor.fetchone()

            # 해당 날짜 분석 통계 계산
            cursor.execute("""
                SELECT
                    COUNT(*) as total_analyses,
                    AVG(ensemble_confidence) as avg_confidence,
                    AVG(consensus_score) as avg_consensus
                FROM analysis_summary WHERE date = ?
            """, (trade_date,))
            analysis_stats = cursor.fetchone()

            # 시그널 분포 계산
            cursor.execute("""
                SELECT ensemble_signal, COUNT(*) as count
                FROM analysis_summary WHERE date = ?
                GROUP BY ensemble_signal
            """, (trade_date,))
            signal_dist = {row['ensemble_signal']: row['count'] for row in cursor.fetchall()}

            # 실현 손익 계산 (매도 금액 - 매수 금액)
            realized_pnl = (trade_stats['total_sell_amount'] or 0) - (trade_stats['total_buy_amount'] or 0)

            # 일별 통계 저장/업데이트
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats (
                    date, total_trades, buy_count, sell_count, success_count,
                    total_buy_amount, total_sell_amount, realized_pnl,
                    total_analyses, avg_confidence, avg_consensus,
                    signal_distribution, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_date,
                trade_stats['total_trades'] or 0,
                trade_stats['buy_count'] or 0,
                trade_stats['sell_count'] or 0,
                trade_stats['success_count'] or 0,
                trade_stats['total_buy_amount'] or 0,
                trade_stats['total_sell_amount'] or 0,
                realized_pnl,
                analysis_stats['total_analyses'] or 0,
                analysis_stats['avg_confidence'] or 0,
                analysis_stats['avg_consensus'] or 0,
                json.dumps(signal_dist, ensure_ascii=False),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()

        finally:
            if should_close:
                conn.close()

    def get_daily_stats(self, date: str = None) -> Dict:
        """
        일별 통계 조회

        Args:
            date: 조회 날짜 (None이면 오늘)

        Returns:
            Dict: 일별 통계
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (date,))
            row = cursor.fetchone()

            if row:
                d = dict(row)
                if d.get('signal_distribution'):
                    try:
                        d['signal_distribution'] = json.loads(d['signal_distribution'])
                    except json.JSONDecodeError:
                        pass
                return d

            return {
                'date': date,
                'total_trades': 0,
                'buy_count': 0,
                'sell_count': 0,
                'success_count': 0,
                'total_buy_amount': 0,
                'total_sell_amount': 0,
                'realized_pnl': 0,
                'total_analyses': 0,
                'avg_confidence': 0,
                'avg_consensus': 0,
                'signal_distribution': {}
            }

    def get_stats_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        기간별 통계 조회

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            List[Dict]: 일별 통계 리스트
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_stats
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """, (start_date, end_date))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                d = dict(row)
                if d.get('signal_distribution'):
                    try:
                        d['signal_distribution'] = json.loads(d['signal_distribution'])
                    except json.JSONDecodeError:
                        pass
                results.append(d)
            return results

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_stock_history(self, stock_code: str, days: int = 30) -> Dict:
        """
        특정 종목의 거래/분석 히스토리 조회

        Args:
            stock_code: 종목코드
            days: 조회 일수

        Returns:
            Dict: 종목 히스토리
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 거래 내역
            cursor.execute("""
                SELECT * FROM trades
                WHERE stock_code = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (stock_code, days * 10))
            trades = [dict(row) for row in cursor.fetchall()]

            # 분석 내역
            cursor.execute("""
                SELECT * FROM analysis_summary
                WHERE stock_code = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (stock_code, days * 10))

            analyses = []
            for row in cursor.fetchall():
                d = dict(row)
                for field in ['models_used', 'signal_votes', 'trend_votes', 'technical_summary']:
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except json.JSONDecodeError:
                            pass
                analyses.append(d)

            return {
                'stock_code': stock_code,
                'trades': trades,
                'analyses': analyses,
                'total_trades': len(trades),
                'total_analyses': len(analyses)
            }

    def export_to_csv(self, output_dir: str, date: str = None) -> Dict[str, str]:
        """
        데이터를 CSV로 내보내기

        Args:
            output_dir: 출력 디렉토리
            date: 날짜 (None이면 전체)

        Returns:
            Dict[str, str]: 생성된 파일 경로
        """
        import csv
        os.makedirs(output_dir, exist_ok=True)

        exported = {}

        # 거래 내역 내보내기
        trades = self.get_trades(date=date, limit=10000)
        if trades:
            trades_file = os.path.join(output_dir, f"trades_{date or 'all'}.csv")
            with open(trades_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)
            exported['trades'] = trades_file
            logger.info(f"거래 내역 내보내기: {trades_file} ({len(trades)}건)")

        # 분석 요약 내보내기
        analyses = self.get_analyses(date=date, limit=10000)
        if analyses:
            analyses_file = os.path.join(output_dir, f"analyses_{date or 'all'}.csv")

            # JSON 필드를 문자열로 변환
            for a in analyses:
                for field in ['models_used', 'signal_votes', 'trend_votes', 'technical_summary']:
                    if isinstance(a.get(field), (dict, list)):
                        a[field] = json.dumps(a[field], ensure_ascii=False)

            with open(analyses_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=analyses[0].keys())
                writer.writeheader()
                writer.writerows(analyses)
            exported['analyses'] = analyses_file
            logger.info(f"분석 요약 내보내기: {analyses_file} ({len(analyses)}건)")

        return exported


# 전역 인스턴스
_trade_history_db: Optional[TradeHistoryDB] = None


def get_trade_history_db(db_path: str = None) -> TradeHistoryDB:
    """전역 TradeHistoryDB 인스턴스 반환"""
    global _trade_history_db
    if _trade_history_db is None:
        _trade_history_db = TradeHistoryDB(db_path)
    return _trade_history_db


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = TradeHistoryDB()

    # 테스트 거래 저장
    trade_id = db.save_trade(
        stock_code="005930",
        stock_name="삼성전자",
        action="BUY",
        order_qty=10,
        order_price=55000,
        order_no="ORD001",
        success=True,
        ensemble_signal="STRONG_BUY",
        confidence=0.85,
        consensus=0.75,
        technical_score=65.5,
        trend="UP",
        reason="앙상블 신호 + 기술적 강세"
    )
    print(f"거래 저장됨: ID={trade_id}")

    # 테스트 분석 저장
    analysis_id = db.save_analysis(
        stock_code="005930",
        stock_name="삼성전자",
        ensemble_signal="STRONG_BUY",
        ensemble_confidence=0.85,
        ensemble_trend="UP",
        consensus_score=0.75,
        current_price=55000,
        models_used=["deepseek-r1:8b", "qwen3:8b", "solar:10.7b"],
        models_agreed=3,
        total_models=3,
        signal_votes={"STRONG_BUY": 2, "BUY": 1},
        total_processing_time=15.5
    )
    print(f"분석 저장됨: ID={analysis_id}")

    # 조회 테스트
    print("\n=== 오늘 거래 ===")
    for t in db.get_today_trades():
        print(f"  {t['timestamp']} {t['stock_name']} {t['action']} {t['order_qty']}주")

    print("\n=== 오늘 분석 ===")
    for a in db.get_today_analyses():
        print(f"  {a['timestamp']} {a['stock_name']} -> {a['ensemble_signal']}")

    print("\n=== 일별 통계 ===")
    stats = db.get_daily_stats()
    print(f"  총 거래: {stats['total_trades']}, 매수: {stats['buy_count']}, 매도: {stats['sell_count']}")
    print(f"  성공: {stats['success_count']}, 매수금액: {stats['total_buy_amount']:,}원")
