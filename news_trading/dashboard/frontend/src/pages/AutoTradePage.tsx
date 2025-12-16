import { useState, useEffect } from 'react';
import {
  Zap,
  Play,
  Pause,
  Settings,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Brain,
} from 'lucide-react';
import clsx from 'clsx';
import axios from 'axios';
import { useDashboardStore } from '../store';

interface TradeConfig {
  enabled: boolean;
  maxOrderAmount: number;
  minConfidence: number;
  stopLoss: number;
  takeProfit: number;
  maxDailyTrades: number;
  scanInterval: number;
}

interface TradeResult {
  id: string;
  timestamp: string;
  stockCode: string;
  stockName: string;
  signal: string;
  confidence: number;
  action: 'buy' | 'sell' | 'hold';
  price: number;
  quantity: number;
  result: 'success' | 'failed' | 'pending';
  profit?: number;
}

function ConfigCard({ title, children, icon }: { title: string; children: React.ReactNode; icon: React.ReactNode }) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <h4 className="font-medium text-white">{title}</h4>
      </div>
      {children}
    </div>
  );
}

function TradeHistoryRow({ trade }: { trade: TradeResult }) {
  const statusConfig = {
    success: { icon: <CheckCircle className="w-4 h-4" />, color: 'text-green-400' },
    failed: { icon: <XCircle className="w-4 h-4" />, color: 'text-red-400' },
    pending: { icon: <Clock className="w-4 h-4" />, color: 'text-yellow-400' },
  };

  const { icon, color } = statusConfig[trade.result];

  return (
    <tr className="hover:bg-gray-800/50 transition-colors border-b border-gray-800">
      <td className="px-4 py-3 text-sm text-gray-400">
        {new Date(trade.timestamp).toLocaleTimeString('ko-KR')}
      </td>
      <td className="px-4 py-3">
        <div>
          <p className="text-sm font-medium text-white">{trade.stockName}</p>
          <p className="text-xs text-gray-500">{trade.stockCode}</p>
        </div>
      </td>
      <td className="px-4 py-3">
        <span className={clsx(
          'px-2 py-1 text-xs font-medium rounded',
          trade.action === 'buy' ? 'bg-red-500/20 text-red-400' :
          trade.action === 'sell' ? 'bg-blue-500/20 text-blue-400' : 'bg-gray-500/20 text-gray-400'
        )}>
          {trade.action === 'buy' ? '매수' : trade.action === 'sell' ? '매도' : '관망'}
        </span>
      </td>
      <td className="px-4 py-3 text-sm text-white">
        {trade.price.toLocaleString()}원
      </td>
      <td className="px-4 py-3 text-sm text-gray-300">
        {trade.quantity}주
      </td>
      <td className="px-4 py-3 text-sm">
        {trade.confidence}%
      </td>
      <td className="px-4 py-3">
        <div className={clsx('flex items-center gap-1', color)}>
          {icon}
          <span className="text-sm">{trade.result === 'success' ? '체결' : trade.result === 'failed' ? '실패' : '대기'}</span>
        </div>
      </td>
      <td className="px-4 py-3 text-sm">
        {trade.profit !== undefined && (
          <span className={trade.profit >= 0 ? 'text-red-400' : 'text-blue-400'}>
            {trade.profit >= 0 ? '+' : ''}{trade.profit.toLocaleString()}원
          </span>
        )}
      </td>
    </tr>
  );
}

function LLMOutputPanel() {
  const { llmOutputs } = useDashboardStore();
  const recentOutputs = llmOutputs.slice(-5);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-5 h-5 text-purple-400" />
        <h3 className="font-semibold text-white">LLM 분석 출력</h3>
      </div>
      <div className="flex-1 overflow-y-auto space-y-2 font-mono text-sm">
        {recentOutputs.length === 0 ? (
          <p className="text-gray-500">자동 매매 시작 시 LLM 분석 결과가 표시됩니다.</p>
        ) : (
          recentOutputs.map((output, index) => (
            <div key={index} className="p-2 bg-gray-800/50 rounded text-gray-300">
              <span className="text-purple-400">[{output.timestamp}]</span> {output.content}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function AutoTradePage() {
  const { isAutoTrading, setAutoTrading, scanResults } = useDashboardStore();
  const [config, setConfig] = useState<TradeConfig>({
    enabled: false,
    maxOrderAmount: 100000,
    minConfidence: 70,
    stopLoss: 3,
    takeProfit: 5,
    maxDailyTrades: 10,
    scanInterval: 60,
  });
  const [tradeHistory, setTradeHistory] = useState<TradeResult[]>([]);
  const [stats, setStats] = useState({
    todayTrades: 0,
    successRate: 0,
    totalProfit: 0,
  });

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await axios.get('/api/v1/auto-trade/status');
        setConfig((prev) => ({ ...prev, ...res.data }));
      } catch (err) {
        console.error('상태 조회 오류:', err);
      }
    };
    fetchStatus();
  }, []);

  const handleToggle = async () => {
    try {
      if (isAutoTrading) {
        await axios.post('/api/v1/auto-trade/stop');
        setAutoTrading(false);
      } else {
        await axios.post('/api/v1/auto-trade/start', { interval: config.scanInterval });
        setAutoTrading(true);
      }
    } catch (err) {
      console.error('자동매매 토글 오류:', err);
    }
  };

  const handleManualScan = async () => {
    try {
      await axios.post('/api/v1/auto-trade/scan');
    } catch (err) {
      console.error('수동 스캔 오류:', err);
    }
  };

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={clsx('p-2 rounded-lg', isAutoTrading ? 'bg-green-500/10' : 'bg-yellow-500/10')}>
            <Zap className={clsx('w-6 h-6', isAutoTrading ? 'text-green-400' : 'text-yellow-400')} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">자동 매매</h1>
            <p className="text-sm text-gray-400">LLM 기반 자동 매매 시스템</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={handleManualScan}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            수동 스캔
          </button>
          <button
            onClick={handleToggle}
            className={clsx(
              'flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-colors',
              isAutoTrading
                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                : 'bg-green-500 text-white hover:bg-green-600'
            )}
          >
            {isAutoTrading ? (
              <>
                <Pause className="w-5 h-5" />
                정지
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                시작
              </>
            )}
          </button>
        </div>
      </div>

      {/* 상태 카드 */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">상태</p>
          <div className="flex items-center gap-2">
            <div className={clsx('w-3 h-3 rounded-full', isAutoTrading ? 'bg-green-400 animate-pulse' : 'bg-gray-500')} />
            <span className="text-lg font-semibold text-white">{isAutoTrading ? '실행 중' : '정지됨'}</span>
          </div>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">오늘 거래</p>
          <p className="text-lg font-semibold text-white">{stats.todayTrades} / {config.maxDailyTrades}회</p>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">성공률</p>
          <p className="text-lg font-semibold text-white">{stats.successRate}%</p>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">오늘 손익</p>
          <p className={clsx('text-lg font-semibold', stats.totalProfit >= 0 ? 'text-red-400' : 'text-blue-400')}>
            {stats.totalProfit >= 0 ? '+' : ''}{stats.totalProfit.toLocaleString()}원
          </p>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* 설정 */}
        <div className="col-span-5 bg-gray-900 rounded-xl border border-gray-800 p-6 space-y-4">
          <div className="flex items-center gap-2 mb-4">
            <Settings className="w-5 h-5 text-gray-400" />
            <h3 className="text-lg font-semibold text-white">설정</h3>
          </div>

          <ConfigCard title="주문 설정" icon={<TrendingUp className="w-4 h-4 text-purple-400" />}>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-400 mb-1">최대 주문 금액</label>
                <input
                  type="number"
                  value={config.maxOrderAmount}
                  onChange={(e) => setConfig((prev) => ({ ...prev, maxOrderAmount: Number(e.target.value) }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">최소 신뢰도 (%)</label>
                <input
                  type="number"
                  value={config.minConfidence}
                  onChange={(e) => setConfig((prev) => ({ ...prev, minConfidence: Number(e.target.value) }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 outline-none"
                  min="0"
                  max="100"
                />
              </div>
            </div>
          </ConfigCard>

          <ConfigCard title="리스크 관리" icon={<AlertTriangle className="w-4 h-4 text-yellow-400" />}>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-400 mb-1">손절 (%)</label>
                <input
                  type="number"
                  value={config.stopLoss}
                  onChange={(e) => setConfig((prev) => ({ ...prev, stopLoss: Number(e.target.value) }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">익절 (%)</label>
                <input
                  type="number"
                  value={config.takeProfit}
                  onChange={(e) => setConfig((prev) => ({ ...prev, takeProfit: Number(e.target.value) }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 outline-none"
                />
              </div>
            </div>
          </ConfigCard>

          <ConfigCard title="스캔 설정" icon={<Clock className="w-4 h-4 text-blue-400" />}>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-400 mb-1">스캔 주기 (초)</label>
                <input
                  type="number"
                  value={config.scanInterval}
                  onChange={(e) => setConfig((prev) => ({ ...prev, scanInterval: Number(e.target.value) }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 outline-none"
                  min="30"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">일일 최대 거래 횟수</label>
                <input
                  type="number"
                  value={config.maxDailyTrades}
                  onChange={(e) => setConfig((prev) => ({ ...prev, maxDailyTrades: Number(e.target.value) }))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 outline-none"
                />
              </div>
            </div>
          </ConfigCard>
        </div>

        {/* LLM 출력 */}
        <div className="col-span-7">
          <LLMOutputPanel />
        </div>
      </div>

      {/* 실시간 스캔 결과 */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">실시간 스캔 결과</h3>
          <span className="text-sm text-gray-400">{scanResults.length}개 종목</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">시간</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">종목</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">액션</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">가격</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">수량</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">신뢰도</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">상태</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">손익</th>
              </tr>
            </thead>
            <tbody>
              {scanResults.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                    스캔 결과가 없습니다. 자동매매를 시작하세요.
                  </td>
                </tr>
              ) : (
                scanResults.map((result: any, index: number) => (
                  <TradeHistoryRow key={index} trade={{
                    id: String(index),
                    timestamp: result.detected_at || new Date().toISOString(),
                    stockCode: result.code,
                    stockName: result.name,
                    signal: result.signal,
                    confidence: result.llm_confidence || result.surge_score,
                    action: result.signal?.includes('BUY') ? 'buy' : result.signal?.includes('SELL') ? 'sell' : 'hold',
                    price: result.price,
                    quantity: 0,
                    result: 'pending',
                  }} />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
