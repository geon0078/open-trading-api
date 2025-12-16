import { useState, useEffect } from 'react';
import { History, RefreshCw, Download, Filter, Brain, TrendingUp, BarChart3, FileText, ChevronRight, Clock } from 'lucide-react';
import clsx from 'clsx';
import axios from 'axios';

// 타입 정의
interface TradeRecord {
  id: number;
  timestamp: string;
  date: string;
  stock_code: string;
  stock_name: string;
  action: string;
  order_qty: number;
  order_price: number;
  order_amount: number;
  order_no?: string;
  success: boolean;
  ensemble_signal?: string;
  confidence: number;
  consensus: number;
  technical_score: number;
  trend?: string;
  reason?: string;
  llm_log_path?: string;
}

interface AnalysisSummary {
  id: number;
  analysis_id: string;
  timestamp: string;
  stock_code: string;
  stock_name: string;
  ensemble_signal?: string;
  ensemble_confidence: number;
  ensemble_trend?: string;
  consensus_score: number;
  current_price: number;
  models_used: string[];
  models_agreed: number;
  total_models: number;
  signal_votes: Record<string, number>;
  total_processing_time: number;
  llm_log_path?: string;
}

interface DailyStats {
  date: string;
  total_trades: number;
  buy_count: number;
  sell_count: number;
  success_count: number;
  total_buy_amount: number;
  total_sell_amount: number;
  realized_pnl: number;
  total_analyses: number;
  avg_confidence: number;
  avg_consensus: number;
  signal_distribution: Record<string, number>;
}

interface LLMLogEntry {
  log_id: string;
  timestamp: string;
  stock_code: string;
  stock_name: string;
  analysis_type: string;
  signal?: string;
  confidence: number;
}

interface StorageStats {
  base_dir: string;
  total_files: number;
  total_size_mb: number;
  date_range: { oldest?: string; newest?: string; total_days: number };
}

interface HistoryOverview {
  today_stats: DailyStats;
  recent_trades: TradeRecord[];
  recent_analyses: AnalysisSummary[];
  storage_stats?: StorageStats;
}

type TabType = 'overview' | 'trades' | 'analyses' | 'llm-logs';

export function HistoryPage() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState<HistoryOverview | null>(null);
  const [trades, setTrades] = useState<TradeRecord[]>([]);
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [llmLogs, setLlmLogs] = useState<LLMLogEntry[]>([]);
  const [selectedLog, setSelectedLog] = useState<any>(null);
  const [dateFilter, setDateFilter] = useState<string>('');
  const [signalFilter, setSignalFilter] = useState<string>('all');

  // 개요 조회
  const fetchOverview = async () => {
    setLoading(true);
    try {
      const res = await axios.get<HistoryOverview>('/api/v1/history/overview');
      setOverview(res.data);
    } catch (err) {
      console.error('히스토리 개요 조회 오류:', err);
    } finally {
      setLoading(false);
    }
  };

  // 거래 내역 조회
  const fetchTrades = async () => {
    setLoading(true);
    try {
      const params: any = { limit: 100 };
      if (dateFilter) params.date = dateFilter;
      const res = await axios.get<TradeRecord[]>('/api/v1/history/trades', { params });
      setTrades(res.data);
    } catch (err) {
      console.error('거래 내역 조회 오류:', err);
    } finally {
      setLoading(false);
    }
  };

  // 분석 내역 조회
  const fetchAnalyses = async () => {
    setLoading(true);
    try {
      const params: any = { limit: 100 };
      if (dateFilter) params.date = dateFilter;
      if (signalFilter !== 'all') params.signal = signalFilter;
      const res = await axios.get<AnalysisSummary[]>('/api/v1/history/analyses', { params });
      setAnalyses(res.data);
    } catch (err) {
      console.error('분석 내역 조회 오류:', err);
    } finally {
      setLoading(false);
    }
  };

  // LLM 로그 인덱스 조회
  const fetchLlmLogs = async () => {
    setLoading(true);
    try {
      const params: any = {};
      if (dateFilter) params.date = dateFilter;
      const res = await axios.get('/api/v1/history/llm-logs/index', { params });
      setLlmLogs(res.data.entries || []);
    } catch (err) {
      console.error('LLM 로그 조회 오류:', err);
    } finally {
      setLoading(false);
    }
  };

  // LLM 로그 상세 조회
  const fetchLogDetail = async (logPath: string) => {
    try {
      const res = await axios.get('/api/v1/history/llm-logs/detail', {
        params: { log_path: logPath }
      });
      setSelectedLog(res.data);
    } catch (err) {
      console.error('LLM 로그 상세 조회 오류:', err);
    }
  };

  // CSV 내보내기
  const handleExport = async () => {
    try {
      const params: any = {};
      if (dateFilter) params.date = dateFilter;
      const res = await axios.post('/api/v1/history/export/csv', null, { params });
      alert(`CSV 내보내기 완료!\n${JSON.stringify(res.data.files, null, 2)}`);
    } catch (err) {
      console.error('CSV 내보내기 오류:', err);
      alert('CSV 내보내기 실패');
    }
  };

  useEffect(() => {
    if (activeTab === 'overview') fetchOverview();
    else if (activeTab === 'trades') fetchTrades();
    else if (activeTab === 'analyses') fetchAnalyses();
    else if (activeTab === 'llm-logs') fetchLlmLogs();
  }, [activeTab, dateFilter, signalFilter]);

  // 시그널 색상
  const getSignalColor = (signal?: string) => {
    if (!signal) return 'text-gray-400';
    if (signal.includes('STRONG_BUY')) return 'text-green-400 bg-green-500/20';
    if (signal.includes('BUY')) return 'text-emerald-400 bg-emerald-500/20';
    if (signal.includes('STRONG_SELL')) return 'text-red-400 bg-red-500/20';
    if (signal.includes('SELL')) return 'text-orange-400 bg-orange-500/20';
    return 'text-gray-400 bg-gray-500/20';
  };

  // 탭 목록
  const tabs = [
    { id: 'overview', label: '개요', icon: BarChart3 },
    { id: 'trades', label: '거래 내역', icon: TrendingUp },
    { id: 'analyses', label: 'LLM 분석', icon: Brain },
    { id: 'llm-logs', label: 'LLM 로그', icon: FileText },
  ];

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-500/10 rounded-lg">
            <History className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">히스토리</h1>
            <p className="text-sm text-gray-400">거래 내역 및 LLM 분석 기록</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
          />
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            CSV
          </button>
          <button
            onClick={() => {
              if (activeTab === 'overview') fetchOverview();
              else if (activeTab === 'trades') fetchTrades();
              else if (activeTab === 'analyses') fetchAnalyses();
              else if (activeTab === 'llm-logs') fetchLlmLogs();
            }}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <RefreshCw className={clsx("w-4 h-4", loading && "animate-spin")} />
          </button>
        </div>
      </div>

      {/* 탭 */}
      <div className="flex gap-2 border-b border-gray-800 pb-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as TabType)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              activeTab === tab.id
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* 개요 탭 */}
      {activeTab === 'overview' && overview && (
        <div className="space-y-6">
          {/* 오늘 통계 */}
          <div className="grid grid-cols-6 gap-4">
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <p className="text-sm text-gray-400 mb-1">총 거래</p>
              <p className="text-xl font-bold text-white">{overview.today_stats.total_trades}건</p>
            </div>
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <p className="text-sm text-gray-400 mb-1">매수</p>
              <p className="text-xl font-bold text-red-400">{overview.today_stats.buy_count}건</p>
            </div>
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <p className="text-sm text-gray-400 mb-1">매도</p>
              <p className="text-xl font-bold text-blue-400">{overview.today_stats.sell_count}건</p>
            </div>
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <p className="text-sm text-gray-400 mb-1">LLM 분석</p>
              <p className="text-xl font-bold text-purple-400">{overview.today_stats.total_analyses}건</p>
            </div>
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <p className="text-sm text-gray-400 mb-1">평균 신뢰도</p>
              <p className="text-xl font-bold text-yellow-400">{(overview.today_stats.avg_confidence * 100).toFixed(0)}%</p>
            </div>
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <p className="text-sm text-gray-400 mb-1">실현 손익</p>
              <p className={clsx(
                "text-xl font-bold",
                overview.today_stats.realized_pnl >= 0 ? "text-green-400" : "text-red-400"
              )}>
                {overview.today_stats.realized_pnl.toLocaleString()}원
              </p>
            </div>
          </div>

          {/* 시그널 분포 */}
          {overview.today_stats.signal_distribution && Object.keys(overview.today_stats.signal_distribution).length > 0 && (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">시그널 분포</h3>
              <div className="flex gap-4">
                {Object.entries(overview.today_stats.signal_distribution).map(([signal, count]) => (
                  <div key={signal} className="flex items-center gap-2">
                    <span className={clsx('px-2 py-1 rounded text-xs font-medium', getSignalColor(signal))}>
                      {signal}
                    </span>
                    <span className="text-white font-medium">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 최근 분석 */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">최근 LLM 분석</h3>
            <div className="space-y-2">
              {overview.recent_analyses.slice(0, 5).map((a) => (
                <div key={a.analysis_id} className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
                  <div className="flex items-center gap-3">
                    <span className={clsx('px-2 py-1 rounded text-xs font-medium', getSignalColor(a.ensemble_signal))}>
                      {a.ensemble_signal || 'N/A'}
                    </span>
                    <div>
                      <p className="text-sm text-white">{a.stock_name}</p>
                      <p className="text-xs text-gray-500">{a.timestamp}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-yellow-400">{(a.ensemble_confidence * 100).toFixed(0)}%</p>
                    <p className="text-xs text-gray-500">{a.models_agreed}/{a.total_models} 합의</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 저장소 정보 */}
          {overview.storage_stats && (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">저장소 정보</h3>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">총 파일</p>
                  <p className="text-white font-medium">{overview.storage_stats.total_files}개</p>
                </div>
                <div>
                  <p className="text-gray-500">용량</p>
                  <p className="text-white font-medium">{overview.storage_stats.total_size_mb} MB</p>
                </div>
                <div>
                  <p className="text-gray-500">보관 기간</p>
                  <p className="text-white font-medium">{overview.storage_stats.date_range.total_days}일</p>
                </div>
                <div>
                  <p className="text-gray-500">기간</p>
                  <p className="text-white font-medium">
                    {overview.storage_stats.date_range.oldest || '-'} ~ {overview.storage_stats.date_range.newest || '-'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 거래 내역 탭 */}
      {activeTab === 'trades' && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
            </div>
          ) : (
            <table className="w-full">
              <thead className="bg-gray-800/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">시간</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">종목</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">구분</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">시그널</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">수량</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">가격</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">금액</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">신뢰도</th>
                </tr>
              </thead>
              <tbody>
                {trades.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                      거래 내역이 없습니다
                    </td>
                  </tr>
                ) : (
                  trades.map((trade) => (
                    <tr key={trade.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                      <td className="px-4 py-3 text-sm text-gray-400">{trade.timestamp}</td>
                      <td className="px-4 py-3">
                        <p className="text-sm font-medium text-white">{trade.stock_name}</p>
                        <p className="text-xs text-gray-500">{trade.stock_code}</p>
                      </td>
                      <td className="px-4 py-3">
                        <span className={clsx(
                          'px-2 py-1 text-xs font-medium rounded',
                          trade.action === 'BUY' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'
                        )}>
                          {trade.action}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={clsx('px-2 py-1 text-xs font-medium rounded', getSignalColor(trade.ensemble_signal))}>
                          {trade.ensemble_signal || '-'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-sm text-white">{trade.order_qty.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right text-sm text-white">{trade.order_price.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right text-sm text-yellow-400">{trade.order_amount.toLocaleString()}</td>
                      <td className="px-4 py-3 text-right text-sm text-purple-400">{(trade.confidence * 100).toFixed(0)}%</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* LLM 분석 탭 */}
      {activeTab === 'analyses' && (
        <div className="space-y-4">
          {/* 시그널 필터 */}
          <div className="flex items-center gap-4">
            <Filter className="w-5 h-5 text-gray-400" />
            <div className="flex gap-2">
              {['all', 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'].map((s) => (
                <button
                  key={s}
                  onClick={() => setSignalFilter(s)}
                  className={clsx(
                    'px-3 py-1 rounded text-xs font-medium transition-colors',
                    signalFilter === s
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:text-white'
                  )}
                >
                  {s === 'all' ? '전체' : s}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
              </div>
            ) : (
              <table className="w-full">
                <thead className="bg-gray-800/50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">시간</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">종목</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">시그널</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">추세</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">신뢰도</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">합의도</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">모델</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">처리시간</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">상세</th>
                  </tr>
                </thead>
                <tbody>
                  {analyses.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-4 py-8 text-center text-gray-500">
                        분석 내역이 없습니다
                      </td>
                    </tr>
                  ) : (
                    analyses.map((a) => (
                      <tr key={a.analysis_id} className="border-b border-gray-800 hover:bg-gray-800/50">
                        <td className="px-4 py-3 text-sm text-gray-400">{a.timestamp}</td>
                        <td className="px-4 py-3">
                          <p className="text-sm font-medium text-white">{a.stock_name}</p>
                          <p className="text-xs text-gray-500">{a.stock_code}</p>
                        </td>
                        <td className="px-4 py-3">
                          <span className={clsx('px-2 py-1 text-xs font-medium rounded', getSignalColor(a.ensemble_signal))}>
                            {a.ensemble_signal || '-'}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-300">{a.ensemble_trend || '-'}</td>
                        <td className="px-4 py-3 text-right text-sm text-yellow-400">{(a.ensemble_confidence * 100).toFixed(0)}%</td>
                        <td className="px-4 py-3 text-right text-sm text-purple-400">{(a.consensus_score * 100).toFixed(0)}%</td>
                        <td className="px-4 py-3 text-sm text-gray-300">{a.models_agreed}/{a.total_models}</td>
                        <td className="px-4 py-3 text-right text-sm text-gray-400">{a.total_processing_time.toFixed(1)}s</td>
                        <td className="px-4 py-3 text-center">
                          {a.llm_log_path && (
                            <button
                              onClick={() => fetchLogDetail(a.llm_log_path!)}
                              className="text-purple-400 hover:text-purple-300"
                            >
                              <ChevronRight className="w-4 h-4" />
                            </button>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* LLM 로그 탭 */}
      {activeTab === 'llm-logs' && (
        <div className="grid grid-cols-2 gap-6">
          {/* 로그 목록 */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">로그 목록</h3>
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
              </div>
            ) : (
              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {llmLogs.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">로그가 없습니다</p>
                ) : (
                  llmLogs.map((log) => (
                    <button
                      key={log.log_id}
                      onClick={() => {
                        const logPath = `${dateFilter || new Date().toISOString().split('T')[0]}/${log.log_id}.json`;
                        fetchLogDetail(logPath);
                      }}
                      className="w-full flex items-center justify-between p-3 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors text-left"
                    >
                      <div>
                        <p className="text-sm text-white">{log.stock_name}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <Clock className="w-3 h-3 text-gray-500" />
                          <span className="text-xs text-gray-500">{log.timestamp}</span>
                          <span className="text-xs text-gray-600">|</span>
                          <span className="text-xs text-purple-400">{log.analysis_type}</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className={clsx('px-2 py-1 text-xs font-medium rounded', getSignalColor(log.signal))}>
                          {log.signal || '-'}
                        </span>
                      </div>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* 로그 상세 */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">로그 상세</h3>
            {selectedLog ? (
              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {/* 기본 정보 */}
                <div className="bg-gray-800 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">종목</p>
                  <p className="text-sm text-white">{selectedLog.stock_name} ({selectedLog.stock_code})</p>
                </div>

                {/* 입력 프롬프트 */}
                <div className="bg-gray-800 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">입력 프롬프트</p>
                  <pre className="text-xs text-gray-300 whitespace-pre-wrap max-h-40 overflow-y-auto">
                    {selectedLog.input?.prompt?.substring(0, 1000)}...
                  </pre>
                </div>

                {/* 모델별 출력 */}
                <div className="space-y-2">
                  <p className="text-xs text-gray-500">모델별 응답</p>
                  {selectedLog.model_outputs?.map((output: any, idx: number) => (
                    <div key={idx} className="bg-gray-800 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium text-purple-400">{output.model_name}</span>
                        <span className="text-xs text-gray-500">{output.processing_time?.toFixed(1)}s</span>
                      </div>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={clsx('px-2 py-0.5 text-xs rounded', getSignalColor(output.parsed_result?.signal))}>
                          {output.parsed_result?.signal || '-'}
                        </span>
                        <span className="text-xs text-yellow-400">
                          {((output.parsed_result?.confidence || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                      <details className="text-xs">
                        <summary className="text-gray-500 cursor-pointer hover:text-gray-400">원본 응답 보기</summary>
                        <pre className="mt-2 text-gray-400 whitespace-pre-wrap max-h-40 overflow-y-auto bg-gray-900 p-2 rounded">
                          {output.raw_output?.substring(0, 1500)}
                        </pre>
                      </details>
                    </div>
                  ))}
                </div>

                {/* 앙상블 결과 */}
                <div className="bg-gray-800 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-2">앙상블 결과</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">시그널: </span>
                      <span className={clsx('px-2 py-0.5 rounded', getSignalColor(selectedLog.ensemble_result?.ensemble_signal))}>
                        {selectedLog.ensemble_result?.ensemble_signal}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">신뢰도: </span>
                      <span className="text-yellow-400">
                        {((selectedLog.ensemble_result?.ensemble_confidence || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">합의도: </span>
                      <span className="text-purple-400">
                        {((selectedLog.ensemble_result?.consensus_score || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">처리시간: </span>
                      <span className="text-gray-300">
                        {selectedLog.ensemble_result?.total_processing_time?.toFixed(1)}s
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center py-12 text-gray-500">
                로그를 선택하세요
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
