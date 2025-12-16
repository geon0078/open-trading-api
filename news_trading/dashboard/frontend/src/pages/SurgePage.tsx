import { useState } from 'react';
import {
  TrendingUp,
  RefreshCw,
  Filter,
  ArrowUpDown,
  Zap,
  Activity,
  Volume2,
  BarChart3,
  Clock,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useDashboardStore } from '../store';
import clsx from 'clsx';

// 필터 옵션
const filterOptions = {
  market: [
    { value: 'all', label: '전체' },
    { value: 'kospi', label: '코스피' },
    { value: 'kosdaq', label: '코스닥' },
  ],
  signal: [
    { value: 'all', label: '전체' },
    { value: 'STRONG_BUY', label: '강력 매수' },
    { value: 'BUY', label: '매수' },
    { value: 'HOLD', label: '관망' },
  ],
  minScore: [
    { value: 0, label: '전체' },
    { value: 50, label: '50점 이상' },
    { value: 70, label: '70점 이상' },
    { value: 80, label: '80점 이상' },
  ],
};

function SignalBadge({ signal }: { signal: string }) {
  const config: Record<string, { label: string; color: string }> = {
    STRONG_BUY: { label: '강력매수', color: 'bg-red-500/20 text-red-400 border-red-500/30' },
    BUY: { label: '매수', color: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
    HOLD: { label: '관망', color: 'bg-gray-500/20 text-gray-400 border-gray-500/30' },
    SELL: { label: '매도', color: 'bg-blue-500/20 text-blue-400 border-blue-500/30' },
  };

  const { label, color } = config[signal] || config.HOLD;

  return (
    <span className={clsx('px-2 py-1 text-xs font-medium rounded border', color)}>
      {label}
    </span>
  );
}

function ScoreBar({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 80) return 'bg-red-500';
    if (s >= 60) return 'bg-orange-500';
    if (s >= 40) return 'bg-yellow-500';
    return 'bg-gray-500';
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all', getColor(score))}
          style={{ width: `${score}%` }}
        />
      </div>
      <span className="text-sm font-medium text-white w-8">{score}</span>
    </div>
  );
}

function SurgeStockRow({ stock, rank }: { stock: any; rank: number }) {
  const [expanded, setExpanded] = useState(false);

  // 미니 차트 데이터
  const chartData = Array.from({ length: 20 }, (_, i) => ({
    time: i,
    price: stock.price * (1 + (Math.random() - 0.5) * 0.02),
  }));

  return (
    <>
      <tr
        onClick={() => setExpanded(!expanded)}
        className="hover:bg-gray-800/50 cursor-pointer transition-colors border-b border-gray-800"
      >
        <td className="px-4 py-3">
          <span
            className={clsx(
              'w-6 h-6 flex items-center justify-center rounded text-sm font-medium',
              rank <= 3 ? 'bg-purple-500/20 text-purple-400' : 'bg-gray-700 text-gray-400'
            )}
          >
            {rank}
          </span>
        </td>
        <td className="px-4 py-3">
          <div>
            <p className="font-medium text-white">{stock.name}</p>
            <p className="text-xs text-gray-500">{stock.code}</p>
          </div>
        </td>
        <td className="px-4 py-3 text-right">
          <span className="font-medium text-white">{stock.price.toLocaleString()}</span>
        </td>
        <td className="px-4 py-3 text-right">
          <span className={clsx('font-medium', stock.change_rate >= 0 ? 'text-red-400' : 'text-blue-400')}>
            {stock.change_rate >= 0 ? '+' : ''}{stock.change_rate.toFixed(2)}%
          </span>
        </td>
        <td className="px-4 py-3 text-right">
          <span className="text-gray-300">{stock.volume_power?.toFixed(1) || '-'}</span>
        </td>
        <td className="px-4 py-3">
          <ScoreBar score={stock.surge_score} />
        </td>
        <td className="px-4 py-3">
          <SignalBadge signal={stock.signal} />
        </td>
        <td className="px-4 py-3">
          <div className="w-20 h-8">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke={stock.change_rate >= 0 ? '#f87171' : '#60a5fa'}
                  fill={stock.change_rate >= 0 ? 'rgba(248, 113, 113, 0.1)' : 'rgba(96, 165, 250, 0.1)'}
                  strokeWidth={1.5}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </td>
      </tr>
      {expanded && (
        <tr className="bg-gray-800/30">
          <td colSpan={8} className="px-4 py-4">
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">매수/매도 거래량</p>
                <div className="flex items-center gap-2">
                  <span className="text-red-400">{stock.buy_volume?.toLocaleString() || '-'}</span>
                  <span className="text-gray-500">/</span>
                  <span className="text-blue-400">{stock.sell_volume?.toLocaleString() || '-'}</span>
                </div>
              </div>
              <div className="bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">탐지 시간</p>
                <p className="text-white">{stock.detected_at?.split(' ')[1]?.substring(0, 8) || '-'}</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">탐지 사유</p>
                <div className="space-y-1">
                  {stock.reasons?.slice(0, 2).map((reason: string, i: number) => (
                    <p key={i} className="text-xs text-gray-300">{reason}</p>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button className="flex-1 py-2 bg-red-500/20 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/30 transition-colors">
                  매수
                </button>
                <button className="flex-1 py-2 bg-purple-500/20 text-purple-400 rounded-lg text-sm font-medium hover:bg-purple-500/30 transition-colors">
                  LLM 분석
                </button>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export function SurgePage() {
  const { surgeCandidates } = useDashboardStore();
  const [marketFilter, setMarketFilter] = useState('all');
  const [signalFilter, setSignalFilter] = useState('all');
  const [minScore, setMinScore] = useState(0);
  const [sortBy, setSortBy] = useState<'score' | 'change_rate' | 'volume_power'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // 필터링 및 정렬
  const filteredStocks = surgeCandidates
    .filter((s) => signalFilter === 'all' || s.signal === signalFilter)
    .filter((s) => s.surge_score >= minScore)
    .sort((a, b) => {
      const aVal = a[sortBy] || 0;
      const bVal = b[sortBy] || 0;
      return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
    });

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-yellow-500/10 rounded-lg">
            <Zap className="w-6 h-6 text-yellow-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">급등 종목</h1>
            <p className="text-sm text-gray-400">실시간 급등 종목 탐지</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Activity className="w-4 h-4 text-green-400 animate-pulse" />
            <span>실시간 업데이트</span>
          </div>
          <div className="px-3 py-1.5 bg-gray-800 rounded-lg text-sm">
            <span className="text-gray-400">탐지 종목: </span>
            <span className="text-white font-medium">{surgeCandidates.length}개</span>
          </div>
        </div>
      </div>

      {/* 필터 */}
      <div className="flex items-center gap-4 p-4 bg-gray-900 rounded-xl border border-gray-800">
        <Filter className="w-5 h-5 text-gray-400" />

        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">시장:</span>
          <select
            value={marketFilter}
            onChange={(e) => setMarketFilter(e.target.value)}
            className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white focus:border-purple-500 outline-none"
          >
            {filterOptions.market.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">시그널:</span>
          <select
            value={signalFilter}
            onChange={(e) => setSignalFilter(e.target.value)}
            className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white focus:border-purple-500 outline-none"
          >
            {filterOptions.signal.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">최소 점수:</span>
          <select
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
            className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white focus:border-purple-500 outline-none"
          >
            {filterOptions.minScore.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div className="flex-1" />

        <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
          <RefreshCw className="w-4 h-4" />
          새로고침
        </button>
      </div>

      {/* 테이블 */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-800/50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                #
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                종목
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                현재가
              </th>
              <th
                onClick={() => handleSort('change_rate')}
                className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white"
              >
                <div className="flex items-center justify-end gap-1">
                  등락률
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th
                onClick={() => handleSort('volume_power')}
                className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white"
              >
                <div className="flex items-center justify-end gap-1">
                  체결강도
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th
                onClick={() => handleSort('score')}
                className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white"
              >
                <div className="flex items-center gap-1">
                  점수
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                시그널
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                추이
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredStocks.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                  조건에 맞는 급등 종목이 없습니다
                </td>
              </tr>
            ) : (
              filteredStocks.map((stock, index) => (
                <SurgeStockRow key={stock.code} stock={stock} rank={index + 1} />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
