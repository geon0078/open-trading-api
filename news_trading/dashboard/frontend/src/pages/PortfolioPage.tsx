import { useState, useEffect } from 'react';
import {
  Briefcase,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  DollarSign,
  PieChart as PieChartIcon,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
} from 'recharts';
import clsx from 'clsx';
import axios from 'axios';

const COLORS = ['#8b5cf6', '#ec4899', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#84cc16'];

interface Position {
  pdno: string;
  prdt_name: string;
  hldg_qty: string;
  pchs_avg_pric: string;
  pchs_amt: string;
  prpr: string;
  evlu_amt: string;
  evlu_pfls_amt: string;
  evlu_pfls_rt: string;
}

interface AccountSummary {
  dnca_tot_amt: string;
  tot_evlu_amt: string;
  evlu_pfls_smtl_amt: string;
  evlu_pfls_rt: string;
  positions: Position[];
}

function StatCard({ title, value, subtitle, icon, color, change }: any) {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 hover:border-gray-700 transition-colors">
      <div className="flex items-center justify-between mb-4">
        <div className={clsx('p-3 rounded-xl', color)}>{icon}</div>
        {change !== undefined && (
          <div className={clsx('flex items-center gap-1 text-sm', change >= 0 ? 'text-red-400' : 'text-blue-400')}>
            {change >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
            <span>{change >= 0 ? '+' : ''}{change.toFixed(2)}%</span>
          </div>
        )}
      </div>
      <p className="text-sm text-gray-400 mb-1">{title}</p>
      <p className="text-2xl font-bold text-white">{value}</p>
      {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );
}

function PositionRow({ position, index }: { position: Position; index: number }) {
  const profit = Number(position.evlu_pfls_amt);
  const profitRate = Number(position.evlu_pfls_rt);
  const isProfit = profit >= 0;

  return (
    <tr className="hover:bg-gray-800/50 transition-colors border-b border-gray-800">
      <td className="px-4 py-4">
        <div className="flex items-center gap-3">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: COLORS[index % COLORS.length] }}
          />
          <div>
            <p className="font-medium text-white">{position.prdt_name}</p>
            <p className="text-xs text-gray-500">{position.pdno}</p>
          </div>
        </div>
      </td>
      <td className="px-4 py-4 text-right">
        <span className="text-white">{Number(position.hldg_qty).toLocaleString()}주</span>
      </td>
      <td className="px-4 py-4 text-right">
        <span className="text-gray-300">{Number(position.pchs_avg_pric).toLocaleString()}원</span>
      </td>
      <td className="px-4 py-4 text-right">
        <span className="text-white font-medium">{Number(position.prpr).toLocaleString()}원</span>
      </td>
      <td className="px-4 py-4 text-right">
        <span className="text-white">{Number(position.evlu_amt).toLocaleString()}원</span>
      </td>
      <td className="px-4 py-4 text-right">
        <div className={clsx('flex items-center justify-end gap-1', isProfit ? 'text-red-400' : 'text-blue-400')}>
          {isProfit ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span>{profit.toLocaleString()}원</span>
        </div>
      </td>
      <td className="px-4 py-4 text-right">
        <span className={clsx('font-medium', isProfit ? 'text-red-400' : 'text-blue-400')}>
          {isProfit ? '+' : ''}{profitRate.toFixed(2)}%
        </span>
      </td>
      <td className="px-4 py-4">
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded text-sm hover:bg-blue-500/30 transition-colors">
            매도
          </button>
        </div>
      </td>
    </tr>
  );
}

export function PortfolioPage() {
  const [account, setAccount] = useState<AccountSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAccount = async () => {
      try {
        const res = await axios.get('/api/v1/account/balance');
        setAccount(res.data);
      } catch (err) {
        console.error('계좌 정보 조회 오류:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchAccount();
    const interval = setInterval(fetchAccount, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 text-purple-400 animate-spin" />
      </div>
    );
  }

  const positions = account?.positions || [];
  const totalValue = Number(account?.tot_evlu_amt) || 0;
  const totalProfit = Number(account?.evlu_pfls_smtl_amt) || 0;
  const profitRate = Number(account?.evlu_pfls_rt) || 0;
  const cash = Number(account?.dnca_tot_amt) || 0;

  // 파이 차트 데이터
  const pieData = positions.map((pos) => ({
    name: pos.prdt_name,
    value: Number(pos.evlu_amt),
  }));

  // 손익 바 차트 데이터
  const profitData = positions
    .map((pos) => ({
      name: pos.prdt_name.slice(0, 6),
      profit: Number(pos.evlu_pfls_amt),
      rate: Number(pos.evlu_pfls_rt),
    }))
    .sort((a, b) => b.profit - a.profit);

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg">
            <Briefcase className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">포트폴리오</h1>
            <p className="text-sm text-gray-400">보유 종목 및 자산 현황</p>
          </div>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors">
          <RefreshCw className="w-4 h-4" />
          새로고침
        </button>
      </div>

      {/* 요약 카드 */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          title="총 평가금액"
          value={`${totalValue.toLocaleString()}원`}
          icon={<DollarSign className="w-6 h-6 text-purple-400" />}
          color="bg-purple-500/10"
        />
        <StatCard
          title="예수금"
          value={`${cash.toLocaleString()}원`}
          icon={<BarChart3 className="w-6 h-6 text-blue-400" />}
          color="bg-blue-500/10"
        />
        <StatCard
          title="총 손익"
          value={`${totalProfit.toLocaleString()}원`}
          icon={totalProfit >= 0 ? <TrendingUp className="w-6 h-6 text-red-400" /> : <TrendingDown className="w-6 h-6 text-blue-400" />}
          color={totalProfit >= 0 ? 'bg-red-500/10' : 'bg-blue-500/10'}
          change={profitRate}
        />
        <StatCard
          title="보유 종목"
          value={`${positions.length}개`}
          icon={<PieChartIcon className="w-6 h-6 text-green-400" />}
          color="bg-green-500/10"
        />
      </div>

      {/* 차트 영역 */}
      <div className="grid grid-cols-2 gap-6">
        {/* 포트폴리오 구성 */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">자산 구성</h3>
          {positions.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-gray-500">
              보유 종목이 없습니다
            </div>
          ) : (
            <div className="flex items-center gap-6">
              <div className="w-48 h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                      formatter={(value: number) => [`${value.toLocaleString()}원`, '평가금액']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex-1 space-y-2">
                {positions.slice(0, 5).map((pos, index) => {
                  const percentage = (Number(pos.evlu_amt) / totalValue * 100).toFixed(1);
                  return (
                    <div key={pos.pdno} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: COLORS[index % COLORS.length] }}
                        />
                        <span className="text-sm text-gray-300">{pos.prdt_name}</span>
                      </div>
                      <span className="text-sm text-gray-400">{percentage}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* 종목별 손익 */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">종목별 손익</h3>
          {positions.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-gray-500">
              보유 종목이 없습니다
            </div>
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={profitData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#9ca3af" tickFormatter={(v) => `${(v / 10000).toFixed(0)}만`} />
                  <YAxis type="category" dataKey="name" stroke="#9ca3af" width={60} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${value.toLocaleString()}원`, '손익']}
                  />
                  <Bar
                    dataKey="profit"
                    fill={(entry: any) => (entry.profit >= 0 ? '#f87171' : '#60a5fa')}
                  >
                    {profitData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.profit >= 0 ? '#f87171' : '#60a5fa'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* 보유 종목 테이블 */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800">
          <h3 className="text-lg font-semibold text-white">보유 종목</h3>
        </div>
        <table className="w-full">
          <thead className="bg-gray-800/50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                종목
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                보유수량
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                매입단가
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                현재가
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                평가금액
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                평가손익
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                수익률
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">

              </th>
            </tr>
          </thead>
          <tbody>
            {positions.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-500">
                  보유 종목이 없습니다
                </td>
              </tr>
            ) : (
              positions.map((position, index) => (
                <PositionRow key={position.pdno} position={position} index={index} />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
