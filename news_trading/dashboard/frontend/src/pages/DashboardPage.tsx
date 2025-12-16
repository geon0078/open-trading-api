import { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Wallet,
  BarChart3,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Zap,
  Clock,
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { useDashboardStore } from '../store';
import clsx from 'clsx';
import axios from 'axios';

// 샘플 데이터
const profitData = [
  { time: '09:00', profit: 0 },
  { time: '09:30', profit: 25000 },
  { time: '10:00', profit: 18000 },
  { time: '10:30', profit: 45000 },
  { time: '11:00', profit: 32000 },
  { time: '11:30', profit: 58000 },
  { time: '12:00', profit: 72000 },
  { time: '12:30', profit: 65000 },
  { time: '13:00', profit: 89000 },
  { time: '13:30', profit: 95000 },
  { time: '14:00', profit: 110000 },
  { time: '14:30', profit: 125000 },
  { time: '15:00', profit: 140000 },
];

const COLORS = ['#8b5cf6', '#ec4899', '#3b82f6', '#10b981', '#f59e0b'];

interface StatCardProps {
  title: string;
  value: string;
  change?: number;
  icon: React.ReactNode;
  color: string;
}

function StatCard({ title, value, change, icon, color }: StatCardProps) {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 hover:border-gray-700 transition-colors">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-gray-400">{title}</span>
        <div className={clsx('p-2 rounded-lg', color)}>{icon}</div>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold text-white">{value}</span>
        {change !== undefined && (
          <div
            className={clsx(
              'flex items-center gap-1 text-sm',
              change >= 0 ? 'text-red-400' : 'text-blue-400'
            )}
          >
            {change >= 0 ? (
              <ArrowUpRight className="w-4 h-4" />
            ) : (
              <ArrowDownRight className="w-4 h-4" />
            )}
            <span>{change >= 0 ? '+' : ''}{change}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

function AccountSummary() {
  const [accountData, setAccountData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAccount = async () => {
      try {
        const res = await axios.get('/api/v1/account/balance');
        setAccountData(res.data);
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
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-gray-900 rounded-xl border border-gray-800 p-4 animate-pulse">
            <div className="h-4 bg-gray-800 rounded w-1/2 mb-3" />
            <div className="h-8 bg-gray-800 rounded w-3/4" />
          </div>
        ))}
      </div>
    );
  }

  const stats = [
    {
      title: '예수금',
      value: accountData?.dnca_tot_amt ? `${Number(accountData.dnca_tot_amt).toLocaleString()}원` : '0원',
      icon: <Wallet className="w-5 h-5 text-purple-400" />,
      color: 'bg-purple-500/10',
    },
    {
      title: '총평가금액',
      value: accountData?.tot_evlu_amt ? `${Number(accountData.tot_evlu_amt).toLocaleString()}원` : '0원',
      icon: <BarChart3 className="w-5 h-5 text-pink-400" />,
      color: 'bg-pink-500/10',
      change: Number(accountData?.evlu_pfls_rt) || 0,
    },
    {
      title: '총 손익',
      value: accountData?.evlu_pfls_smtl_amt ? `${Number(accountData.evlu_pfls_smtl_amt).toLocaleString()}원` : '0원',
      icon: Number(accountData?.evlu_pfls_smtl_amt) >= 0 ? <TrendingUp className="w-5 h-5 text-red-400" /> : <TrendingDown className="w-5 h-5 text-blue-400" />,
      color: Number(accountData?.evlu_pfls_smtl_amt) >= 0 ? 'bg-red-500/10' : 'bg-blue-500/10',
    },
    {
      title: '보유 종목',
      value: accountData?.positions?.length ? `${accountData.positions.length}개` : '0개',
      icon: <Activity className="w-5 h-5 text-green-400" />,
      color: 'bg-green-500/10',
    },
  ];

  return (
    <div className="grid grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <StatCard key={index} {...stat} />
      ))}
    </div>
  );
}

function ProfitChart() {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">오늘의 손익</h3>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Clock className="w-4 h-4" />
          <span>실시간</span>
        </div>
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={profitData}>
            <defs>
              <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={12} tickFormatter={(value) => `${(value / 10000).toFixed(0)}만`} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
              }}
              formatter={(value: number) => [`${value.toLocaleString()}원`, '손익']}
            />
            <Area
              type="monotone"
              dataKey="profit"
              stroke="#8b5cf6"
              strokeWidth={2}
              fill="url(#profitGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SurgeStocksWidget() {
  const { surgeCandidates } = useDashboardStore();
  const topStocks = surgeCandidates.slice(0, 5);

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          급등 종목
        </h3>
        <a href="/surge" className="text-sm text-purple-400 hover:text-purple-300">
          더보기
        </a>
      </div>
      <div className="space-y-3">
        {topStocks.length === 0 ? (
          <div className="text-center text-gray-500 py-8">급등 종목이 없습니다</div>
        ) : (
          topStocks.map((stock, index) => (
            <div
              key={stock.code}
              className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-colors cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 flex items-center justify-center bg-purple-500/20 text-purple-400 rounded text-sm font-medium">
                  {index + 1}
                </span>
                <div>
                  <p className="text-sm font-medium text-white">{stock.name}</p>
                  <p className="text-xs text-gray-500">{stock.code}</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-white">
                  {stock.price.toLocaleString()}원
                </p>
                <p className={clsx('text-xs', stock.change_rate >= 0 ? 'text-red-400' : 'text-blue-400')}>
                  {stock.change_rate >= 0 ? '+' : ''}{stock.change_rate.toFixed(2)}%
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function PositionsWidget() {
  const [positions, setPositions] = useState<any[]>([]);

  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const res = await axios.get('/api/v1/account/balance');
        setPositions(res.data?.positions || []);
      } catch (err) {
        console.error('보유 종목 조회 오류:', err);
      }
    };
    fetchPositions();
    const interval = setInterval(fetchPositions, 30000);
    return () => clearInterval(interval);
  }, []);

  const pieData = positions.map((pos) => ({
    name: pos.prdt_name,
    value: Number(pos.evlu_amt) || 0,
  }));

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">포트폴리오 구성</h3>
        <a href="/portfolio" className="text-sm text-purple-400 hover:text-purple-300">
          더보기
        </a>
      </div>
      {positions.length === 0 ? (
        <div className="text-center text-gray-500 py-8">보유 종목이 없습니다</div>
      ) : (
        <div className="flex items-center gap-4">
          <div className="w-40 h-40">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={60}
                  paddingAngle={5}
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
            {positions.slice(0, 4).map((pos, index) => (
              <div key={pos.pdno} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-sm text-gray-300">{pos.prdt_name}</span>
                </div>
                <span className={clsx('text-sm', Number(pos.evlu_pfls_rt) >= 0 ? 'text-red-400' : 'text-blue-400')}>
                  {Number(pos.evlu_pfls_rt) >= 0 ? '+' : ''}{Number(pos.evlu_pfls_rt).toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RecentOrdersWidget() {
  const [orders, setOrders] = useState<any[]>([]);

  useEffect(() => {
    const fetchOrders = async () => {
      try {
        const res = await axios.get('/api/v1/account/orders/today');
        setOrders(res.data?.orders || []);
      } catch (err) {
        console.error('주문 내역 조회 오류:', err);
      }
    };
    fetchOrders();
  }, []);

  // 시간 포맷팅 함수
  const formatTime = (timeStr: string) => {
    if (!timeStr || timeStr.length < 4) return '';
    return `${timeStr.substring(0, 2)}:${timeStr.substring(2, 4)}`;
  };

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">오늘의 주문</h3>
        <a href="/history" className="text-sm text-purple-400 hover:text-purple-300">
          더보기
        </a>
      </div>
      <div className="space-y-2">
        {orders.length === 0 ? (
          <div className="text-center text-gray-500 py-8">오늘 주문 내역이 없습니다</div>
        ) : (
          orders.slice(0, 5).map((order, index) => {
            const isBuy = order.side === '매수';
            return (
              <div
                key={order.order_id || index}
                className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={clsx(
                      'px-2 py-1 text-xs font-medium rounded',
                      isBuy ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'
                    )}
                  >
                    {order.side}
                  </span>
                  <div>
                    <p className="text-sm font-medium text-white">{order.stock_name}</p>
                    <p className="text-xs text-gray-500">{formatTime(order.order_time)}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-white">
                    {order.exec_price?.toLocaleString() || 0}원
                  </p>
                  <p className="text-xs text-gray-500">{order.exec_qty?.toLocaleString() || 0}주</p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* 계좌 요약 */}
      <AccountSummary />

      {/* 차트 및 위젯 */}
      <div className="grid grid-cols-12 gap-6">
        {/* 손익 차트 */}
        <div className="col-span-8">
          <ProfitChart />
        </div>

        {/* 급등 종목 */}
        <div className="col-span-4">
          <SurgeStocksWidget />
        </div>
      </div>

      {/* 하단 위젯 */}
      <div className="grid grid-cols-12 gap-6">
        {/* 포트폴리오 구성 */}
        <div className="col-span-4">
          <PositionsWidget />
        </div>

        {/* 오늘의 주문 */}
        <div className="col-span-8">
          <RecentOrdersWidget />
        </div>
      </div>
    </div>
  );
}
