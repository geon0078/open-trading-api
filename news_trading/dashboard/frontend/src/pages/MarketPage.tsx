import { useState, useEffect } from 'react';
import {
  Search,
  TrendingUp,
  TrendingDown,
  Star,
  StarOff,
  RefreshCw,
  BarChart3,
  Clock,
  Volume2,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import clsx from 'clsx';
import axios from 'axios';

interface StockInfo {
  code: string;
  name: string;
  price: number;
  change: number;
  changeRate: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  prevClose: number;
}

// 미니 차트 데이터 생성 (실제로는 API에서 가져와야 함)
const generateMiniChartData = () => {
  const data = [];
  let price = 50000 + Math.random() * 10000;
  for (let i = 0; i < 20; i++) {
    price += (Math.random() - 0.5) * 1000;
    data.push({ time: i, price: Math.round(price) });
  }
  return data;
};

function StockSearchResult({ stock, onSelect }: { stock: any; onSelect: (code: string) => void }) {
  const chartData = generateMiniChartData();
  const isUp = stock.change_rate >= 0;

  return (
    <div
      onClick={() => onSelect(stock.code)}
      className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg hover:bg-gray-800 cursor-pointer transition-colors border border-gray-700/50 hover:border-purple-500/50"
    >
      <div className="flex items-center gap-4">
        <div>
          <p className="font-medium text-white">{stock.name}</p>
          <p className="text-sm text-gray-500">{stock.code}</p>
        </div>
      </div>
      <div className="w-24 h-10">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <Line
              type="monotone"
              dataKey="price"
              stroke={isUp ? '#f87171' : '#60a5fa'}
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="text-right">
        <p className="font-medium text-white">{stock.price?.toLocaleString()}원</p>
        <p className={clsx('text-sm', isUp ? 'text-red-400' : 'text-blue-400')}>
          {isUp ? '+' : ''}{stock.change_rate?.toFixed(2)}%
        </p>
      </div>
      <button className="p-2 text-gray-400 hover:text-yellow-400 transition-colors">
        <Star className="w-5 h-5" />
      </button>
    </div>
  );
}

function StockDetailPanel({ stockCode }: { stockCode: string }) {
  const [stockData, setStockData] = useState<StockInfo | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!stockCode) return;

    const fetchStockData = async () => {
      setLoading(true);
      try {
        // 실제로는 API 호출
        // const res = await axios.get(`/api/v1/stock/${stockCode}/price`);
        // setStockData(res.data);

        // 임시 데이터
        setStockData({
          code: stockCode,
          name: '삼성전자',
          price: 71500,
          change: 1500,
          changeRate: 2.14,
          volume: 12345678,
          high: 72000,
          low: 70500,
          open: 70800,
          prevClose: 70000,
        });
      } catch (err) {
        console.error('종목 정보 조회 오류:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchStockData();
  }, [stockCode]);

  if (!stockCode) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        종목을 선택하세요
      </div>
    );
  }

  if (loading || !stockData) {
    return (
      <div className="h-full flex items-center justify-center">
        <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
      </div>
    );
  }

  const isUp = stockData.changeRate >= 0;
  const chartData = generateMiniChartData();

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">{stockData.name}</h2>
          <p className="text-gray-500">{stockData.code}</p>
        </div>
        <button className="p-2 text-gray-400 hover:text-yellow-400 transition-colors">
          <Star className="w-6 h-6" />
        </button>
      </div>

      {/* 가격 정보 */}
      <div className="flex items-end gap-4">
        <span className="text-4xl font-bold text-white">
          {stockData.price.toLocaleString()}원
        </span>
        <div className={clsx('flex items-center gap-1', isUp ? 'text-red-400' : 'text-blue-400')}>
          {isUp ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
          <span className="text-lg font-medium">
            {isUp ? '+' : ''}{stockData.change.toLocaleString()}원 ({stockData.changeRate.toFixed(2)}%)
          </span>
        </div>
      </div>

      {/* 차트 */}
      <div className="h-64 bg-gray-800/50 rounded-lg p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" domain={['dataMin - 500', 'dataMax + 500']} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
              }}
            />
            <Line
              type="monotone"
              dataKey="price"
              stroke={isUp ? '#f87171' : '#60a5fa'}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 상세 정보 */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">시가</span>
              <span className="text-white">{stockData.open.toLocaleString()}원</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">고가</span>
              <span className="text-red-400">{stockData.high.toLocaleString()}원</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">저가</span>
              <span className="text-blue-400">{stockData.low.toLocaleString()}원</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">전일종가</span>
              <span className="text-white">{stockData.prevClose.toLocaleString()}원</span>
            </div>
          </div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">거래량</span>
              <span className="text-white">{stockData.volume.toLocaleString()}주</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">거래대금</span>
              <span className="text-white">{(stockData.volume * stockData.price / 100000000).toFixed(1)}억원</span>
            </div>
          </div>
        </div>
      </div>

      {/* 주문 버튼 */}
      <div className="flex gap-4">
        <button className="flex-1 py-3 bg-red-500/20 text-red-400 rounded-lg font-medium hover:bg-red-500/30 transition-colors">
          매수
        </button>
        <button className="flex-1 py-3 bg-blue-500/20 text-blue-400 rounded-lg font-medium hover:bg-blue-500/30 transition-colors">
          매도
        </button>
      </div>
    </div>
  );
}

export function MarketPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [popularStocks, setPopularStocks] = useState<any[]>([]);

  useEffect(() => {
    // 인기 종목 로드
    const mockPopular = [
      { code: '005930', name: '삼성전자', price: 71500, change_rate: 2.14 },
      { code: '000660', name: 'SK하이닉스', price: 198000, change_rate: -1.23 },
      { code: '035720', name: '카카오', price: 45500, change_rate: 0.88 },
      { code: '035420', name: 'NAVER', price: 187500, change_rate: 1.35 },
      { code: '051910', name: 'LG화학', price: 365000, change_rate: -0.54 },
      { code: '006400', name: '삼성SDI', price: 385000, change_rate: 2.67 },
      { code: '005380', name: '현대차', price: 215000, change_rate: 1.89 },
      { code: '003670', name: '포스코퓨처엠', price: 298500, change_rate: -2.15 },
    ];
    setPopularStocks(mockPopular);
  }, []);

  useEffect(() => {
    if (searchQuery.length >= 2) {
      // 검색 실행
      const filtered = popularStocks.filter(
        (s) => s.name.includes(searchQuery) || s.code.includes(searchQuery)
      );
      setSearchResults(filtered);
    } else {
      setSearchResults([]);
    }
  }, [searchQuery, popularStocks]);

  return (
    <div className="grid grid-cols-12 gap-6 h-full">
      {/* 왼쪽: 종목 검색 및 리스트 */}
      <div className="col-span-5 flex flex-col gap-4">
        {/* 검색 */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <div className="flex items-center gap-3 px-4 py-3 bg-gray-800 rounded-lg border border-gray-700 focus-within:border-purple-500 transition-colors">
            <Search className="w-5 h-5 text-gray-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="종목명 또는 코드 검색..."
              className="flex-1 bg-transparent text-white placeholder-gray-500 outline-none"
            />
          </div>
        </div>

        {/* 종목 리스트 */}
        <div className="flex-1 bg-gray-900 rounded-xl border border-gray-800 p-4 overflow-hidden flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">
              {searchQuery ? '검색 결과' : '인기 종목'}
            </h3>
            <button className="p-2 text-gray-400 hover:text-white transition-colors">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2">
            {(searchQuery ? searchResults : popularStocks).map((stock) => (
              <StockSearchResult
                key={stock.code}
                stock={stock}
                onSelect={setSelectedStock}
              />
            ))}
          </div>
        </div>
      </div>

      {/* 오른쪽: 종목 상세 정보 */}
      <div className="col-span-7 bg-gray-900 rounded-xl border border-gray-800 p-6">
        <StockDetailPanel stockCode={selectedStock} />
      </div>
    </div>
  );
}
