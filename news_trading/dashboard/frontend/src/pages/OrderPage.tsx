import { useState, useEffect } from 'react';
import {
  ShoppingCart,
  Search,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Check,
  X,
  Loader2,
  Calculator,
  Wallet,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import clsx from 'clsx';
import axios from 'axios';

type OrderType = 'buy' | 'sell';
type PriceType = 'market' | 'limit';

interface OrderForm {
  stockCode: string;
  stockName: string;
  orderType: OrderType;
  priceType: PriceType;
  price: number;
  quantity: number;
}

// 미니 차트 데이터 생성
const generateChartData = (basePrice: number) => {
  const data = [];
  let price = basePrice;
  for (let i = 0; i < 30; i++) {
    price += (Math.random() - 0.5) * basePrice * 0.01;
    data.push({ time: i, price: Math.round(price) });
  }
  return data;
};

function OrderTypeSelector({ orderType, onChange }: { orderType: OrderType; onChange: (type: OrderType) => void }) {
  return (
    <div className="flex bg-gray-800 rounded-lg p-1">
      <button
        onClick={() => onChange('buy')}
        className={clsx(
          'flex-1 py-3 rounded-md font-medium transition-colors',
          orderType === 'buy'
            ? 'bg-red-500 text-white'
            : 'text-gray-400 hover:text-white'
        )}
      >
        매수
      </button>
      <button
        onClick={() => onChange('sell')}
        className={clsx(
          'flex-1 py-3 rounded-md font-medium transition-colors',
          orderType === 'sell'
            ? 'bg-blue-500 text-white'
            : 'text-gray-400 hover:text-white'
        )}
      >
        매도
      </button>
    </div>
  );
}

function StockSearch({ onSelect }: { onSelect: (code: string, name: string) => void }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [showResults, setShowResults] = useState(false);

  // 임시 검색 결과
  const mockStocks = [
    { code: '005930', name: '삼성전자', price: 71500 },
    { code: '000660', name: 'SK하이닉스', price: 198000 },
    { code: '035720', name: '카카오', price: 45500 },
    { code: '035420', name: 'NAVER', price: 187500 },
    { code: '051910', name: 'LG화학', price: 365000 },
  ];

  useEffect(() => {
    if (query.length >= 1) {
      const filtered = mockStocks.filter(
        (s) => s.name.includes(query) || s.code.includes(query)
      );
      setResults(filtered);
      setShowResults(true);
    } else {
      setResults([]);
      setShowResults(false);
    }
  }, [query]);

  return (
    <div className="relative">
      <div className="flex items-center gap-2 px-4 py-3 bg-gray-800 rounded-lg border border-gray-700 focus-within:border-purple-500">
        <Search className="w-5 h-5 text-gray-500" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query && setShowResults(true)}
          placeholder="종목명 또는 코드 검색..."
          className="flex-1 bg-transparent text-white placeholder-gray-500 outline-none"
        />
      </div>
      {showResults && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-gray-800 border border-gray-700 rounded-lg overflow-hidden z-10 shadow-xl">
          {results.map((stock) => (
            <button
              key={stock.code}
              onClick={() => {
                onSelect(stock.code, stock.name);
                setQuery(`${stock.name} (${stock.code})`);
                setShowResults(false);
              }}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-700 transition-colors"
            >
              <div>
                <p className="font-medium text-white">{stock.name}</p>
                <p className="text-sm text-gray-500">{stock.code}</p>
              </div>
              <span className="text-white">{stock.price.toLocaleString()}원</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function OrderBook() {
  // 임시 호가 데이터
  const asks = [
    { price: 71800, volume: 1234 },
    { price: 71700, volume: 2345 },
    { price: 71600, volume: 3456 },
    { price: 71500, volume: 4567 },
    { price: 71400, volume: 5678 },
  ];
  const bids = [
    { price: 71300, volume: 5678 },
    { price: 71200, volume: 4567 },
    { price: 71100, volume: 3456 },
    { price: 71000, volume: 2345 },
    { price: 70900, volume: 1234 },
  ];
  const maxVolume = Math.max(...asks.map((a) => a.volume), ...bids.map((b) => b.volume));

  return (
    <div className="space-y-1">
      <h4 className="text-sm font-medium text-gray-400 mb-2">호가</h4>
      <div className="space-y-0.5">
        {asks.reverse().map((ask) => (
          <div key={ask.price} className="flex items-center gap-2 relative">
            <div
              className="absolute right-0 top-0 bottom-0 bg-blue-500/10"
              style={{ width: `${(ask.volume / maxVolume) * 100}%` }}
            />
            <span className="flex-1 text-sm text-blue-400 relative z-10">
              {ask.price.toLocaleString()}
            </span>
            <span className="text-sm text-gray-400 relative z-10">
              {ask.volume.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
      <div className="py-2 text-center border-y border-gray-700">
        <span className="text-lg font-bold text-white">71,500</span>
      </div>
      <div className="space-y-0.5">
        {bids.map((bid) => (
          <div key={bid.price} className="flex items-center gap-2 relative">
            <div
              className="absolute left-0 top-0 bottom-0 bg-red-500/10"
              style={{ width: `${(bid.volume / maxVolume) * 100}%` }}
            />
            <span className="flex-1 text-sm text-red-400 relative z-10">
              {bid.price.toLocaleString()}
            </span>
            <span className="text-sm text-gray-400 relative z-10">
              {bid.volume.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function OrderConfirmModal({
  order,
  onConfirm,
  onCancel,
}: {
  order: OrderForm;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const totalAmount = order.price * order.quantity;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 w-96">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">주문 확인</h3>
          <button onClick={onCancel} className="text-gray-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4 mb-6">
          <div className="flex justify-between">
            <span className="text-gray-400">종목</span>
            <span className="text-white">{order.stockName} ({order.stockCode})</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">주문 유형</span>
            <span className={order.orderType === 'buy' ? 'text-red-400' : 'text-blue-400'}>
              {order.orderType === 'buy' ? '매수' : '매도'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">가격</span>
            <span className="text-white">
              {order.priceType === 'market' ? '시장가' : `${order.price.toLocaleString()}원`}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">수량</span>
            <span className="text-white">{order.quantity.toLocaleString()}주</span>
          </div>
          <div className="border-t border-gray-700 pt-4 flex justify-between">
            <span className="text-gray-400">총 금액</span>
            <span className="text-xl font-bold text-white">{totalAmount.toLocaleString()}원</span>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-3 bg-gray-800 text-gray-300 rounded-lg font-medium hover:bg-gray-700 transition-colors"
          >
            취소
          </button>
          <button
            onClick={onConfirm}
            className={clsx(
              'flex-1 py-3 rounded-lg font-medium transition-colors',
              order.orderType === 'buy'
                ? 'bg-red-500 text-white hover:bg-red-600'
                : 'bg-blue-500 text-white hover:bg-blue-600'
            )}
          >
            {order.orderType === 'buy' ? '매수 주문' : '매도 주문'}
          </button>
        </div>
      </div>
    </div>
  );
}

export function OrderPage() {
  const [orderForm, setOrderForm] = useState<OrderForm>({
    stockCode: '',
    stockName: '',
    orderType: 'buy',
    priceType: 'limit',
    price: 0,
    quantity: 1,
  });
  const [showConfirm, setShowConfirm] = useState(false);
  const [cash, setCash] = useState(0);
  const [stockPrice, setStockPrice] = useState(71500);

  useEffect(() => {
    const fetchCash = async () => {
      try {
        const res = await axios.get('/api/v1/account/balance');
        setCash(Number(res.data?.dnca_tot_amt) || 0);
      } catch (err) {
        console.error('예수금 조회 오류:', err);
      }
    };
    fetchCash();
  }, []);

  const handleStockSelect = (code: string, name: string) => {
    setOrderForm((prev) => ({
      ...prev,
      stockCode: code,
      stockName: name,
      price: stockPrice,
    }));
  };

  const handleSubmit = () => {
    if (!orderForm.stockCode || orderForm.quantity <= 0) {
      alert('종목과 수량을 입력해주세요.');
      return;
    }
    setShowConfirm(true);
  };

  const handleConfirm = async () => {
    // 실제 주문 API 호출
    console.log('주문 실행:', orderForm);
    setShowConfirm(false);
    alert('주문이 접수되었습니다.');
  };

  const chartData = generateChartData(stockPrice);
  const totalAmount = (orderForm.priceType === 'market' ? stockPrice : orderForm.price) * orderForm.quantity;

  return (
    <div className="grid grid-cols-12 gap-6 h-full">
      {/* 왼쪽: 주문 폼 */}
      <div className="col-span-5 space-y-6">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <ShoppingCart className="w-6 h-6 text-purple-400" />
            </div>
            <h2 className="text-xl font-bold text-white">주문</h2>
          </div>

          {/* 종목 검색 */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-400 mb-2">종목</label>
            <StockSearch onSelect={handleStockSelect} />
          </div>

          {/* 매수/매도 선택 */}
          <div className="mb-6">
            <OrderTypeSelector
              orderType={orderForm.orderType}
              onChange={(type) => setOrderForm((prev) => ({ ...prev, orderType: type }))}
            />
          </div>

          {/* 가격 유형 */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-400 mb-2">주문 유형</label>
            <div className="flex gap-2">
              <button
                onClick={() => setOrderForm((prev) => ({ ...prev, priceType: 'limit' }))}
                className={clsx(
                  'flex-1 py-2 rounded-lg text-sm font-medium transition-colors',
                  orderForm.priceType === 'limit'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white'
                )}
              >
                지정가
              </button>
              <button
                onClick={() => setOrderForm((prev) => ({ ...prev, priceType: 'market' }))}
                className={clsx(
                  'flex-1 py-2 rounded-lg text-sm font-medium transition-colors',
                  orderForm.priceType === 'market'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:text-white'
                )}
              >
                시장가
              </button>
            </div>
          </div>

          {/* 가격 입력 */}
          {orderForm.priceType === 'limit' && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-400 mb-2">가격</label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={orderForm.price || ''}
                  onChange={(e) => setOrderForm((prev) => ({ ...prev, price: Number(e.target.value) }))}
                  className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-purple-500 outline-none"
                  placeholder="0"
                />
                <span className="text-gray-400">원</span>
              </div>
            </div>
          )}

          {/* 수량 입력 */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-400 mb-2">수량</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={orderForm.quantity || ''}
                onChange={(e) => setOrderForm((prev) => ({ ...prev, quantity: Number(e.target.value) }))}
                className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-purple-500 outline-none"
                placeholder="0"
                min="1"
              />
              <span className="text-gray-400">주</span>
            </div>
            <div className="flex gap-2 mt-2">
              {[10, 25, 50, 100].map((pct) => (
                <button
                  key={pct}
                  onClick={() => {
                    const maxQty = Math.floor(cash / (orderForm.price || stockPrice));
                    setOrderForm((prev) => ({ ...prev, quantity: Math.floor(maxQty * pct / 100) }));
                  }}
                  className="flex-1 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700 hover:text-white transition-colors"
                >
                  {pct}%
                </button>
              ))}
            </div>
          </div>

          {/* 예상 금액 */}
          <div className="p-4 bg-gray-800/50 rounded-lg mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400">예상 금액</span>
              <span className="text-xl font-bold text-white">{totalAmount.toLocaleString()}원</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">주문 가능</span>
              <div className="flex items-center gap-1 text-gray-400">
                <Wallet className="w-4 h-4" />
                <span>{cash.toLocaleString()}원</span>
              </div>
            </div>
          </div>

          {/* 주문 버튼 */}
          <button
            onClick={handleSubmit}
            disabled={!orderForm.stockCode}
            className={clsx(
              'w-full py-4 rounded-lg font-bold text-lg transition-colors',
              !orderForm.stockCode
                ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                : orderForm.orderType === 'buy'
                  ? 'bg-red-500 text-white hover:bg-red-600'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
            )}
          >
            {orderForm.orderType === 'buy' ? '매수' : '매도'}
          </button>
        </div>
      </div>

      {/* 오른쪽: 차트 및 호가 */}
      <div className="col-span-7 space-y-6">
        {/* 미니 차트 */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white">
                {orderForm.stockName || '종목을 선택하세요'}
              </h3>
              {orderForm.stockCode && (
                <p className="text-sm text-gray-500">{orderForm.stockCode}</p>
              )}
            </div>
            {orderForm.stockCode && (
              <div className="text-right">
                <p className="text-2xl font-bold text-white">{stockPrice.toLocaleString()}원</p>
                <p className="text-sm text-red-400">+1,500원 (+2.14%)</p>
              </div>
            )}
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis dataKey="time" hide />
                <YAxis hide domain={['dataMin - 500', 'dataMax + 500']} />
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
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 호가창 */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <OrderBook />
        </div>
      </div>

      {/* 주문 확인 모달 */}
      {showConfirm && (
        <OrderConfirmModal
          order={orderForm}
          onConfirm={handleConfirm}
          onCancel={() => setShowConfirm(false)}
        />
      )}
    </div>
  );
}
