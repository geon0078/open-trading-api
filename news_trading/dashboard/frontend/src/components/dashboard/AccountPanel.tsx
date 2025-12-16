import { useEffect, useCallback } from 'react';
import { Wallet, RefreshCw, TrendingUp, TrendingDown } from 'lucide-react';
import { useDashboardStore } from '../../store';
import { accountApi } from '../../api/client';
import clsx from 'clsx';

export function AccountPanel() {
  const {
    accountBalance,
    todayOrders,
    lastAccountUpdate,
    setAccountBalance,
    setTodayOrders,
  } = useDashboardStore();

  const fetchAccountData = useCallback(async () => {
    try {
      const [balance, orders] = await Promise.all([
        accountApi.getBalance(),
        accountApi.getTodayOrders(),
      ]);
      setAccountBalance(balance);
      setTodayOrders(orders.orders);
    } catch (error) {
      console.error('계좌 정보 조회 오류:', error);
    }
  }, [setAccountBalance, setTodayOrders]);

  useEffect(() => {
    fetchAccountData();
    const interval = setInterval(fetchAccountData, 30000); // 30초마다 갱신
    return () => clearInterval(interval);
  }, [fetchAccountData]);

  const formatMoney = (num: number) => {
    return num.toLocaleString('ko-KR') + '원';
  };

  const formatPercent = (num: number) => {
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}%`;
  };

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Wallet className="w-5 h-5 text-blue-400" />
          <h2 className="panel-title">계좌 정보</h2>
        </div>
        <div className="flex items-center gap-3">
          {lastAccountUpdate && (
            <span className="text-xs text-gray-500">
              {lastAccountUpdate.toLocaleTimeString('ko-KR')}
            </span>
          )}
          <button
            onClick={fetchAccountData}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {accountBalance ? (
          <>
            {/* 잔고 요약 */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">예수금</div>
                <div className="text-xl font-bold font-mono">
                  {formatMoney(accountBalance.deposit)}
                </div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">총평가금액</div>
                <div className="text-xl font-bold font-mono">
                  {formatMoney(accountBalance.total_eval)}
                </div>
              </div>
            </div>

            {/* 손익 정보 */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-700/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">총 손익</div>
                <div
                  className={clsx(
                    'text-lg font-bold font-mono',
                    accountBalance.total_pnl >= 0
                      ? 'text-red-400'
                      : 'text-blue-400'
                  )}
                >
                  {formatMoney(accountBalance.total_pnl)}
                </div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">수익률</div>
                <div
                  className={clsx(
                    'text-lg font-bold font-mono flex items-center gap-1',
                    accountBalance.pnl_rate >= 0
                      ? 'text-red-400'
                      : 'text-blue-400'
                  )}
                >
                  {accountBalance.pnl_rate >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  {formatPercent(accountBalance.pnl_rate)}
                </div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">매입금액</div>
                <div className="text-lg font-bold font-mono">
                  {formatMoney(accountBalance.total_purchase)}
                </div>
              </div>
            </div>

            {/* 보유 종목 */}
            {accountBalance.holdings.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">
                  보유 종목 ({accountBalance.holdings.length})
                </h3>
                <div className="space-y-2">
                  {accountBalance.holdings.map((holding) => (
                    <div
                      key={holding.code}
                      className="bg-gray-800 rounded-lg p-3 flex items-center justify-between"
                    >
                      <div>
                        <div className="font-medium">{holding.name}</div>
                        <div className="text-xs text-gray-500">
                          {holding.code} · {holding.quantity}주
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono">
                          {formatMoney(holding.eval_amount)}
                        </div>
                        <div
                          className={clsx(
                            'text-sm font-mono',
                            holding.pnl_rate >= 0
                              ? 'text-red-400'
                              : 'text-blue-400'
                          )}
                        >
                          {formatPercent(holding.pnl_rate)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 오늘의 주문 */}
            {todayOrders.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">
                  오늘의 주문 ({todayOrders.length})
                </h3>
                <div className="space-y-2">
                  {todayOrders.slice(0, 5).map((order, idx) => (
                    <div
                      key={idx}
                      className="bg-gray-800 rounded-lg p-3 flex items-center justify-between"
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={clsx(
                            'px-2 py-0.5 rounded text-xs font-medium',
                            order.side === 'BUY'
                              ? 'bg-red-500/20 text-red-400'
                              : 'bg-blue-500/20 text-blue-400'
                          )}
                        >
                          {order.side === 'BUY' ? '매수' : '매도'}
                        </span>
                        <div>
                          <div className="font-medium">{order.stock_name}</div>
                          <div className="text-xs text-gray-500">
                            {order.order_time}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono">
                          {order.order_price.toLocaleString()}원
                        </div>
                        <div className="text-xs text-gray-500">
                          {order.order_qty}주 ·{' '}
                          <span
                            className={clsx(
                              order.status === 'FILLED'
                                ? 'text-green-400'
                                : order.status === 'PARTIAL'
                                  ? 'text-yellow-400'
                                  : 'text-gray-400'
                            )}
                          >
                            {order.status === 'FILLED'
                              ? '체결'
                              : order.status === 'PARTIAL'
                                ? '부분체결'
                                : '대기'}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            계좌 정보를 불러오는 중...
          </div>
        )}
      </div>
    </div>
  );
}
