import { useCallback } from 'react';
import { RefreshCw, TrendingUp, Zap } from 'lucide-react';
import { useDashboardStore } from '../../store';
import { surgeApi } from '../../api/client';
import type { SurgeCandidate, SignalType } from '../../types';
import clsx from 'clsx';

const signalColors: Record<SignalType, string> = {
  STRONG_BUY: 'text-green-400 bg-green-400/10',
  BUY: 'text-green-500 bg-green-500/10',
  WATCH: 'text-yellow-500 bg-yellow-500/10',
  NEUTRAL: 'text-gray-400 bg-gray-400/10',
};

const signalLabels: Record<SignalType, string> = {
  STRONG_BUY: '강력매수',
  BUY: '매수',
  WATCH: '관심',
  NEUTRAL: '중립',
};

interface SurgePanelProps {
  onStockSelect?: (stock: SurgeCandidate) => void;
}

export function SurgePanel({ onStockSelect }: SurgePanelProps) {
  const {
    surgeCandidates,
    lastSurgeUpdate,
    isScanning,
    setIsScanning,
    setSurgeCandidates,
    selectedStock,
    setSelectedStock,
  } = useDashboardStore();

  const handleRefresh = useCallback(async () => {
    setIsScanning(true);
    try {
      const result = await surgeApi.getCandidates(50, 20, true);
      setSurgeCandidates(result.candidates);
    } catch (error) {
      console.error('스캔 오류:', error);
    } finally {
      setIsScanning(false);
    }
  }, [setIsScanning, setSurgeCandidates]);

  const handleRowClick = (stock: SurgeCandidate) => {
    setSelectedStock(stock);
    onStockSelect?.(stock);
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString('ko-KR');
  };

  const formatPercent = (num: number) => {
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}%`;
  };

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-green-400" />
          <h2 className="panel-title">급등 종목</h2>
          <span className="text-sm text-gray-400">
            ({surgeCandidates.length}개)
          </span>
        </div>
        <div className="flex items-center gap-3">
          {lastSurgeUpdate && (
            <span className="text-xs text-gray-500">
              {lastSurgeUpdate.toLocaleTimeString('ko-KR')}
            </span>
          )}
          <button
            onClick={handleRefresh}
            disabled={isScanning}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              isScanning
                ? 'bg-gray-700 text-gray-500'
                : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
            )}
          >
            <RefreshCw
              className={clsx('w-4 h-4', isScanning && 'animate-spin')}
            />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <table className="data-table">
          <thead className="sticky top-0 bg-gray-800">
            <tr>
              <th className="w-8">#</th>
              <th>종목</th>
              <th className="text-right">현재가</th>
              <th className="text-right">등락률</th>
              <th className="text-right">
                <span className="flex items-center justify-end gap-1">
                  <Zap className="w-3 h-3" />
                  체결강도
                </span>
              </th>
              <th className="text-right">점수</th>
              <th className="text-center">시그널</th>
            </tr>
          </thead>
          <tbody>
            {surgeCandidates.map((stock) => (
              <tr
                key={stock.code}
                onClick={() => handleRowClick(stock)}
                className={clsx(
                  'cursor-pointer transition-colors',
                  selectedStock?.code === stock.code && 'bg-blue-900/30'
                )}
              >
                <td className="text-gray-500">{stock.rank}</td>
                <td>
                  <div>
                    <div className="font-medium">{stock.name}</div>
                    <div className="text-xs text-gray-500">{stock.code}</div>
                  </div>
                </td>
                <td className="text-right font-mono">
                  {formatNumber(stock.price)}
                </td>
                <td
                  className={clsx(
                    'text-right font-mono',
                    stock.change_rate >= 0 ? 'text-red-400' : 'text-blue-400'
                  )}
                >
                  {formatPercent(stock.change_rate)}
                </td>
                <td
                  className={clsx(
                    'text-right font-mono',
                    stock.volume_power >= 150
                      ? 'text-yellow-400'
                      : stock.volume_power >= 120
                        ? 'text-green-400'
                        : 'text-gray-400'
                  )}
                >
                  {stock.volume_power.toFixed(1)}
                </td>
                <td className="text-right font-mono">
                  <span
                    className={clsx(
                      'px-2 py-0.5 rounded',
                      stock.surge_score >= 70
                        ? 'bg-green-500/20 text-green-400'
                        : stock.surge_score >= 50
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'text-gray-400'
                    )}
                  >
                    {stock.surge_score.toFixed(0)}
                  </span>
                </td>
                <td className="text-center">
                  <span
                    className={clsx(
                      'px-2 py-1 rounded text-xs font-medium',
                      signalColors[stock.signal]
                    )}
                  >
                    {signalLabels[stock.signal]}
                  </span>
                </td>
              </tr>
            ))}
            {surgeCandidates.length === 0 && (
              <tr>
                <td colSpan={7} className="text-center py-8 text-gray-500">
                  {isScanning ? '스캔 중...' : '급등 종목이 없습니다.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
