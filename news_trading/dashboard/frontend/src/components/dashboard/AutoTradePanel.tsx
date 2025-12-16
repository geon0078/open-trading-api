import { useState, useCallback, useEffect } from 'react';
import {
  Play,
  Square,
  Settings,
  History,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Zap,
  TrendingUp,
  TrendingDown,
  Newspaper,
  Clock,
} from 'lucide-react';
import { useDashboardStore } from '../../store';
import { autoTradeApi } from '../../api/client';
import type { AutoTradeConfig, AutoTradeResult, AutoTradeHistoryItem } from '../../types';
import clsx from 'clsx';

const signalColors: Record<string, string> = {
  STRONG_BUY: 'text-green-400 bg-green-400/10',
  BUY: 'text-green-500 bg-green-500/10',
  HOLD: 'text-yellow-500 bg-yellow-500/10',
  SELL: 'text-red-500 bg-red-500/10',
  STRONG_SELL: 'text-red-400 bg-red-400/10',
};

const actionIcons: Record<string, JSX.Element> = {
  BUY: <TrendingUp className="w-4 h-4 text-green-400" />,
  SELL: <TrendingDown className="w-4 h-4 text-red-400" />,
  HOLD: <span className="w-4 h-4 text-yellow-400">-</span>,
  SKIP: <span className="w-4 h-4 text-gray-400">-</span>,
  ERROR: <XCircle className="w-4 h-4 text-red-500" />,
};

interface ConfigModalProps {
  config: AutoTradeConfig;
  onSave: (config: Partial<AutoTradeConfig>) => void;
  onClose: () => void;
}

function ConfigModal({ config, onSave, onClose }: ConfigModalProps) {
  const [formData, setFormData] = useState<Partial<AutoTradeConfig>>({
    env_dv: config.env_dv,
    max_order_amount: config.max_order_amount,
    min_confidence: config.min_confidence,
    min_consensus: config.min_consensus,
    stop_loss_pct: config.stop_loss_pct,
    take_profit_pct: config.take_profit_pct,
    max_daily_trades: config.max_daily_trades,
    max_daily_loss: config.max_daily_loss,
    min_surge_score: config.min_surge_score,
    max_stocks_per_scan: config.max_stocks_per_scan,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
        <h3 className="text-lg font-bold mb-4">자동 매매 설정</h3>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">환경</label>
            <div className="w-full bg-gray-700 rounded px-3 py-2 text-red-400 font-medium">
              실전투자
            </div>
            <input type="hidden" value="real" />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">
              1회 주문 한도 (원)
            </label>
            <input
              type="number"
              value={formData.max_order_amount}
              onChange={(e) =>
                setFormData({ ...formData, max_order_amount: Number(e.target.value) })
              }
              className="w-full bg-gray-700 rounded px-3 py-2"
              min={1000}
              step={10000}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                최소 신뢰도
              </label>
              <input
                type="number"
                value={formData.min_confidence}
                onChange={(e) =>
                  setFormData({ ...formData, min_confidence: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={0}
                max={1}
                step={0.05}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                최소 합의도
              </label>
              <input
                type="number"
                value={formData.min_consensus}
                onChange={(e) =>
                  setFormData({ ...formData, min_consensus: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={0}
                max={1}
                step={0.05}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                손절률 (%)
              </label>
              <input
                type="number"
                value={formData.stop_loss_pct}
                onChange={(e) =>
                  setFormData({ ...formData, stop_loss_pct: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={0.1}
                max={5}
                step={0.1}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                익절률 (%)
              </label>
              <input
                type="number"
                value={formData.take_profit_pct}
                onChange={(e) =>
                  setFormData({ ...formData, take_profit_pct: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={0.1}
                max={10}
                step={0.1}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                일일 최대 거래
              </label>
              <input
                type="number"
                value={formData.max_daily_trades}
                onChange={(e) =>
                  setFormData({ ...formData, max_daily_trades: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={1}
                max={100}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                일일 최대 손실 (원)
              </label>
              <input
                type="number"
                value={formData.max_daily_loss}
                onChange={(e) =>
                  setFormData({ ...formData, max_daily_loss: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={1000}
                step={10000}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                최소 급등 점수
              </label>
              <input
                type="number"
                value={formData.min_surge_score}
                onChange={(e) =>
                  setFormData({ ...formData, min_surge_score: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={0}
                max={100}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                최대 종목 수
              </label>
              <input
                type="number"
                value={formData.max_stocks_per_scan}
                onChange={(e) =>
                  setFormData({ ...formData, max_stocks_per_scan: Number(e.target.value) })
                }
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={1}
                max={20}
              />
            </div>
          </div>

          <div className="flex justify-end gap-2 mt-6">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600"
            >
              취소
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-purple-600 rounded hover:bg-purple-500"
            >
              저장
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function ResultRow({ result }: { result: AutoTradeResult | AutoTradeHistoryItem }) {
  const isHistory = 'order_amount' in result;

  return (
    <tr className={clsx(result.success ? 'bg-green-900/10' : '')}>
      <td className="text-xs text-gray-500">
        {'timestamp' in result ? result.timestamp.split(' ')[1] : ''}
      </td>
      <td>
        <div className="flex items-center gap-2">
          {actionIcons[result.action] || actionIcons.SKIP}
          <div>
            <div className="font-medium">{result.stock_name}</div>
            <div className="text-xs text-gray-500">{result.stock_code}</div>
          </div>
        </div>
      </td>
      <td className="text-center">
        <span className={clsx('px-2 py-0.5 rounded text-xs', signalColors[result.ensemble_signal])}>
          {result.ensemble_signal}
        </span>
      </td>
      <td className="text-right font-mono text-sm">
        {(result.confidence * 100).toFixed(0)}%
      </td>
      <td className="text-right font-mono text-sm">
        {result.order_qty > 0 ? (
          <>
            {result.order_qty}주
            <div className="text-xs text-gray-500">
              @{result.order_price.toLocaleString()}
            </div>
          </>
        ) : (
          '-'
        )}
      </td>
      <td className="text-center">
        {result.success ? (
          <CheckCircle className="w-4 h-4 text-green-400 inline" />
        ) : (
          <XCircle className="w-4 h-4 text-gray-500 inline" />
        )}
      </td>
    </tr>
  );
}

export function AutoTradePanel() {
  const {
    autoTradeStatus,
    autoTradeResults,
    autoTradeHistory,
    isAutoTrading,
    lastAutoTradeUpdate,
    tradingMode,
    setAutoTradeStatus,
    setAutoTradeResults,
    setAutoTradeHistory,
    setIsAutoTrading,
    setTradingMode,
  } = useDashboardStore();

  const [showConfig, setShowConfig] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'results' | 'history'>('results');
  const [scanInterval, setScanInterval] = useState(60);

  // 상태 로드
  const loadStatus = useCallback(async () => {
    try {
      const [status, mode] = await Promise.all([
        autoTradeApi.getStatus(),
        autoTradeApi.getMode(),
      ]);
      setAutoTradeStatus(status);
      setTradingMode(mode);
    } catch (error) {
      console.error('상태 로드 실패:', error);
    }
  }, [setAutoTradeStatus, setTradingMode]);

  // 히스토리 로드
  const loadHistory = useCallback(async () => {
    try {
      const history = await autoTradeApi.getHistory(50);
      setAutoTradeHistory(history.items);
    } catch (error) {
      console.error('히스토리 로드 실패:', error);
    }
  }, [setAutoTradeHistory]);

  // 초기 로드
  useEffect(() => {
    loadStatus();
    loadHistory();

    // 30초마다 상태 업데이트
    const interval = setInterval(loadStatus, 30000);
    return () => clearInterval(interval);
  }, [loadStatus, loadHistory]);

  // 시작/중지
  const handleToggle = async () => {
    setIsLoading(true);
    try {
      if (isAutoTrading) {
        await autoTradeApi.stop();
        setIsAutoTrading(false);
      } else {
        await autoTradeApi.start(scanInterval);
        setIsAutoTrading(true);
      }
      await loadStatus();
    } catch (error) {
      console.error('토글 실패:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // 수동 스캔
  const handleManualScan = async () => {
    setIsLoading(true);
    try {
      const results = await autoTradeApi.scan(
        autoTradeStatus?.config?.min_surge_score,
        autoTradeStatus?.config?.max_stocks_per_scan,
        true
      );
      setAutoTradeResults(results);
      await loadHistory();
    } catch (error) {
      console.error('스캔 실패:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // 스캘핑 실행
  const handleScalping = async () => {
    setIsLoading(true);
    try {
      const results = await autoTradeApi.scalping();
      setAutoTradeResults(results);
      await loadHistory();
    } catch (error) {
      console.error('스캘핑 실패:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // 설정 저장
  const handleSaveConfig = async (config: Partial<AutoTradeConfig>) => {
    try {
      await autoTradeApi.updateConfig(config);
      await loadStatus();
    } catch (error) {
      console.error('설정 저장 실패:', error);
    }
  };

  const config = autoTradeStatus?.config;

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Zap className={clsx('w-5 h-5', isAutoTrading ? 'text-yellow-400' : 'text-gray-400')} />
          <h2 className="panel-title">자동 매매</h2>
          {isAutoTrading && (
            <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded animate-pulse">
              실행중
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {lastAutoTradeUpdate && (
            <span className="text-xs text-gray-500">
              {lastAutoTradeUpdate.toLocaleTimeString('ko-KR')}
            </span>
          )}
          <button
            onClick={() => setShowConfig(true)}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 현재 모드 표시 */}
      {tradingMode && (
        <div className="px-4 py-2 border-b border-gray-700 flex items-center justify-between bg-gray-800/50">
          <div className="flex items-center gap-2">
            {tradingMode.mode === 'NEWS' ? (
              <Newspaper className="w-4 h-4 text-blue-400" />
            ) : (
              <Zap className="w-4 h-4 text-yellow-400" />
            )}
            <span className="text-sm">{tradingMode.mode_description}</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Clock className="w-3 h-3" />
            <span>{tradingMode.market_status}</span>
          </div>
        </div>
      )}

      {/* 상태 요약 */}
      <div className="px-4 py-3 border-b border-gray-700 space-y-2">
        {/* 환경 & 거래 가능 여부 */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-500/20 text-red-400">
              실전투자
            </span>
            {autoTradeStatus?.can_trade ? (
              <span className="flex items-center gap-1 text-green-400 text-xs">
                <CheckCircle className="w-3 h-3" /> 거래 가능
              </span>
            ) : (
              <span className="flex items-center gap-1 text-yellow-400 text-xs">
                <AlertTriangle className="w-3 h-3" />
                {autoTradeStatus?.market_status?.reason ||
                  autoTradeStatus?.risk_status?.reason ||
                  '거래 불가'}
              </span>
            )}
          </div>
          <div className="text-xs text-gray-400">
            오늘 {autoTradeStatus?.today_trades || 0}/{config?.max_daily_trades || 10}회
          </div>
        </div>

        {/* 설정 요약 */}
        <div className="grid grid-cols-4 gap-2 text-xs">
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-500">주문한도</div>
            <div className="font-mono">{(config?.max_order_amount || 0).toLocaleString()}</div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-500">신뢰도</div>
            <div className="font-mono">{((config?.min_confidence || 0) * 100).toFixed(0)}%</div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-500">손절</div>
            <div className="font-mono text-red-400">-{config?.stop_loss_pct || 0}%</div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-500">익절</div>
            <div className="font-mono text-green-400">+{config?.take_profit_pct || 0}%</div>
          </div>
        </div>

        {/* 컨트롤 버튼 */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 flex-1">
            <input
              type="number"
              value={scanInterval}
              onChange={(e) => setScanInterval(Number(e.target.value))}
              className="w-16 bg-gray-700 rounded px-2 py-1 text-sm"
              min={10}
              max={300}
              disabled={isAutoTrading}
            />
            <span className="text-xs text-gray-500">초</span>
          </div>

          <button
            onClick={handleToggle}
            disabled={isLoading}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
              isAutoTrading
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : 'bg-green-600 hover:bg-green-500 text-white',
              isLoading && 'opacity-50'
            )}
          >
            {isAutoTrading ? (
              <>
                <Square className="w-4 h-4" /> 중지
              </>
            ) : (
              <>
                <Play className="w-4 h-4" /> 시작
              </>
            )}
          </button>

          <button
            onClick={handleManualScan}
            disabled={isLoading || isAutoTrading}
            className={clsx(
              'p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300',
              (isLoading || isAutoTrading) && 'opacity-50'
            )}
            title="수동 스캔"
          >
            <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
          </button>

          <button
            onClick={handleScalping}
            disabled={isLoading || isAutoTrading}
            className={clsx(
              'px-3 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-sm',
              (isLoading || isAutoTrading) && 'opacity-50'
            )}
            title="스캘핑 모드"
          >
            스캘핑
          </button>
        </div>
      </div>

      {/* 탭 */}
      <div className="flex border-b border-gray-700">
        <button
          onClick={() => setActiveTab('results')}
          className={clsx(
            'flex-1 py-2 text-sm font-medium transition-colors',
            activeTab === 'results'
              ? 'text-purple-400 border-b-2 border-purple-400'
              : 'text-gray-400 hover:text-gray-300'
          )}
        >
          실시간 결과 ({autoTradeResults.length})
        </button>
        <button
          onClick={() => {
            setActiveTab('history');
            loadHistory();
          }}
          className={clsx(
            'flex-1 py-2 text-sm font-medium transition-colors flex items-center justify-center gap-1',
            activeTab === 'history'
              ? 'text-purple-400 border-b-2 border-purple-400'
              : 'text-gray-400 hover:text-gray-300'
          )}
        >
          <History className="w-4 h-4" /> 히스토리 ({autoTradeHistory.length})
        </button>
      </div>

      {/* 결과 테이블 */}
      <div className="flex-1 overflow-auto">
        <table className="data-table">
          <thead className="sticky top-0 bg-gray-800">
            <tr>
              <th className="w-16">시간</th>
              <th>종목</th>
              <th className="text-center">시그널</th>
              <th className="text-right">신뢰도</th>
              <th className="text-right">주문</th>
              <th className="text-center w-12">결과</th>
            </tr>
          </thead>
          <tbody>
            {activeTab === 'results' ? (
              autoTradeResults.length > 0 ? (
                autoTradeResults.map((result, idx) => (
                  <ResultRow key={`${result.stock_code}-${idx}`} result={result} />
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="text-center py-8 text-gray-500">
                    {isAutoTrading ? '대기 중...' : '스캔 결과가 없습니다.'}
                  </td>
                </tr>
              )
            ) : autoTradeHistory.length > 0 ? (
              autoTradeHistory.map((item, idx) => (
                <ResultRow key={`${item.stock_code}-${idx}`} result={item} />
              ))
            ) : (
              <tr>
                <td colSpan={6} className="text-center py-8 text-gray-500">
                  거래 히스토리가 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* 모델 정보 */}
      {autoTradeStatus?.ensemble_models && autoTradeStatus.ensemble_models.length > 0 && (
        <div className="px-4 py-2 border-t border-gray-700 text-xs text-gray-500">
          <span className="text-gray-400">모델:</span>{' '}
          {autoTradeStatus.ensemble_models.join(', ')}
          {autoTradeStatus.main_model && (
            <span className="ml-2 text-purple-400">
              (메인: {autoTradeStatus.main_model.split('/').pop()})
            </span>
          )}
        </div>
      )}

      {/* 설정 모달 */}
      {showConfig && config && (
        <ConfigModal
          config={config}
          onSave={handleSaveConfig}
          onClose={() => setShowConfig(false)}
        />
      )}
    </div>
  );
}
