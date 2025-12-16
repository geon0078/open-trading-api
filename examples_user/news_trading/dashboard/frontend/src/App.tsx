import { useState } from 'react';
import { Activity, Wifi, WifiOff, LayoutGrid, Zap } from 'lucide-react';
import { useDashboardStore } from './store';
import { useSSE } from './hooks/useSSE';
import { useWebSocket } from './hooks/useWebSocket';
import { SurgePanel } from './components/dashboard/SurgePanel';
import { LLMPanel } from './components/dashboard/LLMPanel';
import { AccountPanel } from './components/dashboard/AccountPanel';
import { AutoTradePanel } from './components/dashboard/AutoTradePanel';
import { NewsAnalysisPanel } from './components/dashboard/NewsAnalysisPanel';
import { LLMOutputPanel } from './components/dashboard/LLMOutputPanel';

function ConnectionStatus() {
  const { sseConnected, wsConnected } = useDashboardStore();

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1">
        <span className="text-xs text-gray-500">SSE:</span>
        {sseConnected ? (
          <Wifi className="w-3 h-3 text-green-400" />
        ) : (
          <WifiOff className="w-3 h-3 text-red-400" />
        )}
      </div>
      <div className="flex items-center gap-1">
        <span className="text-xs text-gray-500">WS:</span>
        {wsConnected ? (
          <Wifi className="w-3 h-3 text-green-400" />
        ) : (
          <WifiOff className="w-3 h-3 text-yellow-400" />
        )}
      </div>
    </div>
  );
}

type ViewMode = 'monitor' | 'auto-trade';

function Header({
  viewMode,
  onViewModeChange,
}: {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}) {
  const { isAutoTrading } = useDashboardStore();

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-purple-400" />
          <div>
            <h1 className="text-xl font-bold">Trading Dashboard</h1>
            <p className="text-sm text-gray-400">실시간 트레이딩 모니터링</p>
          </div>
        </div>

        {/* 뷰 모드 탭 */}
        <div className="flex items-center gap-4">
          <div className="flex bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => onViewModeChange('monitor')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'monitor'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <LayoutGrid className="w-4 h-4" />
              모니터링
            </button>
            <button
              onClick={() => onViewModeChange('auto-trade')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'auto-trade'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Zap className={`w-4 h-4 ${isAutoTrading ? 'text-yellow-400' : ''}`} />
              자동매매
              {isAutoTrading && (
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              )}
            </button>
          </div>
          <ConnectionStatus />
        </div>
      </div>
    </header>
  );
}

function MonitorView() {
  return (
    <div className="grid grid-cols-12 gap-4 h-full">
      {/* 왼쪽: 급등 종목 */}
      <div className="col-span-5 h-full">
        <SurgePanel />
      </div>

      {/* 중앙: LLM 분석 */}
      <div className="col-span-4 h-full">
        <LLMPanel />
      </div>

      {/* 오른쪽: 계좌 정보 */}
      <div className="col-span-3 h-full">
        <AccountPanel />
      </div>
    </div>
  );
}

function AutoTradeView() {
  return (
    <div className="grid grid-cols-12 gap-4 h-full">
      {/* 왼쪽: 자동 매매 + LLM 출력 */}
      <div className="col-span-5 h-full flex flex-col gap-4">
        <div className="flex-[2]">
          <AutoTradePanel />
        </div>
        <div className="flex-1">
          <LLMOutputPanel />
        </div>
      </div>

      {/* 중앙: 뉴스 분석 */}
      <div className="col-span-4 h-full">
        <NewsAnalysisPanel />
      </div>

      {/* 오른쪽: 계좌 정보 + 급등 종목 */}
      <div className="col-span-3 h-full flex flex-col gap-4">
        <div className="flex-1">
          <AccountPanel />
        </div>
        <div className="flex-1">
          <SurgePanel />
        </div>
      </div>
    </div>
  );
}

function Dashboard({ viewMode }: { viewMode: ViewMode }) {
  return (
    <div className="flex-1 p-4 overflow-hidden">
      {viewMode === 'monitor' ? <MonitorView /> : <AutoTradeView />}
    </div>
  );
}

export default function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('auto-trade');

  // SSE 연결 초기화
  useSSE();

  // WebSocket 연결 초기화 (계좌 업데이트, LLM 출력 등)
  useWebSocket();

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
      <Header viewMode={viewMode} onViewModeChange={setViewMode} />
      <Dashboard viewMode={viewMode} />
    </div>
  );
}
