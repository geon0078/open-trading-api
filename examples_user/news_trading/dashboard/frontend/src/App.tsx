import { useEffect } from 'react';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { useDashboardStore } from './store';
import { useSSE } from './hooks/useSSE';
import { SurgePanel } from './components/dashboard/SurgePanel';
import { LLMPanel } from './components/dashboard/LLMPanel';
import { AccountPanel } from './components/dashboard/AccountPanel';

function ConnectionStatus() {
  const { sseConnected } = useDashboardStore();

  return (
    <div className="flex items-center gap-2">
      {sseConnected ? (
        <>
          <Wifi className="w-4 h-4 text-green-400" />
          <span className="text-sm text-green-400">연결됨</span>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4 text-red-400" />
          <span className="text-sm text-red-400">연결 끊김</span>
        </>
      )}
    </div>
  );
}

function Header() {
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
        <ConnectionStatus />
      </div>
    </header>
  );
}

function Dashboard() {
  return (
    <div className="flex-1 p-4 overflow-hidden">
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
    </div>
  );
}

export default function App() {
  // SSE 연결 초기화
  useSSE();

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
      <Header />
      <Dashboard />
    </div>
  );
}
