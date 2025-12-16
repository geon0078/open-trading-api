import { useState, useEffect } from 'react';
import {
  Wifi,
  WifiOff,
  Moon,
  Sun,
  RefreshCw,
  Clock,
  TrendingUp,
  TrendingDown,
} from 'lucide-react';
import { useDashboardStore } from '../../store';
import clsx from 'clsx';

function ConnectionStatus() {
  const { sseConnected, wsConnected } = useDashboardStore();

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg">
        <span className="text-xs text-gray-500">SSE</span>
        <div
          className={clsx(
            'w-2 h-2 rounded-full',
            sseConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
          )}
        />
      </div>
      <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg">
        <span className="text-xs text-gray-500">WS</span>
        <div
          className={clsx(
            'w-2 h-2 rounded-full',
            wsConnected ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'
          )}
        />
      </div>
    </div>
  );
}

function MarketStatus() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState<'pre' | 'open' | 'closed'>('closed');

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setCurrentTime(now);

      const hours = now.getHours();
      const minutes = now.getMinutes();
      const time = hours * 60 + minutes;

      // 장 시간 체크 (9:00 ~ 15:30)
      if (time >= 540 && time < 930) {
        setMarketStatus('open');
      } else if (time >= 480 && time < 540) {
        setMarketStatus('pre');
      } else {
        setMarketStatus('closed');
      }
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const statusConfig = {
    pre: { label: '장 시작 전', color: 'text-yellow-400', bgColor: 'bg-yellow-400/10' },
    open: { label: '장중', color: 'text-green-400', bgColor: 'bg-green-400/10' },
    closed: { label: '장 마감', color: 'text-gray-400', bgColor: 'bg-gray-400/10' },
  };

  const config = statusConfig[marketStatus];

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2 text-gray-400">
        <Clock className="w-4 h-4" />
        <span className="text-sm font-mono">
          {currentTime.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </span>
      </div>
      <div className={clsx('flex items-center gap-2 px-3 py-1.5 rounded-lg', config.bgColor)}>
        <div className={clsx('w-2 h-2 rounded-full', config.color.replace('text-', 'bg-'))} />
        <span className={clsx('text-sm font-medium', config.color)}>{config.label}</span>
      </div>
    </div>
  );
}

function IndexTicker() {
  // 실제로는 API에서 가져와야 함
  const indices = [
    { name: 'KOSPI', value: 2456.23, change: 1.24, isUp: true },
    { name: 'KOSDAQ', value: 678.45, change: -0.87, isUp: false },
  ];

  return (
    <div className="flex items-center gap-4">
      {indices.map((index) => (
        <div key={index.name} className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg">
          <span className="text-sm text-gray-400">{index.name}</span>
          <span className="text-sm font-medium text-white">{index.value.toLocaleString()}</span>
          <div className={clsx('flex items-center gap-0.5', index.isUp ? 'text-red-400' : 'text-blue-400')}>
            {index.isUp ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            <span className="text-xs">{index.isUp ? '+' : ''}{index.change}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function Header() {
  const [isDarkMode, setIsDarkMode] = useState(true);

  return (
    <header className="flex items-center justify-between h-14 px-4 bg-gray-900 border-b border-gray-800">
      <div className="flex items-center gap-6">
        <IndexTicker />
      </div>

      <div className="flex items-center gap-4">
        <MarketStatus />
        <ConnectionStatus />

        <div className="flex items-center gap-2">
          <button
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            title="새로고침"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            title={isDarkMode ? '라이트 모드' : '다크 모드'}
          >
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </header>
  );
}
