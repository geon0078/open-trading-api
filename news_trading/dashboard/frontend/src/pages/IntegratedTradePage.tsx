import { useState, useEffect, useCallback } from 'react';
import {
  Brain,
  Play,
  Pause,
  Newspaper,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  Shield,
  Zap,
  Send,
  Eye,
  XCircle,
  ArrowRight,
} from 'lucide-react';
import clsx from 'clsx';
import axios from 'axios';

interface TradeSignal {
  stock_code: string;
  stock_name: string;
  action: string;
  confidence: number;
  urgency: string;
  reason: string;
  news_title: string;
  news_sentiment: string;
  technical_score: number;
  trend: string;
  support_price: number;
  resistance_price: number;
  suggested_price: number;
  stop_loss_condition: string;
  take_profit_condition: string;
  timestamp: string;
}

interface ExitDecision {
  stock_code: string;
  stock_name: string;
  current_pnl: number;
  current_pnl_rate: number;
  exit_type: string;
  exit_ratio: number;
  should_exit: boolean;
  reason: string;
  suggested_price: number;
  timestamp: string;
}

interface Position {
  stock_code: string;
  stock_name: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  pnl_rate: number;
  entry_reason: string;
}

interface TradingStatus {
  is_running: boolean;
  initialized: boolean;
  current_mode: string;
  market_sentiment: string;
  market_context: string;
  today_trades: number;
  today_pnl: number;
  positions_count: number;
  positions: Position[];
}

function SignalCard({ signal }: { signal: TradeSignal }) {
  const actionConfig: Record<string, { color: string; bg: string; icon: React.ReactNode }> = {
    BUY: { color: 'text-red-400', bg: 'bg-red-500/10', icon: <TrendingUp className="w-5 h-5" /> },
    SELL: { color: 'text-blue-400', bg: 'bg-blue-500/10', icon: <TrendingDown className="w-5 h-5" /> },
    HOLD: { color: 'text-gray-400', bg: 'bg-gray-500/10', icon: <Clock className="w-5 h-5" /> },
    WATCH: { color: 'text-yellow-400', bg: 'bg-yellow-500/10', icon: <Eye className="w-5 h-5" /> },
  };

  const urgencyConfig: Record<string, string> = {
    LOW: 'bg-gray-500/20 text-gray-400',
    NORMAL: 'bg-blue-500/20 text-blue-400',
    HIGH: 'bg-yellow-500/20 text-yellow-400',
    CRITICAL: 'bg-red-500/20 text-red-400 animate-pulse',
  };

  const config = actionConfig[signal.action] || actionConfig.HOLD;

  return (
    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={clsx('p-2 rounded-lg', config.bg)}>
            <span className={config.color}>{config.icon}</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-white">{signal.stock_name}</span>
              <span className="text-xs text-gray-500">{signal.stock_code}</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className={clsx('px-2 py-0.5 text-xs font-medium rounded', config.bg, config.color)}>
                {signal.action}
              </span>
              <span className={clsx('px-2 py-0.5 text-xs font-medium rounded', urgencyConfig[signal.urgency])}>
                {signal.urgency}
              </span>
              <span className="text-xs text-gray-400">
                신뢰도: {(signal.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
        <span className="text-xs text-gray-500">{signal.timestamp}</span>
      </div>

      {signal.news_title && (
        <div className="mb-3 p-2 bg-gray-900/50 rounded flex items-start gap-2">
          <Newspaper className="w-4 h-4 text-purple-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-gray-300 line-clamp-2">{signal.news_title}</p>
        </div>
      )}

      <p className="text-sm text-gray-300 mb-3">{signal.reason}</p>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="p-2 bg-red-500/5 rounded border border-red-500/20">
          <div className="flex items-center gap-1 text-red-400 text-xs mb-1">
            <Shield className="w-3 h-3" />
            <span>손절 조건 (LLM 판단)</span>
          </div>
          <p className="text-gray-300 text-xs">{signal.stop_loss_condition || '조건 미제시'}</p>
        </div>
        <div className="p-2 bg-green-500/5 rounded border border-green-500/20">
          <div className="flex items-center gap-1 text-green-400 text-xs mb-1">
            <Target className="w-3 h-3" />
            <span>익절 조건 (LLM 판단)</span>
          </div>
          <p className="text-gray-300 text-xs">{signal.take_profit_condition || '조건 미제시'}</p>
        </div>
      </div>

      {(signal.support_price > 0 || signal.resistance_price > 0) && (
        <div className="flex items-center gap-4 mt-3 pt-3 border-t border-gray-700 text-xs text-gray-400">
          {signal.support_price > 0 && (
            <span>지지선: {signal.support_price.toLocaleString()}원</span>
          )}
          {signal.resistance_price > 0 && (
            <span>저항선: {signal.resistance_price.toLocaleString()}원</span>
          )}
          {signal.trend && <span>추세: {signal.trend}</span>}
        </div>
      )}
    </div>
  );
}

function ExitDecisionCard({ decision }: { decision: ExitDecision }) {
  const exitTypeConfig: Record<string, { color: string; icon: React.ReactNode }> = {
    HOLD: { color: 'text-gray-400', icon: <Clock className="w-4 h-4" /> },
    TAKE_PROFIT: { color: 'text-green-400', icon: <CheckCircle className="w-4 h-4" /> },
    STOP_LOSS: { color: 'text-red-400', icon: <XCircle className="w-4 h-4" /> },
    PARTIAL_EXIT: { color: 'text-yellow-400', icon: <ArrowRight className="w-4 h-4" /> },
    URGENT_EXIT: { color: 'text-red-400', icon: <AlertTriangle className="w-4 h-4 animate-pulse" /> },
  };

  const config = exitTypeConfig[decision.exit_type] || exitTypeConfig.HOLD;

  return (
    <div className={clsx(
      'p-3 rounded-lg border',
      decision.should_exit ? 'bg-red-500/5 border-red-500/30' : 'bg-gray-800/50 border-gray-700'
    )}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={config.color}>{config.icon}</span>
          <span className="font-medium text-white">{decision.stock_name}</span>
          <span className={clsx('text-xs px-2 py-0.5 rounded', config.color, 'bg-gray-800')}>
            {decision.exit_type}
          </span>
        </div>
        <span className="text-xs text-gray-500">{decision.timestamp}</span>
      </div>
      <div className="flex items-center gap-4 text-sm mb-2">
        <span className={decision.current_pnl >= 0 ? 'text-red-400' : 'text-blue-400'}>
          손익: {decision.current_pnl >= 0 ? '+' : ''}{decision.current_pnl.toLocaleString()}원 ({decision.current_pnl_rate >= 0 ? '+' : ''}{decision.current_pnl_rate.toFixed(2)}%)
        </span>
        {decision.should_exit && (
          <span className="text-yellow-400">청산 비율: {(decision.exit_ratio * 100).toFixed(0)}%</span>
        )}
      </div>
      <p className="text-sm text-gray-300">{decision.reason}</p>
    </div>
  );
}

function PositionCard({ position }: { position: Position }) {
  const isProfitable = position.pnl >= 0;

  return (
    <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <div>
          <span className="font-medium text-white">{position.stock_name}</span>
          <span className="text-xs text-gray-500 ml-2">{position.stock_code}</span>
        </div>
        <span className={clsx('font-semibold', isProfitable ? 'text-red-400' : 'text-blue-400')}>
          {isProfitable ? '+' : ''}{position.pnl.toLocaleString()}원
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs text-gray-400">
        <div>
          <span className="text-gray-500">수량:</span> {position.quantity}주
        </div>
        <div>
          <span className="text-gray-500">평균가:</span> {position.avg_price.toLocaleString()}원
        </div>
        <div>
          <span className="text-gray-500">현재가:</span> {position.current_price.toLocaleString()}원
        </div>
      </div>
      {position.entry_reason && (
        <p className="text-xs text-gray-500 mt-2 truncate">{position.entry_reason}</p>
      )}
    </div>
  );
}

export function IntegratedTradePage() {
  const [status, setStatus] = useState<TradingStatus | null>(null);
  const [signals, setSignals] = useState<TradeSignal[]>([]);
  const [exitDecisions, setExitDecisions] = useState<ExitDecision[]>([]);
  const [newsInput, setNewsInput] = useState('');
  const [stockCodeInput, setStockCodeInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<TradeSignal[]>([]);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await axios.get('/api/v1/integrated/status');
      setStatus(res.data);
    } catch (err) {
      console.error('상태 조회 오류:', err);
    }
  }, []);

  const fetchSignals = useCallback(async () => {
    try {
      const res = await axios.get('/api/v1/integrated/signals?limit=10');
      setSignals(res.data);
    } catch (err) {
      console.error('시그널 조회 오류:', err);
    }
  }, []);

  const fetchExitDecisions = useCallback(async () => {
    try {
      const res = await axios.get('/api/v1/integrated/exit-decisions?limit=10');
      setExitDecisions(res.data);
    } catch (err) {
      console.error('청산 결정 조회 오류:', err);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchSignals();
    fetchExitDecisions();

    const interval = setInterval(() => {
      fetchStatus();
      fetchSignals();
      fetchExitDecisions();
    }, 10000);

    return () => clearInterval(interval);
  }, [fetchStatus, fetchSignals, fetchExitDecisions]);

  const handleToggle = async () => {
    try {
      if (status?.is_running) {
        await axios.post('/api/v1/integrated/stop');
      } else {
        await axios.post('/api/v1/integrated/start');
      }
      await fetchStatus();
    } catch (err) {
      console.error('토글 오류:', err);
    }
  };

  const handleAnalyzeNews = async () => {
    if (!newsInput.trim()) return;

    setIsAnalyzing(true);
    try {
      const res = await axios.post('/api/v1/integrated/analyze-news', {
        news_title: newsInput,
        stock_code: stockCodeInput || null,
      });
      setAnalysisResult(res.data.signals || []);
    } catch (err) {
      console.error('뉴스 분석 오류:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const modeConfig: Record<string, { color: string; label: string }> = {
    INIT: { color: 'text-gray-400', label: '초기화' },
    PRE_MARKET: { color: 'text-yellow-400', label: '장 시작 전' },
    SCALPING: { color: 'text-orange-400', label: '스캘핑 (09:00~09:30)' },
    REGULAR: { color: 'text-green-400', label: '정규장 (09:30~15:20)' },
    POST_MARKET: { color: 'text-blue-400', label: '장 마감 후' },
    WEEKEND: { color: 'text-purple-400', label: '주말' },
  };

  const sentimentConfig: Record<string, { color: string; icon: React.ReactNode }> = {
    BULLISH: { color: 'text-red-400', icon: <TrendingUp className="w-4 h-4" /> },
    BEARISH: { color: 'text-blue-400', icon: <TrendingDown className="w-4 h-4" /> },
    NEUTRAL: { color: 'text-gray-400', icon: <Clock className="w-4 h-4" /> },
  };

  const currentMode = modeConfig[status?.current_mode || 'INIT'];
  const currentSentiment = sentimentConfig[status?.market_sentiment || 'NEUTRAL'];

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={clsx('p-2 rounded-lg', status?.is_running ? 'bg-green-500/10' : 'bg-purple-500/10')}>
            <Brain className={clsx('w-6 h-6', status?.is_running ? 'text-green-400' : 'text-purple-400')} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">LLM 통합 트레이딩</h1>
            <p className="text-sm text-gray-400">동적 포지션 관리 + 실시간 뉴스 분석</p>
          </div>
        </div>
        <button
          onClick={handleToggle}
          className={clsx(
            'flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-colors',
            status?.is_running
              ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
              : 'bg-green-500 text-white hover:bg-green-600'
          )}
        >
          {status?.is_running ? (
            <>
              <Pause className="w-5 h-5" />
              정지
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              시작
            </>
          )}
        </button>
      </div>

      {/* 상태 카드 */}
      <div className="grid grid-cols-5 gap-4">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">상태</p>
          <div className="flex items-center gap-2">
            <div className={clsx('w-3 h-3 rounded-full', status?.is_running ? 'bg-green-400 animate-pulse' : 'bg-gray-500')} />
            <span className="text-lg font-semibold text-white">{status?.is_running ? '실행 중' : '정지됨'}</span>
          </div>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">현재 모드</p>
          <span className={clsx('text-lg font-semibold', currentMode.color)}>{currentMode.label}</span>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">시장 심리</p>
          <div className="flex items-center gap-2">
            <span className={currentSentiment.color}>{currentSentiment.icon}</span>
            <span className={clsx('text-lg font-semibold', currentSentiment.color)}>{status?.market_sentiment || 'NEUTRAL'}</span>
          </div>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">보유 포지션</p>
          <span className="text-lg font-semibold text-white">{status?.positions_count || 0}개</span>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-1">오늘 손익</p>
          <span className={clsx('text-lg font-semibold', (status?.today_pnl || 0) >= 0 ? 'text-red-400' : 'text-blue-400')}>
            {(status?.today_pnl || 0) >= 0 ? '+' : ''}{(status?.today_pnl || 0).toLocaleString()}원
          </span>
        </div>
      </div>

      {/* 뉴스 분석 입력 */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Newspaper className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">뉴스 즉시 분석</h3>
          <span className="text-xs text-gray-500">(LLM이 뉴스를 분석하고 매매 시그널을 생성합니다)</span>
        </div>
        <div className="flex gap-4">
          <input
            type="text"
            value={newsInput}
            onChange={(e) => setNewsInput(e.target.value)}
            placeholder="뉴스 헤드라인을 입력하세요..."
            className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-purple-500 outline-none"
          />
          <input
            type="text"
            value={stockCodeInput}
            onChange={(e) => setStockCodeInput(e.target.value)}
            placeholder="종목코드 (선택)"
            className="w-32 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-purple-500 outline-none"
          />
          <button
            onClick={handleAnalyzeNews}
            disabled={isAnalyzing || !newsInput.trim()}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
              isAnalyzing || !newsInput.trim()
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-purple-500 text-white hover:bg-purple-600'
            )}
          >
            {isAnalyzing ? (
              <>
                <Brain className="w-4 h-4 animate-spin" />
                분석 중...
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                분석
              </>
            )}
          </button>
        </div>

        {/* 분석 결과 */}
        {analysisResult.length > 0 && (
          <div className="mt-4 space-y-3">
            <h4 className="text-sm font-medium text-gray-400">분석 결과</h4>
            {analysisResult.map((signal, index) => (
              <SignalCard key={index} signal={signal} />
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* 매매 시그널 */}
        <div className="col-span-7 bg-gray-900 rounded-xl border border-gray-800 p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              <h3 className="font-semibold text-white">LLM 매매 시그널</h3>
            </div>
            <span className="text-xs text-gray-500">{signals.length}개</span>
          </div>
          <div className="space-y-3 max-h-[500px] overflow-y-auto">
            {signals.length === 0 ? (
              <p className="text-gray-500 text-center py-8">시그널이 없습니다. 트레이딩을 시작하세요.</p>
            ) : (
              signals.map((signal, index) => (
                <SignalCard key={index} signal={signal} />
              ))
            )}
          </div>
        </div>

        {/* 청산 결정 + 포지션 */}
        <div className="col-span-5 space-y-6">
          {/* 청산 결정 */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-5 h-5 text-red-400" />
              <h3 className="font-semibold text-white">LLM 청산 결정</h3>
            </div>
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {exitDecisions.length === 0 ? (
                <p className="text-gray-500 text-center py-4 text-sm">청산 결정이 없습니다</p>
              ) : (
                exitDecisions.map((decision, index) => (
                  <ExitDecisionCard key={index} decision={decision} />
                ))
              )}
            </div>
          </div>

          {/* 현재 포지션 */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                <h3 className="font-semibold text-white">현재 포지션</h3>
              </div>
              <span className="text-xs text-gray-500">{status?.positions_count || 0}개</span>
            </div>
            <div className="space-y-2 max-h-[250px] overflow-y-auto">
              {!status?.positions?.length ? (
                <p className="text-gray-500 text-center py-4 text-sm">보유 포지션이 없습니다</p>
              ) : (
                status.positions.map((position, index) => (
                  <PositionCard key={index} position={position} />
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 시장 컨텍스트 */}
      {status?.market_context && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Brain className="w-5 h-5 text-purple-400" />
            <h3 className="font-semibold text-white">시장 분석 (LLM)</h3>
          </div>
          <p className="text-gray-300 text-sm">{status.market_context}</p>
        </div>
      )}
    </div>
  );
}
