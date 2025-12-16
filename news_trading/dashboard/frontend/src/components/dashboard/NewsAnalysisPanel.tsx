import { useState, useCallback, useEffect } from 'react';
import {
  Newspaper,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  Clock,
  Tag,
  Eye,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { useDashboardStore } from '../../store';
import { autoTradeApi } from '../../api/client';
import clsx from 'clsx';

const sentimentConfig = {
  BULLISH: {
    icon: TrendingUp,
    color: 'text-green-400',
    bg: 'bg-green-400/10',
    label: '강세',
  },
  BEARISH: {
    icon: TrendingDown,
    color: 'text-red-400',
    bg: 'bg-red-400/10',
    label: '약세',
  },
  NEUTRAL: {
    icon: Minus,
    color: 'text-yellow-400',
    bg: 'bg-yellow-400/10',
    label: '중립',
  },
};

export function NewsAnalysisPanel() {
  const {
    tradingMode,
    newsAnalysis,
    isAnalyzingNews,
    lastNewsUpdate,
    setTradingMode,
    setNewsAnalysis,
    setIsAnalyzingNews,
  } = useDashboardStore();

  const [isExpanded, setIsExpanded] = useState(false);
  const [showNewsList, setShowNewsList] = useState(false);

  // 모드 로드
  const loadMode = useCallback(async () => {
    try {
      const mode = await autoTradeApi.getMode();
      setTradingMode(mode);
    } catch (error) {
      console.error('모드 로드 실패:', error);
    }
  }, [setTradingMode]);

  // 뉴스 분석 실행
  const handleAnalyze = async () => {
    setIsAnalyzingNews(true);
    try {
      const result = await autoTradeApi.runNewsAnalysis(30);
      setNewsAnalysis(result);
      await loadMode();
    } catch (error) {
      console.error('뉴스 분석 실패:', error);
    } finally {
      setIsAnalyzingNews(false);
    }
  };

  // 초기 로드
  useEffect(() => {
    loadMode();
    const interval = setInterval(loadMode, 30000);
    return () => clearInterval(interval);
  }, [loadMode]);

  const sentiment = newsAnalysis?.market_sentiment || 'NEUTRAL';
  const SentimentIcon = sentimentConfig[sentiment]?.icon || Minus;
  const sentimentStyle = sentimentConfig[sentiment] || sentimentConfig.NEUTRAL;

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Newspaper className="w-5 h-5 text-blue-400" />
          <h2 className="panel-title">뉴스 분석</h2>
          {tradingMode?.mode === 'NEWS' && (
            <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded">
              분석 모드
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {lastNewsUpdate && (
            <span className="text-xs text-gray-500">
              {lastNewsUpdate.toLocaleTimeString('ko-KR')}
            </span>
          )}
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzingNews}
            className={clsx(
              'p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300',
              isAnalyzingNews && 'opacity-50'
            )}
            title="뉴스 분석 실행"
          >
            <RefreshCw className={clsx('w-4 h-4', isAnalyzingNews && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* 모드 표시 */}
      <div className="px-4 py-2 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-gray-400">
            {tradingMode?.mode_description || '초기화 중...'}
          </span>
        </div>
        <span className="text-xs text-gray-500">
          {tradingMode?.market_status || ''}
        </span>
      </div>

      {/* 시장 심리 */}
      <div className="px-4 py-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={clsx('p-3 rounded-xl', sentimentStyle.bg)}>
              <SentimentIcon className={clsx('w-8 h-8', sentimentStyle.color)} />
            </div>
            <div>
              <div className="text-sm text-gray-400">시장 심리</div>
              <div className={clsx('text-2xl font-bold', sentimentStyle.color)}>
                {sentimentStyle.label}
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">분석 뉴스</div>
            <div className="text-xl font-mono">{newsAnalysis?.news_count || 0}건</div>
          </div>
        </div>
      </div>

      {/* 주요 테마 */}
      {newsAnalysis?.key_themes && newsAnalysis.key_themes.length > 0 && (
        <div className="px-4 py-3 border-b border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <Tag className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">주요 테마</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {newsAnalysis.key_themes.map((theme, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-purple-500/20 text-purple-300 text-sm rounded-full"
              >
                {theme}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 주목 종목 */}
      <div className="flex-1 overflow-auto">
        {newsAnalysis?.attention_stocks && newsAnalysis.attention_stocks.length > 0 ? (
          <div className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <Eye className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">
                주목 종목 ({newsAnalysis.attention_stocks.length})
              </span>
            </div>
            <div className="space-y-2">
              {newsAnalysis.attention_stocks.map((stock, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-gray-800 rounded-lg hover:bg-gray-750 transition-colors"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-blue-400">{stock.name}</span>
                    <span className="text-xs text-gray-500">{stock.code}</span>
                  </div>
                  <p className="text-sm text-gray-400 line-clamp-2">{stock.reason}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            {isAnalyzingNews ? '분석 중...' : '뉴스 분석을 실행하세요'}
          </div>
        )}
      </div>

      {/* 시장 전망 */}
      {newsAnalysis?.market_outlook && (
        <div className="px-4 py-3 border-t border-gray-700">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-300 w-full"
          >
            {isExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
            시장 전망
          </button>
          {isExpanded && (
            <p className="mt-2 text-sm text-gray-300 leading-relaxed">
              {newsAnalysis.market_outlook}
            </p>
          )}
        </div>
      )}

      {/* 뉴스 목록 (토글) */}
      {newsAnalysis?.news_list && newsAnalysis.news_list.length > 0 && (
        <div className="border-t border-gray-700">
          <button
            onClick={() => setShowNewsList(!showNewsList)}
            className="w-full px-4 py-2 flex items-center justify-between text-sm text-gray-400 hover:bg-gray-800"
          >
            <span>수집된 뉴스 ({newsAnalysis.news_list.length}건)</span>
            {showNewsList ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
          {showNewsList && (
            <div className="max-h-48 overflow-y-auto px-4 pb-3">
              {newsAnalysis.news_list.slice(0, 10).map((news, idx) => (
                <div
                  key={idx}
                  className="py-2 border-b border-gray-700 last:border-0 text-sm text-gray-300"
                >
                  {idx + 1}. {news}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
