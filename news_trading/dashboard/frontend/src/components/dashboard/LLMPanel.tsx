import { useState } from 'react';
import { Brain, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import { useDashboardStore } from '../../store';
import { useLLMStream } from '../../hooks/useSSE';
import type { EnsembleAnalysisResult, ModelResult } from '../../types';
import clsx from 'clsx';

const signalColors: Record<string, string> = {
  STRONG_BUY: 'text-green-400',
  BUY: 'text-green-500',
  HOLD: 'text-gray-400',
  SELL: 'text-red-500',
  STRONG_SELL: 'text-red-400',
};

export function LLMPanel() {
  const {
    llmAnalyses,
    currentAnalysis,
    isAnalyzing,
    analysisProgress,
    selectedStock,
  } = useDashboardStore();
  const { startAnalysis, stopAnalysis } = useLLMStream();
  const [expandedAnalysis, setExpandedAnalysis] = useState<string | null>(null);

  const handleAnalyze = () => {
    if (selectedStock) {
      startAnalysis(selectedStock.code, selectedStock.name);
    }
  };

  const toggleExpand = (id: string) => {
    setExpandedAnalysis(expandedAnalysis === id ? null : id);
  };

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <h2 className="panel-title">LLM 분석</h2>
        </div>
        <div className="flex items-center gap-3">
          {isAnalyzing && analysisProgress && (
            <span className="text-xs text-gray-400 flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              {analysisProgress.model}
            </span>
          )}
          <button
            onClick={isAnalyzing ? stopAnalysis : handleAnalyze}
            disabled={!selectedStock && !isAnalyzing}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              isAnalyzing
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : selectedStock
                  ? 'bg-purple-600 hover:bg-purple-500 text-white'
                  : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            )}
          >
            {isAnalyzing ? '중지' : '분석'}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-3">
        {llmAnalyses.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            {selectedStock
              ? `"${selectedStock.name}" 종목을 분석하려면 분석 버튼을 클릭하세요.`
              : '급등 종목을 선택하면 LLM 분석을 실행할 수 있습니다.'}
          </div>
        ) : (
          llmAnalyses.map((analysis) => (
            <AnalysisCard
              key={analysis.id}
              analysis={analysis}
              isExpanded={expandedAnalysis === analysis.id}
              onToggle={() => toggleExpand(analysis.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}

interface AnalysisCardProps {
  analysis: EnsembleAnalysisResult;
  isExpanded: boolean;
  onToggle: () => void;
}

function AnalysisCard({ analysis, isExpanded, onToggle }: AnalysisCardProps) {
  return (
    <div className="bg-gray-700/50 rounded-lg border border-gray-600">
      {/* 헤더 */}
      <div
        className="p-3 cursor-pointer flex items-center justify-between"
        onClick={onToggle}
      >
        <div className="flex items-center gap-3">
          <div>
            <div className="font-medium">{analysis.stock_name}</div>
            <div className="text-xs text-gray-500">{analysis.stock_code}</div>
          </div>
          <span
            className={clsx(
              'px-2 py-1 rounded text-sm font-bold',
              signalColors[analysis.ensemble_signal]
            )}
          >
            {analysis.ensemble_signal}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right text-sm">
            <div className="text-gray-400">신뢰도</div>
            <div className="font-mono">
              {(analysis.ensemble_confidence * 100).toFixed(0)}%
            </div>
          </div>
          <div className="text-right text-sm">
            <div className="text-gray-400">합의도</div>
            <div className="font-mono">
              {(analysis.consensus_score * 100).toFixed(0)}%
            </div>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </div>
      </div>

      {/* 상세 내용 */}
      {isExpanded && (
        <div className="border-t border-gray-600 p-4 space-y-4">
          {/* 투표 결과 */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm text-gray-400 mb-2">시그널 투표</h4>
              <div className="flex gap-2 flex-wrap">
                {Object.entries(analysis.signal_votes).map(([signal, count]) => (
                  <span
                    key={signal}
                    className={clsx(
                      'px-2 py-1 rounded text-xs',
                      signalColors[signal],
                      'bg-gray-800'
                    )}
                  >
                    {signal}: {count}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-sm text-gray-400 mb-2">추세 투표</h4>
              <div className="flex gap-2 flex-wrap">
                {Object.entries(analysis.trend_votes).map(([trend, count]) => (
                  <span
                    key={trend}
                    className="px-2 py-1 rounded text-xs bg-gray-800"
                  >
                    {trend}: {count}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* 모델별 결과 */}
          <div>
            <h4 className="text-sm text-gray-400 mb-2">모델별 결과</h4>
            <div className="space-y-2">
              {analysis.model_results.map((result, idx) => (
                <ModelResultCard key={idx} result={result} />
              ))}
            </div>
          </div>

          {/* 입력 프롬프트 */}
          {analysis.input_prompt && (
            <div>
              <h4 className="text-sm text-gray-400 mb-2">입력 프롬프트</h4>
              <div className="code-block max-h-40 overflow-auto text-xs">
                {analysis.input_prompt}
              </div>
            </div>
          )}

          {/* 관련 뉴스 */}
          {analysis.news_list.length > 0 && (
            <div>
              <h4 className="text-sm text-gray-400 mb-2">관련 뉴스</h4>
              <ul className="text-sm space-y-1">
                {analysis.news_list.slice(0, 5).map((news, idx) => (
                  <li key={idx} className="text-gray-300 truncate">
                    • {news}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 메타 정보 */}
          <div className="text-xs text-gray-500 flex gap-4">
            <span>처리시간: {analysis.total_processing_time.toFixed(1)}초</span>
            <span>
              모델: {analysis.models_agreed}/{analysis.total_models} 합의
            </span>
            <span>
              {new Date(analysis.timestamp).toLocaleString('ko-KR')}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function ModelResultCard({ result }: { result: ModelResult }) {
  const [showRaw, setShowRaw] = useState(false);

  return (
    <div className="bg-gray-800 rounded p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">{result.model_name}</span>
        <div className="flex items-center gap-3">
          <span className={clsx('font-bold text-sm', signalColors[result.signal])}>
            {result.signal}
          </span>
          <span className="text-xs text-gray-500">
            {result.processing_time.toFixed(1)}초
          </span>
          <button
            onClick={() => setShowRaw(!showRaw)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {showRaw ? '숨기기' : 'Raw'}
          </button>
        </div>
      </div>

      {result.reasoning && (
        <p className="text-sm text-gray-400 mb-2">{result.reasoning}</p>
      )}

      {showRaw && result.raw_output && (
        <div className="code-block max-h-32 overflow-auto text-xs mt-2">
          {result.raw_output}
        </div>
      )}

      {result.error_message && (
        <div className="text-xs text-red-400 mt-1">{result.error_message}</div>
      )}
    </div>
  );
}
