import { useRef, useEffect } from 'react';
import { Brain, Trash2, Loader2 } from 'lucide-react';
import { useDashboardStore, type LLMOutput } from '../../store';
import clsx from 'clsx';

const outputTypeConfig: Record<
  string,
  { color: string; bg: string; label: string }
> = {
  start: { color: 'text-blue-400', bg: 'bg-blue-500/10', label: 'START' },
  thinking: { color: 'text-yellow-400', bg: 'bg-yellow-500/10', label: 'THINK' },
  response: { color: 'text-green-400', bg: 'bg-green-500/10', label: 'RESP' },
  signal: { color: 'text-purple-400', bg: 'bg-purple-500/10', label: 'SIGNAL' },
  complete: { color: 'text-cyan-400', bg: 'bg-cyan-500/10', label: 'DONE' },
  error: { color: 'text-red-400', bg: 'bg-red-500/10', label: 'ERROR' },
};

export function LLMOutputPanel() {
  const { llmOutputs, currentLLMStock, clearLLMOutputs, isAnalyzing } =
    useDashboardStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  // 자동 스크롤
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [llmOutputs]);

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <h2 className="panel-title">LLM 출력</h2>
          {isAnalyzing && (
            <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />
          )}
        </div>
        <div className="flex items-center gap-2">
          {currentLLMStock && (
            <span className="text-xs text-gray-400 px-2 py-1 bg-gray-700 rounded">
              {currentLLMStock.name} ({currentLLMStock.code})
            </span>
          )}
          <button
            onClick={clearLLMOutputs}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300"
            title="출력 지우기"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-auto p-3 space-y-2">
        {llmOutputs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            자동 매매 분석 시 LLM 출력이 여기에 표시됩니다
          </div>
        ) : (
          llmOutputs.map((output, idx) => (
            <LLMOutputItem key={idx} output={output} />
          ))
        )}
      </div>
    </div>
  );
}

function LLMOutputItem({ output }: { output: LLMOutput }) {
  const config = outputTypeConfig[output.output_type] || outputTypeConfig.response;
  const timestamp = new Date(output.timestamp).toLocaleTimeString('ko-KR');

  return (
    <div className={clsx('rounded-lg p-2 text-sm', config.bg)}>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              'px-1.5 py-0.5 rounded text-xs font-medium',
              config.color,
              'bg-gray-800/50'
            )}
          >
            {config.label}
          </span>
          <span className="text-gray-400 text-xs">{output.model_name}</span>
        </div>
        <span className="text-xs text-gray-500">{timestamp}</span>
      </div>

      <div className={clsx('whitespace-pre-wrap break-words', config.color)}>
        {output.output_type === 'thinking' ? (
          // thinking 출력은 축약해서 표시
          <details className="cursor-pointer">
            <summary className="text-yellow-400/70">
              {output.content.slice(0, 100)}
              {output.content.length > 100 && '...'}
            </summary>
            <div className="mt-2 text-yellow-400/50 text-xs">
              {output.content}
            </div>
          </details>
        ) : (
          output.content
        )}
      </div>
    </div>
  );
}
