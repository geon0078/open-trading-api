import { useEffect, useRef, useCallback } from 'react';
import { useDashboardStore } from '../store';
import type { SurgeCandidate, EnsembleAnalysisResult } from '../types';

const SSE_BASE = '/api/v1/stream';

export function useSSE() {
  const eventSourceRef = useRef<EventSource | null>(null);
  const {
    setSurgeCandidates,
    setIsScanning,
    addLLMAnalysis,
    setCurrentAnalysis,
    setIsAnalyzing,
    setAnalysisProgress,
    updateConnectionStatus,
  } = useDashboardStore();

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource(`${SSE_BASE}/all`);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      console.log('SSE 연결됨');
      updateConnectionStatus('sse', true);
    };

    eventSource.onerror = () => {
      console.error('SSE 연결 오류');
      updateConnectionStatus('sse', false);
      // 5초 후 재연결
      setTimeout(connect, 5000);
    };

    // 급등 종목 업데이트
    eventSource.addEventListener('surge_update', (event) => {
      try {
        const data = JSON.parse(event.data);
        setSurgeCandidates(data.candidates as SurgeCandidate[]);
        setIsScanning(false);
      } catch (e) {
        console.error('surge_update 파싱 오류:', e);
      }
    });

    // LLM 분석 업데이트
    eventSource.addEventListener('llm_update', (event) => {
      try {
        const data = JSON.parse(event.data) as EnsembleAnalysisResult;
        addLLMAnalysis(data);
        setIsAnalyzing(false);
        setAnalysisProgress(null);
      } catch (e) {
        console.error('llm_update 파싱 오류:', e);
      }
    });

    // 상태 업데이트
    eventSource.addEventListener('status', (event) => {
      try {
        const data = JSON.parse(event.data);
        setIsScanning(data.surge_scanning);
        setIsAnalyzing(data.llm_analyzing);
      } catch (e) {
        console.error('status 파싱 오류:', e);
      }
    });

    // 하트비트
    eventSource.addEventListener('heartbeat', () => {
      // 연결 유지 확인
    });

    return eventSource;
  }, [
    setSurgeCandidates,
    setIsScanning,
    addLLMAnalysis,
    setIsAnalyzing,
    setAnalysisProgress,
    updateConnectionStatus,
  ]);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      updateConnectionStatus('sse', false);
    }
  }, [updateConnectionStatus]);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { connect, disconnect };
}

// LLM 분석 스트리밍용 훅
export function useLLMStream() {
  const eventSourceRef = useRef<EventSource | null>(null);
  const {
    setCurrentAnalysis,
    setIsAnalyzing,
    setAnalysisProgress,
    addLLMAnalysis,
  } = useDashboardStore();

  const startAnalysis = useCallback(
    (stockCode: string, stockName: string = '') => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      setIsAnalyzing(true);
      setAnalysisProgress({ model: 'initializing', status: 'starting' });

      const url = `${SSE_BASE}/llm/analyze?stock_code=${stockCode}&stock_name=${encodeURIComponent(stockName)}`;
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.addEventListener('analysis_started', (event) => {
        const data = JSON.parse(event.data);
        console.log('분석 시작:', data);
      });

      eventSource.addEventListener('prompt_generated', (event) => {
        const data = JSON.parse(event.data);
        console.log('프롬프트 생성됨:', data.prompt?.substring(0, 100));
      });

      eventSource.addEventListener('model_started', (event) => {
        const data = JSON.parse(event.data);
        setAnalysisProgress({ model: data.model, status: 'processing' });
      });

      eventSource.addEventListener('model_completed', (event) => {
        const data = JSON.parse(event.data);
        setAnalysisProgress({ model: data.model, status: 'completed' });
        console.log('모델 완료:', data.model, data.result?.signal);
      });

      eventSource.addEventListener('model_error', (event) => {
        const data = JSON.parse(event.data);
        console.error('모델 오류:', data.model, data.error);
      });

      eventSource.addEventListener('analysis_completed', (event) => {
        const data = JSON.parse(event.data) as EnsembleAnalysisResult;
        addLLMAnalysis(data);
        setIsAnalyzing(false);
        setAnalysisProgress(null);
        eventSource.close();
      });

      eventSource.addEventListener('analysis_error', (event) => {
        const data = JSON.parse(event.data);
        console.error('분석 오류:', data.error);
        setIsAnalyzing(false);
        setAnalysisProgress(null);
        eventSource.close();
      });

      eventSource.addEventListener('error', (event) => {
        const messageEvent = event as MessageEvent;
        if (messageEvent.data) {
          const data = JSON.parse(messageEvent.data);
          console.error('오류:', data.error);
        }
        setIsAnalyzing(false);
        eventSource.close();
      });

      eventSource.onerror = () => {
        setIsAnalyzing(false);
        eventSource.close();
      };
    },
    [setIsAnalyzing, setAnalysisProgress, addLLMAnalysis]
  );

  const stopAnalysis = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsAnalyzing(false);
      setAnalysisProgress(null);
    }
  }, [setIsAnalyzing, setAnalysisProgress]);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return { startAnalysis, stopAnalysis };
}
