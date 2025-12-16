import { create } from 'zustand';
import type {
  SurgeCandidate,
  EnsembleAnalysisResult,
  AccountBalance,
  Order,
  AutoTradeStatus,
  AutoTradeResult,
  AutoTradeHistoryItem,
  NewsAnalysisResult,
  TradingMode,
} from '../types';

// LLM 실시간 출력 타입
export interface LLMOutput {
  stock_code: string;
  stock_name: string;
  model_name: string;
  output_type: 'thinking' | 'response' | 'signal' | 'error' | 'start' | 'complete';
  content: string;
  timestamp: string;
}

// 포지션 알림 타입
export interface PositionAlert {
  stock_code: string;
  stock_name: string;
  alert_type: 'stop_loss' | 'take_profit' | 'warning';
  avg_price: number;
  current_price: number;
  quantity: number;
  pnl: number;
  pnl_rate: number;
  threshold: number;
  action_taken: string;
  order_no?: string;
  timestamp: string;
}

interface DashboardState {
  // 연결 상태
  wsConnected: boolean;
  sseConnected: boolean;

  // 급등 종목
  surgeCandidates: SurgeCandidate[];
  lastSurgeUpdate: Date | null;
  isScanning: boolean;

  // LLM 분석
  llmAnalyses: EnsembleAnalysisResult[];
  currentAnalysis: EnsembleAnalysisResult | null;
  isAnalyzing: boolean;
  analysisProgress: {
    model: string;
    status: string;
  } | null;

  // LLM 실시간 출력
  llmOutputs: LLMOutput[];
  currentLLMStock: { code: string; name: string } | null;

  // 포지션 알림
  positionAlerts: PositionAlert[];

  // 계좌
  accountBalance: AccountBalance | null;
  todayOrders: Order[];
  lastAccountUpdate: Date | null;

  // 선택된 종목
  selectedStock: SurgeCandidate | null;

  // 자동 트레이딩
  autoTradeStatus: AutoTradeStatus | null;
  autoTradeHistory: AutoTradeHistoryItem[];
  autoTradeResults: AutoTradeResult[];
  isAutoTrading: boolean;
  lastAutoTradeUpdate: Date | null;

  // 뉴스 분석
  tradingMode: TradingMode | null;
  newsAnalysis: NewsAnalysisResult | null;
  isAnalyzingNews: boolean;
  lastNewsUpdate: Date | null;

  // Actions
  setSurgeCandidates: (candidates: SurgeCandidate[]) => void;
  setIsScanning: (scanning: boolean) => void;
  addLLMAnalysis: (analysis: EnsembleAnalysisResult) => void;
  setCurrentAnalysis: (analysis: EnsembleAnalysisResult | null) => void;
  setIsAnalyzing: (analyzing: boolean) => void;
  setAnalysisProgress: (progress: { model: string; status: string } | null) => void;
  setAccountBalance: (balance: AccountBalance) => void;
  setTodayOrders: (orders: Order[]) => void;
  setSelectedStock: (stock: SurgeCandidate | null) => void;
  updateConnectionStatus: (type: 'ws' | 'sse', connected: boolean) => void;

  // LLM Output Actions
  addLLMOutput: (output: LLMOutput) => void;
  clearLLMOutputs: () => void;

  // Position Alert Actions
  addPositionAlert: (alert: PositionAlert) => void;
  clearPositionAlerts: () => void;

  // Auto Trade Actions
  setAutoTradeStatus: (status: AutoTradeStatus) => void;
  setAutoTradeHistory: (history: AutoTradeHistoryItem[]) => void;
  addAutoTradeResult: (result: AutoTradeResult) => void;
  setAutoTradeResults: (results: AutoTradeResult[]) => void;
  setIsAutoTrading: (trading: boolean) => void;

  // News Analysis Actions
  setTradingMode: (mode: TradingMode) => void;
  setNewsAnalysis: (analysis: NewsAnalysisResult) => void;
  setIsAnalyzingNews: (analyzing: boolean) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  // 초기 상태
  wsConnected: false,
  sseConnected: false,
  surgeCandidates: [],
  lastSurgeUpdate: null,
  isScanning: false,
  llmAnalyses: [],
  currentAnalysis: null,
  isAnalyzing: false,
  analysisProgress: null,
  llmOutputs: [],
  currentLLMStock: null,
  positionAlerts: [],
  accountBalance: null,
  todayOrders: [],
  lastAccountUpdate: null,
  selectedStock: null,
  autoTradeStatus: null,
  autoTradeHistory: [],
  autoTradeResults: [],
  isAutoTrading: false,
  lastAutoTradeUpdate: null,
  tradingMode: null,
  newsAnalysis: null,
  isAnalyzingNews: false,
  lastNewsUpdate: null,

  // Actions
  setSurgeCandidates: (candidates) =>
    set({
      surgeCandidates: candidates,
      lastSurgeUpdate: new Date(),
    }),

  setIsScanning: (scanning) => set({ isScanning: scanning }),

  addLLMAnalysis: (analysis) =>
    set((state) => ({
      llmAnalyses: [analysis, ...state.llmAnalyses].slice(0, 50),
      currentAnalysis: analysis,
    })),

  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),

  setIsAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),

  setAnalysisProgress: (progress) => set({ analysisProgress: progress }),

  setAccountBalance: (balance) =>
    set({
      accountBalance: balance,
      lastAccountUpdate: new Date(),
    }),

  setTodayOrders: (orders) => set({ todayOrders: orders }),

  setSelectedStock: (stock) => set({ selectedStock: stock }),

  updateConnectionStatus: (type, connected) =>
    set({
      [type === 'ws' ? 'wsConnected' : 'sseConnected']: connected,
    }),

  // LLM Output Actions
  addLLMOutput: (output) =>
    set((state) => {
      // 새로운 종목이면 이전 출력 초기화
      const isNewStock =
        state.currentLLMStock?.code !== output.stock_code;

      return {
        llmOutputs: isNewStock
          ? [output]
          : [...state.llmOutputs, output].slice(-100), // 최대 100개
        currentLLMStock: {
          code: output.stock_code,
          name: output.stock_name,
        },
      };
    }),

  clearLLMOutputs: () =>
    set({
      llmOutputs: [],
      currentLLMStock: null,
    }),

  // Position Alert Actions
  addPositionAlert: (alert) =>
    set((state) => ({
      positionAlerts: [alert, ...state.positionAlerts].slice(0, 50),
    })),

  clearPositionAlerts: () => set({ positionAlerts: [] }),

  // Auto Trade Actions
  setAutoTradeStatus: (status) =>
    set({
      autoTradeStatus: status,
      isAutoTrading: status.is_running,
      lastAutoTradeUpdate: new Date(),
    }),

  setAutoTradeHistory: (history) => set({ autoTradeHistory: history }),

  addAutoTradeResult: (result) =>
    set((state) => ({
      autoTradeResults: [result, ...state.autoTradeResults].slice(0, 50),
      lastAutoTradeUpdate: new Date(),
    })),

  setAutoTradeResults: (results) =>
    set({
      autoTradeResults: results,
      lastAutoTradeUpdate: new Date(),
    }),

  setIsAutoTrading: (trading) => set({ isAutoTrading: trading }),

  // News Analysis Actions
  setTradingMode: (mode) =>
    set({
      tradingMode: mode,
      newsAnalysis: mode.last_news_analysis,
      lastNewsUpdate: new Date(),
    }),

  setNewsAnalysis: (analysis) =>
    set({
      newsAnalysis: analysis,
      lastNewsUpdate: new Date(),
    }),

  setIsAnalyzingNews: (analyzing) => set({ isAnalyzingNews: analyzing }),
}));
