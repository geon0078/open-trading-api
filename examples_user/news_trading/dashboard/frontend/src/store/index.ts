import { create } from 'zustand';
import type {
  SurgeCandidate,
  EnsembleAnalysisResult,
  AccountBalance,
  Order,
} from '../types';

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

  // 계좌
  accountBalance: AccountBalance | null;
  todayOrders: Order[];
  lastAccountUpdate: Date | null;

  // 선택된 종목
  selectedStock: SurgeCandidate | null;

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
  accountBalance: null,
  todayOrders: [],
  lastAccountUpdate: null,
  selectedStock: null,

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
}));
