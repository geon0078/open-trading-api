import axios from 'axios';
import type {
  SurgeCandidateList,
  EnsembleAnalysisResult,
  AnalysisHistoryItem,
  AccountBalance,
  OrdersList,
} from '../types';

const API_BASE = '/api/v1';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 급등 종목 API
export const surgeApi = {
  getCandidates: async (
    minScore = 50,
    limit = 20,
    refresh = false
  ): Promise<SurgeCandidateList> => {
    const { data } = await client.get('/surge/candidates', {
      params: { min_score: minScore, limit, refresh },
    });
    return data;
  },

  scan: async (minScore = 50, limit = 20) => {
    const { data } = await client.post('/surge/scan', null, {
      params: { min_score: minScore, limit },
    });
    return data;
  },

  getStatus: async () => {
    const { data } = await client.get('/surge/status');
    return data;
  },
};

// LLM API
export const llmApi = {
  analyze: async (
    stockCode: string,
    stockName?: string
  ): Promise<EnsembleAnalysisResult> => {
    const { data } = await client.post('/llm/analyze', {
      stock_code: stockCode,
      stock_name: stockName,
    });
    return data;
  },

  getHistory: async (limit = 20): Promise<AnalysisHistoryItem[]> => {
    const { data } = await client.get('/llm/history', {
      params: { limit },
    });
    return data;
  },

  getAnalysisDetail: async (id: string): Promise<EnsembleAnalysisResult> => {
    const { data } = await client.get(`/llm/history/${id}`);
    return data;
  },

  getStatus: async () => {
    const { data } = await client.get('/llm/status');
    return data;
  },

  getModels: async () => {
    const { data } = await client.get('/llm/models');
    return data;
  },

  setPreset: async (preset: string) => {
    const { data } = await client.post('/llm/preset', null, {
      params: { preset },
    });
    return data;
  },
};

// 계좌 API
export const accountApi = {
  getBalance: async (): Promise<AccountBalance> => {
    const { data } = await client.get('/account/balance');
    return data;
  },

  getHoldings: async () => {
    const { data } = await client.get('/account/holdings');
    return data;
  },

  getTodayOrders: async (): Promise<OrdersList> => {
    const { data } = await client.get('/account/orders/today');
    return data;
  },

  getStatus: async () => {
    const { data } = await client.get('/account/status');
    return data;
  },
};

export default client;
