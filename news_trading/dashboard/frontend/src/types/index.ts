// 급등 종목 타입
export type SignalType = 'STRONG_BUY' | 'BUY' | 'WATCH' | 'NEUTRAL';

export interface SurgeCandidate {
  code: string;
  name: string;
  price: number;
  change: number;
  change_rate: number;
  volume: number;
  volume_power: number;
  buy_volume: number;
  sell_volume: number;
  bid_balance: number;
  ask_balance: number;
  balance_ratio: number;
  surge_score: number;
  rank: number;
  signal: SignalType;
  detected_at: string;
  reasons: string[];
  llm_recommendation?: string;
  llm_confidence?: number;
  llm_analysis?: string;
}

export interface SurgeCandidateList {
  candidates: SurgeCandidate[];
  timestamp: string;
  total_count: number;
  scan_duration_ms: number;
}

// LLM 분석 타입
export interface ModelResult {
  model_name: string;
  signal: string;
  confidence: number;
  trend_prediction: string;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  reasoning: string;
  news_impact: string;
  risk_factors: string[];
  processing_time: number;
  success: boolean;
  raw_output: string;
  error_message: string;
}

export interface EnsembleAnalysisResult {
  id: string;
  timestamp: string;
  stock_code: string;
  stock_name: string;
  ensemble_signal: string;
  ensemble_confidence: number;
  ensemble_trend: string;
  current_price: number;
  avg_entry_price: number;
  avg_stop_loss: number;
  avg_take_profit: number;
  signal_votes: Record<string, number>;
  trend_votes: Record<string, number>;
  model_results: ModelResult[];
  models_used: string[];
  models_agreed: number;
  total_models: number;
  consensus_score: number;
  input_prompt: string;
  input_data: Record<string, unknown>;
  news_list: string[];
  total_processing_time: number;
  success: boolean;
  error_message: string;
}

export interface AnalysisHistoryItem {
  id: string;
  timestamp: string;
  stock_code: string;
  stock_name: string;
  ensemble_signal: string;
  ensemble_confidence: number;
  consensus_score: number;
  models_used: string[];
  success: boolean;
}

// 계좌 타입
export interface Holding {
  code: string;
  name: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  eval_amount: number;
  pnl: number;
  pnl_rate: number;
}

export interface AccountBalance {
  deposit: number;
  total_eval: number;
  total_purchase: number;
  total_pnl: number;
  pnl_rate: number;
  holdings_count: number;
  holdings: Holding[];
  updated_at: string;
}

export interface Order {
  order_id: string;
  order_time: string;
  stock_code: string;
  stock_name: string;
  side: string;
  order_qty: number;
  exec_qty: number;
  order_price: number;
  exec_price: number;
  exec_amount: number;
  status: string;
}

export interface OrdersList {
  orders: Order[];
  total_count: number;
  buy_count: number;
  sell_count: number;
  buy_amount: number;
  sell_amount: number;
  updated_at: string;
}

// SSE 이벤트 타입
export interface SSEEvent {
  event: string;
  data: unknown;
}

// 자동 트레이딩 타입
export interface AutoTradeConfig {
  env_dv: string;
  max_order_amount: number;
  min_confidence: number;
  min_consensus: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  max_daily_trades: number;
  max_daily_loss: number;
  min_surge_score: number;
  max_stocks_per_scan: number;
  ensemble_models?: string[];
  main_model?: string;
}

export interface AutoTradeStatus {
  is_running: boolean;
  initialized: boolean;
  config: AutoTradeConfig | null;
  today_trades: number;
  today_pnl: number;
  today_date: string;
  can_trade: boolean;
  market_status: {
    can_trade: boolean;
    reason: string;
  };
  risk_status: {
    can_trade: boolean;
    reason: string;
  };
  ensemble_models: string[];
  main_model: string;
}

export interface AutoTradeResult {
  success: boolean;
  action: string;
  stock_code: string;
  stock_name: string;
  current_price: number;
  ensemble_signal: string;
  confidence: number;
  consensus: number;
  order_qty: number;
  order_price: number;
  order_no?: string;
  technical_score: number;
  trend: string;
  reason: string;
  timestamp: string;
}

export interface AutoTradeHistoryItem {
  timestamp: string;
  stock_code: string;
  stock_name: string;
  action: string;
  ensemble_signal: string;
  confidence: number;
  consensus: number;
  order_qty: number;
  order_price: number;
  order_amount: number;
  order_no?: string;
  success: boolean;
  reason: string;
  technical_score: number;
}

export interface AutoTradeHistory {
  items: AutoTradeHistoryItem[];
  total_count: number;
  success_count: number;
  total_buy_amount: number;
  total_sell_amount: number;
}

// 뉴스 분석 타입
export interface AttentionStock {
  code: string;
  name: string;
  reason: string;
}

export interface NewsAnalysisResult {
  news_count: number;
  market_sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  key_themes: string[];
  attention_stocks: AttentionStock[];
  market_outlook: string;
  news_list: string[];
  analysis_time: string;
  llm_raw_output?: string;
}

export interface TradingMode {
  mode: 'INIT' | 'NEWS' | 'TRADING' | 'IDLE';
  mode_description: string;
  market_status: string;
  next_scan_time: string;
  last_news_analysis: NewsAnalysisResult | null;
}
