import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useSSE } from './hooks/useSSE';
import { useWebSocket } from './hooks/useWebSocket';
import { Layout } from './components/layout';
import {
  DashboardPage,
  MarketPage,
  SurgePage,
  OrderPage,
  PortfolioPage,
  AutoTradePage,
  IntegratedTradePage,
  NewsPage,
  WatchlistPage,
  HistoryPage,
  RankingsPage,
  NotificationsPage,
  SettingsPage,
} from './pages';

function AppContent() {
  // SSE 연결 초기화
  useSSE();

  // WebSocket 연결 초기화 (계좌 업데이트, LLM 출력 등)
  useWebSocket();

  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<DashboardPage />} />
        <Route path="market" element={<MarketPage />} />
        <Route path="surge" element={<SurgePage />} />
        <Route path="order" element={<OrderPage />} />
        <Route path="portfolio" element={<PortfolioPage />} />
        <Route path="auto-trade" element={<AutoTradePage />} />
        <Route path="llm-trade" element={<IntegratedTradePage />} />
        <Route path="news" element={<NewsPage />} />
        <Route path="watchlist" element={<WatchlistPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="rankings" element={<RankingsPage />} />
        <Route path="notifications" element={<NotificationsPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
