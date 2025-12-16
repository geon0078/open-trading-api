import { useState } from 'react';
import { Star, Plus, Trash2, Bell, TrendingUp, TrendingDown, Edit2, Search } from 'lucide-react';
import clsx from 'clsx';

interface WatchlistItem {
  code: string;
  name: string;
  price: number;
  change: number;
  changeRate: number;
  alertPrice?: number;
}

export function WatchlistPage() {
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([
    { code: '005930', name: '삼성전자', price: 71500, change: 1500, changeRate: 2.14 },
    { code: '000660', name: 'SK하이닉스', price: 198000, change: -2500, changeRate: -1.25 },
    { code: '035720', name: '카카오', price: 45500, change: 500, changeRate: 1.11 },
    { code: '035420', name: 'NAVER', price: 187500, change: 2500, changeRate: 1.35 },
  ]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);

  const removeFromWatchlist = (code: string) => {
    setWatchlist((prev) => prev.filter((item) => item.code !== code));
  };

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-yellow-500/10 rounded-lg">
            <Star className="w-6 h-6 text-yellow-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">관심 종목</h1>
            <p className="text-sm text-gray-400">즐겨찾는 종목을 관리하세요</p>
          </div>
        </div>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          <Plus className="w-4 h-4" />
          종목 추가
        </button>
      </div>

      {/* 검색 */}
      <div className="flex items-center gap-3 px-4 py-3 bg-gray-900 rounded-lg border border-gray-800">
        <Search className="w-5 h-5 text-gray-500" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="관심 종목 검색..."
          className="flex-1 bg-transparent text-white placeholder-gray-500 outline-none"
        />
      </div>

      {/* 관심 종목 그리드 */}
      <div className="grid grid-cols-2 gap-4">
        {watchlist
          .filter((item) => item.name.includes(searchQuery) || item.code.includes(searchQuery))
          .map((item) => {
            const isUp = item.changeRate >= 0;
            return (
              <div
                key={item.code}
                className="bg-gray-900 rounded-xl border border-gray-800 p-6 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{item.name}</h3>
                    <p className="text-sm text-gray-500">{item.code}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button className="p-2 text-gray-400 hover:text-yellow-400 transition-colors">
                      <Bell className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => removeFromWatchlist(item.code)}
                      className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="flex items-end justify-between">
                  <div>
                    <p className="text-2xl font-bold text-white">{item.price.toLocaleString()}원</p>
                    <div className={clsx('flex items-center gap-1 mt-1', isUp ? 'text-red-400' : 'text-blue-400')}>
                      {isUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      <span>
                        {isUp ? '+' : ''}{item.change.toLocaleString()}원 ({item.changeRate.toFixed(2)}%)
                      </span>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded text-sm font-medium hover:bg-red-500/30 transition-colors">
                      매수
                    </button>
                    <button className="px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded text-sm font-medium hover:bg-blue-500/30 transition-colors">
                      매도
                    </button>
                  </div>
                </div>

                {item.alertPrice && (
                  <div className="mt-4 pt-4 border-t border-gray-800 flex items-center gap-2 text-sm text-gray-400">
                    <Bell className="w-4 h-4 text-yellow-400" />
                    <span>알림 설정: {item.alertPrice.toLocaleString()}원</span>
                  </div>
                )}
              </div>
            );
          })}
      </div>

      {watchlist.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <Star className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>관심 종목이 없습니다.</p>
          <p className="text-sm">종목을 추가하여 시작하세요.</p>
        </div>
      )}
    </div>
  );
}
