import { useState } from 'react';
import { BarChart3, TrendingUp, TrendingDown, Volume2, RefreshCw, ArrowUpDown } from 'lucide-react';
import clsx from 'clsx';

type RankingType = 'volume' | 'change' | 'market_cap' | 'volume_power';

const rankingTypes = [
  { id: 'volume', label: '거래량 순위', icon: <Volume2 className="w-4 h-4" /> },
  { id: 'change', label: '등락률 순위', icon: <TrendingUp className="w-4 h-4" /> },
  { id: 'market_cap', label: '시가총액 순위', icon: <BarChart3 className="w-4 h-4" /> },
  { id: 'volume_power', label: '체결강도 순위', icon: <ArrowUpDown className="w-4 h-4" /> },
];

// 임시 데이터
const mockRankings = {
  volume: [
    { rank: 1, code: '005930', name: '삼성전자', price: 71500, changeRate: 2.14, volume: 15234567 },
    { rank: 2, code: '000660', name: 'SK하이닉스', price: 198000, changeRate: -1.25, volume: 8765432 },
    { rank: 3, code: '035720', name: '카카오', price: 45500, changeRate: 1.11, volume: 6543210 },
    { rank: 4, code: '035420', name: 'NAVER', price: 187500, changeRate: 1.35, volume: 5432109 },
    { rank: 5, code: '051910', name: 'LG화학', price: 365000, changeRate: -0.54, volume: 4321098 },
    { rank: 6, code: '006400', name: '삼성SDI', price: 385000, changeRate: 2.67, volume: 3210987 },
    { rank: 7, code: '005380', name: '현대차', price: 215000, changeRate: 1.89, volume: 2109876 },
    { rank: 8, code: '003670', name: '포스코퓨처엠', price: 298500, changeRate: -2.15, volume: 1098765 },
    { rank: 9, code: '207940', name: '삼성바이오로직스', price: 750000, changeRate: 0.27, volume: 987654 },
    { rank: 10, code: '373220', name: 'LG에너지솔루션', price: 380000, changeRate: -0.78, volume: 876543 },
  ],
  change: [
    { rank: 1, code: '123456', name: '급등주A', price: 15000, changeRate: 29.87, volume: 1234567 },
    { rank: 2, code: '234567', name: '급등주B', price: 8500, changeRate: 25.32, volume: 2345678 },
    { rank: 3, code: '345678', name: '급등주C', price: 12300, changeRate: 22.15, volume: 3456789 },
  ],
  market_cap: [
    { rank: 1, code: '005930', name: '삼성전자', price: 71500, changeRate: 2.14, marketCap: 427000000000000 },
    { rank: 2, code: '373220', name: 'LG에너지솔루션', price: 380000, changeRate: -0.78, marketCap: 89000000000000 },
    { rank: 3, code: '000660', name: 'SK하이닉스', price: 198000, changeRate: -1.25, marketCap: 144000000000000 },
  ],
  volume_power: [
    { rank: 1, code: '029460', name: '케이씨', price: 25000, changeRate: 4.82, volumePower: 419.9 },
    { rank: 2, code: '244920', name: '에이플러스에셋', price: 9300, changeRate: 5.44, volumePower: 235.5 },
    { rank: 3, code: '475230', name: '엔알비', price: 14280, changeRate: 3.55, volumePower: 212.1 },
  ],
};

export function RankingsPage() {
  const [selectedType, setSelectedType] = useState<RankingType>('volume');
  const [marketFilter, setMarketFilter] = useState<'all' | 'kospi' | 'kosdaq'>('all');

  const rankings = mockRankings[selectedType];

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/10 rounded-lg">
            <BarChart3 className="w-6 h-6 text-orange-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">순위 분석</h1>
            <p className="text-sm text-gray-400">실시간 시장 순위</p>
          </div>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors">
          <RefreshCw className="w-4 h-4" />
          새로고침
        </button>
      </div>

      {/* 순위 유형 선택 */}
      <div className="flex items-center gap-2">
        {rankingTypes.map((type) => (
          <button
            key={type.id}
            onClick={() => setSelectedType(type.id as RankingType)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              selectedType === type.id
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            )}
          >
            {type.icon}
            {type.label}
          </button>
        ))}
      </div>

      {/* 시장 필터 */}
      <div className="flex items-center gap-2">
        {[
          { value: 'all', label: '전체' },
          { value: 'kospi', label: '코스피' },
          { value: 'kosdaq', label: '코스닥' },
        ].map((f) => (
          <button
            key={f.value}
            onClick={() => setMarketFilter(f.value as any)}
            className={clsx(
              'px-3 py-1.5 rounded text-sm transition-colors',
              marketFilter === f.value
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white'
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* 순위 테이블 */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-800/50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">#</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">종목</th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">현재가</th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">등락률</th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">
                {selectedType === 'volume' && '거래량'}
                {selectedType === 'change' && '거래량'}
                {selectedType === 'market_cap' && '시가총액'}
                {selectedType === 'volume_power' && '체결강도'}
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase"></th>
            </tr>
          </thead>
          <tbody>
            {rankings.map((item: any) => {
              const isUp = item.changeRate >= 0;
              return (
                <tr key={item.code} className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                  <td className="px-4 py-4">
                    <span className={clsx(
                      'w-8 h-8 flex items-center justify-center rounded-lg text-sm font-medium',
                      item.rank <= 3 ? 'bg-purple-500/20 text-purple-400' : 'bg-gray-800 text-gray-400'
                    )}>
                      {item.rank}
                    </span>
                  </td>
                  <td className="px-4 py-4">
                    <div>
                      <p className="font-medium text-white">{item.name}</p>
                      <p className="text-sm text-gray-500">{item.code}</p>
                    </div>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <span className="font-medium text-white">{item.price.toLocaleString()}원</span>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <div className={clsx('flex items-center justify-end gap-1', isUp ? 'text-red-400' : 'text-blue-400')}>
                      {isUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      <span>{isUp ? '+' : ''}{item.changeRate.toFixed(2)}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-4 text-right text-gray-300">
                    {selectedType === 'volume' && `${(item.volume / 1000000).toFixed(1)}M`}
                    {selectedType === 'change' && `${(item.volume / 1000000).toFixed(1)}M`}
                    {selectedType === 'market_cap' && `${(item.marketCap / 1000000000000).toFixed(0)}조`}
                    {selectedType === 'volume_power' && item.volumePower?.toFixed(1)}
                  </td>
                  <td className="px-4 py-4">
                    <div className="flex items-center gap-2 justify-end">
                      <button className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded text-sm hover:bg-red-500/30 transition-colors">
                        매수
                      </button>
                      <button className="px-3 py-1.5 bg-gray-700 text-gray-300 rounded text-sm hover:bg-gray-600 transition-colors">
                        상세
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
