import { useState } from 'react';
import { Newspaper, RefreshCw, TrendingUp, TrendingDown, Clock, ExternalLink, Brain } from 'lucide-react';
import { useDashboardStore } from '../store';
import clsx from 'clsx';

export function NewsPage() {
  const [selectedSentiment, setSelectedSentiment] = useState<'all' | 'positive' | 'negative' | 'neutral'>('all');

  // 임시 뉴스 데이터
  const mockNews = [
    {
      id: 1,
      title: '삼성전자, AI 반도체 투자 확대... 1조원 추가 투입',
      source: '한국경제',
      time: '10분 전',
      sentiment: 'positive',
      stocks: ['005930'],
      summary: '삼성전자가 AI 반도체 시장 선점을 위해 1조원 규모의 추가 투자를 결정했다.',
    },
    {
      id: 2,
      title: 'SK하이닉스, HBM3E 양산 본격화',
      source: '매일경제',
      time: '25분 전',
      sentiment: 'positive',
      stocks: ['000660'],
      summary: 'SK하이닉스가 차세대 고대역폭 메모리 HBM3E의 양산을 본격화한다.',
    },
    {
      id: 3,
      title: '미 연준, 금리 동결 시사... 시장 불확실성 확대',
      source: '연합뉴스',
      time: '1시간 전',
      sentiment: 'negative',
      stocks: [],
      summary: '미 연준이 금리 동결을 시사하며 글로벌 시장의 불확실성이 확대되고 있다.',
    },
    {
      id: 4,
      title: '카카오, 신규 AI 서비스 출시 예고',
      source: '조선비즈',
      time: '2시간 전',
      sentiment: 'positive',
      stocks: ['035720'],
      summary: '카카오가 내년 초 신규 AI 기반 서비스 출시를 예고했다.',
    },
    {
      id: 5,
      title: '국제 유가 하락세... 정유업계 수익성 우려',
      source: '서울경제',
      time: '3시간 전',
      sentiment: 'negative',
      stocks: ['096770', '010950'],
      summary: '국제 유가가 하락세를 보이며 정유업계의 수익성에 대한 우려가 커지고 있다.',
    },
  ];

  const sentimentConfig = {
    positive: { label: '긍정', color: 'text-green-400', bgColor: 'bg-green-500/20' },
    negative: { label: '부정', color: 'text-red-400', bgColor: 'bg-red-500/20' },
    neutral: { label: '중립', color: 'text-gray-400', bgColor: 'bg-gray-500/20' },
  };

  const filteredNews = selectedSentiment === 'all'
    ? mockNews
    : mockNews.filter((n) => n.sentiment === selectedSentiment);

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/10 rounded-lg">
            <Newspaper className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">뉴스 분석</h1>
            <p className="text-sm text-gray-400">실시간 금융 뉴스 및 LLM 분석</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
            <Brain className="w-4 h-4" />
            LLM 분석 실행
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors">
            <RefreshCw className="w-4 h-4" />
            새로고침
          </button>
        </div>
      </div>

      {/* 시장 심리 요약 */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-2">전체 시장 심리</p>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            <span className="text-xl font-bold text-green-400">긍정적</span>
          </div>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-2">긍정 뉴스</p>
          <p className="text-xl font-bold text-white">{mockNews.filter((n) => n.sentiment === 'positive').length}건</p>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-2">부정 뉴스</p>
          <p className="text-xl font-bold text-white">{mockNews.filter((n) => n.sentiment === 'negative').length}건</p>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <p className="text-sm text-gray-400 mb-2">분석된 뉴스</p>
          <p className="text-xl font-bold text-white">{mockNews.length}건</p>
        </div>
      </div>

      {/* 필터 */}
      <div className="flex items-center gap-2">
        {['all', 'positive', 'negative', 'neutral'].map((sentiment) => (
          <button
            key={sentiment}
            onClick={() => setSelectedSentiment(sentiment as any)}
            className={clsx(
              'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              selectedSentiment === sentiment
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            )}
          >
            {sentiment === 'all' ? '전체' : sentimentConfig[sentiment as keyof typeof sentimentConfig].label}
          </button>
        ))}
      </div>

      {/* 뉴스 리스트 */}
      <div className="space-y-4">
        {filteredNews.map((news) => {
          const config = sentimentConfig[news.sentiment as keyof typeof sentimentConfig];
          return (
            <div
              key={news.id}
              className="bg-gray-900 rounded-xl border border-gray-800 p-6 hover:border-gray-700 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={clsx('px-2 py-1 text-xs font-medium rounded', config.bgColor, config.color)}>
                      {config.label}
                    </span>
                    <span className="text-sm text-gray-500">{news.source}</span>
                    <span className="flex items-center gap-1 text-sm text-gray-500">
                      <Clock className="w-3 h-3" />
                      {news.time}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{news.title}</h3>
                  <p className="text-gray-400">{news.summary}</p>
                </div>
                <button className="p-2 text-gray-400 hover:text-white transition-colors">
                  <ExternalLink className="w-5 h-5" />
                </button>
              </div>
              {news.stocks.length > 0 && (
                <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-800">
                  <span className="text-sm text-gray-500">관련 종목:</span>
                  {news.stocks.map((code) => (
                    <span key={code} className="px-2 py-1 bg-gray-800 rounded text-sm text-gray-300">
                      {code}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
