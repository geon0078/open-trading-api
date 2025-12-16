import { useState } from 'react';
import { Bell, Check, Trash2, CheckCheck, AlertCircle, TrendingUp, Zap, ShoppingCart } from 'lucide-react';
import clsx from 'clsx';

interface Notification {
  id: string;
  type: 'order' | 'price' | 'auto_trade' | 'system';
  title: string;
  message: string;
  time: string;
  read: boolean;
}

const mockNotifications: Notification[] = [
  {
    id: '1',
    type: 'order',
    title: '주문 체결 완료',
    message: '삼성전자 10주 매수 주문이 체결되었습니다. (71,500원)',
    time: '5분 전',
    read: false,
  },
  {
    id: '2',
    type: 'auto_trade',
    title: '자동매매 실행',
    message: 'LLM 분석 결과 케이씨(029460) 매수 시그널이 감지되었습니다.',
    time: '15분 전',
    read: false,
  },
  {
    id: '3',
    type: 'price',
    title: '가격 알림',
    message: 'SK하이닉스가 설정한 가격 200,000원에 도달했습니다.',
    time: '1시간 전',
    read: true,
  },
  {
    id: '4',
    type: 'system',
    title: '시스템 알림',
    message: '장 마감까지 30분 남았습니다.',
    time: '2시간 전',
    read: true,
  },
  {
    id: '5',
    type: 'order',
    title: '주문 체결 완료',
    message: 'NAVER 5주 매도 주문이 체결되었습니다. (187,500원)',
    time: '3시간 전',
    read: true,
  },
];

const typeConfig = {
  order: { icon: <ShoppingCart className="w-5 h-5" />, color: 'text-green-400', bgColor: 'bg-green-500/10' },
  price: { icon: <TrendingUp className="w-5 h-5" />, color: 'text-blue-400', bgColor: 'bg-blue-500/10' },
  auto_trade: { icon: <Zap className="w-5 h-5" />, color: 'text-yellow-400', bgColor: 'bg-yellow-500/10' },
  system: { icon: <AlertCircle className="w-5 h-5" />, color: 'text-purple-400', bgColor: 'bg-purple-500/10' },
};

export function NotificationsPage() {
  const [notifications, setNotifications] = useState(mockNotifications);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');

  const unreadCount = notifications.filter((n) => !n.read).length;
  const filteredNotifications = filter === 'all' ? notifications : notifications.filter((n) => !n.read);

  const markAsRead = (id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const markAllAsRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const deleteNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/10 rounded-lg relative">
            <Bell className="w-6 h-6 text-purple-400" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs text-white flex items-center justify-center">
                {unreadCount}
              </span>
            )}
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">알림</h1>
            <p className="text-sm text-gray-400">읽지 않은 알림 {unreadCount}개</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={markAllAsRead}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <CheckCheck className="w-4 h-4" />
            모두 읽음
          </button>
          <button
            onClick={clearAll}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            모두 삭제
          </button>
        </div>
      </div>

      {/* 필터 */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setFilter('all')}
          className={clsx(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            filter === 'all' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
          )}
        >
          전체
        </button>
        <button
          onClick={() => setFilter('unread')}
          className={clsx(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            filter === 'unread' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'
          )}
        >
          읽지 않음 ({unreadCount})
        </button>
      </div>

      {/* 알림 리스트 */}
      <div className="space-y-3">
        {filteredNotifications.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Bell className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>알림이 없습니다.</p>
          </div>
        ) : (
          filteredNotifications.map((notification) => {
            const config = typeConfig[notification.type];
            return (
              <div
                key={notification.id}
                className={clsx(
                  'flex items-start gap-4 p-4 bg-gray-900 rounded-xl border transition-colors',
                  notification.read ? 'border-gray-800' : 'border-purple-500/30 bg-purple-500/5'
                )}
              >
                <div className={clsx('p-2 rounded-lg', config.bgColor, config.color)}>
                  {config.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="font-medium text-white">{notification.title}</h3>
                    {!notification.read && (
                      <span className="w-2 h-2 bg-purple-500 rounded-full" />
                    )}
                  </div>
                  <p className="text-sm text-gray-400 mb-2">{notification.message}</p>
                  <span className="text-xs text-gray-500">{notification.time}</span>
                </div>
                <div className="flex items-center gap-2">
                  {!notification.read && (
                    <button
                      onClick={() => markAsRead(notification.id)}
                      className="p-2 text-gray-400 hover:text-green-400 transition-colors"
                      title="읽음으로 표시"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={() => deleteNotification(notification.id)}
                    className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                    title="삭제"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
