import { useState } from 'react';
import {
  Settings,
  User,
  Bell,
  Shield,
  Palette,
  Globe,
  Database,
  Key,
  Save,
  RefreshCw,
} from 'lucide-react';
import clsx from 'clsx';

interface SettingSection {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const sections: SettingSection[] = [
  { id: 'account', label: '계정 설정', icon: <User className="w-5 h-5" /> },
  { id: 'notification', label: '알림 설정', icon: <Bell className="w-5 h-5" /> },
  { id: 'trading', label: '매매 설정', icon: <Shield className="w-5 h-5" /> },
  { id: 'appearance', label: '외관 설정', icon: <Palette className="w-5 h-5" /> },
  { id: 'api', label: 'API 설정', icon: <Key className="w-5 h-5" /> },
];

function AccountSettings() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">계정 정보</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">계좌 번호</label>
            <input
              type="text"
              value="****-01"
              disabled
              className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-gray-400"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">투자 환경</label>
            <div className="flex gap-3">
              <button className="flex-1 py-3 bg-purple-600 text-white rounded-lg font-medium">
                실전투자
              </button>
              <button className="flex-1 py-3 bg-gray-800 text-gray-400 rounded-lg font-medium hover:bg-gray-700 transition-colors">
                모의투자
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function NotificationSettings() {
  const [notifications, setNotifications] = useState({
    orderComplete: true,
    priceAlert: true,
    newsAlert: false,
    autoTradeAlert: true,
    dailyReport: false,
  });

  const toggleNotification = (key: keyof typeof notifications) => {
    setNotifications((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const notificationItems = [
    { key: 'orderComplete', label: '주문 체결 알림', description: '주문이 체결되면 알림을 받습니다.' },
    { key: 'priceAlert', label: '가격 알림', description: '설정한 가격에 도달하면 알림을 받습니다.' },
    { key: 'newsAlert', label: '뉴스 알림', description: '관심 종목 관련 뉴스가 있으면 알림을 받습니다.' },
    { key: 'autoTradeAlert', label: '자동매매 알림', description: '자동매매 실행 시 알림을 받습니다.' },
    { key: 'dailyReport', label: '일일 리포트', description: '매일 거래 요약 리포트를 받습니다.' },
  ] as const;

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white mb-4">알림 설정</h3>
      <div className="space-y-4">
        {notificationItems.map((item) => (
          <div key={item.key} className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
            <div>
              <p className="font-medium text-white">{item.label}</p>
              <p className="text-sm text-gray-500">{item.description}</p>
            </div>
            <button
              onClick={() => toggleNotification(item.key)}
              className={clsx(
                'w-12 h-6 rounded-full transition-colors relative',
                notifications[item.key] ? 'bg-purple-600' : 'bg-gray-700'
              )}
            >
              <div
                className={clsx(
                  'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform',
                  notifications[item.key] ? 'translate-x-7' : 'translate-x-1'
                )}
              />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

function TradingSettings() {
  const [settings, setSettings] = useState({
    defaultOrderType: 'limit',
    confirmBeforeOrder: true,
    maxOrderAmount: 1000000,
    riskLevel: 'medium',
  });

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white mb-4">매매 설정</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">기본 주문 유형</label>
          <div className="flex gap-3">
            <button
              onClick={() => setSettings((prev) => ({ ...prev, defaultOrderType: 'limit' }))}
              className={clsx(
                'flex-1 py-3 rounded-lg font-medium transition-colors',
                settings.defaultOrderType === 'limit'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              )}
            >
              지정가
            </button>
            <button
              onClick={() => setSettings((prev) => ({ ...prev, defaultOrderType: 'market' }))}
              className={clsx(
                'flex-1 py-3 rounded-lg font-medium transition-colors',
                settings.defaultOrderType === 'market'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              )}
            >
              시장가
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
          <div>
            <p className="font-medium text-white">주문 전 확인</p>
            <p className="text-sm text-gray-500">주문 실행 전 확인 창을 표시합니다.</p>
          </div>
          <button
            onClick={() => setSettings((prev) => ({ ...prev, confirmBeforeOrder: !prev.confirmBeforeOrder }))}
            className={clsx(
              'w-12 h-6 rounded-full transition-colors relative',
              settings.confirmBeforeOrder ? 'bg-purple-600' : 'bg-gray-700'
            )}
          >
            <div
              className={clsx(
                'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform',
                settings.confirmBeforeOrder ? 'translate-x-7' : 'translate-x-1'
              )}
            />
          </button>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">최대 주문 금액</label>
          <input
            type="number"
            value={settings.maxOrderAmount}
            onChange={(e) => setSettings((prev) => ({ ...prev, maxOrderAmount: Number(e.target.value) }))}
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-purple-500 outline-none"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">리스크 수준</label>
          <div className="flex gap-3">
            {['low', 'medium', 'high'].map((level) => (
              <button
                key={level}
                onClick={() => setSettings((prev) => ({ ...prev, riskLevel: level }))}
                className={clsx(
                  'flex-1 py-3 rounded-lg font-medium capitalize transition-colors',
                  settings.riskLevel === level
                    ? level === 'low' ? 'bg-green-600 text-white' :
                      level === 'medium' ? 'bg-yellow-600 text-white' : 'bg-red-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                )}
              >
                {level === 'low' ? '낮음' : level === 'medium' ? '보통' : '높음'}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function AppearanceSettings() {
  const [theme, setTheme] = useState('dark');
  const [accentColor, setAccentColor] = useState('purple');

  const colors = [
    { id: 'purple', color: '#8b5cf6' },
    { id: 'blue', color: '#3b82f6' },
    { id: 'green', color: '#10b981' },
    { id: 'pink', color: '#ec4899' },
    { id: 'orange', color: '#f59e0b' },
  ];

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white mb-4">외관 설정</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">테마</label>
          <div className="flex gap-3">
            <button
              onClick={() => setTheme('dark')}
              className={clsx(
                'flex-1 py-3 rounded-lg font-medium transition-colors',
                theme === 'dark' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              )}
            >
              다크 모드
            </button>
            <button
              onClick={() => setTheme('light')}
              className={clsx(
                'flex-1 py-3 rounded-lg font-medium transition-colors',
                theme === 'light' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              )}
            >
              라이트 모드
            </button>
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">강조 색상</label>
          <div className="flex gap-3">
            {colors.map((c) => (
              <button
                key={c.id}
                onClick={() => setAccentColor(c.id)}
                className={clsx(
                  'w-10 h-10 rounded-full border-2 transition-all',
                  accentColor === c.id ? 'border-white scale-110' : 'border-transparent'
                )}
                style={{ backgroundColor: c.color }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function APISettings() {
  const [apiKey, setApiKey] = useState('********');
  const [apiSecret, setApiSecret] = useState('********');

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white mb-4">API 설정</h3>
      <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg mb-4">
        <p className="text-sm text-yellow-400">
          API 키는 kis_devlp.yaml 파일에서 관리됩니다. 보안을 위해 여기서는 변경할 수 없습니다.
        </p>
      </div>
      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">APP Key</label>
          <input
            type="password"
            value={apiKey}
            disabled
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-gray-400"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-400 mb-2">APP Secret</label>
          <input
            type="password"
            value={apiSecret}
            disabled
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-gray-400"
          />
        </div>
      </div>
    </div>
  );
}

export function SettingsPage() {
  const [activeSection, setActiveSection] = useState('account');

  const renderSection = () => {
    switch (activeSection) {
      case 'account':
        return <AccountSettings />;
      case 'notification':
        return <NotificationSettings />;
      case 'trading':
        return <TradingSettings />;
      case 'appearance':
        return <AppearanceSettings />;
      case 'api':
        return <APISettings />;
      default:
        return <AccountSettings />;
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6 h-full">
      {/* 사이드 네비게이션 */}
      <div className="col-span-3 bg-gray-900 rounded-xl border border-gray-800 p-4">
        <div className="flex items-center gap-2 mb-6 px-2">
          <Settings className="w-5 h-5 text-purple-400" />
          <h2 className="text-lg font-semibold text-white">설정</h2>
        </div>
        <nav className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={clsx(
                'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors',
                activeSection === section.id
                  ? 'bg-purple-600/20 text-purple-400'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              )}
            >
              {section.icon}
              <span className="font-medium">{section.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* 설정 내용 */}
      <div className="col-span-9 bg-gray-900 rounded-xl border border-gray-800 p-6">
        {renderSection()}

        {/* 저장 버튼 */}
        <div className="mt-8 pt-6 border-t border-gray-800 flex justify-end gap-3">
          <button className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors">
            <RefreshCw className="w-4 h-4" />
            초기화
          </button>
          <button className="flex items-center gap-2 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
            <Save className="w-4 h-4" />
            저장
          </button>
        </div>
      </div>
    </div>
  );
}
