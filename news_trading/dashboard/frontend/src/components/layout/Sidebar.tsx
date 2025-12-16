import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  ShoppingCart,
  Briefcase,
  Zap,
  Brain,
  Newspaper,
  LineChart,
  Star,
  Settings,
  ChevronLeft,
  ChevronRight,
  Activity,
  BarChart3,
  Wallet,
  History,
  Bell,
  Search,
} from 'lucide-react';
import clsx from 'clsx';

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  path: string;
  badge?: number;
  children?: NavItem[];
}

const navItems: NavItem[] = [
  {
    id: 'dashboard',
    label: '대시보드',
    icon: <LayoutDashboard className="w-5 h-5" />,
    path: '/',
  },
  {
    id: 'market',
    label: '시세 조회',
    icon: <LineChart className="w-5 h-5" />,
    path: '/market',
  },
  {
    id: 'surge',
    label: '급등 종목',
    icon: <TrendingUp className="w-5 h-5" />,
    path: '/surge',
  },
  {
    id: 'order',
    label: '주문',
    icon: <ShoppingCart className="w-5 h-5" />,
    path: '/order',
  },
  {
    id: 'portfolio',
    label: '포트폴리오',
    icon: <Briefcase className="w-5 h-5" />,
    path: '/portfolio',
  },
  {
    id: 'auto-trade',
    label: '자동 매매',
    icon: <Zap className="w-5 h-5" />,
    path: '/auto-trade',
  },
  {
    id: 'llm-trade',
    label: 'LLM 트레이딩',
    icon: <Brain className="w-5 h-5" />,
    path: '/llm-trade',
  },
  {
    id: 'news',
    label: '뉴스 분석',
    icon: <Newspaper className="w-5 h-5" />,
    path: '/news',
  },
  {
    id: 'watchlist',
    label: '관심 종목',
    icon: <Star className="w-5 h-5" />,
    path: '/watchlist',
  },
  {
    id: 'history',
    label: '거래 내역',
    icon: <History className="w-5 h-5" />,
    path: '/history',
  },
  {
    id: 'rankings',
    label: '순위 분석',
    icon: <BarChart3 className="w-5 h-5" />,
    path: '/rankings',
  },
];

const bottomNavItems: NavItem[] = [
  {
    id: 'notifications',
    label: '알림',
    icon: <Bell className="w-5 h-5" />,
    path: '/notifications',
    badge: 3,
  },
  {
    id: 'settings',
    label: '설정',
    icon: <Settings className="w-5 h-5" />,
    path: '/settings',
  },
];

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const location = useLocation();

  return (
    <aside
      className={clsx(
        'flex flex-col bg-gray-900 border-r border-gray-800 transition-all duration-300',
        isCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-gray-800">
        {!isCollapsed && (
          <div className="flex items-center gap-2">
            <Activity className="w-8 h-8 text-purple-500" />
            <span className="text-lg font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
              KIS Trading
            </span>
          </div>
        )}
        {isCollapsed && <Activity className="w-8 h-8 text-purple-500 mx-auto" />}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
        >
          {isCollapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <ChevronLeft className="w-5 h-5" />
          )}
        </button>
      </div>

      {/* Search */}
      {!isCollapsed && (
        <div className="px-3 py-3">
          <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg border border-gray-700 focus-within:border-purple-500 transition-colors">
            <Search className="w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="종목 검색..."
              className="flex-1 bg-transparent text-sm text-gray-300 placeholder-gray-500 outline-none"
            />
          </div>
        </div>
      )}

      {/* Main Navigation */}
      <nav className="flex-1 px-2 py-2 space-y-1 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.id}
            to={item.path}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                isActive
                  ? 'bg-purple-600/20 text-purple-400 border-l-2 border-purple-500'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white',
                isCollapsed && 'justify-center'
              )
            }
          >
            {item.icon}
            {!isCollapsed && (
              <span className="text-sm font-medium">{item.label}</span>
            )}
            {!isCollapsed && item.badge && (
              <span className="ml-auto px-2 py-0.5 text-xs bg-purple-600 rounded-full">
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Divider */}
      <div className="mx-3 border-t border-gray-800" />

      {/* Bottom Navigation */}
      <nav className="px-2 py-2 space-y-1">
        {bottomNavItems.map((item) => (
          <NavLink
            key={item.id}
            to={item.path}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                isActive
                  ? 'bg-purple-600/20 text-purple-400'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white',
                isCollapsed && 'justify-center'
              )
            }
          >
            <div className="relative">
              {item.icon}
              {item.badge && item.badge > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 text-xs bg-red-500 rounded-full flex items-center justify-center">
                  {item.badge}
                </span>
              )}
            </div>
            {!isCollapsed && (
              <span className="text-sm font-medium">{item.label}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User Info */}
      {!isCollapsed && (
        <div className="p-3 mx-2 mb-2 bg-gray-800/50 rounded-lg border border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Wallet className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-200 truncate">실전투자</p>
              <p className="text-xs text-gray-500">계좌: ****-01</p>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
