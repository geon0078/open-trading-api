import { useEffect, useRef, useCallback, useState } from 'react';
import { useDashboardStore, type LLMOutput, type PositionAlert } from '../store';

// WebSocket 엔드포인트
const WS_URL = `ws://${window.location.host}/api/v1/ws/dashboard`;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  const {
    setAccountBalance,
    addLLMOutput,
    addPositionAlert,
    updateConnectionStatus,
  } = useDashboardStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket 연결됨');
        setWsConnected(true);
        updateConnectionStatus('ws', true);
      };

      ws.onclose = () => {
        console.log('WebSocket 연결 종료');
        setWsConnected(false);
        updateConnectionStatus('ws', false);

        // 5초 후 재연결
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, 5000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket 오류:', error);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleMessage(message);
        } catch (e) {
          console.error('WebSocket 메시지 파싱 오류:', e);
        }
      };
    } catch (error) {
      console.error('WebSocket 연결 오류:', error);
    }
  }, [updateConnectionStatus]);

  const handleMessage = useCallback(
    (message: { event: string; data: any; timestamp: string }) => {
      const { event, data, timestamp } = message;

      switch (event) {
        case 'connected':
          console.log('WebSocket 연결 확인:', data);
          break;

        case 'llm_output':
          // LLM 출력 처리
          addLLMOutput({
            ...data,
            timestamp,
          } as LLMOutput);
          break;

        case 'execution_notice':
        case 'execution':
          // 체결 통보
          console.log('체결 통보:', data);
          // 계좌 갱신 트리거
          break;

        case 'account_update':
          // 계좌 정보 업데이트
          if (data.holdings) {
            setAccountBalance({
              deposit: data.deposit || 0,
              total_eval: data.total_eval || 0,
              total_purchase: data.total_purchase || 0,
              total_pnl: data.total_pnl || 0,
              pnl_rate: data.pnl_rate || 0,
              holdings_count: data.count || data.holdings.length,
              holdings: data.holdings,
              updated_at: timestamp,
            });
          }
          break;

        case 'position_alert':
          // 포지션 알림 (손절/익절)
          addPositionAlert(data as PositionAlert);
          break;

        case 'refresh_account':
          // 계좌 갱신 트리거
          console.log('계좌 갱신 요청:', data.reason);
          break;

        case 'trade_result':
          // 자동 매매 결과
          console.log('매매 결과:', data);
          break;

        case 'pong':
          // 핑퐁 응답
          break;

        default:
          console.log('알 수 없는 WebSocket 이벤트:', event, data);
      }
    },
    [addLLMOutput, addPositionAlert, setAccountBalance]
  );

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsConnected(false);
    updateConnectionStatus('ws', false);
  }, [updateConnectionStatus]);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);

  // 연결 시작
  useEffect(() => {
    connect();

    // 30초마다 핑 전송
    const pingInterval = setInterval(sendPing, 30000);

    return () => {
      clearInterval(pingInterval);
      disconnect();
    };
  }, [connect, disconnect, sendPing]);

  return {
    isConnected: wsConnected,
    connect,
    disconnect,
    sendPing,
  };
}
