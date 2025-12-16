# -*- coding: utf-8 -*-
"""WebSocket API 엔드포인트 - 실시간 데이터 스트리밍."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """WebSocket 연결 관리자."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """새 연결 수락."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket 연결: 현재 {len(self.active_connections)}개 연결")

    async def disconnect(self, websocket: WebSocket):
        """연결 해제."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket 연결 해제: 현재 {len(self.active_connections)}개 연결")

    async def send_personal(self, websocket: WebSocket, message: dict):
        """개인 메시지 전송."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
        except Exception as e:
            logger.error(f"개인 메시지 전송 실패: {e}")

    async def broadcast(self, event_type: str, data: dict):
        """모든 연결에 브로드캐스트."""
        message = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        disconnected = []
        async with self._lock:
            for connection in self.active_connections:
                try:
                    if connection.client_state == WebSocketState.CONNECTED:
                        await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"브로드캐스트 실패: {e}")
                    disconnected.append(connection)

            # 끊어진 연결 제거
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# 전역 연결 관리자
manager = ConnectionManager()


async def broadcast_event(event_type: str, data: dict):
    """외부에서 브로드캐스트를 호출할 수 있는 함수."""
    await manager.broadcast(event_type, data)


async def broadcast_llm_output(stock_code: str, stock_name: str, model_name: str,
                                output_type: str, content: str, **kwargs):
    """LLM 출력 브로드캐스트."""
    await manager.broadcast("llm_output", {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "model_name": model_name,
        "output_type": output_type,  # "thinking", "response", "signal", "error"
        "content": content,
        **kwargs
    })


async def broadcast_execution(order_data: dict):
    """주문 체결 브로드캐스트."""
    await manager.broadcast("execution", order_data)


async def broadcast_account_update(balance_data: dict):
    """계좌 잔고 업데이트 브로드캐스트."""
    await manager.broadcast("account_update", balance_data)


async def broadcast_trade_result(result_data: dict):
    """자동 매매 결과 브로드캐스트."""
    await manager.broadcast("trade_result", result_data)


@router.websocket("/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """대시보드 WebSocket 엔드포인트."""
    await manager.connect(websocket)

    try:
        # 연결 확인 메시지
        await manager.send_personal(websocket, {
            "event": "connected",
            "data": {
                "message": "대시보드 WebSocket 연결 성공",
                "connections": manager.connection_count
            },
            "timestamp": datetime.now().isoformat()
        })

        # 클라이언트 메시지 수신 루프
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # 클라이언트 명령 처리
                if message.get("type") == "ping":
                    await manager.send_personal(websocket, {
                        "event": "pong",
                        "data": {},
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.get("type") == "subscribe":
                    # 특정 이벤트 구독 (향후 확장용)
                    pass

            except json.JSONDecodeError:
                logger.warning("잘못된 JSON 메시지 수신")

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info("클라이언트 연결 종료")
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        await manager.disconnect(websocket)


@router.get("/connections")
async def get_connections():
    """현재 WebSocket 연결 수 조회."""
    return {
        "connections": manager.connection_count,
        "timestamp": datetime.now().isoformat()
    }
