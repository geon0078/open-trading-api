# -*- coding: utf-8 -*-
"""KIS 웹소켓 서비스 - 실시간 체결 통보 및 계좌 업데이트."""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Callable, Any, Dict
from dataclasses import dataclass, asdict

import websockets
import pandas as pd
from io import StringIO

# 기존 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # news_trading
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))  # open-trading-api
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "examples_user"))  # examples_user (for domestic_stock)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "examples_user" / "domestic_stock"))  # kis_auth

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExecutionNotice:
    """체결 통보 데이터."""
    cust_id: str = ""           # 고객ID
    acnt_no: str = ""           # 계좌번호
    order_no: str = ""          # 주문번호
    order_qty: int = 0          # 주문수량
    side: str = ""              # 매도/매수 구분 (01:매도, 02:매수)
    stock_code: str = ""        # 종목코드
    exec_qty: int = 0           # 체결수량
    exec_price: int = 0         # 체결단가
    exec_time: str = ""         # 체결시간
    is_execution: bool = False  # 체결여부 (True:체결, False:주문접수)
    stock_name: str = ""        # 종목명
    order_price: int = 0        # 주문가격
    timestamp: str = ""         # 타임스탬프


class KISWebSocketService:
    """KIS 웹소켓 서비스."""

    def __init__(self):
        self._authenticated = False
        self._ka = None
        self._trenv = None
        self._ws_connection = None
        self._running = False
        self._clients: Set[asyncio.Queue] = set()
        self._data_map: Dict[str, dict] = {}
        self._reconnect_task: Optional[asyncio.Task] = None

    async def ensure_auth(self) -> bool:
        """웹소켓 인증 확인."""
        if self._authenticated and self._ka is not None:
            return True

        try:
            loop = asyncio.get_event_loop()

            def _auth():
                import kis_auth as ka
                svr = settings.kis_env
                ka.auth(svr=svr)
                ka.auth_ws(svr=svr)  # 웹소켓 인증 추가
                return ka, ka.getTREnv()

            self._ka, self._trenv = await loop.run_in_executor(None, _auth)
            self._authenticated = True
            logger.info("KIS 웹소켓 인증 성공")
            return True
        except Exception as e:
            logger.error(f"KIS 웹소켓 인증 실패: {e}")
            self._authenticated = False
            return False

    def add_client(self, queue: asyncio.Queue):
        """클라이언트 큐 추가."""
        self._clients.add(queue)
        logger.info(f"WebSocket 클라이언트 추가 (총 {len(self._clients)}개)")

    def remove_client(self, queue: asyncio.Queue):
        """클라이언트 큐 제거."""
        self._clients.discard(queue)
        logger.info(f"WebSocket 클라이언트 제거 (총 {len(self._clients)}개)")

    async def broadcast(self, event_type: str, data: dict):
        """모든 클라이언트에 메시지 브로드캐스트."""
        message = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        dead_clients = []
        for client_queue in self._clients:
            try:
                client_queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("클라이언트 큐가 가득 참")
            except Exception as e:
                logger.error(f"브로드캐스트 오류: {e}")
                dead_clients.append(client_queue)

        # 죽은 클라이언트 제거
        for client in dead_clients:
            self._clients.discard(client)

    def _aes_cbc_base64_dec(self, key: str, iv: str, cipher_text: str) -> str:
        """AES CBC 복호화."""
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        import base64

        cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
        decrypted = cipher.decrypt(base64.b64decode(cipher_text))
        return unpad(decrypted, AES.block_size).decode('utf-8')

    def _parse_execution_notice(self, data: str, columns: list) -> Optional[ExecutionNotice]:
        """체결 통보 데이터 파싱."""
        try:
            df = pd.read_csv(
                StringIO(data),
                header=None,
                sep="^",
                names=columns,
                dtype=object
            )

            if df.empty:
                return None

            row = df.iloc[0]

            # 체결여부 확인 (CNTG_YN: 1=주문접수, 2=체결)
            cntg_yn = str(row.get("CNTG_YN", ""))
            is_execution = (cntg_yn == "2")

            # 매도/매수 구분
            side_code = str(row.get("SELN_BYOV_CLS", ""))
            side = "매수" if side_code == "02" else "매도"

            return ExecutionNotice(
                cust_id=str(row.get("CUST_ID", "")),
                acnt_no=str(row.get("ACNT_NO", "")),
                order_no=str(row.get("ODER_NO", "")),
                order_qty=int(row.get("ODER_QTY", 0) or 0),
                side=side,
                stock_code=str(row.get("STCK_SHRN_ISCD", "")),
                exec_qty=int(row.get("CNTG_QTY", 0) or 0),
                exec_price=int(row.get("CNTG_UNPR", 0) or 0),
                exec_time=str(row.get("STCK_CNTG_HOUR", "")),
                is_execution=is_execution,
                stock_name=str(row.get("CNTG_ISNM40", "")).strip(),
                order_price=int(row.get("ODER_PRC", 0) or 0),
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"체결 통보 파싱 오류: {e}")
            return None

    async def _handle_ws_message(self, raw: str):
        """웹소켓 메시지 처리."""
        try:
            if raw[0] in ["0", "1"]:
                # 실시간 데이터
                parts = raw.split("|")
                if len(parts) < 4:
                    return

                tr_id = parts[1]
                data = parts[3]

                # 암호화된 데이터 복호화
                dm = self._data_map.get(tr_id, {})
                if dm.get("encrypt") == "Y":
                    key = dm.get("key", "")
                    iv = dm.get("iv", "")
                    if key and iv:
                        data = self._aes_cbc_base64_dec(key, iv, data)

                # 체결 통보 처리 (H0STCNI0, H0STCNI9)
                if tr_id in ["H0STCNI0", "H0STCNI9"]:
                    columns = [
                        "CUST_ID", "ACNT_NO", "ODER_NO", "ODER_QTY", "SELN_BYOV_CLS", "RCTF_CLS",
                        "ODER_KIND", "ODER_COND", "STCK_SHRN_ISCD", "CNTG_QTY", "CNTG_UNPR",
                        "STCK_CNTG_HOUR", "RFUS_YN", "CNTG_YN", "ACPT_YN", "BRNC_NO", "ACNT_NO2",
                        "ACNT_NAME", "ORD_COND_PRC", "ORD_EXG_GB", "POPUP_YN", "FILLER", "CRDT_CLS",
                        "CRDT_LOAN_DATE", "CNTG_ISNM40", "ODER_PRC"
                    ]

                    notice = self._parse_execution_notice(data, columns)
                    if notice:
                        logger.info(f"체결 통보: {notice.stock_name} {notice.side} {notice.exec_qty}주 @ {notice.exec_price}원")
                        await self.broadcast("execution_notice", asdict(notice))

                        # 체결시 계좌 정보 갱신 트리거
                        if notice.is_execution:
                            await self.broadcast("refresh_account", {"reason": "execution"})
            else:
                # 시스템 메시지 (구독 응답, 핑퐁 등)
                try:
                    rsp = json.loads(raw)

                    # 구독 응답 처리
                    if "header" in rsp:
                        tr_id = rsp.get("header", {}).get("tr_id", "")

                        # 암호화 키 저장
                        body = rsp.get("body", {}).get("output", {})
                        encrypt = body.get("iv", None) is not None

                        if tr_id:
                            self._data_map[tr_id] = {
                                "encrypt": "Y" if encrypt else "N",
                                "key": body.get("key", ""),
                                "iv": body.get("iv", "")
                            }
                            logger.debug(f"TR {tr_id} 등록 완료")

                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.error(f"웹소켓 메시지 처리 오류: {e}")

    async def _ws_subscriber(self, ws):
        """웹소켓 구독 루프."""
        async for raw in ws:
            try:
                # 핑퐁 처리
                if '"tr_id":"PINGPONG"' in raw:
                    await ws.pong(raw)
                    continue

                await self._handle_ws_message(raw)
            except websockets.ConnectionClosed:
                logger.warning("웹소켓 연결 종료")
                break
            except Exception as e:
                logger.error(f"웹소켓 수신 오류: {e}")

    async def _subscribe_execution_notice(self, ws):
        """체결 통보 구독."""
        if not self._trenv:
            return

        # 체결 통보 구독 요청
        tr_id = "H0STCNI0" if settings.kis_env == "prod" else "H0STCNI9"

        # HTS ID로 체결 통보 구독
        hts_id = self._trenv.my_htsid

        msg = {
            "header": {
                "approval_key": self._ka._base_headers_ws.get("approval_key", ""),
                "custtype": "P",
                "tr_type": "1",  # 구독
                "content-type": "utf-8"
            },
            "body": {
                "input": {
                    "tr_id": tr_id,
                    "tr_key": hts_id
                }
            }
        }

        await ws.send(json.dumps(msg))
        logger.info(f"체결 통보 구독 요청: {tr_id}, HTS ID: {hts_id}")

    async def connect(self):
        """KIS 웹소켓 연결 시작."""
        if self._running:
            logger.warning("이미 웹소켓이 실행 중입니다")
            return

        if not await self.ensure_auth():
            logger.error("웹소켓 인증 실패")
            return

        self._running = True

        ws_url = f"{self._trenv.my_url_ws}/tryitout/H0STCNI0"
        logger.info(f"웹소켓 연결 시작: {ws_url}")

        retry_count = 0
        max_retries = 5

        while self._running and retry_count < max_retries:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws_connection = ws
                    retry_count = 0  # 연결 성공시 재시도 횟수 초기화

                    logger.info("KIS 웹소켓 연결 성공")

                    # 체결 통보 구독
                    await self._subscribe_execution_notice(ws)

                    # 연결 성공 브로드캐스트
                    await self.broadcast("ws_connected", {"status": "connected"})

                    # 메시지 수신 루프
                    await self._ws_subscriber(ws)

            except websockets.ConnectionClosed as e:
                logger.warning(f"웹소켓 연결 종료: {e}")
            except Exception as e:
                logger.error(f"웹소켓 연결 오류: {e}")

            if self._running:
                retry_count += 1
                wait_time = min(30, 5 * retry_count)
                logger.info(f"웹소켓 재연결 대기 ({wait_time}초)...")
                await asyncio.sleep(wait_time)

        self._running = False
        self._ws_connection = None
        logger.info("웹소켓 연결 종료")

    async def disconnect(self):
        """웹소켓 연결 종료."""
        self._running = False
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
        logger.info("웹소켓 연결 해제")

    def is_connected(self) -> bool:
        """연결 상태 확인."""
        return self._running and self._ws_connection is not None


# 싱글톤 인스턴스
ws_service = KISWebSocketService()
