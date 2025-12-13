# -*- coding: utf-8 -*-
"""급등 종목 탐지 서비스."""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 기존 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # news_trading

from config import settings
from models.surge import ScanSettings, SignalType, SurgeCandidate, SurgeCandidateList

logger = logging.getLogger(__name__)


class SurgeService:
    """급등 종목 탐지 서비스."""

    def __init__(self):
        self._detector = None
        self._last_scan: Optional[datetime] = None
        self._cached_candidates: List[SurgeCandidate] = []
        self._is_scanning = False
        self._scan_settings = ScanSettings()

    def _ensure_detector(self):
        """SurgeDetector 초기화."""
        if self._detector is None:
            try:
                from modules.surge_detector import SurgeDetector
                self._detector = SurgeDetector()
                logger.info("SurgeDetector 초기화 완료")
            except ImportError as e:
                logger.error(f"SurgeDetector 임포트 실패: {e}")
                raise

    async def scan_stocks(
        self,
        min_score: float = None,
        limit: int = None
    ) -> SurgeCandidateList:
        """비동기 급등 종목 스캔."""
        if self._is_scanning:
            # 이미 스캔 중이면 캐시된 결과 반환
            return SurgeCandidateList(
                candidates=self._cached_candidates,
                timestamp=self._last_scan or datetime.now(),
                total_count=len(self._cached_candidates),
                scan_duration_ms=0.0
            )

        self._is_scanning = True
        start_time = time.time()

        try:
            self._ensure_detector()

            min_score = min_score or self._scan_settings.min_score
            limit = limit or self._scan_settings.limit

            # CPU 바운드 작업을 스레드 풀에서 실행
            loop = asyncio.get_event_loop()
            candidates = await loop.run_in_executor(
                None,
                lambda: self._detector.scan_surge_stocks(min_score)
            )

            # 기존 dataclass를 Pydantic 모델로 변환
            self._cached_candidates = [
                self._convert_candidate(c, idx + 1)
                for idx, c in enumerate(candidates[:limit])
            ]
            self._last_scan = datetime.now()

            scan_duration = (time.time() - start_time) * 1000

            logger.info(f"급등 종목 스캔 완료: {len(self._cached_candidates)}개 ({scan_duration:.1f}ms)")

            return SurgeCandidateList(
                candidates=self._cached_candidates,
                timestamp=self._last_scan,
                total_count=len(self._cached_candidates),
                scan_duration_ms=scan_duration
            )

        except Exception as e:
            logger.error(f"급등 종목 스캔 실패: {e}")
            return SurgeCandidateList(
                candidates=[],
                timestamp=datetime.now(),
                total_count=0,
                scan_duration_ms=0.0
            )
        finally:
            self._is_scanning = False

    def _convert_candidate(self, original, rank: int) -> SurgeCandidate:
        """기존 dataclass를 Pydantic 모델로 변환."""
        # signal 변환
        signal_map = {
            "STRONG_BUY": SignalType.STRONG_BUY,
            "BUY": SignalType.BUY,
            "WATCH": SignalType.WATCH,
            "NEUTRAL": SignalType.NEUTRAL,
        }
        signal = signal_map.get(getattr(original, "signal", "NEUTRAL"), SignalType.NEUTRAL)

        return SurgeCandidate(
            code=getattr(original, "code", ""),
            name=getattr(original, "name", ""),
            price=int(getattr(original, "price", 0)),
            change=int(getattr(original, "change", 0)),
            change_rate=float(getattr(original, "change_rate", 0)),
            volume=int(getattr(original, "volume", 0)),
            volume_power=float(getattr(original, "volume_power", 0)),
            buy_volume=int(getattr(original, "buy_volume", 0)),
            sell_volume=int(getattr(original, "sell_volume", 0)),
            bid_balance=int(getattr(original, "bid_balance", 0)),
            ask_balance=int(getattr(original, "ask_balance", 0)),
            balance_ratio=float(getattr(original, "balance_ratio", 1.0)),
            surge_score=float(getattr(original, "surge_score", 0)),
            rank=rank,
            signal=signal,
            detected_at=datetime.now(),
            reasons=list(getattr(original, "reasons", []))
        )

    def get_cached_candidates(self) -> SurgeCandidateList:
        """캐시된 급등 종목 반환."""
        return SurgeCandidateList(
            candidates=self._cached_candidates,
            timestamp=self._last_scan or datetime.now(),
            total_count=len(self._cached_candidates),
            scan_duration_ms=0.0
        )

    def get_settings(self) -> ScanSettings:
        """스캔 설정 조회."""
        return self._scan_settings

    def update_settings(self, new_settings: ScanSettings) -> ScanSettings:
        """스캔 설정 업데이트."""
        self._scan_settings = new_settings
        logger.info(f"스캔 설정 업데이트: {new_settings}")
        return self._scan_settings

    @property
    def is_scanning(self) -> bool:
        """스캔 중 여부."""
        return self._is_scanning

    @property
    def last_scan_time(self) -> Optional[datetime]:
        """마지막 스캔 시간."""
        return self._last_scan


# 싱글톤 인스턴스
surge_service = SurgeService()
