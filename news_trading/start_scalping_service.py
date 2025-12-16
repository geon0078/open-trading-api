# -*- coding: utf-8 -*-
"""
스캘핑 서비스 24/365 실행기

백그라운드에서 웹 대시보드를 실행하고 자동 재시작 기능을 제공합니다.

기능:
- 웹 대시보드 백그라운드 실행
- 자동 재시작 (크래시 복구)
- 로그 파일 기록
- 시스템 상태 모니터링

사용법:
    # 서비스 시작
    python start_scalping_service.py start

    # 서비스 중지
    python start_scalping_service.py stop

    # 상태 확인
    python start_scalping_service.py status

    # 직접 실행 (포그라운드)
    python start_scalping_service.py run
"""

import os
import sys
import io
import time
import signal
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path

# UTF-8 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 경로 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
PID_FILE = SCRIPT_DIR / "scalping_service.pid"
LOG_FILE = SCRIPT_DIR / "logs" / f"scalping_{datetime.now().strftime('%Y%m%d')}.log"

# 로그 디렉토리 생성
LOG_FILE.parent.mkdir(exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScalpingService:
    """스캘핑 서비스 관리자"""

    def __init__(self):
        self.process = None
        self.running = False

    def start_dashboard(self):
        """대시보드 시작"""
        dashboard_script = SCRIPT_DIR / "scalping_dashboard.py"

        if not dashboard_script.exists():
            logger.error(f"대시보드 스크립트를 찾을 수 없습니다: {dashboard_script}")
            return None

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        process = subprocess.Popen(
            [sys.executable, str(dashboard_script)],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )

        logger.info(f"대시보드 시작됨 (PID: {process.pid})")
        return process

    def run(self, auto_restart=True, restart_delay=10):
        """서비스 실행 (포그라운드)"""
        self.running = True

        # PID 파일 생성
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))

        # 시그널 핸들러
        def signal_handler(signum, frame):
            logger.info("종료 신호 수신")
            self.running = False
            if self.process:
                self.process.terminate()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("=" * 60)
        logger.info("스캘핑 서비스 시작")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"로그 파일: {LOG_FILE}")
        logger.info("=" * 60)

        restart_count = 0

        while self.running:
            try:
                self.process = self.start_dashboard()

                if not self.process:
                    logger.error("대시보드 시작 실패")
                    break

                # 프로세스 출력 모니터링
                while self.running and self.process.poll() is None:
                    try:
                        line = self.process.stdout.readline()
                        if line:
                            print(line.rstrip())
                            if 'Running on' in line:
                                logger.info("웹 서버 시작 완료")
                    except Exception as e:
                        pass

                exit_code = self.process.poll()

                if not self.running:
                    logger.info("서비스 정상 종료")
                    break

                if auto_restart and self.running:
                    restart_count += 1
                    logger.warning(f"대시보드 종료됨 (exit code: {exit_code})")
                    logger.info(f"재시작 중... (시도: {restart_count})")
                    time.sleep(restart_delay)
                else:
                    break

            except Exception as e:
                logger.error(f"서비스 오류: {e}")
                if auto_restart and self.running:
                    time.sleep(restart_delay)
                else:
                    break

        # PID 파일 삭제
        if PID_FILE.exists():
            PID_FILE.unlink()

        logger.info("서비스 종료")

    def start_background(self):
        """백그라운드 시작 (Windows)"""
        if sys.platform == 'win32':
            # Windows에서 백그라운드 실행
            script = str(SCRIPT_DIR / "start_scalping_service.py")
            log_file = str(LOG_FILE)

            cmd = f'start /B python "{script}" run > "{log_file}" 2>&1'
            os.system(cmd)

            logger.info("백그라운드 서비스 시작됨")
            logger.info(f"로그 파일: {log_file}")
            logger.info(f"대시보드 URL: http://localhost:5000")
        else:
            # Unix 시스템
            import daemon
            with daemon.DaemonContext():
                self.run()

    def stop(self):
        """서비스 중지"""
        if not PID_FILE.exists():
            logger.info("실행 중인 서비스가 없습니다")
            return

        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())

        try:
            if sys.platform == 'win32':
                os.system(f'taskkill /F /PID {pid} /T')
            else:
                os.kill(pid, signal.SIGTERM)

            logger.info(f"서비스 중지됨 (PID: {pid})")

        except ProcessLookupError:
            logger.info("프로세스가 이미 종료됨")

        if PID_FILE.exists():
            PID_FILE.unlink()

    def status(self):
        """서비스 상태 확인"""
        print("\n" + "=" * 50)
        print("스캘핑 서비스 상태")
        print("=" * 50)

        if not PID_FILE.exists():
            print("상태: 중지됨")
            print("=" * 50)
            return

        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())

        # 프로세스 확인
        try:
            if sys.platform == 'win32':
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}'],
                    capture_output=True,
                    text=True
                )
                running = str(pid) in result.stdout
            else:
                os.kill(pid, 0)
                running = True
        except (ProcessLookupError, PermissionError):
            running = False

        print(f"상태: {'실행중' if running else '중지됨'}")
        print(f"PID: {pid}")
        print(f"로그 파일: {LOG_FILE}")
        print(f"대시보드 URL: http://localhost:5000")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='스캘핑 서비스 관리')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'run', 'restart'],
                        help='실행할 명령')
    parser.add_argument('--port', type=int, default=5000, help='웹 서버 포트')
    parser.add_argument('--no-restart', action='store_true', help='자동 재시작 비활성화')

    args = parser.parse_args()

    service = ScalpingService()

    if args.command == 'start':
        service.start_background()

    elif args.command == 'stop':
        service.stop()

    elif args.command == 'status':
        service.status()

    elif args.command == 'run':
        service.run(auto_restart=not args.no_restart)

    elif args.command == 'restart':
        service.stop()
        time.sleep(2)
        service.start_background()


if __name__ == "__main__":
    main()
