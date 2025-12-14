# -*- coding: utf-8 -*-
"""
환경 설정 로더 모듈

.env 파일에서 API 키를 읽어 KIS API 인증에 사용합니다.
kis_devlp.yaml 파일을 자동 생성/업데이트합니다.

사용 예시:
    >>> from modules.config_loader import setup_kis_config
    >>> setup_kis_config()  # .env에서 설정 로드 및 yaml 생성
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_env_file(env_path: Optional[str] = None) -> dict:
    """
    .env 파일에서 환경 변수 로드

    Args:
        env_path: .env 파일 경로 (None이면 기본 경로 사용)

    Returns:
        dict: 환경 변수 딕셔너리
    """
    if env_path is None:
        # 기본 경로: 현재 모듈 기준 상위 폴더의 .env
        env_path = Path(__file__).parent.parent / ".env"
    else:
        env_path = Path(env_path)

    env_vars = {}

    if not env_path.exists():
        logger.warning(f".env 파일을 찾을 수 없습니다: {env_path}")
        return env_vars

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 주석이나 빈 줄 무시
            if not line or line.startswith("#"):
                continue

            # KEY=VALUE 형식 파싱
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    logger.info(f".env 파일 로드 완료: {len(env_vars)}개 변수")
    return env_vars


def setup_kis_config(
    env_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    server: str = "prod"
) -> str:
    """
    KIS API 설정 파일(kis_devlp.yaml) 생성/업데이트

    Args:
        env_path: .env 파일 경로
        config_dir: 설정 파일 저장 디렉토리 (기본: ~/KIS/config/)
        server: 서버 유형 ("prod": 실전, "vps": 모의)

    Returns:
        str: 생성된 설정 파일 경로
    """
    # .env에서 환경 변수 로드
    env_vars = load_env_file(env_path)

    # 필수 키 확인
    app_key = env_vars.get("APP_KEY", "")
    app_secret = env_vars.get("APP_SECRET", "")

    if not app_key or not app_secret:
        raise ValueError("APP_KEY와 APP_SECRET이 .env 파일에 설정되어야 합니다.")

    # 계좌 정보 (선택)
    cano = env_vars.get("CANO", "00000000")
    acnt_prdt_cd = env_vars.get("ACNT_PRDT_CD", "01")
    hts_id = env_vars.get("HTS_ID", "")

    # 설정 디렉토리 생성
    if config_dir is None:
        config_dir = Path.home() / "KIS" / "config"
    else:
        config_dir = Path(config_dir)

    config_dir.mkdir(parents=True, exist_ok=True)

    # YAML 설정 내용 생성
    yaml_content = f"""# KIS Open API 설정 파일
# 자동 생성됨 - .env 파일에서 로드

# 실전투자 API 키
my_app: "{app_key}"
my_sec: "{app_secret}"

# 모의투자 API 키 (실전과 동일하게 설정, 필요시 변경)
paper_app: "{app_key}"
paper_sec: "{app_secret}"

# 계좌 정보
my_acct_stock: "{cano}"
my_acct_future: "{cano}"
my_paper_stock: "{cano}"
my_paper_future: "{cano}"
my_prod: "{acnt_prdt_cd}"
my_htsid: "{hts_id}"
my_agent: "Mozilla/5.0"
my_token: ""

# 서버 URL
prod: "https://openapi.koreainvestment.com:9443"
vps: "https://openapivts.koreainvestment.com:29443"
ops: "ws://ops.koreainvestment.com:21000"
vops: "ws://ops.koreainvestment.com:31000"
"""

    # 파일 저장
    config_path = config_dir / "kis_devlp.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info(f"KIS 설정 파일 생성: {config_path}")
    return str(config_path)


def get_api_credentials(env_path: Optional[str] = None) -> dict:
    """
    API 인증 정보 반환

    Args:
        env_path: .env 파일 경로

    Returns:
        dict: API 인증 정보 (app_key, app_secret, cano, acnt_prdt_cd)
    """
    env_vars = load_env_file(env_path)

    return {
        "app_key": env_vars.get("APP_KEY", ""),
        "app_secret": env_vars.get("APP_SECRET", ""),
        "cano": env_vars.get("CANO", ""),
        "acnt_prdt_cd": env_vars.get("ACNT_PRDT_CD", "01"),
        "env_dv": env_vars.get("ENV_DV", "prod"),
    }


# =====================================================
# 사용 예제
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 설정 파일 생성
    try:
        config_path = setup_kis_config()
        print(f"설정 파일 생성 완료: {config_path}")

        # 인증 정보 확인
        creds = get_api_credentials()
        print(f"APP_KEY: {creds['app_key'][:10]}...")
        print(f"APP_SECRET: {creds['app_secret'][:10]}...")
    except ValueError as e:
        print(f"오류: {e}")
