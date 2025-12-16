# -*- coding: utf-8 -*-
"""
KIS API 연결 테스트 스크립트

.env 파일의 API 키로 실제 API 연결을 테스트합니다.
주문 API는 사용하지 않고, 조회 API만 테스트합니다.

테스트 항목:
1. 설정 파일 생성
2. API 인증 (토큰 발급)
3. 현재가 조회 API
4. 뉴스 조회 API

실행:
    python test_api_connection.py
"""

import sys
import os
import io
import logging
from datetime import datetime
from pathlib import Path

# Windows 콘솔 UTF-8 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 모듈 경로 설정
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "modules"))
sys.path.insert(0, str(current_dir.parent.parent / "examples_llm"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_config_setup():
    """1. 설정 파일 생성 테스트"""
    print("\n" + "=" * 60)
    print("1. Config file creation test")
    print("=" * 60)

    try:
        from modules.config_loader import setup_kis_config, get_api_credentials

        # 설정 파일 생성
        config_path = setup_kis_config()
        print(f"[OK] Config file created: {config_path}")

        # 인증 정보 확인
        creds = get_api_credentials()
        print(f"[OK] APP_KEY: {creds['app_key'][:15]}...")
        print(f"[OK] APP_SECRET: {creds['app_secret'][:15]}...")

        return True
    except Exception as e:
        print(f"[FAIL] Config setup failed: {e}")
        return False


def test_api_auth():
    """2. API 인증 테스트"""
    print("\n" + "=" * 60)
    print("2. API Authentication test (Token)")
    print("=" * 60)

    try:
        import kis_auth as ka

        # 실전투자 인증
        ka.auth(svr="prod", product="01")
        print("[OK] API authentication success")

        # 환경 정보 확인
        env = ka.getTREnv()
        print(f"[OK] Server URL: {env.my_url}")
        print(f"[OK] Token issued")

        return True
    except Exception as e:
        print(f"[FAIL] API authentication failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_price_api():
    """3. 현재가 조회 API 테스트"""
    print("\n" + "=" * 60)
    print("3. Stock Price API test")
    print("=" * 60)

    try:
        from modules.price_checker import PriceChecker

        checker = PriceChecker(env_dv="prod")

        # 삼성전자 현재가 조회
        print("\n[Samsung Electronics (005930) Price Query]")
        price = checker.get_price("005930")

        if price:
            print(f"[OK] Stock Name: {price.stock_name}")
            print(f"[OK] Current Price: {price.current_price:,} KRW")
            print(f"[OK] Change: {price.change:+,} KRW ({price.change_rate:+.2f}%)")
            print(f"[OK] Open: {price.open_price:,} KRW")
            print(f"[OK] High: {price.high_price:,} KRW")
            print(f"[OK] Low: {price.low_price:,} KRW")
            print(f"[OK] Volume: {price.volume:,}")
            print(f"[OK] 52W High: {price.week52_high:,} KRW")
            print(f"[OK] 52W Low: {price.week52_low:,} KRW")
            print(f"[OK] PER: {price.per:.2f}")
            print(f"[OK] PBR: {price.pbr:.2f}")

            # 여러 종목 조회
            print("\n[Multiple Stock Price Query]")
            watch_list = ["005930", "000660", "035720"]
            df = checker.get_price_df(watch_list)

            if not df.empty:
                print(df[['stock_code', 'stock_name', 'current_price', 'change_rate']].to_string())
                print("[OK] Multiple stock query success")

            return True
        else:
            print("[FAIL] Price query failed (no data)")
            return False

    except Exception as e:
        print(f"[FAIL] Price query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_news_api():
    """4. 뉴스 조회 API 테스트"""
    print("\n" + "=" * 60)
    print("4. News API test")
    print("=" * 60)

    try:
        from domestic_stock.news_title.news_title import news_title

        # 전체 뉴스 조회
        print("\n[Latest News Query]")
        df = news_title(
            fid_news_ofer_entp_code="",
            fid_cond_mrkt_cls_code="",
            fid_input_iscd="",  # 전체 뉴스
            fid_titl_cntt="",
            fid_input_date_1="",
            fid_input_hour_1="",
            fid_rank_sort_cls_code="",
            fid_input_srno="",
            max_depth=1
        )

        if df is not None and not df.empty:
            print(f"[OK] News query success: {len(df)} items")
            print("\nLatest 5 news:")
            for idx, row in df.head(5).iterrows():
                title = row.get('titl', '')[:50]
                date = row.get('data_dt', '')
                time = row.get('data_tm', '')
                print(f"  [{date} {time}] {title}...")
            return True
        else:
            print("[FAIL] News query failed (no data)")
            return False

    except ImportError as e:
        print(f"[FAIL] News API module load failed: {e}")
        print("  domestic_stock/news_title module required")
        return False
    except Exception as e:
        print(f"[FAIL] News query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("KIS API Connection Test")
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # 1. 설정 파일 생성
    results["config"] = test_config_setup()

    if not results["config"]:
        print("\n[WARNING] Config setup failed. Check .env file.")
        return

    # 2. API 인증
    results["auth"] = test_api_auth()

    if not results["auth"]:
        print("\n[WARNING] API auth failed. Check APP_KEY and APP_SECRET.")
        return

    # 3. 현재가 조회
    results["price"] = test_price_api()

    # 4. 뉴스 조회
    results["news"] = test_news_api()

    # 결과 요약
    print("\n" + "=" * 60)
    print("Test Result Summary")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test_name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n*** All API tests passed! ***")
    else:
        print("\n[WARNING] Some tests failed. Check logs.")


if __name__ == "__main__":
    main()
