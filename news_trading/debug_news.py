# -*- coding: utf-8 -*-
"""뉴스 데이터 구조 확인"""
import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'examples_llm'))

from modules.config_loader import setup_kis_config, load_env_file
load_env_file()
setup_kis_config()

import kis_auth as ka
ka.auth(svr="prod")

from domestic_stock.news_title.news_title import news_title

news_df = news_title(
    fid_news_ofer_entp_code="",
    fid_cond_mrkt_cls_code="",
    fid_input_iscd="",
    fid_titl_cntt="",
    fid_input_date_1="",
    fid_input_hour_1="",
    fid_rank_sort_cls_code="",
    fid_input_srno="",
    max_depth=1
)

print("=== News DataFrame Columns ===")
print(news_df.columns.tolist())
print()

print("=== First 5 Rows ===")
print(news_df.head())
print()

print("=== Sample Data ===")
for idx, row in news_df.head(5).iterrows():
    print(f"Row {idx}:")
    for col in news_df.columns:
        val = row.get(col, '')
        print(f"  {col}: {val}")
    print()
