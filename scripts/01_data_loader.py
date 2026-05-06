"""
01_data_loader.py - 数据加载与清洗
读取5个CSV文件，合并为统一数据集，清洗并输出
"""
import pandas as pd
import numpy as np
import ast
import json
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 文件配置
FILES = [
    {'path': '高市早苗-日语.csv', 'leader': 'Takaichi', 'language': 'JP'},
    {'path': '高市早苗-英文.csv', 'leader': 'Takaichi', 'language': 'EN'},
    {'path': '高市早苗-中文.csv', 'leader': 'Takaichi', 'language': 'ZH'},
    {'path': '勒庞-中文.csv', 'leader': 'LePen', 'language': 'ZH'},
    {'path': '勒庞-英文.csv', 'leader': 'LePen', 'language': 'EN'},
]


def parse_statistics(stat_str):
    """安全解析statistics字典字符串"""
    if pd.isna(stat_str) or not isinstance(stat_str, str) or stat_str.strip() == '':
        return {}
    try:
        result = ast.literal_eval(stat_str)
        if isinstance(result, dict):
            return result
        return {}
    except (ValueError, SyntaxError):
        return {}


def load_and_clean():
    """加载所有CSV并合并"""
    dfs = []
    file_stats = {}

    for f in FILES:
        filepath = os.path.join(BASE_DIR, f['path'])
        print(f"Loading {f['path']}...")

        df = pd.read_csv(filepath, low_memory=False)
        original_rows = len(df)

        # 添加元数据列
        df['leader'] = f['leader']
        df['language'] = f['language']

        file_stats[f['path']] = {
            'original_rows': original_rows,
            'leader': f['leader'],
            'language': f['language']
        }

        dfs.append(df)
        print(f"  Loaded {original_rows} rows")

    # 合并所有数据
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal combined rows: {len(df)}")

    # === 清洗步骤 ===
    print("\n=== Cleaning data ===")

    # 1. 解析created_at
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    invalid_dates = df['created_at'].isna().sum()
    print(f"Invalid dates: {invalid_dates}")

    # 2. 解析statistics列
    print("Parsing statistics column...")
    stats_expanded = df['statistics'].apply(parse_statistics)
    stats_df = pd.json_normalize(stats_expanded)

    # 合并解析后的统计列
    for col in stats_df.columns:
        if col not in df.columns:
            df[col] = stats_df[col].values
        else:
            df[col] = df[col].fillna(stats_df[col].values)

    # 标准化互动数据为数值
    for col in ['like_count', 'retweet_count', 'reply_count', 'view_count', 'quote_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 如果没有独立的like_count，尝试从total_interactions_count使用
    if 'total_interactions_count' in df.columns:
        df['total_interactions_count'] = pd.to_numeric(df['total_interactions_count'], errors='coerce')

    # 3. 过滤空文本
    empty_text = df['text'].isna() | (df['text'].astype(str).str.strip() == '') | (df['text'].astype(str) == 'nan')
    print(f"Empty text rows: {empty_text.sum()}")
    df = df[~empty_text].copy()

    # 4. 标准化sentiment和emotion
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
    df['sentiment'] = df['sentiment'].replace({'nan': np.nan, '': np.nan})

    df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()
    df['emotion'] = df['emotion'].replace({'nan': np.nan, '': np.nan})

    # 5. 标记相关内容
    df['is_relevant'] = df['sentiment'].apply(
        lambda x: False if pd.isna(x) or x == 'irrelevant' else True
    )

    # 6. 计算文本长度
    df['text_length'] = df['text'].astype(str).str.len()

    # 7. 标准化account_type
    df['account_type'] = df['account_type'].astype(str).str.strip()
    df['account_type'] = df['account_type'].replace({'nan': 'Unknown', '': 'Unknown'})

    # 8. 标准化platform
    df['platform'] = df['platform'].astype(str).str.strip().str.lower()

    # 9. 标准化country
    df['country'] = df['country'].astype(str).str.strip()
    df['country'] = df['country'].replace({'nan': np.nan, '': np.nan})

    # 修复国家名不一致
    country_fixes = {
        '东京': 'Japan', '東京': 'Japan', '日本': 'Japan',
        '美国': 'United States', '英国': 'United Kingdom',
        '法国': 'France', '中国': 'China', '韩国': 'South Korea',
        '俄罗斯': 'Russia', '澳大利亚': 'Australia',
    }
    df['country'] = df['country'].replace(country_fixes)

    # 10. 年月列用于时间分析
    df['year_month'] = df['created_at'].dt.to_period('M').astype(str)

    cleaned_rows = len(df)
    relevant_rows = df['is_relevant'].sum()

    print(f"\nCleaned rows: {cleaned_rows}")
    print(f"Relevant rows: {relevant_rows}")
    print(f"Irrelevant rows: {cleaned_rows - relevant_rows}")

    # === 输出 ===
    # 保存清洗后数据
    output_csv = os.path.join(OUTPUT_DIR, 'cleaned_data.csv')
    # 选择关键列输出（减小文件大小）
    key_cols = [
        'leader', 'language', 'platform', 'text', 'created_at', 'year_month',
        'sentiment', 'emotion', 'is_relevant', 'text_length',
        'total_interactions_count', 'like_count', 'retweet_count',
        'reply_count', 'view_count', 'quote_count',
        'username', 'followers_count', 'friends_count',
        'account_type', 'country', 'country_code',
        'sentiment_target', 'sentiment_reason', 'emotion'
    ]
    existing_cols = [c for c in key_cols if c in df.columns]
    df[existing_cols].to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nSaved cleaned data to {output_csv}")

    # 保存数据概览JSON
    overview = {
        'total_rows': int(cleaned_rows),
        'relevant_rows': int(relevant_rows),
        'irrelevant_rows': int(cleaned_rows - relevant_rows),
        'leaders': ['Takaichi', 'LePen'],
        'languages': ['JP', 'EN', 'ZH'],
        'date_range': {
            'min': str(df['created_at'].min()),
            'max': str(df['created_at'].max())
        },
        'file_stats': file_stats,
        'sentiment_values': sorted(df['sentiment'].dropna().unique().tolist()),
        'emotion_values': sorted(df['emotion'].dropna().unique().tolist()),
        'platforms': sorted(df['platform'].unique().tolist()),
        'account_types': sorted(df['account_type'].unique().tolist()),
        'rows_by_segment': {}
    }

    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        overview['rows_by_segment'][key] = {
            'total': int(len(group)),
            'relevant': int(group['is_relevant'].sum()),
            'irrelevant': int((~group['is_relevant']).sum())
        }

    overview_path = os.path.join(OUTPUT_DIR, 'data_overview.json')
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    print(f"Saved overview to {overview_path}")

    return df


if __name__ == '__main__':
    df = load_and_clean()
    print("\n=== Data Overview ===")
    print(f"Total: {len(df)} rows")
    print(f"Relevant: {df['is_relevant'].sum()}")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    print(f"\nBy segment:")
    for (leader, lang), group in df.groupby(['leader', 'language']):
        print(f"  {leader}-{lang}: {len(group)} rows, {group['is_relevant'].sum()} relevant")
