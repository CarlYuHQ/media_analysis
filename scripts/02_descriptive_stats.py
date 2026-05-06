"""
02_descriptive_stats.py - 描述性统计
"""
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), low_memory=False)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    result = {}

    # 1. 平台分布
    platform_dist = {}
    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        platform_dist[key] = group['platform'].value_counts().to_dict()
    result['platform_distribution'] = platform_dist

    # 2. 账户类型分布
    account_dist = {}
    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        valid = group[group['account_type'] != 'Unknown']
        account_dist[key] = valid['account_type'].value_counts().to_dict()
    result['account_type_distribution'] = account_dist

    # 3. 国家分布 (Top 15 per segment)
    country_dist = {}
    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        valid = group['country'].dropna()
        country_dist[key] = valid.value_counts().head(15).to_dict()
    result['country_distribution'] = country_dist

    # 4. 时间分布 (月度)
    temporal_dist = {}
    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        valid = group.dropna(subset=['created_at'])
        monthly = valid.set_index('created_at').resample('M').size()
        temporal_dist[key] = {
            str(k.date()): int(v) for k, v in monthly.items()
        }
    result['temporal_distribution'] = temporal_dist

    # 5. 互动统计
    interaction_stats = {}
    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        stats = {}
        for col in ['total_interactions_count', 'like_count', 'retweet_count', 'view_count', 'reply_count']:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors='coerce').dropna()
                if len(vals) > 0:
                    stats[col] = {
                        'mean': float(vals.mean()),
                        'median': float(vals.median()),
                        'std': float(vals.std()),
                        'min': float(vals.min()),
                        'max': float(vals.max()),
                        'count': int(len(vals))
                    }
        interaction_stats[key] = stats
    result['interaction_stats'] = interaction_stats

    # 6. 文本长度统计
    text_length_stats = {}
    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        vals = group['text_length'].dropna()
        text_length_stats[key] = {
            'mean': float(vals.mean()),
            'median': float(vals.median()),
            'std': float(vals.std()),
            'min': float(vals.min()),
            'max': float(vals.max()),
            'histogram': {}
        }
        # 直方图数据 (分桶)
        bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, float('inf')]
        labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1001-2000', '2001-5000', '5000+']
        cuts = pd.cut(vals, bins=bins, labels=labels, right=True)
        text_length_stats[key]['histogram'] = cuts.value_counts().sort_index().to_dict()

    result['text_length_stats'] = text_length_stats

    # 7. 总体统计
    result['summary'] = {
        'total_rows': int(len(df)),
        'relevant_rows': int(df['is_relevant'].sum()),
        'leaders': ['Takaichi', 'LePen'],
        'languages': ['JP', 'EN', 'ZH'],
        'date_range': {
            'min': str(df['created_at'].min()),
            'max': str(df['created_at'].max())
        },
        'platforms': sorted(df['platform'].unique().tolist()),
    }

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'descriptive_stats.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved descriptive stats to {output_path}")
    print(f"Segments: {list(result['platform_distribution'].keys())}")

if __name__ == '__main__':
    main()
