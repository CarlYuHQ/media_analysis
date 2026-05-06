"""
07_temporal_analysis.py - 时间趋势分析
"""
import pandas as pd
import numpy as np
import json
import os
from scipy import stats as scipy_stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), low_memory=False)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    relevant = df[(df['is_relevant'] == True) & (df['created_at'].notna())].copy()

    result = {}

    # 1. 月度情感趋势
    monthly_sentiment = {}
    for (leader, lang), group in relevant.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        group = group.sort_values('created_at')
        monthly = group.set_index('created_at').resample('M')

        monthly_data = []
        for month, mgroup in monthly:
            if len(mgroup) == 0:
                continue
            counts = mgroup['sentiment'].value_counts()
            total = counts.sum()
            neg = counts.get('negative', 0)
            pos = counts.get('positive', 0)
            neu = counts.get('neutral', 0)

            monthly_data.append({
                'month': str(month.date())[:7],
                'positive': int(pos),
                'negative': int(neg),
                'neutral': int(neu),
                'total': int(total),
                'neg_ratio': round(neg / total * 100, 1) if total > 0 else 0,
                'pos_ratio': round(pos / total * 100, 1) if total > 0 else 0,
            })

        monthly_sentiment[key] = monthly_data

    result['monthly_sentiment'] = monthly_sentiment

    # 2. 发布量与负面情感比例的相关性
    correlations = {}
    for key, monthly_data in monthly_sentiment.items():
        if len(monthly_data) < 5:
            continue
        totals = [m['total'] for m in monthly_data]
        neg_ratios = [m['neg_ratio'] for m in monthly_data]
        if len(set(totals)) > 1 and len(set(neg_ratios)) > 1:
            r, p = scipy_stats.pearsonr(totals, neg_ratios)
            correlations[key] = {
                'r': round(float(r), 3),
                'p_value': float(p),
                'significant': bool(p < 0.05)
            }
    result['volume_sentiment_correlation'] = correlations

    # 3. 关键事件标注
    key_events = []
    for (leader, lang), group in relevant.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        group = group.sort_values('created_at')
        monthly = group.set_index('created_at').resample('M').size()
        if len(monthly) == 0:
            continue
        mean_vol = monthly.mean()
        peak_months = monthly[monthly > mean_vol * 2.5]
        for month, count in peak_months.items():
            spike_factor = round(count / mean_vol, 1)
            key_events.append({
                'date': str(month.date())[:7],
                'leader': leader,
                'language': lang,
                'segment': key,
                'post_count': int(count),
                'spike_factor': spike_factor
            })

    # 手动添加已知事件
    known_events = [
        {'date': '2022-04', 'leader': 'LePen', 'event': 'French Presidential Election Runoff', 'segment': 'LePen-EN'},
        {'date': '2024-06', 'leader': 'LePen', 'event': 'European Parliament Elections', 'segment': 'LePen-EN'},
        {'date': '2024-09', 'leader': 'Takaichi', 'event': 'LDP Leadership Election', 'segment': 'Takaichi-JP'},
        {'date': '2024-10', 'leader': 'Takaichi', 'event': 'LDP Leadership Race / PM Campaign', 'segment': 'Takaichi-ZH'},
        {'date': '2025-01', 'leader': 'Takaichi', 'event': 'Regular Diet Session / Policy Debate', 'segment': 'Takaichi-JP'},
        {'date': '2025-03', 'leader': 'LePen', 'event': 'Embezzlement Trial Verdict', 'segment': 'LePen-EN'},
    ]
    result['key_events'] = key_events
    result['known_events'] = known_events

    # 4. 累积情感轨迹 (按领导人汇总)
    cumulative = {}
    for leader in ['Takaichi', 'LePen']:
        group = relevant[relevant['leader'] == leader].sort_values('created_at')
        monthly = group.set_index('created_at').resample('M')
        cum_data = []
        cum_pos, cum_neg = 0, 0
        for month, mgroup in monthly:
            counts = mgroup['sentiment'].value_counts()
            cum_pos += counts.get('positive', 0)
            cum_neg += counts.get('negative', 0)
            cum_data.append({
                'month': str(month.date())[:7],
                'cumulative_positive': int(cum_pos),
                'cumulative_negative': int(cum_neg),
            })
        cumulative[leader] = cum_data
    result['cumulative_sentiment'] = cumulative

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'temporal_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved temporal analysis to {output_path}")

    # Print
    print("\n=== Key Events (volume spikes) ===")
    for evt in sorted(key_events, key=lambda x: x['date']):
        print(f"  {evt['date']} {evt['segment']}: {evt['post_count']} posts ({evt['spike_factor']}x avg)")

    print("\n=== Correlations (volume vs negative sentiment) ===")
    for key, corr in correlations.items():
        sig = "***" if corr['significant'] else "n.s."
        print(f"  {key}: r={corr['r']}, p={corr['p_value']:.4f} {sig}")

if __name__ == '__main__':
    main()
