"""
03_sentiment_analysis.py - 情感分析与统计检验
"""
import pandas as pd
import numpy as np
import json
import os
from scipy.stats import chi2_contingency

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def cramers_v(contingency_table):
    """计算Cramer's V效应量"""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), low_memory=False)

    result = {}

    # 1. 原始情感计数
    raw_counts = {}
    for (leader, lang), group in df.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        raw_counts[key] = group['sentiment'].value_counts().to_dict()
    result['raw_counts'] = raw_counts

    # 2. 排除irrelevant后的情感比例
    relevant = df[df['is_relevant'] == True].copy()
    proportions = {}
    for (leader, lang), group in relevant.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        counts = group['sentiment'].value_counts()
        total = counts.sum()
        proportions[key] = {
            k: round(v / total * 100, 1) for k, v in counts.items()
        }
        proportions[key]['total'] = int(total)
    result['proportions'] = proportions

    # 3. 卡方检验
    chi_tests = []

    def do_chi_test(test_name, df_a, df_b, label_a, label_b):
        """安全执行卡方检验"""
        combined = pd.concat([
            df_a[['sentiment']].assign(group=label_a),
            df_b[['sentiment']].assign(group=label_b)
        ], ignore_index=True)
        ct = pd.crosstab(combined['group'], combined['sentiment'])
        if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
            return None
        chi2, p, dof, expected = chi2_contingency(ct)
        return {
            'test': test_name,
            'chi2': round(float(chi2), 2),
            'p_value': float(p),
            'df': int(dof),
            'cramers_v': round(cramers_v(ct), 3),
            'significant': bool(p < 0.05)
        }

    # 高市 vs 勒庞 (中文)
    taka_zh = relevant[(relevant['leader'] == 'Takaichi') & (relevant['language'] == 'ZH')]
    lepen_zh = relevant[(relevant['leader'] == 'LePen') & (relevant['language'] == 'ZH')]
    r = do_chi_test('Takaichi vs LePen (Chinese)', taka_zh, lepen_zh, 'Takaichi', 'LePen')
    if r: chi_tests.append(r)

    # 高市 vs 勒庞 (英文)
    taka_en = relevant[(relevant['leader'] == 'Takaichi') & (relevant['language'] == 'EN')]
    lepen_en = relevant[(relevant['leader'] == 'LePen') & (relevant['language'] == 'EN')]
    r = do_chi_test('Takaichi vs LePen (English)', taka_en, lepen_en, 'Takaichi', 'LePen')
    if r: chi_tests.append(r)

    # 高市内部语言差异
    taka_all = relevant[relevant['leader'] == 'Takaichi']
    if len(taka_all) > 0:
        ct = pd.crosstab(taka_all['language'], taka_all['sentiment'])
        chi2, p, dof, expected = chi2_contingency(ct)
        chi_tests.append({
            'test': 'Takaichi across languages (JP vs EN vs ZH)',
            'chi2': round(float(chi2), 2),
            'p_value': float(p),
            'df': int(dof),
            'cramers_v': round(cramers_v(ct), 3),
            'significant': bool(p < 0.05)
        })

    # 勒庞内部语言差异
    lepen_all = relevant[relevant['leader'] == 'LePen']
    if len(lepen_all) > 0:
        ct = pd.crosstab(lepen_all['language'], lepen_all['sentiment'])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(ct)
            chi_tests.append({
                'test': 'LePen across languages (EN vs ZH)',
                'chi2': round(float(chi2), 2),
                'p_value': float(p),
                'df': int(dof),
                'cramers_v': round(cramers_v(ct), 3),
                'significant': bool(p < 0.05)
            })

    result['chi_square_tests'] = chi_tests

    # 4. 按平台的情感分布
    sentiment_by_platform = {}
    for platform, group in relevant.groupby('platform'):
        counts = group['sentiment'].value_counts()
        total = counts.sum()
        sentiment_by_platform[platform] = {
            k: round(v / total * 100, 1) for k, v in counts.items()
        }
        sentiment_by_platform[platform]['total'] = int(total)
    result['sentiment_by_platform'] = sentiment_by_platform

    # 5. 按账户类型的情感分布
    sentiment_by_account = {}
    for atype, group in relevant[relevant['account_type'] != 'Unknown'].groupby('account_type'):
        counts = group['sentiment'].value_counts()
        total = counts.sum()
        sentiment_by_account[atype] = {
            k: round(v / total * 100, 1) for k, v in counts.items()
        }
        sentiment_by_account[atype]['total'] = int(total)
    result['sentiment_by_account_type'] = sentiment_by_account

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'sentiment_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved sentiment analysis to {output_path}")

    # Print key findings
    print("\n=== Key Findings ===")
    for key, props in proportions.items():
        neg = props.get('negative', 0)
        pos = props.get('positive', 0)
        print(f"  {key}: negative={neg}%, positive={pos}%")
    print("\nChi-square tests:")
    for t in chi_tests:
        sig = "***" if t['significant'] else "n.s."
        print(f"  {t['test']}: chi2={t['chi2']}, p={t['p_value']:.4f} {sig}, V={t['cramers_v']}")

if __name__ == '__main__':
    main()
