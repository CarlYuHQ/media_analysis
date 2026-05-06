"""
04_emotion_analysis.py - 情绪分布分析
"""
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

EMOTION_ORDER = ['anger', 'disgust', 'fear', 'sadness', 'surprise', 'happiness', 'hope', 'neutral']


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), low_memory=False)
    relevant = df[df['is_relevant'] == True].copy()

    result = {}

    # 1. 情绪计数和比例
    emotion_counts = {}
    emotion_proportions = {}
    for (leader, lang), group in relevant.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        valid = group['emotion'].dropna()
        valid = valid[valid != 'irrelevant']
        counts = valid.value_counts()
        total = counts.sum()
        emotion_counts[key] = {k: int(v) for k, v in counts.items()}
        emotion_proportions[key] = {
            k: round(v / total * 100, 1) for k, v in counts.items()
        }
        emotion_proportions[key]['total'] = int(total)

    result['emotion_counts'] = emotion_counts
    result['emotion_proportions'] = emotion_proportions

    # 2. 雷达图数据 (标准化到0-1)
    all_emotions = set()
    for counts in emotion_counts.values():
        all_emotions.update(counts.keys())
    all_emotions = sorted(all_emotions)

    # 获取每个segment的情绪比例，标准化
    radar_data = {}
    max_vals = {}
    for emo in all_emotions:
        vals = []
        for key in emotion_proportions:
            vals.append(emotion_proportions[key].get(emo, 0))
        max_vals[emo] = max(vals) if vals else 1

    for key in emotion_proportions:
        radar_data[key] = []
        for emo in all_emotions:
            val = emotion_proportions[key].get(emo, 0)
            max_v = max_vals[emo] if max_vals[emo] > 0 else 1
            radar_data[key].append(round(val / max_v, 3))

    result['radar_data'] = {
        'emotions': all_emotions,
        'series': radar_data
    }

    # 3. 情感-情绪交叉分析
    emotion_sentiment_cross = {}
    for (leader, lang), group in relevant.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        valid = group.dropna(subset=['emotion', 'sentiment'])
        valid = valid[valid['emotion'] != 'irrelevant']
        cross = pd.crosstab(valid['emotion'], valid['sentiment'])
        emotion_sentiment_cross[key] = {}
        for emo in cross.index:
            emotion_sentiment_cross[key][emo] = cross.loc[emo].to_dict()

    result['emotion_sentiment_cross'] = emotion_sentiment_cross

    # 4. 按领导人的总情绪分布（汇总所有语言）
    leader_emotion = {}
    for leader in ['Takaichi', 'LePen']:
        group = relevant[(relevant['leader'] == leader)]
        valid = group['emotion'].dropna()
        valid = valid[valid != 'irrelevant']
        counts = valid.value_counts()
        total = counts.sum()
        leader_emotion[leader] = {
            'counts': {k: int(v) for k, v in counts.items()},
            'proportions': {k: round(v / total * 100, 1) for k, v in counts.items()},
            'total': int(total)
        }
    result['leader_emotion_summary'] = leader_emotion

    # 5. 关键发现
    findings = []
    # 找出负面情绪最高的segment
    neg_emotions = ['anger', 'disgust']
    for key, props in emotion_proportions.items():
        neg_total = sum(props.get(e, 0) for e in neg_emotions)
        pos_total = props.get('happiness', 0)
        if neg_total > 40:
            findings.append(f"{key} has high negative emotions: anger+disgust={neg_total}% vs happiness={pos_total}%")
    result['key_findings'] = findings

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'emotion_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved emotion analysis to {output_path}")

    # Print
    print("\n=== Emotion Distribution ===")
    for leader in ['Takaichi', 'LePen']:
        print(f"\n{leader}:")
        data = leader_emotion[leader]
        for emo, pct in sorted(data['proportions'].items(), key=lambda x: -x[1]):
            print(f"  {emo}: {pct}%")

    print("\n=== Key Findings ===")
    for f in findings:
        print(f"  - {f}")

if __name__ == '__main__':
    main()
