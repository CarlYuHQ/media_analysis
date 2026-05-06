"""
08_comparative_analysis.py - 比较与框架分析
"""
import pandas as pd
import numpy as np
import json
import os
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 性别框架关键词模式
GENDER_FRAMES = {
    'iron_lady': {
        'label_cn': '铁娘子框架',
        'label_en': 'Iron Lady Frame',
        'description': '强人、坚定、果断、铁腕等领导力描述',
        'patterns_zh': ['铁娘子', '强硬', '铁腕', '坚定', '果断', '鹰派', '右翼', '保守', '民族主义', '强硬派', '不妥协', '强势'],
        'patterns_en': ['iron lady', 'strong', 'tough', 'firm', 'resolute', 'hawkish', 'right-wing', 'conservative', 'nationalist', 'decisive', 'uncompromising', 'powerful'],
        'patterns_jp': ['鉄の女', '強硬', 'タカ派', '保守', '民族主義', '断固', '強い', '毅然']
    },
    'emotional_woman': {
        'label_cn': '情感女性框架',
        'label_en': 'Emotional Woman Frame',
        'description': '情绪化、非理性、感性等性别刻板印象',
        'patterns_zh': ['情绪化', '感性', '哭泣', '脆弱', '敏感', '冲动', '不理性', '感情用事', '暴躁', '激动'],
        'patterns_en': ['emotional', 'hysterical', 'irrational', 'sensitive', 'fragile', 'crying', 'overreact', 'dramatic', 'unstable'],
        'patterns_jp': ['感情的', 'ヒステリ', '泣き', '繊細', '不安定', '衝動的']
    },
    'appearance': {
        'label_cn': '外貌/身体框架',
        'label_en': 'Appearance/Body Frame',
        'description': '关注外貌、穿着、年龄、身材等身体特征',
        'patterns_zh': ['漂亮', '美丽', '妆容', '穿着', '发型', '年龄', '身材', '衣服', '时尚', '外表', '年轻', '老'],
        'patterns_en': ['beautiful', 'pretty', 'makeup', 'outfit', 'dress', 'hair', 'age', 'look', 'appearance', 'stylish', 'young', 'old'],
        'patterns_jp': ['美人', '化粧', '服装', '髪型', '若い', '年齢', 'スタイル', 'ファッション']
    },
    'competence': {
        'label_cn': '能力质疑框架',
        'label_en': 'Competence Questioning Frame',
        'description': '质疑其能力、资格、经验等',
        'patterns_zh': ['能力', '资格', '经验', '不配', '不够', '质疑', '胜任', '水平', '能力不足', '不合格'],
        'patterns_en': ['incompetent', 'unqualified', 'inexperienced', 'incapable', 'not fit', 'questionable', 'lack of', 'unable'],
        'patterns_jp': ['能力', '資格', '経験不足', '不適格', '疑問', '無能']
    },
    'ideology': {
        'label_cn': '意识形态框架',
        'label_en': 'Ideological Frame',
        'description': '强调极右翼、民粹主义、排外等意识形态标签',
        'patterns_zh': ['极右', '民粹', '排外', '法西斯', '纳粹', '种族主义', '反移民', '仇外', '极端', '危险'],
        'patterns_en': ['far-right', 'extreme right', 'populist', 'xenophobic', 'fascist', 'nazi', 'racist', 'anti-immigrant', 'dangerous', 'extremist', 'radical'],
        'patterns_jp': ['極右', 'ポピュリスト', '排外', 'ファシスト', 'ネオナチ', '人種差別', '反移民', '危険']
    }
}

# Femonationalism指标
FEMONATIONALISM_PATTERNS = {
    'patterns_zh': ['女性权利', '保护女性', '穆斯林女性', '移民', '伊斯兰', '传统价值', '家庭价值', '母亲', '母亲角色', '女性尊严'],
    'patterns_en': ['women rights', 'protect women', 'muslim women', 'immigrant', 'islam', 'traditional values', 'family values', 'mother', 'maternal', 'women dignity'],
    'patterns_jp': ['女性の権利', '女性保護', 'ムスリム女性', '移民', 'イスラム', '伝統的価値', '家族の価値', '母親']
}


def count_frame_matches(text, patterns):
    """计算文本中匹配的模式数"""
    if not isinstance(text, str):
        return 0, []
    text_lower = text.lower()
    matches = []
    for p in patterns:
        if p.lower() in text_lower:
            matches.append(p)
    return len(matches), matches


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), low_memory=False)
    relevant = df[df['is_relevant'] == True].copy()

    # Load sentiment and tfidf data
    with open(os.path.join(OUTPUT_DIR, 'sentiment_analysis.json'), 'r', encoding='utf-8') as f:
        sentiment_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'tfidf_keywords.json'), 'r', encoding='utf-8') as f:
        tfidf_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'emotion_analysis.json'), 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)

    result = {}

    # 1. 领导人比较矩阵
    leader_comparison = {}
    for leader in ['Takaichi', 'LePen']:
        group = relevant[relevant['leader'] == leader]
        sent_props = sentiment_data['proportions']
        # Aggregate sentiment across languages
        total_pos, total_neg, total_neu, total_all = 0, 0, 0, 0
        for key, props in sent_props.items():
            if key.startswith(leader):
                t = props.get('total', 1)
                total_pos += props.get('positive', 0) * t / 100
                total_neg += props.get('negative', 0) * t / 100
                total_neu += props.get('neutral', 0) * t / 100
                total_all += t
        if total_all > 0:
            leader_comparison[leader] = {
                'sentiment': {
                    'positive': round(total_pos / total_all * 100, 1),
                    'negative': round(total_neg / total_all * 100, 1),
                    'neutral': round(total_neu / total_all * 100, 1)
                },
                'total_posts': int(total_all),
                'emotion': emotion_data['leader_emotion_summary'].get(leader, {}).get('proportions', {})
            }
    result['leader_comparison'] = leader_comparison

    # 2. 性别框架分析
    print("Analyzing gender frames...")
    gender_frames_result = {}
    for leader in ['Takaichi', 'LePen']:
        leader_frames = {}
        group = relevant[relevant['leader'] == leader]

        for frame_id, frame_def in GENDER_FRAMES.items():
            frame_counts = {}
            frame_examples = []

            for (lang), lang_group in group.groupby('language'):
                if lang == 'ZH':
                    patterns = frame_def['patterns_zh']
                elif lang == 'JP':
                    patterns = frame_def['patterns_jp'] + frame_def['patterns_zh']
                else:
                    patterns = frame_def['patterns_en']

                matched = 0
                examples = []
                for text in lang_group['text'].dropna():
                    count, matches = count_frame_matches(str(text), patterns)
                    if count > 0:
                        matched += 1
                        if len(examples) < 3:
                            examples.append({
                                'text': str(text)[:200],
                                'matched_patterns': matches[:3],
                                'language': lang
                            })

                frame_counts[lang] = {
                    'matched': matched,
                    'total': len(lang_group),
                    'ratio': round(matched / max(len(lang_group), 1) * 100, 1)
                }
                frame_examples.extend(examples)

            leader_frames[frame_id] = {
                'label': frame_def['label_cn'],
                'description': frame_def['description'],
                'counts_by_language': frame_counts,
                'total_matched': sum(c['matched'] for c in frame_counts.values()),
                'examples': frame_examples[:5]
            }

        gender_frames_result[leader] = leader_frames

    result['gender_frames'] = gender_frames_result

    # 3. Femonationalism分析
    print("Analyzing femonationalism indicators...")
    femono_result = {}
    for leader in ['Takaichi', 'LePen']:
        group = relevant[relevant['leader'] == leader]
        femono_counts = {}
        femono_examples = []

        for lang, lang_group in group.groupby('language'):
            if lang == 'ZH':
                patterns = FEMONATIONALISM_PATTERNS['patterns_zh']
            elif lang == 'JP':
                patterns = FEMONATIONALISM_PATTERNS['patterns_jp'] + FEMONATIONALISM_PATTERNS['patterns_zh']
            else:
                patterns = FEMONATIONALISM_PATTERNS['patterns_en']

            matched = 0
            for text in lang_group['text'].dropna():
                count, matches = count_frame_matches(str(text), patterns)
                if count > 0:
                    matched += 1
                    if len(femono_examples) < 3:
                        femono_examples.append({
                            'text': str(text)[:200],
                            'matched': matches[:3],
                            'language': lang
                        })

            femono_counts[lang] = {
                'matched': matched,
                'total': len(lang_group),
                'ratio': round(matched / max(len(lang_group), 1) * 100, 1)
            }

        femono_result[leader] = {
            'counts_by_language': femono_counts,
            'examples': femono_examples
        }

    result['femonationalism'] = femono_result

    # 4. 双重束缚得分 (Double Bind Score)
    # 计算：当负面情感帖子中同时包含"女性"和"领导力"相关词汇的比例
    print("Computing double bind scores...")
    gender_leadership_patterns = {
        'ZH': {
            'gender': ['女性', '女人', '她', '女', '母亲', '妻子', '女政治家', '女领导', '第一位女性'],
            'leadership': ['领导', '首相', '总统', '权力', '执政', '统治', '管理', '决策', '掌权']
        },
        'EN': {
            'gender': ['woman', 'female', 'she', 'her', 'mother', 'wife', 'lady', 'girls'],
            'leadership': ['leader', 'prime minister', 'president', 'power', 'rule', 'govern', 'control']
        },
        'JP': {
            'gender': ['女性', '女', '彼女', '母親', '妻', '女性政治家'],
            'leadership': ['リーダー', '首相', '大統領', '権力', '支配', '統治']
        }
    }

    double_bind = {}
    for leader in ['Takaichi', 'LePen']:
        group = relevant[(relevant['leader'] == leader) & (relevant['sentiment'] == 'negative')]
        bind_scores = {}

        for lang, lang_group in group.groupby('language'):
            if lang not in gender_leadership_patterns:
                continue
            gp = gender_leadership_patterns[lang]['gender']
            lp = gender_leadership_patterns[lang]['leadership']

            has_gender = 0
            has_leadership = 0
            has_both = 0
            total = 0

            for text in lang_group['text'].dropna():
                text_lower = str(text).lower()
                total += 1
                g_match = any(p in text_lower for p in gp)
                l_match = any(p in text_lower for p in lp)
                if g_match: has_gender += 1
                if l_match: has_leadership += 1
                if g_match and l_match: has_both += 1

            bind_scores[lang] = {
                'total_negative_posts': total,
                'gender_mentioned': has_gender,
                'leadership_mentioned': has_leadership,
                'both_mentioned': has_both,
                'double_bind_ratio': round(has_both / max(total, 1) * 100, 1)
            }

        double_bind[leader] = bind_scores

    result['double_bind'] = double_bind

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'comparative_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved comparative analysis to {output_path}")

    # Print findings
    print("\n=== Gender Frame Analysis ===")
    for leader, frames in gender_frames_result.items():
        print(f"\n{leader}:")
        for frame_id, frame_data in frames.items():
            total = frame_data['total_matched']
            langs = ', '.join([f"{k}:{v['ratio']}%" for k, v in frame_data['counts_by_language'].items()])
            print(f"  {frame_data['label']}: {total} posts ({langs})")

    print("\n=== Double Bind Scores ===")
    for leader, scores in double_bind.items():
        print(f"\n{leader}:")
        for lang, s in scores.items():
            print(f"  {lang}: {s['double_bind_ratio']}% double bind ({s['both_mentioned']}/{s['total_negative_posts']})")


if __name__ == '__main__':
    main()
