"""
06_topic_modeling.py - 主题建模 (NMF)
"""
import pandas as pd
import numpy as np
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 复用05的停用词和清洗逻辑
CHINESE_STOPWORDS = set(
    '的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 '
    '这 他 她 它 们 那 里 后 来 中 大 为 以 还 个 之 们 可以 什么 但 被 从 这个 他们 我们 '
    '你们 现在 如何 就是 怎么 如果 因为 所以 但是 或者 虽然 而且 然而 不过 因此 于是 '
    '已 已经 而将 把 与 及 其 该 每 各 某 本 此 另 此外 另外 可能 能 应 该 让 被 给 向 对 '
    '关于 通过 根据 按照 由于 既然 只要 只有 无论 不管 即使 尽管 比 更 最 非常 十分 '
    '非常 特别 一些 这些 那些 这样 那样 上面 下面 前面 后面 里头 外面 之间 其中 '
    '并 并且 或 乃至 乃至 以至 从而 进而 反而 却 仍 仍然 尚 且 既然 固然 固 '
    '高市 早苗 勒庞 https http www com cn org net 她的 他的 他们 我们 你们 '
    'the a an is are was were be been being have has had do does did '
    'will would shall should may might can could must need to of in for '
    'on with at by from as into through during before after above below '
    'between out off over under again further then once here there when '
    'where why how all both each few more most other some such no nor '
    'not only own same so than too very just don should now '
    'takaichi lepen marine sanae this that these those she her he him '
    'they them their what which who whom and but if or because until while '
    'about against between through during before after'.split()
)

JAPANESE_STOPWORDS = set(
    'の に は を た が で て と し れ さ ある いる も する から な こと として '
    'い や れる など なっ ない その あっ よう によって により たい まで これ '
    'しかし また そして しかし ため なので でも ところ その後 もの ので '
    'です ます だ である する した される できる および なら について '
    'でも しかし ただし また なお さらに そして または それ この その あの '
    '高市 早苗 https http www com takaichi lepen she her he they'
    'the a an is are was were have has had do does did will would shall '
    'should may might can could must to of in for on with at by from'.split()
)


def preprocess_texts(texts, language):
    """预处理文本并返回空格分隔的文档列表"""
    import jieba

    processed = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        # 清洗
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\S+', '', text)

        if language == 'ZH':
            words = jieba.lcut(text)
            words = [w.strip() for w in words if len(w.strip()) > 1 and w.strip() not in CHINESE_STOPWORDS]
        elif language == 'JP':
            try:
                import fugashi
                tagger = fugashi.Tagger()
                words = []
                for word in tagger(text):
                    w = word.surface.strip()
                    if len(w) <= 1 or w in JAPANESE_STOPWORDS:
                        continue
                    pos1 = getattr(word.feature, 'pos1', '') if word.feature else ''
                    if pos1 in ('名詞', '動詞', '形容詞', '副詞'):
                        if not (len(w) <= 2 and all('\u3040' <= c <= '\u309f' for c in w)):
                            words.append(w)
            except ImportError:
                words = jieba.lcut(text)
                words = [w.strip() for w in words if len(w.strip()) > 1 and w.strip() not in JAPANESE_STOPWORDS]
        else:  # EN - only pure ASCII English words
            words = text.lower().split()
            words = [w for w in words if len(w) > 2 and w.isalpha() and w.isascii() and w not in CHINESE_STOPWORDS]

        if words:
            processed.append(' '.join(words))

    return processed


def run_topic_modeling(documents, n_topics=6, n_top_words=10):
    """运行NMF主题建模"""
    if len(documents) < n_topics * 5:
        n_topics = max(2, len(documents) // 5)

    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=2,
        max_features=5000
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()

    if tfidf_matrix.shape[1] < n_topics:
        n_topics = max(2, tfidf_matrix.shape[1])

    nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
    nmf.fit(tfidf_matrix)

    # 文档-主题分布
    doc_topic = nmf.transform(tfidf_matrix)
    topic_assignments = doc_topic.argmax(axis=1)

    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        keywords = [feature_names[i] for i in top_indices]
        doc_count = int((topic_assignments == topic_idx).sum())
        topics.append({
            'id': int(topic_idx),
            'keywords': keywords,
            'doc_count': doc_count,
            'weight': float(topic.sum() / nmf.components_.sum() * 100)
        })

    return topics, topic_assignments


def main():
    print("Loading data...")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), low_memory=False)
    relevant = df[df['is_relevant'] == True].copy()

    result = {'topics': {}, 'cross_leader_themes': []}

    for (leader, lang), group in relevant.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        print(f"\nProcessing {key} ({len(group)} docs)...")

        texts = group['text'].dropna().tolist()
        sentiments = group.loc[group['text'].notna(), 'sentiment'].tolist()

        if len(texts) < 30:
            print(f"  Skipping {key}: too few documents")
            continue

        processed = preprocess_texts(texts, lang)
        if len(processed) < 30:
            print(f"  Skipping {key}: too few after preprocessing")
            continue

        n_topics = 6 if len(processed) > 200 else 4
        topic_result = run_topic_modeling(processed, n_topics=n_topics)

        if not topic_result:
            print(f"  Skipping {key}: topic modeling failed")
            continue

        topics, topic_assignments = topic_result

        # 为每个主题计算情感分布
        for i, topic in enumerate(topics):
            topic_docs_indices = [j for j, t in enumerate(topic_assignments) if t == i]
            topic_sentiments = [sentiments[j] for j in topic_docs_indices if j < len(sentiments)]
            from collections import Counter
            sent_counts = Counter(topic_sentiments)
            total_s = sum(sent_counts.values())
            topic['sentiment_dist'] = {
                k: round(v / total_s * 100, 1) for k, v in sent_counts.items()
            } if total_s > 0 else {}

        result['topics'][key] = topics
        print(f"  Found {len(topics)} topics")
        for t in topics:
            print(f"    Topic {t['id']}: {', '.join(t['keywords'][:5])} ({t['doc_count']} docs)")

    # 跨领导人共同主题识别
    all_keywords = {}
    for key, topics in result['topics'].items():
        seg_words = set()
        for t in topics:
            seg_words.update(t['keywords'])
        all_keywords[key] = seg_words

    # 找共同关键词
    if 'Takaichi-ZH' in all_keywords and 'LePen-ZH' in all_keywords:
        common_zh = all_keywords['Takaichi-ZH'] & all_keywords['LePen-ZH']
        if common_zh:
            result['cross_leader_themes'].append({
                'language': 'ZH',
                'shared_keywords': list(common_zh),
                'count': len(common_zh)
            })

    if 'Takaichi-EN' in all_keywords and 'LePen-EN' in all_keywords:
        common_en = all_keywords['Takaichi-EN'] & all_keywords['LePen-EN']
        if common_en:
            result['cross_leader_themes'].append({
                'language': 'EN',
                'shared_keywords': list(common_en),
                'count': len(common_en)
            })

    output_path = os.path.join(OUTPUT_DIR, 'topic_modeling.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved topic modeling to {output_path}")


if __name__ == '__main__':
    main()
