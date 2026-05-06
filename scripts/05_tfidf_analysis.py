"""
05_tfidf_analysis.py - TF-IDF关键词分析（多语言）
"""
import pandas as pd
import numpy as np
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# === 停用词 ===
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


def clean_text(text):
    """清洗文本：移除URL、@、#等"""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_japanese(text):
    """日文分词：字符级 + jieba混合"""
    import jieba
    # 对日文文本使用jieba（能处理汉字部分）
    words = jieba.lcut(text)
    # 过滤
    words = [w.strip() for w in words if len(w.strip()) > 1]
    words = [w for w in words if w not in JAPANESE_STOPWORDS]
    # 额外过滤纯平假名单词（太短无意义）
    words = [w for w in words if not (len(w) <= 2 and all('\u3040' <= c <= '\u309f' for c in w))]
    return words


def tokenize_chinese(text):
    """中文分词"""
    import jieba
    words = jieba.lcut(text)
    words = [w.strip() for w in words if len(w.strip()) > 1]
    words = [w for w in words if w not in CHINESE_STOPWORDS]
    return words


def tokenize_english(text):
    """英文分词"""
    from collections import Counter
    words = text.lower().split()
    words = [w for w in words if len(w) > 2 and w.isalpha() and w not in CHINESE_STOPWORDS]
    return words


def tokenize(text, language):
    """根据语言选择分词器"""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if language == 'JP':
        return tokenize_japanese(cleaned)
    elif language == 'ZH':
        return tokenize_chinese(cleaned)
    else:
        return tokenize_english(cleaned)


def compute_tfidf(corpus_dict):
    """
    简单TF-IDF计算
    corpus_dict: {segment_key: [tokenized_documents]}
    返回每个segment的top关键词
    """
    from collections import Counter

    # 计算DF (document frequency)
    all_terms = set()
    doc_freq = Counter()
    for seg, docs in corpus_dict.items():
        seg_terms = set()
        for doc in docs:
            seg_terms.update(doc)
        for term in seg_terms:
            doc_freq[term] += 1
        all_terms.update(seg_terms)

    N = len(corpus_dict)  # segment数作为"语料库大小"

    results = {}
    for seg, docs in corpus_dict.items():
        if not docs:
            results[seg] = []
            continue

        # 计算TF (在该segment中)
        tf_counter = Counter()
        total_terms = 0
        for doc in docs:
            tf_counter.update(doc)
            total_terms += len(doc)

        if total_terms == 0:
            results[seg] = []
            continue

        # TF-IDF
        tfidf_scores = {}
        for term, count in tf_counter.items():
            tf = count / total_terms
            df = doc_freq.get(term, 0)
            idf = np.log((N + 1) / (df + 1)) + 1
            tfidf_scores[term] = tf * idf

        # 排序取Top50
        sorted_terms = sorted(tfidf_scores.items(), key=lambda x: -x[1])[:50]
        results[seg] = [
            {'word': word, 'score': round(score, 4)}
            for word, score in sorted_terms
        ]

    return results


def main():
    print("Loading data...")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), low_memory=False)
    relevant = df[df['is_relevant'] == True].copy()

    print("Tokenizing texts...")
    # 为每个segment构建语料
    corpus = {}
    for (leader, lang), group in relevant.groupby(['leader', 'language']):
        key = f"{leader}-{lang}"
        print(f"  Tokenizing {key} ({len(group)} docs)...")
        docs = []
        for text in group['text'].dropna():
            tokens = tokenize(str(text), lang)
            if tokens:
                docs.append(tokens)
        corpus[key] = docs

    # 计算TF-IDF (按segment)
    print("\nComputing TF-IDF by segment...")
    keywords_by_segment = compute_tfidf(corpus)

    # 计算TF-IDF (按segment + sentiment)
    print("Computing TF-IDF by segment and sentiment...")
    corpus_sentiment = {}
    for (leader, lang, sentiment), group in relevant.groupby(['leader', 'language', 'sentiment']):
        key = f"{leader}-{lang}-{sentiment}"
        docs = []
        for text in group['text'].dropna():
            tokens = tokenize(str(text), lang)
            if tokens:
                docs.append(tokens)
        corpus_sentiment[key] = docs

    keywords_by_sentiment = compute_tfidf(corpus_sentiment)

    # 词云数据 (ECharts格式)
    wordcloud_data = {}
    for key, words in keywords_by_segment.items():
        wordcloud_data[key] = [
            {'name': w['word'], 'value': round(w['score'] * 1000)}
            for w in words[:80]
        ]

    result = {
        'keywords_by_segment': keywords_by_segment,
        'keywords_by_sentiment': keywords_by_sentiment,
        'wordcloud_data': wordcloud_data,
    }

    output_path = os.path.join(OUTPUT_DIR, 'tfidf_keywords.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved TF-IDF analysis to {output_path}")

    # Print top keywords
    print("\n=== Top Keywords by Segment ===")
    for key, words in keywords_by_segment.items():
        top5 = [f"{w['word']}({w['score']})" for w in words[:5]]
        print(f"  {key}: {', '.join(top5)}")


if __name__ == '__main__':
    main()
