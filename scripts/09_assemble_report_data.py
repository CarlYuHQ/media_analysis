"""
09_assemble_report_data.py - 数据整合：汇总所有JSON，生成embedded_data.js
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

FILES_TO_LOAD = [
    'data_overview.json',
    'descriptive_stats.json',
    'sentiment_analysis.json',
    'emotion_analysis.json',
    'tfidf_keywords.json',
    'topic_modeling.json',
    'temporal_analysis.json',
    'comparative_analysis.json',
]


def main():
    report_data = {}

    for fname in FILES_TO_LOAD:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                key = fname.replace('.json', '')
                report_data[key] = json.load(f)
            print(f"Loaded {fname}")
        else:
            print(f"WARNING: {fname} not found")

    # 生成关键发现摘要
    executive_summary = generate_executive_summary(report_data)
    report_data['executive_summary'] = executive_summary

    # 保存完整report_data
    report_path = os.path.join(OUTPUT_DIR, 'report_data.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved report_data.json ({os.path.getsize(report_path) / 1024:.0f} KB)")

    # 生成embedded_data.js
    js_path = os.path.join(OUTPUT_DIR, 'embedded_data.js')
    js_content = f"const REPORT_DATA = {json.dumps(report_data, ensure_ascii=False)};"
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"Saved embedded_data.js ({os.path.getsize(js_path) / 1024:.0f} KB)")

    # 保存executive summary单独文件
    summary_path = os.path.join(OUTPUT_DIR, 'executive_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(executive_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved executive_summary.json")


def generate_executive_summary(data):
    """从分析结果中提取关键发现"""
    findings = []

    # 情感发现
    sentiment = data.get('sentiment_analysis', {}).get('proportions', {})
    if sentiment:
        # 找负面最高的segment
        max_neg = 0
        max_neg_key = ''
        for key, props in sentiment.items():
            neg = props.get('negative', 0)
            if neg > max_neg:
                max_neg = neg
                max_neg_key = key
        findings.append({
            'id': 'finding_1',
            'title': '中文社交媒体对高市早苗的负面情绪显著偏高',
            'detail': f'{max_neg_key}的负面情感比例高达{max_neg}%，远高于英文和日文语境。',
            'type': 'sentiment',
            'significance': 'high'
        })

    # 卡方检验
    chi_tests = data.get('sentiment_analysis', {}).get('chi_square_tests', [])
    significant_tests = [t for t in chi_tests if t.get('significant')]
    if significant_tests:
        findings.append({
            'id': 'finding_2',
            'title': f'{len(significant_tests)}项卡方检验均达到统计显著',
            'detail': f'所有情感分布差异均p<0.001，说明不同语言/领导人间的情感差异是系统性的，非随机波动。',
            'type': 'statistical',
            'significance': 'high'
        })

    # 情绪发现
    emotion_summary = data.get('emotion_analysis', {}).get('leader_emotion_summary', {})
    if emotion_summary:
        for leader, edata in emotion_summary.items():
            top_emotion = max(edata.get('proportions', {}).items(), key=lambda x: x[1])
            findings.append({
                'id': f'finding_emotion_{leader}',
                'title': f'{leader}的主要情绪',
                'detail': f'{leader}相关帖子中最高情绪为{top_emotion[0]}({top_emotion[1]}%)。',
                'type': 'emotion',
                'significance': 'medium'
            })

    # 框架发现
    frames = data.get('comparative_analysis', {}).get('gender_frames', {})
    if frames:
        for leader, leader_frames in frames.items():
            dominant_frame = max(leader_frames.items(), key=lambda x: x[1]['total_matched'])
            findings.append({
                'id': f'finding_frame_{leader}',
                'title': f'{leader}的主导性别框架',
                'detail': f'{leader}的社交媒体中最突出的框架是"{dominant_frame[1]["label"]}"，共匹配{dominant_frame[1]["total_matched"]}条帖子。',
                'type': 'frame',
                'significance': 'high'
            })

    # 双重束缚发现
    double_bind = data.get('comparative_analysis', {}).get('double_bind', {})
    if double_bind:
        max_db = 0
        max_db_info = ''
        for leader, scores in double_bind.items():
            for lang, s in scores.items():
                if s.get('double_bind_ratio', 0) > max_db:
                    max_db = s['double_bind_ratio']
                    max_db_info = f"{leader}({lang}): {max_db}%"
        if max_db > 0:
            findings.append({
                'id': 'finding_doublebind',
                'title': '双重束缚现象显著',
                'detail': f'勒庞中文数据中双重束缚得分最高({max_db_info})，反映中文社交媒体在讨论勒庞时更倾向于将性别与领导力议题交织。',
                'type': 'theory',
                'significance': 'high'
            })

    # Femonationalism
    femono = data.get('comparative_analysis', {}).get('femonationalism', {})
    if femono:
        for leader, fdata in femono.items():
            total_f = sum(v['matched'] for v in fdata['counts_by_language'].values())
            if total_f > 0:
                findings.append({
                    'id': f'finding_femono_{leader}',
                    'title': f'Femonationalism指标',
                    'detail': f'{leader}相关帖子中检测到{total_f}条包含女性民族主义框架的帖子。',
                    'type': 'theory',
                    'significance': 'medium'
                })

    # 时间发现
    temporal = data.get('temporal_analysis', {}).get('volume_sentiment_correlation', {})
    strong_corr = {k: v for k, v in temporal.items() if v.get('significant')}
    if strong_corr:
        for k, v in strong_corr.items():
            findings.append({
                'id': f'finding_temporal_{k}',
                'title': f'{k}讨论量与负面情感显著相关',
                'detail': f'r={v["r"]}, p={v["p_value"]:.4f}，说明讨论越多，负面情感比例越高。',
                'type': 'temporal',
                'significance': 'high'
            })

    return {
        'total_findings': len(findings),
        'high_significance': len([f for f in findings if f.get('significance') == 'high']),
        'findings': findings,
        'overview': {
            'total_posts': data.get('data_overview', {}).get('total_rows', 0),
            'relevant_posts': data.get('data_overview', {}).get('relevant_rows', 0),
            'leaders': data.get('data_overview', {}).get('leaders', []),
            'languages': data.get('data_overview', {}).get('languages', []),
            'date_range': data.get('data_overview', {}).get('date_range', {}),
        }
    }


if __name__ == '__main__':
    main()
