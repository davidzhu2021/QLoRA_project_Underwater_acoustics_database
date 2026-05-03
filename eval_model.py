from nltk.translate.bleu_score import sentence_bleu
import json

def evaluate_bleu(model, test_data, tokenizer):
    scores = []
    for item in test_data:
        pred = generate(model, item['instruction'])
        ref = item['output'].split()
        hyp = pred.split()
        score = sentence_bleu([ref], hyp)
        scores.append(score)
    return sum(scores) / len(scores)

def evaluate_professionalism(outputs):
    """模拟人工评估：检测非专业输出"""
    non_pro_patterns = [
        '我觉得', '可能是', '大概', '也许',  # 不确定表述
        '百度', '谷歌', '网上说',            # 引用非专业来源
        '不知道', '不清楚'                    # 拒答
    ]
    non_pro_count = 0
    for out in outputs:
        if any(p in out for p in non_pro_patterns):
            non_pro_count += 1
    return non_pro_count / len(outputs)

def evaluate_format_compliance(outputs):
    """格式合规率：段落完整、有结论句"""
    compliant = 0
    for out in outputs:
        if len(out) > 100 and ('。' in out[-20:]):  # 有完整结句
            compliant += 1
    return compliant / len(outputs)