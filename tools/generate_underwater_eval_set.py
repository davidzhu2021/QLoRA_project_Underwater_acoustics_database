import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "under_acoustic_data" / "eval_underwater_qa.json"


TOPICS = [
    ("主动声呐", ["发射脉冲", "目标回波", "双程传播损失", "混响"], ["fabricated_range"]),
    ("被动声呐", ["辐射噪声", "阵列增益", "方位估计", "环境噪声"], ["overconfident_detection"]),
    ("水声信道", ["低声速", "多途传播", "时变信道", "吸收损失"], ["missing_boundary_conditions"]),
    ("多途效应", ["海面反射", "海底反射", "时延扩展", "码间干扰"], ["single-cause_answer"]),
    ("混响", ["体混响", "海面混响", "海底混响", "虚警概率"], ["confuse_noise_reverberation"]),
    ("匹配滤波", ["脉冲压缩", "互相关", "信噪比", "距离分辨率"], ["claim_universal_optimum"]),
    ("阵列处理", ["波束形成", "旁瓣", "阵元间距", "空间采样"], ["ignore_grating_lobes"]),
    ("传播损失", ["几何扩展", "吸收", "边界散射", "声速剖面"], ["invent_numeric_loss"]),
    ("声速剖面", ["温度", "盐度", "静压力", "折射"], ["ignore_profile_variation"]),
    ("目标强度", ["散射截面", "姿态角", "频率依赖", "目标材质"], ["fixed_ts_claim"]),
]

QUESTION_PATTERNS = [
    "{topic}在水下探测中的核心机理是什么？",
    "设计{topic}相关系统时需要关注哪些工程限制？",
    "{topic}会怎样影响水声通信或声呐探测性能？",
    "解释{topic}时，哪些条件不足会导致结论不可靠？",
    "如何从专业角度评价{topic}相关结果是否可信？",
    "{topic}在浅海环境中通常会遇到哪些问题？",
    "{topic}与声速剖面、海况或阵列参数有什么关系？",
    "为什么不能脱离频率、深度和海底条件讨论{topic}？",
    "请说明{topic}的常见误区以及正确表述方式。",
    "在工程应用中，如何缓解{topic}带来的不利影响？",
]


def make_reference(topic, keywords):
    core = "、".join(keywords[:3])
    return (
        f"{topic}的专业分析应先说明物理定义，再结合{core}等因素解释机理。"
        f"在水声场景中，结论通常依赖频率、深度、海况、声速剖面、海底底质、阵列孔径和平台自噪声，"
        f"不能脱离边界条件给出固定数值或绝对判断。工程上需要同时评估信噪比、传播损失、检测阈值、"
        f"分辨率和虚警风险；条件不足时应明确说明不确定性，并给出需要补充的观测或参数。"
    )


def main():
    data = []
    idx = 1
    for topic, keywords, risk_flags in TOPICS:
        for pattern in QUESTION_PATTERNS:
            data.append(
                {
                    "id": f"uwqa-{idx:03d}",
                    "question": pattern.format(topic=topic),
                    "reference": make_reference(topic, keywords),
                    "keywords": keywords,
                    "risk_flags": risk_flags,
                }
            )
            idx += 1

    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} examples to {OUT}")


if __name__ == "__main__":
    main()
