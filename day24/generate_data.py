"""
generate_data.py — Day 24 作业：微调数据集制作
蒸馏生成 ~600 条考研复习问答 → 清洗 → 存 Alpaca 格式 jsonl → 质量统计 + 抽查

用法:
    python generate_data.py --test          # 小批量测试（10 条）
    python generate_data.py                 # 正式生成 600 条
    python generate_data.py --model glm-4   # 用更强模型（加分项，质量更高但更慢）
"""

import os
import re
import sys
import json
import time
import random
import argparse
from collections import Counter

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_PROJ, "day01_03") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJ, "day01_03"))

from hello_glm import ask_glm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 子主题配置（加分项：多子主题覆盖 + 分类标签） ────────

SUBJECTS = {
    "考研政治": [
        "马克思主义基本原理（唯物论、辩证法、认识论、历史唯物主义）",
        "毛泽东思想和中国特色社会主义理论体系概论",
        "中国近现代史纲要（重大事件与时间线）",
        "思想道德与法治",
        "形势与政策（当年时政热点）",
    ],
    "考研英语": [
        "核心词汇辨析与词根词缀",
        "长难句分析与句子结构",
        "阅读理解解题技巧",
        "写作（小作文书信 + 大作文图表/议论）",
        "翻译与完形填空",
    ],
    "考研数学": [
        "高等数学（极限、微分、积分、级数）",
        "线性代数（矩阵、行列式、特征值）",
        "概率论与数理统计",
        "解题技巧与常见陷阱",
    ],
    "计算机专业课408": [
        "数据结构（线性表、树、图、排序、查找）",
        "计算机组成原理（数据表示、CPU、存储层次）",
        "操作系统（进程、内存、文件、死锁）",
        "计算机网络（TCP/IP、HTTP、路由）",
    ],
    "备考方法": [
        "复习规划与时间管理",
        "各科复习顺序与阶段安排",
        "真题使用方法",
        "心态调整与应试技巧",
    ],
}

# 每批生成的条数
BATCH_SIZE = 15


# ── 1. 蒸馏生成 ─────────────────────────────────────────

def build_prompt(subject: str, topics: list[str], n: int, existing: list[str]) -> str:
    """构造生成提示词"""
    avoid = ""
    if existing:
        avoid = "\n\n**不要重复以下已有问题：**\n" + "\n".join(f"- {q}" for q in existing[-25:])

    return f"""你是考研辅导专家。请生成 {n} 条「{subject}」科目的高质量问答训练数据。

覆盖这些子方向（尽量分散）：
{chr(10).join(f'- {t}' for t in topics)}{avoid}

要求：
1. 每条是一问一答：问题要具体（像学生真会问的），答案要准确、有干货
2. **答案长度 100-200 字**，条理清晰，不要空话套话
3. 涉及考点要准确（时间、公式、定义不能错）
4. 问题要多样化：概念解释、对比辨析、解题方法、易错点、记忆技巧

**只输出 JSON 数组**，不要 markdown 代码块，不要任何解释文字。格式：
[
  {{"instruction": "问题", "output": "答案"}},
  ...
]"""


def parse_json_array(text: str) -> list[dict]:
    """从 LLM 输出里稳健地抠出 JSON 数组"""
    text = text.strip()
    # 去掉 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # 退而求其次：找第一个 [ ... ] 块
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # 再退：逐行抠 JSON 对象
    items = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def generate_training_data(target: int, model: str, test: bool = False) -> list[dict]:
    """蒸馏生成 target 条原始数据（多子主题轮转）"""
    all_items = []
    subject_names = list(SUBJECTS.keys())
    # 每个子主题的目标条数
    per_subject = target // len(subject_names)
    plan = {s: per_subject for s in subject_names}
    # 余数分给前几个（408 和数学更缺）
    for i in range(target - per_subject * len(subject_names)):
        plan[subject_names[i % len(subject_names)]] += 1

    print(f"📋 生成计划（目标 {target} 条）:")
    for s, n in plan.items():
        print(f"   {s}: {n} 条")

    for subject, want in plan.items():
        topics = SUBJECTS[subject]
        got = []
        seen_q = []
        attempts = 0
        while len(got) < want and attempts < want:      # 加重试上限
            attempts += 1
            n = min(BATCH_SIZE, want - len(got))
            prompt = build_prompt(subject, topics, n, seen_q)
            try:
                reply = ask_glm(prompt, "你是考研辅导专家，只输出 JSON 数组。", model=model)
                items = parse_json_array(reply)
            except Exception as e:
                print(f"   ⚠️ {subject} 第 {attempts} 批失败: {type(e).__name__}: {str(e)[:80]}")
                time.sleep(2)
                continue

            added = 0
            for it in items:
                q = str(it.get("instruction", "")).strip()
                a = str(it.get("output", "")).strip()
                if not q or not a:
                    continue
                got.append({"instruction": q, "input": "", "output": a, "subject": subject})
                seen_q.append(q)
                added += 1
            print(f"   {subject}: +{added} → {len(got)}/{want}")
            if added == 0:
                break
            time.sleep(0.5)                              # 轻量限速

        all_items.extend(got)

    return all_items


# ── 2. 清洗 ─────────────────────────────────────────────

BAD_PATTERNS = [
    r"^作为(一个)?(AI|人工智能|语言模型)",
    r"^(抱歉|对不起)[,，]",
    r"我(无法|不能)(提供|回答)",
    r"^以下是",
    r"^(当然|好的)[,，]",
]


def clean_data(items: list[dict]) -> tuple[list[dict], dict]:
    """去重 / 去空 / 去乱码 / 去太短太长"""
    stats = {"原始": len(items), "空字段": 0, "重复": 0,
             "长度不合格": 0, "含模板话术": 0, "乱码": 0}
    cleaned, seen = [], set()

    for it in items:
        q = re.sub(r"\s+", " ", str(it.get("instruction", ""))).strip()
        a = re.sub(r"\s+", " ", str(it.get("output", ""))).strip()

        if not q or not a:
            stats["空字段"] += 1
            continue

        # 去重（问题归一化后比对）
        key = re.sub(r"[^\w\u4e00-\u9fff]", "", q)
        if key in seen:
            stats["重复"] += 1
            continue

        # 长度：output 10-500 字（中文按字符数）
        if not (10 <= len(a) <= 500):
            stats["长度不合格"] += 1
            continue

        # 去除明显是模型话术/拒答的
        if any(re.search(p, a) for p in BAD_PATTERNS):
            stats["含模板话术"] += 1
            continue

        # 乱码检测：替换字符、过高的非中英数比例
        garbage = a.count("�") + a.count("\ufffd")
        if garbage > 0:
            stats["乱码"] += 1
            continue
        meaningful = len(re.findall(r"[\u4e00-\u9fff\w]", a))
        if meaningful / max(len(a), 1) < 0.5:
            stats["乱码"] += 1
            continue

        seen.add(key)
        cleaned.append({"instruction": q, "input": "", "output": a,
                        "subject": it.get("subject", "未分类")})

    return cleaned, stats


# ── 3. 保存 ─────────────────────────────────────────────

def save_jsonl(items: list[dict], path: str, strict_alpaca: bool = True):
    """保存 jsonl。strict_alpaca=True 时只写三字段（严格 Alpaca 格式）"""
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            if strict_alpaca:
                row = {"instruction": it["instruction"], "input": it.get("input", ""),
                       "output": it["output"]}
            else:
                row = it
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# ── 4. 质量统计 ─────────────────────────────────────────

def stats_data(items: list[dict]) -> dict:
    lens = [len(it["output"]) for it in items]
    lens.sort()
    n = len(lens)
    return {
        "总条数": n,
        "output 字数": {
            "最小": lens[0], "最大": lens[-1],
            "平均": round(sum(lens) / n, 1),
            "中位数": lens[n // 2],
            "10-500 合规": sum(1 for x in lens if 10 <= x <= 500),
        },
        "子主题分布": dict(Counter(it["subject"] for it in items).most_common()),
        "问题平均字数": round(sum(len(it["instruction"]) for it in items) / n, 1),
    }


# ── 5. 抽查 ─────────────────────────────────────────────

def check_quality(items: list[dict], n: int = 10, seed: int = 42):
    """随机抽查 n 条供人工核对"""
    random.seed(seed)
    sample = random.sample(items, min(n, len(items)))
    print(f"\n{'═'*70}\n🔍 随机抽查 {len(sample)} 条（人工核对事实准确性）\n{'═'*70}")
    for i, it in enumerate(sample, 1):
        print(f"\n[{i}] 【{it['subject']}】")
        print(f"Q: {it['instruction']}")
        print(f"A: {it['output']}（{len(it['output'])} 字）")
    return sample


# ── 主流程 ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600, help="生成目标条数")
    ap.add_argument("--model", default="glm-4-flash", help="生成模型")
    ap.add_argument("--test", action="store_true", help="小批量测试")
    args = ap.parse_args()

    target = 15 if args.test else args.target
    suffix = "_test" if args.test else ""

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║   📊 微调数据集制作（Day 24）                    ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"模型: {args.model}   目标: {target} 条\n")

    t0 = time.time()
    raw = generate_training_data(target, args.model, test=args.test)
    print(f"\n✅ 生成完成: {len(raw)} 条（耗时 {time.time()-t0:.0f}s）")

    cleaned, stats = clean_data(raw)
    print(f"\n🧹 清洗统计:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    print(f"   保留: {len(cleaned)} 条")

    # 保存
    main_path = save_jsonl(cleaned, os.path.join(OUT_DIR, f"train{suffix}.jsonl"), strict_alpaca=True)
    tagged_path = save_jsonl(cleaned, os.path.join(OUT_DIR, f"train_with_tags{suffix}.jsonl"), strict_alpaca=False)
    print(f"\n💾 已保存:")
    print(f"   {main_path}          （严格 Alpaca 三字段，提交用）")
    print(f"   {tagged_path}（含 subject 分类标签）")

    # 统计
    st = stats_data(cleaned)
    print(f"\n📈 质量统计:")
    print(json.dumps(st, ensure_ascii=False, indent=2))

    # 抽查
    if not args.test:
        check_quality(cleaned)


if __name__ == "__main__":
    main()
