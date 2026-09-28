"""
run_eval.py — Day 27：一键跑出评估报告
用法: python run_eval.py

流程：准备测试题 → 两模型各跑一遍 → LLM 裁判评分 → 生成 txt + json 报告
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval_model import (prepare_test_questions, compare_models, evaluate_all,
                        generate_report, BASE_MODEL, ADAPTER)


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   📊 微调效果评估（Day 27 · Phase 4 收官）                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. 测试题
    questions = prepare_test_questions()
    print(f"\n📝 阶段1: 准备测试题 —— {len(questions)} 道（训练集外）")
    subj = {}
    for q in questions:
        subj[q["subject"]] = subj.get(q["subject"], 0) + 1
    for s, c in subj.items():
        print(f"   {s}: {c} 道")

    # 2. 对比
    print(f"\n🔬 阶段2: 原模型 vs 微调模型 各跑一遍（同 temperature/max_tokens）...")
    t0 = time.time()
    results = compare_models(questions)
    print(f"   完成，耗时 {time.time()-t0:.0f}s")

    # 3. 评分
    print(f"\n🎯 阶段3: LLM 裁判自动评分（准确性/完整性/简洁性 各 5 分）...")
    t1 = time.time()
    results = evaluate_all(results)
    print(f"   完成，耗时 {time.time()-t1:.0f}s")

    # 4. 报告
    print(f"\n📄 阶段4: 生成报告...")
    model_info = {
        "基础模型": BASE_MODEL,
        "LoRA 权重": ADAPTER,
        "采样参数": "temperature=0.7, top_p=0.9, repetition_penalty=1.05, max_new_tokens=256",
        "量化": "4bit NF4（QLoRA 推理）",
        "裁判模型": "glm-4-flash（rubric: 准确性/完整性/简洁性，各 1-5 分）",
    }
    txt_path, json_path = generate_report(results, model_info)

    print(f"\n✅ 完成!")
    print(f"   {txt_path}")
    print(f"   {json_path}")

    # 摘要
    n = len(results)
    w = sum(1 for r in results if r.get("winner") == "B")
    l = sum(1 for r in results if r.get("winner") == "A")
    t = sum(1 for r in results if r.get("winner") == "tie")
    print(f"\n   微调模型胜 {w}/{n}，原模型胜 {l}/{n}，平手 {t}/{n}")


if __name__ == "__main__":
    main()
