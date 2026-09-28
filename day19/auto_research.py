"""
auto_research.py — Day 19 作业：自动调研 Agent
三阶段：规划拆解 → 逐步执行+反思 → 汇总报告
加分项：追加问题、参考资料、plan_X.json 保存规划
"""

import os
import sys
import json

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from planning_agent import research, MAX_STEPS, MAX_RETRIES
from hello_glm import ask_glm


def save_outputs(theme: str, report: str, steps: list[str], results: list[dict]):
    """保存报告和规划（加分项：plan_X.json）"""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in theme)
    report_path = os.path.join(_ROOT, f"report_{safe}.md")
    plan_path = os.path.join(_ROOT, f"plan_{safe}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump({"theme": theme, "steps": steps,
                   "results": [{"step": r["step"], "retries": r["retries"],
                                "passed": r["passed"]} for r in results]},
                  f, ensure_ascii=False, indent=2)
    return report_path, plan_path


def follow_up(theme: str, report: str):
    """加分项：报告生成后继续追问，Agent 基于已有调研结果回答"""
    print("\n（可以继续追问相关问题，输入 q 返回）")
    while True:
        try:
            q = input("追问: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        prompt = f"""以下是关于「{theme}」的调研报告：

{report[:3000]}

用户追问：{q}

请基于上述调研报告回答。如果报告里没有相关信息，说明并给出你自己的补充。"""
        print("\n" + ask_glm(prompt, "你是技术调研专家。") + "\n")


def main():
    print("╔══════════════════════════════════════╗")
    print("║      📊 自动调研 Agent               ║")
    print("╚══════════════════════════════════════╝")
    print(f"（自动规划拆解 → 逐步执行+反思 → 生成报告；max_steps={MAX_STEPS}, max_retries={MAX_RETRIES}）\n")

    theme = input("请输入要调研的技术主题: ").strip()
    if not theme:
        print("  ✗ 主题不能为空")
        return

    print()
    report, steps, results = research(theme)

    # 保存
    report_path, plan_path = save_outputs(theme, report, steps, results)
    print(f"\n💾 报告已保存: {report_path}")
    print(f"💾 规划已保存: {plan_path}")

    # 查看报告
    if input("\n要不要看报告内容？(y/n): ").strip().lower() == "y":
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)

    # 加分项：追问
    follow_up(theme, report)


if __name__ == "__main__":
    main()
