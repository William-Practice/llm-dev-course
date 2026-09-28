"""
planning_agent.py — Day 19 作业核心：Plan-and-Execute + Reflexion
plan_task()          规划：把主题拆成 5-8 个步骤
execute_step()       执行单步（带前序结果作上下文）
reflect_and_retry()  反思：判断本步质量，不合格则换思路重试
generate_report()    汇总所有步骤结果为 Markdown 报告
"""

import os
import re
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_PROJ, "day01_03") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJ, "day01_03"))

from hello_glm import ask_glm

MAX_STEPS = 8        # 最多规划/执行几步
MAX_RETRIES = 2      # 每步最多重试次数


# ── 阶段1：规划 ─────────────────────────────────────────

def plan_task(theme: str) -> list[str]:
    """把调研主题拆成 5-8 个可执行步骤"""
    prompt = f"""你是调研规划专家。用户要调研技术主题「{theme}」。

请把它拆解成 5-8 个具体的调研子任务，要求：
1. 从基础到深入（是什么 → 核心概念 → 使用方式 → 应用场景 → 对比/局限）
2. 每个子任务是一个明确的问题，可以独立回答
3. 最后一个子任务是"总结整理"

只输出子任务列表，每行一个，不要编号，不要其他内容。"""
    reply = ask_glm(prompt, "你是严谨的调研规划专家。")
    steps = []
    for line in reply.splitlines():
        line = re.sub(r"^[\d一二三四五六七八九十]+[.、)）]\s*", "", line.strip())
        line = re.sub(r"^[-•*]\s*", "", line)
        if line:
            steps.append(line)
    return steps[:MAX_STEPS]


# ── 阶段2：执行单步 ─────────────────────────────────────

def execute_step(theme: str, step: str, prev_results: list[dict], retry_hint: str = "") -> str:
    """执行单个子任务，把前序结果作为上下文"""
    context = ""
    if prev_results:
        context = "【前序调研结果】\n" + "\n".join(
            f"- {r['step']}：{r['result'][:300]}" for r in prev_results[-3:]
        ) + "\n\n"

    hint = f"\n注意：上一次尝试的问题是「{retry_hint}」，请换个角度回答。\n" if retry_hint else ""

    prompt = f"""你在调研技术主题「{theme}」。

{context}当前子任务：{step}{hint}

请给出这个子任务的具体答案（200-400字，内容要准确、有干货）。"""
    return ask_glm(prompt, "你是技术调研专家，回答准确、具体。")


# ── 反思 ────────────────────────────────────────────────

def reflect(theme: str, step: str, result: str) -> tuple[bool, str]:
    """
    反思本步结果质量
    :return: (是否通过, 不通过原因)
    """
    prompt = f"""请评估下面这个调研子任务的回答质量。

主题：{theme}
子任务：{step}
回答：{result[:600]}

评估标准：
1. 是否直接回答了子任务
2. 内容是否具体（有实质信息，不是空话套话）
3. 是否有明显错误或答非所问

请只输出以下两种格式之一：
PASS
或者
RETRY: 具体原因（20字内）"""
    reply = ask_glm(prompt, "你是严格的答题审核员。").strip()

    if reply.upper().startswith("PASS"):
        return True, ""
    m = re.search(r"RETRY\s*[：:]\s*(.+)", reply)
    return False, (m.group(1).strip() if m else "质量不达标")


def reflect_and_retry(theme: str, step: str, prev_results: list[dict],
                      verbose: bool = True) -> dict:
    """执行 + 反思 + 必要时重试（最多 MAX_RETRIES 次）"""
    hint = ""
    result = ""
    for attempt in range(MAX_RETRIES + 1):
        result = execute_step(theme, step, prev_results, retry_hint=hint)
        passed, reason = reflect(theme, step, result)
        if passed:
            if verbose:
                status = "✅ 通过" if attempt == 0 else f"✅ 反思重试{attempt}次后通过"
                print(f"    {status}")
            return {"step": step, "result": result, "passed": True, "retries": attempt}
        hint = reason
        if verbose:
            print(f"    ⚠️ 反思不通过（{reason}），重试 {attempt+1}/{MAX_RETRIES}")

    if verbose:
        print(f"    ⚠️ 达到最大重试次数，采用当前结果")
    return {"step": step, "result": result, "passed": False, "retries": MAX_RETRIES}


# ── 阶段3：汇总报告 ─────────────────────────────────────

def generate_report(theme: str, results: list[dict], queries: list[str] | None = None) -> str:
    """把所有步骤结果整理成结构化 Markdown 报告（加分项：参考资料）"""
    body = "\n\n".join(
        f"## {i}. {r['step']}\n{r['result']}" for i, r in enumerate(results, 1)
    )
    outline = "\n".join(f"{i}. {r['step']}" for i, r in enumerate(results, 1))

    prompt = f"""请把下面的调研材料整理成一份专业的 Markdown 调研报告。

主题：{theme}

调研内容：
{body}

要求：
1. 报告结构：标题、概述（3-5句）、分章节正文、小结
2. 保留所有关键信息，去掉重复
3. 语言专业、条理清晰
4. 用 Markdown 格式（## 标题、- 列表、必要时表格）

只输出报告内容本身。"""
    report = ask_glm(prompt, "你是资深技术分析师，擅长写调研报告。")

    # 加分项：附参考资料（每步的查询）
    if queries:
        report += "\n\n## 参考资料\n" + "\n".join(f"- {q}" for q in queries)

    return report


# ── 三阶段主流程 ────────────────────────────────────────

def research(theme: str, verbose: bool = True) -> tuple[str, list[str], list[dict]]:
    """
    完整调研流程
    :return: (报告, 步骤列表, 步骤结果)
    """
    # 阶段1：规划
    if verbose:
        print(f"🎯 调研主题: {theme}\n📋 阶段1: 规划任务...")
    steps = plan_task(theme)
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    # 阶段2：执行 + 反思
    print(f"\n🔧 阶段2: 执行 + 反思...")
    results = []
    for i, step in enumerate(steps, 1):
        print(f"  步骤{i}: {step[:40]}")
        r = reflect_and_retry(theme, step, results, verbose=verbose)
        results.append(r)

    # 阶段3：报告
    print(f"\n📄 阶段3: 生成报告...")
    queries = [f"调研子任务: {s}" for s in steps]
    report = generate_report(theme, results, queries=queries)

    return report, steps, results


if __name__ == "__main__":
    theme = sys.argv[1] if len(sys.argv) > 1 else "LangGraph"
    report, steps, results = research(theme)
    print("\n" + "=" * 60)
    print(report[:1500])
