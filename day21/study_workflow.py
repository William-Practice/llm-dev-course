"""
study_workflow.py — Day 21 作业：学习规划工作流（LangGraph）
StateGraph 编排 7 个节点，含 2 处条件分支 + 1 个人工确认节点

START → analyze → calc_days ─┬─(天数够)→ plan
                             └─(不够)→ adjust → plan
plan → confirm ─┬─(y)→ checklist → output → END
                └─(n)→ plan（带反馈重新生成）
"""

import os
import sys
from datetime import date, datetime
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_PROJ, "day01_03") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJ, "day01_03"))

from hello_glm import ask_glm

MAX_REPLAN = 3     # 确认环节最多重新生成几次（防死循环）


# ── State 定义 ───────────────────────────────────────────

class StudyState(TypedDict):
    goal: str           # 学习目标
    target_date: str    # 目标日期 YYYY-MM-DD
    days: int           # 距目标天数
    enough: bool        # 时间是否够
    analysis: str       # 目标内容分析
    adjust_note: str    # 时间不够时的调整建议
    plan: str           # 学习计划
    feedback: str       # 用户对计划的修改意见
    confirmed: bool     # 是否确认
    replan_count: int   # 重新生成次数
    checklist: str      # 每日打卡清单
    saved_path: str     # 保存路径


# ── 节点函数 ─────────────────────────────────────────────

def analyze(state: StudyState) -> dict:
    """节点1：AI 分析目标需要哪些内容"""
    print("🔵 节点1: analyze — 分析学习目标...")
    prompt = f"""用户的学习目标是「{state['goal']}」，目标日期 {state['target_date']}。

请分析这个目标需要掌握哪些学习内容，按模块列出（5-8 个模块），
每个模块简要说明包含什么知识点。用简洁的列表回答。"""
    analysis = ask_glm(prompt, "你是学习规划专家。")
    print(f"   分析完成（{len(analysis)} 字）")
    return {"analysis": analysis}


def calc_days(state: StudyState) -> dict:
    """节点2：计算距目标还有几天"""
    print("🔵 节点2: calc_days — 计算剩余天数...")
    try:
        target = datetime.strptime(state["target_date"], "%Y-%m-%d").date()
        days = (target - date.today()).days
    except ValueError:
        print(f"   ⚠️ 日期格式错误，默认按 90 天算")
        days = 90
    enough = days >= 30          # 分支条件：≥30 天算够
    print(f"   距 {state['target_date']} 还有 {days} 天 → {'时间充足' if enough else '时间紧张'}")
    return {"days": days, "enough": enough}


def plan(state: StudyState) -> dict:
    """节点4：AI 生成按天学习计划（带回退反馈时重新生成）"""
    n = state.get("replan_count", 0)
    print(f"🔵 节点4: plan — 生成学习计划{'（第 %d 次重新生成）' % n if n else ''}...")

    feedback = state.get("feedback", "")
    fb_block = f"\n\n【用户对上一版计划的修改意见（必须采纳）】\n{feedback}" if feedback else ""
    adjust_block = f"\n\n【时间紧张，请参考以下调整建议】\n{state.get('adjust_note','')}" if state.get("adjust_note") and not feedback else ""

    prompt = f"""为下面这个学习目标制定按天计划：

目标：{state['goal']}
目标日期：{state['target_date']}（还有 {state['days']} 天）
需要学习的内容：
{state['analysis']}{fb_block}{adjust_block}

要求：
1. 按阶段划分（如 基础/强化/冲刺），分配到具体天数
2. 每天有明确任务量和产出物
3. 留出复习和模拟测试时间
4. 用 Markdown 格式（## 阶段名、- 每日任务）"""
    p = ask_glm(prompt, "你是学习规划专家，计划要具体可执行。")
    print(f"   计划完成（{len(p)} 字）")
    return {"plan": p, "feedback": ""}      # 清空反馈，避免下次误用


def adjust(state: StudyState) -> dict:
    """节点5：时间不够时的调整建议"""
    print("🔵 节点5: adjust — 时间紧张，生成调整建议...")
    prompt = f"""用户目标「{state['goal']}」只剩 {state['days']} 天，时间紧张。

需要学习的内容：
{state['analysis']}

请给出精简策略：哪些内容是必学的核心（优先），哪些可以略过或延后，
每天需要投入多长时间。简洁明了（150字内）。"""
    note = ask_glm(prompt, "你是务实的备考顾问。")
    print(f"   调整建议完成")
    return {"adjust_note": note}


def confirm(state: StudyState) -> dict:
    """节点6：人工确认（用户可提出修改）"""
    print("\n" + "=" * 60)
    print("📋 当前学习计划：")
    print("=" * 60)
    print(state["plan"][:2000])
    print("=" * 60)

    ans = input("\n是否确认这个计划？(y=确认 / n=提出修改意见): ").strip().lower()
    if ans == "y":
        return {"confirmed": True}
    fb = input("请输入你的修改意见: ").strip()
    return {"confirmed": False, "feedback": fb or "请调整得更精简一些",
            "replan_count": state.get("replan_count", 0) + 1}


def checklist(state: StudyState) -> dict:
    """节点7（加分项）：生成每日打卡清单"""
    print("🔵 节点7: checklist — 生成每日打卡清单...")
    prompt = f"""根据下面的学习计划，生成一份「每日打卡清单」：

目标：{state['goal']}
计划：
{state['plan'][:2000]}

要求：
1. 用 Markdown 复选框格式：- [ ] 任务内容
2. 按天或按阶段分组
3. 包含每日必做的固定项（如复习昨日、做习题）
4. 简洁，方便每天打勾"""
    c = ask_glm(prompt, "你是学习管理助手。")
    print("   打卡清单完成")
    return {"checklist": c}


def output(state: StudyState) -> dict:
    """节点8：输出最终计划并保存文件（加分项）"""
    print("🔵 节点8: output — 输出并保存...")
    content = f"""# 学习规划：{state['goal']}

> 目标日期：{state['target_date']}（剩余 {state['days']} 天）
> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 一、学习内容分析
{state['analysis']}

## 二、学习计划
{state['plan']}
"""
    if state.get("adjust_note"):
        content += f"\n## 三、时间调整建议\n{state['adjust_note']}\n"
    if state.get("checklist"):
        content += f"\n## 四、每日打卡清单\n{state['checklist']}\n"

    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in state["goal"])[:20]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"plan_{safe}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n{'='*60}\n✅ 最终计划已生成，保存到:\n   {path}\n{'='*60}")
    return {"saved_path": path}


# ── 条件分支 ─────────────────────────────────────────────

def route_by_days(state: StudyState) -> str:
    """天数够 → plan；不够 → adjust"""
    return "ok" if state["enough"] else "tight"


def route_by_confirm(state: StudyState) -> str:
    """确认 → checklist；不确认(且未超次数) → plan 重新生成；超次数 → checklist"""
    if state["confirmed"]:
        return "yes"
    if state.get("replan_count", 0) >= MAX_REPLAN:
        print(f"   ⚠️ 已达最大重新生成次数（{MAX_REPLAN}），采用当前计划")
        return "yes"
    return "no"


# ── 构建工作流 ───────────────────────────────────────────

def build_workflow():
    g = StateGraph(StudyState)

    g.add_node("analyze", analyze)
    g.add_node("calc_days", calc_days)
    g.add_node("plan", plan)
    g.add_node("adjust", adjust)
    g.add_node("confirm", confirm)
    g.add_node("checklist", checklist)
    g.add_node("output", output)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "calc_days")
    # 条件分支1：天数够不够
    g.add_conditional_edges("calc_days", route_by_days, {"ok": "plan", "tight": "adjust"})
    g.add_edge("adjust", "plan")
    g.add_edge("plan", "confirm")
    # 条件分支2：是否确认（n 回到 plan 形成循环）
    g.add_conditional_edges("confirm", route_by_confirm, {"yes": "checklist", "no": "plan"})
    g.add_edge("checklist", "output")
    g.add_edge("output", END)

    return g.compile()


def main():
    print("╔══════════════════════════════════════════╗")
    print("║   🎯 学习规划工作流（LangGraph）          ║")
    print("╚══════════════════════════════════════════╝")

    goal = input("请输入学习目标（如：软考软件设计师）: ").strip()
    target = input("目标日期 (YYYY-MM-DD，回车=2026-11-07): ").strip() or "2026-11-07"

    app = build_workflow()
    init: StudyState = {"goal": goal, "target_date": target, "days": 0, "enough": True,
                        "analysis": "", "adjust_note": "", "plan": "", "feedback": "",
                        "confirmed": False, "replan_count": 0, "checklist": "", "saved_path": ""}

    print(f"\n🚀 开始执行工作流：{goal} @ {target}\n" + "─" * 50)
    final = app.invoke(init, {"recursion_limit": 30})

    print(f"\n📊 State 验证（节点间传递的字段）:")
    for k in ("goal", "target_date", "days", "enough", "confirmed", "replan_count", "saved_path"):
        print(f"   {k} = {final.get(k)!r}")
    print(f"   analysis = {len(final.get('analysis',''))} 字 | plan = {len(final.get('plan',''))} 字")


if __name__ == "__main__":
    main()
