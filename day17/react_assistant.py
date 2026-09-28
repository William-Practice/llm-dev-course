"""
react_assistant.py — Day 17 作业：手写 ReAct Agent（不用任何框架）
ReAct 循环：Thought → Action → Observation，直到 Final Answer
正则解析 Action / Final Answer，max_steps=6 防死循环
"""

import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_ROOT)
for p in (os.path.join(_PROJ, "day01_03"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from hello_glm import ask_glm
from tools import tools_description, run_tool, TOOLS

MAX_STEPS = 6                      # 防死循环
LOG_FILE = os.path.join(_ROOT, "agent_log.txt")

REACT_PROMPT = """你是一个能使用工具的智能助手。请用 ReAct 方式解决问题：先思考，需要时调用工具，最后给出答案。

可用工具：
{tools}

回答格式（严格遵守，每次只输出一个步骤）：

Thought: 你的思考过程
Action: 工具名
Action Input: 工具参数

工具执行后会返回 Observation，然后你继续 Thought/Action。

当你已经能回答用户问题时，输出：
Thought: 我已经收集到足够信息
Final Answer: 最终答案（中文，简洁）

【正确示例】
用户问题：今天几号？25*4+30 等于多少？
Thought: 需要先获取今天的日期
Action: get_today
Action Input: none
Observation: 2026-09-27
Thought: 现在需要计算 25*4+30，必须用 calculate 工具
Action: calculate
Action Input: 25*4+30
Observation: 25*4+30 = 130
Thought: 已收集到全部信息
Final Answer: 今天是 2026-09-27，25*4+30 = 130

【重要规则】
1. Action 必须是上面列出的工具名之一
2. Action Input 是工具需要的参数（不需要参数的工具填 none）
3. 每次只输出一个 Thought + Action，或一个 Thought + Final Answer
4. 凡是算术计算，必须调用 calculate 拿到真实结果后才能写 Final Answer，
   严禁自己心算（心算容易出错）
5. 涉及日期、知识库、文件内容，也必须调工具获取，不要凭记忆

用户问题：{question}"""


def parse_action(text: str) -> dict:
    """
    正则解析 AI 输出（兼容中英文冒号）
    只取第一个 Action 块（模型常违反"一次一步"规则，一次吐多步）
    """
    result = {}

    # 找第一个 Action
    m_action = re.search(r"Action\s*[：:]\s*([a-zA-Z_][a-zA-Z0-9_]*)", text)

    if m_action:
        # 截取从该 Action 到下一个 Action 之间的内容（只处理第一个块）
        rest = text[m_action.end():]
        nxt = re.search(r"\n\s*(Thought|Action)\s*[：:]", rest)
        block = rest[:nxt.start()] if nxt else rest

        m_input = re.search(r"Action\s*Input\s*[：:]\s*(.+)", block)
        result["action"] = m_action.group(1).strip()
        if m_input:
            result["action_input"] = m_input.group(1).strip()

        # Thought 在本块之前
        pre = text[:m_action.start()]
        thoughts = re.findall(r"Thought\s*[：:]\s*(.+?)$", pre, re.S | re.M)
        if thoughts:
            result["thought"] = thoughts[-1].strip()

        # 优先执行 Action（即使同时输出了 Final Answer，也先把工具跑完）
        return result

    # 没有 Action → 看是否给出 Final Answer
    m = re.search(r"Final\s*Answer\s*[：:]\s*(.+)", text, re.S | re.I)
    if m:
        result["final"] = m.group(1).strip()
        thoughts = re.findall(r"Thought\s*[：:]\s*(.+?)$", text[:m.start()], re.S | re.M)
        if thoughts:
            result["thought"] = thoughts[-1].strip()
        return result

    # 都没有 → 记录原始思考
    thoughts = re.findall(r"Thought\s*[：:]\s*(.+)", text, re.S)
    if thoughts:
        result["thought"] = thoughts[-1].strip()
    return result


def _extract_math(question: str) -> list[str]:
    """从问题里提取算术表达式（如 88*12+56）"""
    return re.findall(r"\d+(?:\s*[+\-*/]\s*\d+)+", question)


def react_agent(question: str, verbose: bool = True) -> str:
    """
    ReAct 主循环：最多 MAX_STEPS 步
    :return: 最终答案
    """
    prompt = REACT_PROMPT.format(tools=tools_description(), question=question)
    scratchpad = ""        # 只记录「实际执行过的步骤」，格式规范
    used_tools = []
    seen_calls = set()     # 防重复调用

    for step in range(1, MAX_STEPS + 1):
        full_prompt = prompt + "\n\n【已执行步骤】\n" + scratchpad if scratchpad else prompt
        output = ask_glm(full_prompt, "你是一个严谨的 ReAct 智能体。只输出一个步骤。")
        parsed = parse_action(output)

        if verbose and parsed.get("thought"):
            print(f"  💭 Thought: {parsed['thought'][:120]}")

        # 已完成 → 校验后返回
        if "final" in parsed:
            exprs = _extract_math(question)
            if exprs and "calculate" not in used_tools:
                if verbose:
                    print(f"  ⚠ 算术未经工具验证，强制调用 calculate")
                scratchpad += (f"\n[系统] 你的回答包含未经工具验证的算术结果。"
                               f"请立即输出 Action: calculate / Action Input: {exprs[0]}，"
                               f"不要自己心算。\n")
                continue
            if verbose:
                print(f"  ✅ Final Answer（共 {step} 步）")
            prefix = f"（本次使用了工具: {', '.join(used_tools)}）\n" if used_tools else ""
            return prefix + parsed["final"]

        # 执行工具
        if "action" in parsed:
            action = parsed["action"]
            arg = parsed.get("action_input", "")

            # 防重复：同样调用已执行过
            call_key = (action, arg)
            if call_key in seen_calls:
                if verbose:
                    print(f"  ⚠ {action}({arg}) 已执行过，提示模型基于已有信息作答")
                scratchpad += (f"\n[系统] {action}({arg}) 已经调用过，结果在上面的 Observation 里。"
                               f"请不要再重复调用，直接基于已有信息输出 Final Answer。\n")
                continue
            seen_calls.add(call_key)

            if verbose:
                print(f"  🔧 Action: {action}({arg})")
            observation = run_tool(action, arg)
            used_tools.append(action)
            if verbose:
                print(f"  👀 Observation: {observation[:120]}")

            # 只记录规范化的这一步
            thought = parsed.get("thought", "")
            scratchpad += (f"Thought: {thought}\nAction: {action}\n"
                           f"Action Input: {arg}\nObservation: {observation}\n\n")
            _log(f"[Step {step}] Action={action} Input={arg} → {observation[:200]}")
        else:
            # 格式不对：提示模型按格式输出
            scratchpad += f"\n[系统] 你的输出格式不对。请严格按 Thought/Action/Action Input 格式，或输出 Final Answer。\n"

    _log(f"[达到 MAX_STEPS={MAX_STEPS}]")
    return (f"（已达到最大步数 {MAX_STEPS}，未能得出结论）\n"
            f"已执行步骤:\n{scratchpad[-600:]}")


def _log(text: str):
    """加分项：Thought 记录到日志文件"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n" + "-" * 50 + "\n")


def react_assistant():
    """主循环：连续接受用户提问"""
    print("╔══════════════════════════════════════╗")
    print("║      🤖 ReAct 小助手                 ║")
    print("╚══════════════════════════════════════╝")
    print("我能使用工具：查日期、算数学、搜知识库、读文件")
    print("输入 quit 退出\n")

    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("quit", "exit", "q", "退出"):
            print("👋 再见")
            break
        if not q:
            continue

        print("\n思考中...")
        answer = react_agent(q)
        print(f"\n🤖 最终答案: {answer}\n")


if __name__ == "__main__":
    # 支持命令行直接问一句
    if len(sys.argv) > 1:
        print(react_agent(" ".join(sys.argv[1:])))
    else:
        react_assistant()
