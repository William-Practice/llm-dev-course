"""
study_agent.py — Day 22 收官项目：智能学习助手
RAG 知识库工具 + 日期/笔记工具 + LangGraph ReAct Agent + 多轮对话 + 工具调用日志

Phase 3 收官：综合 Day17(ReAct) / Day18(自定义工具) / Day19(多步规划) /
             Phase2(RAG) / Day21(LangGraph) 所学
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

try:
    from langchain.agents import create_agent as create_react_agent
except ImportError:                                        # pragma: no cover
    from langgraph.prebuilt import create_react_agent

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rag_tool import ask_kb                    # RAG 工具
from study_tools import ALL_TOOLS              # 日期/笔记工具

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

SYSTEM_PROMPT = """你是「智能学习助手」，帮用户学习大数据、软考、考研相关知识。

你有这些工具：
- ask_kb：查本地知识库（大数据技术栈、软考考点、考研政治）。**凡涉及这些领域的概念/考点，
  必须先调 ask_kb 检索，基于检索结果回答，不要靠自己的记忆编造**
- get_today：获取今天日期（涉及"今天""还有几天"先调它）
- days_until：算倒计时（软考/考研/毕业可直接写名字）
- write_note：记笔记
- read_note：读回之前的笔记
- list_notes：列出所有笔记

工作原则：
1. 知识类问题先检索，不编造
2. **一个请求包含多个动作时，必须把每个动作都执行完才能给最终回答**
   例如"查一下 Kafka 然后记笔记" = ask_kb（查） + write_note（真正写文件）两步，
   两步都调完工具才能结束；只检索不写文件、或口头说"已保存"都算没完成
3. 说要记笔记就必须真的调用 write_note 工具，绝不凭空声称已保存
4. 回答简洁准确，用中文
"""


def build_agent():
    """构建带记忆的 ReAct Agent"""
    llm = ChatOpenAI(
        model="glm-4-flash",
        api_key=API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.1,
    )
    tools = [ask_kb] + ALL_TOOLS
    try:
        return create_react_agent(llm, tools, system_prompt=SYSTEM_PROMPT,
                                  checkpointer=MemorySaver())
    except TypeError:                                      # 旧版 LangGraph
        return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT,
                                  checkpointer=MemorySaver())


def print_tool_calls(messages):
    """打印本轮的每一步工具调用日志"""
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            args = ", ".join(f"{k}={str(v)[:45]}" for k, v in call.get("args", {}).items())
            print(f"  🔧 调用工具: {call.get('name')}({args})")
        if m.__class__.__name__ == "ToolMessage":
            content = str(m.content).replace("\n", " ")
            print(f"  👀 工具返回: {content[:120]}{'...' if len(content) > 120 else ''}")


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║   🤖 智能学习助手（RAG + Agent · Phase 3 收官）  ║")
    print("╚══════════════════════════════════════════════════╝")
    print("可问：大数据/软考/考研知识、日期倒计时、记笔记")
    print("（输入 q 退出）\n")

    agent = build_agent()
    config = {"configurable": {"thread_id": "study-session"}}   # 多轮对话记忆
    history_len = 0

    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue

        print()
        try:
            result = agent.invoke({"messages": [("user", q)]}, config=config)
            messages = result["messages"]
            print_tool_calls(messages[history_len:])
            history_len = len(messages)

            answer = _final_answer(messages)

            # 程序层兜底：Agent 只调了工具却没有最终回答 → 自动续跑，逼它做完
            for _ in range(2):
                if answer:
                    break
                print("  ↻ 检测到任务可能未完成，自动续跑...")
                result = agent.invoke(
                    {"messages": [("user", "请继续完成我的要求中还没做的步骤。"
                                             "如果全部完成，请给出最终回答。")]}, config=config)
                messages = result["messages"]
                print_tool_calls(messages[history_len:])
                history_len = len(messages)
                answer = _final_answer(messages)

            print(f"\n助手: {answer or '（未能生成回答，请换个说法再试）'}\n")
        except Exception as e:
            print(f"  ✗ 出错: {e}\n")


def _final_answer(messages) -> str:
    """取最后一条有内容、且不是纯工具调用的 AI 回答"""
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage" and m.content and not getattr(m, "tool_calls", None):
            return m.content
    return ""


if __name__ == "__main__":
    main()
