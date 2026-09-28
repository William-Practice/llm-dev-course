"""
study_agent.py — Day 18 作业：学习助手 Agent（LangChain + LangGraph）
5 个 @tool 工具：查资料 / 记笔记 / 算天数 / 查日期 / 读文件
用 langgraph.prebuilt.create_react_agent 创建 Agent
"""

import os
import re
import sys
from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 新版 langchain 把 create_react_agent 移到了 langchain.agents（旧路径仍可用但会警告）
try:
    from langchain.agents import create_agent as create_react_agent   # langchain >= 1.0
except ImportError:                                                    # pragma: no cover
    from langgraph.prebuilt import create_react_agent

_ROOT = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(_ROOT, "notes")
KB_FILE = "/home/wuruiyang/724/考研笔记.txt"
API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

os.makedirs(NOTES_DIR, exist_ok=True)

# 工具间共享状态：缓存最近一次搜索结果（解决小模型传长参数被截断的问题）
_LAST_SEARCH: dict = {"content": ""}


# ── 5 个工具（@tool + 类型注解 + docstring） ──────────────

@tool
def get_today() -> str:
    """获取今天的日期。当需要知道当前日期、或要计算距某天还有多久时，先用这个工具。返回格式 YYYY-MM-DD。"""
    return datetime.now().strftime("%Y-%m-%d")


@tool
def days_until(target_date: str) -> str:
    """计算距离某个日期还有多少天（并换算成周）。
    当用户问"还有几天/几周"到某个日期时使用。
    参数 target_date：目标日期，格式 2026-12-20 或 2026年12月20日。"""
    m = re.match(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", target_date.strip())
    if not m:
        return f"日期格式无法识别: {target_date}（请用 2026-12-20）"
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    delta = (datetime(y, mo, d).date() - datetime.now().date()).days
    if delta >= 0:
        return f"距离 {y}-{mo:02d}-{d:02d} 还有 {delta} 天（约 {delta/7:.1f} 周）"
    return f"距离 {y}-{mo:02d}-{d:02d} 已经过去 {-delta} 天"


@tool
def search_kb(keyword: str) -> str:
    """在本地知识库（考研笔记等）中搜索关键词，返回相关段落。
    当用户想"查资料/查笔记/了解某知识点"时使用。
    参数 keyword：搜索关键词，如 考研政治（传核心名词即可，不必传整句）"""
    if not os.path.exists(KB_FILE):
        return f"知识库文件不存在: {KB_FILE}"
    with open(KB_FILE, encoding="utf-8", errors="replace") as f:
        text = f.read()

    # 生成候选关键词：原串 + 去疑问词 + 拆分（模型常把整句传进来）
    kws = set()
    raw = keyword.strip()
    kws.add(raw)
    cleaned = re.sub(r"(包括|有哪些|是什么|有什么|呢|吗|如何|怎么|请|帮我|查一下|查查|告诉|一下|的)", "", raw).strip()
    if len(cleaned) >= 2:
        kws.add(cleaned)
    for part in re.split(r"[\s，,、。？?！!：:；;]+", raw):
        if len(part) >= 2:
            kws.add(part)

    # 优先按段落匹配（返回内容更完整）
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    hits = [p for p in paragraphs if any(k in p for k in kws if k)]
    if hits:
        result = "\n\n".join(hits[:3])[:1200]
        _LAST_SEARCH["content"] = result
        return result

    # 退化为按行匹配
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits = [ln for ln in lines if any(k in ln for k in kws if k)]
    if not hits:
        return f"知识库中没有找到「{keyword}」相关内容"
    result = "\n".join(hits[:5])
    _LAST_SEARCH["content"] = result
    return result


@tool
def write_note(filename: str, content: str = "") -> str:
    """把内容保存成笔记文件（Markdown）。
    当用户说"记笔记/保存/写下来"时使用。
    参数 filename：文件名，如 Agent.md（会存到 notes/ 目录）
    参数 content：笔记内容（Markdown 格式）。如果是保存刚查到的资料，可以留空，
    系统会自动使用最近一次 search_kb 的完整结果。"""
    name = os.path.basename(filename.strip())
    if not name.endswith(".md"):
        name += ".md"
    path = os.path.join(NOTES_DIR, name)

    # 兜底：小模型传长参数时会截断 content，用最近一次搜索结果补全
    last = _LAST_SEARCH.get("content", "")
    if last and (not content or len(content) < len(last) * 0.6):
        content = last

    if not content:
        return "笔记内容为空，未保存"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已保存笔记到 {path}（{len(content)} 字）"


@tool
def read_file(filepath: str) -> str:
    """读取本地文本/Markdown 文件内容。
    当用户想"看一下之前的笔记/读某个文件"时使用。
    参数 filepath：文件路径（可为 notes/xxx.md 或绝对路径）。"""
    p = filepath.strip()
    if not os.path.isabs(p):
        p = os.path.join(NOTES_DIR, os.path.basename(p))
    if not os.path.exists(p):
        return f"文件不存在: {p}"
    with open(p, encoding="utf-8", errors="replace") as f:
        return f"[{p}]\n{f.read()[:1500]}"


TOOLS = [get_today, days_until, search_kb, write_note, read_file]

SYSTEM_PROMPT = """你是「学习助手 Agent」，帮用户完成学习相关任务。

你可以使用这些工具：
- get_today：查今天日期
- days_until：算距某日期还有几天/几周
- search_kb：在知识库中查资料
- write_note：把内容保存成笔记文件
- read_file：读取已有的笔记文件

工作原则：
1. 需要真实数据（日期、天数、资料、文件）时必须调用工具，不要凭记忆编造
2. 多步任务要连续调用多个工具（例如：先 search_kb 查资料，再 write_note 保存）
3. **调用 write_note 时，content 参数必须传入完整内容**（把 search_kb 查到的内容
   原样完整传过去，不要截断或只传开头）
4. 需要日期计算时，可以用 get_today 获取今天日期后，用 days_until 计算
5. 完成后用简洁中文总结你做了什么、结果是什么"""


def build_agent():
    """创建 Agent（ChatOpenAI 指向智谱 API）
    兼容新旧 API：新版用 system_prompt=，旧版（langgraph）用 prompt=
    """
    llm = ChatOpenAI(
        model="glm-4-flash",
        api_key=API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.2,
    )
    try:
        return create_react_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)   # langchain >= 1.0
    except TypeError:
        return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)         # langgraph 旧版


def ask(agent, question: str, verbose: bool = True) -> str:
    """调用 Agent 并展示工具调用过程"""
    result = agent.invoke({"messages": [("user", question)]})
    messages = result["messages"]

    if verbose:
        # 打印中间的工具调用
        for m in messages:
            for call in getattr(m, "tool_calls", None) or []:
                args = call.get("args", {})
                print(f"  🔧 {call.get('name')}({', '.join(f'{k}={str(v)[:40]}' for k,v in args.items())})")
            if m.__class__.__name__ == "ToolMessage":
                print(f"  👀 {str(m.content)[:100]}")

    # 最后一条 AI 消息 = 最终回答
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage" and m.content:
            return m.content
    return "(无回答)"


def main():
    print("╔══════════════════════════════════════╗")
    print("║      🎓 学习助手 Agent（LangChain）  ║")
    print("╚══════════════════════════════════════╝")
    print("能力：查资料 / 记笔记 / 定计划 / 算天数 / 读文件")
    print("（输入 quit 退出）\n")

    print("初始化 Agent...")
    agent = build_agent()
    print("就绪\n")

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
        print()
        answer = ask(agent, q)
        print(f"\n🤖 {answer}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent = build_agent()
        print(ask(agent, " ".join(sys.argv[1:])))
    else:
        main()
