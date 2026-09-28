"""
mcp_agent.py — Day 20：MCP 客户端 + Agent
connect_mcp_server()  启动文件 MCP Server 并加载其工具
ask_with_mcp()        让 Agent 自主调用 MCP 工具完成任务
"""

import os
import sys
import asyncio

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# 新版 langchain 的 create_agent（兼容旧路径）
try:
    from langchain.agents import create_agent as create_react_agent
except ImportError:                                        # pragma: no cover
    from langgraph.prebuilt import create_react_agent

_ROOT = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

PYTHON = sys.executable

SYSTEM_PROMPT = """你是「MCP 文件助手」，通过 MCP 协议连接了一个文件服务，可以操作 workspace 目录。

可用工具（都由 MCP Server 提供）：
- list_directory：列出目录内容
- read_file：读文件
- write_file：写文件（自动建父目录）
- create_directory：建目录
- move_file：移动文件
- file_size：统计大小

工作原则：
1. 一切文件操作都要通过工具完成，**不要凭空猜测文件内容或文件名**
2. 工具返回的结果是 MCP 格式的列表（如 [{'type': 'text', 'text': '实际内容'}]），
   其中 text 字段才是真实内容——请只依据 text 字段作答，绝不编造
3. 整理文件夹这类多步任务要串联多个工具（先 list_directory 看有什么，
   再 create_directory 建分类目录，再 move_file 逐个归档）
4. 按扩展名分类：图片(jpg/png/gif/webp) → images/，文档(txt/md/pdf/docx) → docs/，
   其他 → others/
5. 完成后用中文简洁汇报做了什么"""


def _flatten_mcp_tools(mcp_tools: list):
    """
    把 MCP 工具包装成返回纯文本的 LangChain 工具。
    MCP 适配器返回 [{'type':'text','text':...}] 这种嵌套格式，小模型解析不了会幻觉，
    这里拍平成纯字符串。
    """
    from langchain_core.tools import StructuredTool

    wrapped = []
    for t in mcp_tools:
        async def _run(_t=t, **kwargs):
            res = await _t.ainvoke(kwargs)
            if isinstance(res, list):
                parts = []
                for item in res:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", item)))
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(res)

        wrapped.append(StructuredTool.from_function(
            coroutine=_run,
            name=t.name,
            description=t.description,
            args_schema=t.args_schema,
        ))
    return wrapped


def connect_mcp_server():
    """连接文件 MCP Server，返回 client"""
    client = MultiServerMCPClient({
        "files": {
            "command": PYTHON,
            "args": [os.path.join(_ROOT, "file_server.py")],
            "transport": "stdio",
        }
    })
    return client


def build_agent(tools):
    """用 MCP 工具构建 Agent"""
    llm = ChatOpenAI(
        model="glm-4-flash",
        api_key=API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.2,
    )
    try:
        return create_react_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    except TypeError:
        return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


# ── 全局（启动时初始化一次） ─────────────────────────────

_AGENT = None
_TOOLS = None


async def _init():
    global _AGENT, _TOOLS
    if _AGENT is None:
        client = connect_mcp_server()
        raw_tools = await client.get_tools()
        _TOOLS = _flatten_mcp_tools(raw_tools)      # 拍平成纯文本返回
        _AGENT = build_agent(_TOOLS)
    return _AGENT, _TOOLS


def init_sync():
    """同步初始化（供普通脚本调用）"""
    return asyncio.run(_init())


async def _ask(question: str, verbose: bool = True) -> str:
    agent, _ = await _init()
    result = await agent.ainvoke({"messages": [("user", question)]})
    messages = result["messages"]
    if verbose:
        for m in messages:
            for call in getattr(m, "tool_calls", None) or []:
                args = ", ".join(f"{k}={str(v)[:35]}" for k, v in call.get("args", {}).items())
                print(f"  🔧 {call.get('name')}({args})")
            if m.__class__.__name__ == "ToolMessage":
                print(f"  👀 {str(m.content)[:110]}")
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage" and m.content:
            return m.content
    return "(无回答)"


def ask_with_mcp(question: str, verbose: bool = True) -> str:
    """让 Agent 用 MCP 工具回答问题（同步接口）"""
    return asyncio.run(_ask(question, verbose))


if __name__ == "__main__":
    tools = init_sync()[1]
    print(f"已从 MCP Server 加载 {len(tools)} 个工具:")
    for t in tools:
        print(f"  - {t.name}")
    print()
    q = " ".join(sys.argv[1:]) or "列出 workspace 目录下有什么"
    print(ask_with_mcp(q))
