"""
tools.py — Day 17：ReAct Agent 工具集
get_today   查今天日期
calculate   算数学表达式
search_kb   在知识库中搜索关键词
read_file   读取本地文本文件（加分项）
"""

import os
import re
from datetime import datetime

# 默认知识库文件（可换成任意 txt/md）
KB_FILE = "/home/wuruiyang/724/考研笔记.txt"


# ── 工具实现 ─────────────────────────────────────────────

def get_today(arg: str = "") -> str:
    """获取今天日期"""
    return datetime.now().strftime("%Y-%m-%d")


def days_until(target_date: str) -> str:
    """算距离某日期还有多少天（负数=已过去）"""
    target_date = target_date.strip().strip("\"'")
    # 容忍 2026年12月20日 / 2026-12-20 两种写法
    m = re.match(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", target_date)
    if not m:
        return f"日期格式无法识别: {target_date}（请用 2026-12-20 格式）"
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        target = datetime(y, mo, d).date()
        delta = (target - datetime.now().date()).days
        if delta >= 0:
            return f"距离 {target} 还有 {delta} 天（约 {delta/7:.1f} 周）"
        return f"距离 {target} 已经过去 {-delta} 天"
    except Exception as e:
        return f"日期解析失败: {e}"


def calculate(expression: str) -> str:
    """计算数学表达式（限制环境，防注入）"""
    # 只允许数字和常见运算符号
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)%]+", expression.strip()):
        return "计算失败：表达式含非法字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression.strip()} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


def search_kb(keyword: str) -> str:
    """在知识库文件中搜索关键词，返回包含关键词的段落"""
    keyword = keyword.strip().strip("\"'")
    if not os.path.exists(KB_FILE):
        return f"知识库文件不存在: {KB_FILE}"

    with open(KB_FILE, encoding="utf-8", errors="replace") as f:
        text = f.read()

    # 按段落找含关键词的内容
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    hits = [p for p in paragraphs if keyword in p]

    if not hits:
        # 退一步：按行找
        lines = [ln.strip() for ln in text.splitlines() if keyword in ln]
        hits = lines[:3]

    if not hits:
        return f"知识库中没有找到「{keyword}」相关内容"
    return "\n---\n".join(h[:300] for h in hits[:3])


def read_file(path: str) -> str:
    """读取本地文本文件（加分项）"""
    path = path.strip().strip("\"'")
    if not os.path.exists(path):
        return f"文件不存在: {path}"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return f"[{path} 前 800 字]\n{content[:800]}"
    except Exception as e:
        return f"读取失败: {e}"


# ── 工具注册（供 Agent 使用） ─────────────────────────────

TOOLS = {
    "get_today": {
        "fn": get_today,
        "desc": "获取今天的日期，无需参数",
    },
    "days_until": {
        "fn": days_until,
        "desc": "计算距离某个日期还有多少天（会自动换算周数）。参数为日期，如 2026-12-20 或 2026年12月20日",
    },
    "calculate": {
        "fn": calculate,
        "desc": "计算数学表达式，参数为表达式，如 3*7+2 或 (100-20)/4",
    },
    "search_kb": {
        "fn": search_kb,
        "desc": "在知识库中搜索关键词，返回相关段落。参数为关键词，如 考研政治",
    },
    "read_file": {
        "fn": read_file,
        "desc": "读取本地文本文件内容，参数为文件绝对路径",
    },
}


def tools_description() -> str:
    """生成工具列表描述（拼进 Prompt）"""
    lines = []
    for name, t in TOOLS.items():
        lines.append(f"- {name}: {t['desc']}")
    return "\n".join(lines)


def run_tool(name: str, arg: str) -> str:
    """执行工具（带错误处理）"""
    t = TOOLS.get(name.strip())
    if not t:
        return f"未知工具「{name}」，可用工具: {', '.join(TOOLS.keys())}"
    try:
        return str(t["fn"](arg))
    except Exception as e:
        return f"工具执行出错: {e}"


if __name__ == "__main__":
    print("工具列表:")
    print(tools_description())
    print("\n测试:")
    print(" ", get_today())
    print(" ", calculate("3*7+2"))
    print(" ", search_kb("考研政治")[:100])
    print(" ", calculate("__import__('os').system('ls')"))   # 安全测试
