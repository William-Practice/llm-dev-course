"""
study_tools.py — Day 22：日期 / 笔记工具集
Agent 可调用的自定义工具：日期计算 + 笔记读写（含加分项的读笔记工具）
"""

import os
import re
from datetime import date, datetime, timedelta

from langchain_core.tools import tool

_ROOT = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(_ROOT, "notes")
os.makedirs(NOTES_DIR, exist_ok=True)

# 常用日期（方便 Agent 直接算倒计时）
KNOWN_DATES = {
    "软考": "2026-11-07", "软件设计师": "2026-11-07",
    "考研": "2026-12-20", "毕业": "2028-06-10",
}


@tool
def get_today() -> str:
    """获取今天的日期（YYYY-MM-DD）和星期。任何涉及"今天""现在"的问题都要先调用这个工具。"""
    today = date.today()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return f"今天是 {today.isoformat()}，星期{weekdays[today.weekday()]}"


@tool
def days_until(target_date: str) -> str:
    """计算从今天到目标日期的天数（倒计时）。
    参数 target_date：目标日期，格式 YYYY-MM-DD，也可以直接写"软考""考研""毕业"。"""
    t = target_date.strip()
    if t in KNOWN_DATES:
        t = KNOWN_DATES[t]
    try:
        target = datetime.strptime(t, "%Y-%m-%d").date()
    except ValueError:
        return f"日期格式错误：{target_date}，请用 YYYY-MM-DD 格式"
    delta = (target - date.today()).days
    if delta < 0:
        return f"{t} 已经过去 {abs(delta)} 天了"
    return f"距离 {t} 还有 {delta} 天（约 {delta/7:.1f} 周，{delta/30:.1f} 个月）"


def _safe_title(title: str) -> str:
    """把标题变成安全的文件名"""
    name = re.sub(r"[^\w\u4e00-\u9fff-]", "_", title.strip())
    return name[:40] or "note"


@tool
def write_note(title: str, content: str) -> str:
    """把内容保存成 Markdown 笔记文件。
    当用户说"记笔记""记下来""保存"时用这个工具。
    参数 title：笔记标题（也作为文件名，如 "Kafka要点"）
    参数 content：笔记正文。**必须是你整理好的正文，不要直接粘贴工具返回的原始检索文本**
    （不要包含"相似度""来源""请基于这些内容回答"这类元信息）"""
    content = _clean_content(content)
    path = os.path.join(NOTES_DIR, f"{_safe_title(title)}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n> 记录时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{content}\n")
    return f"笔记已保存到 notes/{os.path.basename(path)}（{len(content)} 字）"


def _clean_content(content: str) -> str:
    """程序层兜底：去掉检索工具返回的元信息，避免脏数据进笔记"""
    lines = []
    for line in content.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        # 过滤检索提示语和来源标记行
        if s.startswith("检索到以下知识库内容"):
            continue
        if s.startswith("【来源") and "相似度" in s:
            continue
        # 去掉行内的【来源 xx，相似度 0.78】前缀
        s = re.sub(r"【来源[^】]*】", "", s).strip()
        if s:
            lines.append(s)
    result = "\n".join(lines).strip()
    return result or content


@tool
def read_note(title: str) -> str:
    """读取之前保存的笔记（加分项）。当用户问"我之前记的笔记""读一下我的笔记"时用。
    参数 title：笔记标题（支持模糊匹配）"""
    if not os.path.isdir(NOTES_DIR):
        return "还没有任何笔记"
    key = _safe_title(title)
    files = [f for f in os.listdir(NOTES_DIR) if f.endswith(".md")]
    matches = [f for f in files if key in f or f[:-3] in key or title in f]
    if not matches:
        return f"没有找到标题包含「{title}」的笔记。已有笔记：{', '.join(f[:-3] for f in files) or '（无）'}"
    content = open(os.path.join(NOTES_DIR, matches[0]), encoding="utf-8").read()
    return f"[{matches[0]}]\n{content[:1500]}"


@tool
def list_notes() -> str:
    """列出所有已保存的笔记（加分项）。"""
    if not os.path.isdir(NOTES_DIR):
        return "还没有任何笔记"
    files = sorted(f for f in os.listdir(NOTES_DIR) if f.endswith(".md"))
    if not files:
        return "还没有任何笔记"
    return f"共 {len(files)} 篇笔记：\n" + "\n".join(f"- {f[:-3]}" for f in files)


ALL_TOOLS = [get_today, days_until, write_note, read_note, list_notes]

if __name__ == "__main__":
    print(get_today.invoke({}))
    print(days_until.invoke({"target_date": "软考"}))
    print(days_until.invoke({"target_date": "考研"}))
    print(write_note.invoke({"title": "测试笔记", "content": "这是一条测试内容。"}))
    print(list_notes.invoke({}))
