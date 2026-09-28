"""
file_server.py — Day 20：文件操作 MCP Server（stdio）
提供 6 个工具：列目录 / 读文件 / 写文件 / 建目录 / 移动文件 / 文件大小
安全：所有路径限定在 WORKSPACE 目录内（防目录穿越）
"""

import os
import shutil
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# 允许操作的根目录（所有路径限制在这里面）
WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

mcp = FastMCP("file-server")


def _safe(path: str) -> Path:
    """把路径解析到 WORKSPACE 内，拒绝目录穿越"""
    p = (WORKSPACE / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if not str(p).startswith(str(WORKSPACE.resolve())):
        raise ValueError(f"⛔ 越权访问被拒绝（只允许操作 {WORKSPACE}）: {path}")
    return p


# ── MCP 工具 ─────────────────────────────────────────────

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """列出指定目录下的文件和子目录（含类型和大小）。
    参数 path：相对 workspace 的目录路径，根目录用 "."。"""
    p = _safe(path)
    if not p.exists():
        return f"目录不存在: {path}"
    if not p.is_dir():
        return f"不是目录: {path}"
    items = []
    for item in sorted(p.iterdir()):
        if item.is_dir():
            items.append(f"[目录] {item.name}/")
        else:
            items.append(f"[文件] {item.name}  ({item.stat().st_size} 字节)")
    if not items:
        return f"{path} 是空目录"
    return f"{path} 下共 {len(items)} 项:\n" + "\n".join(items)


@mcp.tool()
def read_file(path: str) -> str:
    """读取文本文件内容。
    参数 path：相对 workspace 的文件路径。"""
    p = _safe(path)
    if not p.exists():
        return f"文件不存在: {path}"
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"读取失败: {e}"
    if len(content) > 3000:
        return f"[{path} 前 3000 字，共 {len(content)} 字]\n{content[:3000]}"
    return f"[{path}]\n{content}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """把内容写入文本文件（会创建父目录，覆盖已存在文件）。
    参数 path：相对 workspace 的文件路径，如 notes/idea.txt
    参数 content：要写入的文本内容。"""
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字）"


@mcp.tool()
def create_directory(path: str) -> str:
    """创建目录（已存在则不报错）。
    参数 path：相对 workspace 的目录名，如 images。"""
    p = _safe(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"已创建目录 {path}"


@mcp.tool()
def move_file(source: str, target: str) -> str:
    """把文件移动到另一个位置（目标目录会自动创建）。
    参数 source：源文件路径（相对 workspace）
    参数 target：目标路径（相对 workspace），如 images/photo.jpg"""
    s = _safe(source)
    t = _safe(target)
    if not s.exists():
        return f"源文件不存在: {source}"
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(t))
    return f"已移动 {source} → {target}"


@mcp.tool()
def search_files(keyword: str, path: str = ".") -> str:
    """递归搜索文件名包含关键词的文件（加分项：搜索文件）。
    参数 keyword：文件名关键词（大小写不敏感）
    参数 path：搜索起始目录（相对 workspace，默认根目录）"""
    p = _safe(path)
    if not p.exists():
        return f"路径不存在: {path}"
    kw = keyword.lower()
    hits = []
    for root, _, files in os.walk(p):
        for f in files:
            if kw in f.lower():
                rel = os.path.relpath(os.path.join(root, f), _safe("."))
                hits.append(rel)
    if not hits:
        return f"没有找到文件名包含「{keyword}」的文件"
    return f"找到 {len(hits)} 个匹配「{keyword}」的文件:\n" + "\n".join(f"- {h}" for h in sorted(hits))


@mcp.tool()
def file_size(path: str) -> str:
    """统计指定目录/文件的大小（加分项：文件夹大小统计）。
    参数 path：相对 workspace 的路径。"""
    p = _safe(path)
    if not p.exists():
        return f"路径不存在: {path}"
    if p.is_file():
        return f"{path}: {p.stat().st_size} 字节"
    total = 0
    count = 0
    for root, _, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
            count += 1
    return f"{path}/: {count} 个文件，共 {total} 字节（{total/1024:.1f} KB）"


if __name__ == "__main__":
    mcp.run()      # stdio 传输
