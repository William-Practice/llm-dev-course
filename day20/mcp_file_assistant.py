"""
mcp_file_assistant.py — Day 20 作业：MCP 文件助手
4 个功能（+1 加分项）全部通过 MCP 工具实现，Agent 自主决定调用哪个工具
"""

import os
import sys
import asyncio

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp_agent import _init, _ask


# ── 4 个功能（都通过 ask_with_mcp 让 Agent 调 MCP 工具） ──

async def browse_folder(path: str):
    """功能1：浏览文件夹（列出内容并简介）"""
    print()
    print(await _ask(f"列出 {path} 目录下的内容，并简要说明每个文件/文件夹是做什么的。"))


async def read_and_summarize(path: str):
    """功能2：读文件并总结"""
    print()
    print(await _ask(f"读取文件 {path}，然后总结它的主要内容和用途。"))


async def organize_folder(path: str):
    """功能3：整理文件夹（AI 串联 list → create → move 多个工具）"""
    print("\n（整理是多步任务，Agent 会连续调用多个工具）")
    print(await _ask(
        f"帮我整理 {path} 目录：先列出所有文件，然后按类型归档——"
        f"图片(jpg/png/gif/webp)移到 images/，文档(txt/md/pdf/docx/csv)移到 docs/，"
        f"其他移到 others/。逐个移动，最后报告归档结果。"))


async def save_note(content: str):
    """功能4：记笔记（存成 txt 文件）"""
    filename = input("保存文件名（如 note.txt）: ").strip() or "note.txt"
    print()
    print(await _ask(f"把我的这段内容保存到 {filename}：\"{content}\"，保存后告诉我文件路径和字数。"))


async def search_files(keyword: str):
    """加分项：搜索文件"""
    print()
    print(await _ask(f"在 workspace 目录中找出名字包含「{keyword}」的文件，并列出它们。"))


async def main():
    print("╔══════════════════════════════════════╗")
    print("║      🤖 MCP 文件助手                 ║")
    print("╚══════════════════════════════════════╝")
    print("初始化：连接 MCP 文件 Server...")
    _, tools = await _init()
    print(f"✅ 已加载 {len(tools)} 个 MCP 工具：{', '.join(t.name for t in tools)}")
    print("（所有操作限定在 workspace/ 目录内）\n")

    while True:
        print("""
请选择功能：
  1. 浏览文件夹（列出内容+简介）
  2. 读文件（读取+总结）
  3. 整理文件夹（按类型自动归档）
  4. 记笔记（存成 txt）
  5. 搜索文件（按名字）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            p = input("文件夹路径（回车=根目录 .）: ").strip() or "."
            await browse_folder(p)
        elif choice == "2":
            p = input("文件路径（如 notes.txt）: ").strip()
            if p:
                await read_and_summarize(p)
        elif choice == "3":
            p = input("要整理的文件夹（回车=根目录 .）: ").strip() or "."
            await organize_folder(p)
        elif choice == "4":
            c = input("笔记内容: ").strip()
            if c:
                await save_note(c)
        elif choice == "5":
            k = input("文件名关键词: ").strip()
            if k:
                await search_files(k)
        else:
            print("  ✗ 请输入 0-5")


if __name__ == "__main__":
    asyncio.run(main())
