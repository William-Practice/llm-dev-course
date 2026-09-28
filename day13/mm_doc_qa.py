"""
mm_doc_qa.py — Day 13 作业：多模态文档问答系统
加载带图 PDF（文字 + GLM-4V 图片描述一起入库）→ 问答（标注来源）→ 查看知识库
"""

import os
import sys
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mm_rag import load_pdf, ask_question

KB: list[dict] = []    # 全局知识库


def load_document():
    """功能1：加载 PDF 建库"""
    global KB
    path = input("PDF 路径: ").strip()
    if not os.path.exists(path):
        print("  ✗ 文件不存在")
        return
    full = input("是否整页渲染描述（捕获矢量图表，较慢）？(y/n，默认 n): ").strip().lower() == "y"
    print("正在加载（图片会调 GLM-4V 描述，首次较慢，结果会缓存）...")
    KB = load_pdf(path, full_page=full)
    n_text = sum(1 for k in KB if k["type"] == "text")
    n_img = sum(1 for k in KB if k["type"] == "image")
    n_page = sum(1 for k in KB if k["type"] == "page")
    print(f"✅ 知识库就绪：{n_text} 文字片段 + {n_img} 图片描述 + {n_page} 整页描述")


def ask():
    """功能2：提问（答案标注来源）"""
    if not KB:
        print("  ⚠ 请先加载文档（功能1）")
        return
    q = input("❓ 你的问题: ").strip()
    if not q:
        return
    print("\n检索中...\n")
    print(ask_question(KB, q))


def show_kb():
    """功能3：查看知识库内容"""
    if not KB:
        print("  ⚠ 知识库为空，请先加载文档")
        return
    print(f"\n◎ 知识库内容（共 {len(KB)} 条）")
    print("─" * 56)
    for i, item in enumerate(KB, 1):
        label = "📄 文字" if item["type"] == "text" else "🖼️  图片"
        print(f"[{i}] {label} | {item['source']}")
        preview = item["content"].replace("\n", " ")[:90]
        print(f"    {preview}{'...' if len(item['content']) > 90 else ''}")
        print("─" * 56)


def open_image():
    """加分项：用系统看图器打开某张图片"""
    if not KB:
        return
    n = input("图片编号（见功能3列表）: ").strip()
    try:
        item = KB[int(n) - 1]
    except (ValueError, IndexError):
        print("  ✗ 编号无效")
        return
    if item.get("image_path"):
        subprocess.Popen(["xdg-open", item["image_path"]])
        print(f"  ✓ 已打开 {item['image_path']}")
    else:
        print("  ✗ 该条目不是图片")


def main():
    print("╔══════════════════════════════════════╗")
    print("║    🤖 多模态文档问答系统             ║")
    print("╚══════════════════════════════════════╝")
    print("（支持带图表的 PDF：文字和图片都能被检索问答）")

    while True:
        print("""
请选择功能：
  1. 加载 PDF 文档（文字 + 图片一起入库）
  2. 提问问答（支持问图片里的信息）
  3. 查看知识库内容
  4. 打开某张图片（加分项）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            load_document()
        elif choice == "2":
            ask()
        elif choice == "3":
            show_kb()
        elif choice == "4":
            open_image()
        else:
            print("  ✗ 请输入 0-4")


if __name__ == "__main__":
    main()
