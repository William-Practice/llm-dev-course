"""
doc_processor.py — Day 05 作业：多格式文档处理器
load_any() 自动识别格式（txt/pdf/docx/md）→ 总结（CoT）/ 问答（RAG）/ 出题（Few-shot）
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_ROOT)
sys.path.insert(0, os.path.join(_PROJ, "day01_03"))   # hello_glm
sys.path.insert(0, _ROOT)                              # doc_loader

from hello_glm import ask_glm, save_output
from doc_loader import load_any, describe
from knowledge_qa import ask_with_context


def summarize(context: str) -> str:
    """功能1：CoT 总结"""
    prompt = f"""请总结下面文档的核心内容。请一步一步思考：
1. 先列出文档涉及的主要知识点
2. 再说明每个知识点的要点
3. 最后一句话概括主旨

【文档内容】
{context}"""
    return ask_glm(prompt, "你是学习助手，擅长提炼文档核心。")


def quiz(context: str) -> str:
    """功能3：Few-shot 出题（题目+选项+答案）"""
    prompt = f"""根据下面文档出 3 道选择题，严格按示例格式：

示例：
题目：xxx？
A. xxx
B. xxx
C. xxx
D. xxx
答案：A

【文档内容】
{context}"""
    return ask_glm(prompt, "你是出题老师，格式统一。")


def mindmap(context: str) -> str:
    """加分项：生成 Markdown 思维导图"""
    prompt = f"""请根据下面文档生成思维导图，用 Markdown 层级列表表示：

# 文档主题
## 一级要点1
### 二级要点
## 一级要点2
...

【文档内容】
{context}"""
    return ask_glm(prompt, "你是知识整理专家。")


def main():
    print("╔══════════════════════════════════════╗")
    print("║     📦 多格式文档处理器              ║")
    print("╚══════════════════════════════════════╝")

    # 加分项：支持多文件
    paths_input = input("\n请输入文档路径（支持 txt/pdf/docx/md，多个用逗号分隔）:\n> ").strip()
    if not paths_input:
        print("  ✗ 路径不能为空")
        return

    paths = [p.strip() for p in paths_input.split(",") if p.strip()]
    contexts = []
    for p in paths:
        try:
            doc = load_any(p)
            contexts.append(f"【来自 {os.path.basename(p)}】\n{doc['text']}")
            print(f"✅ {describe(doc)}  ← {os.path.basename(p)}")
        except Exception as e:
            print(f"  ✗ 跳过 {p}: {e}")

    if not contexts:
        print("  ✗ 没有成功加载任何文档")
        return

    # 3 个功能共用加载好的文档，不重复加载
    context = "\n\n".join(contexts)
    total = sum(len(c) for c in contexts)
    print(f"\n📊 合计 {len(contexts)} 个文档，共 {total} 字")

    while True:
        print("""
请选择功能：
  1. 总结（CoT 梳理核心）
  2. 问答（只根据文档回答）
  3. 出题（3 道选择题）
  4. 思维导图（Markdown）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            print("\n总结中...\n")
            ans = summarize(context)
            print(ans)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【文档总结】\n{ans}")
        elif choice == "2":
            # 加分项：多轮问答
            print("（可连续提问，输入 q 返回菜单）")
            while True:
                q = input("\n你的问题: ").strip()
                if q.lower() in ("q", "quit", "exit"):
                    break
                if not q:
                    continue
                print("\n思考中...\n")
                print(ask_with_context(context, q))
        elif choice == "3":
            print("\n出题中...\n")
            ans = quiz(context)
            print(ans)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【练习题】\n{ans}")
        elif choice == "4":
            print("\n生成思维导图中...\n")
            print(mindmap(context))
        else:
            print("  ✗ 请输入 0-4")


if __name__ == "__main__":
    main()
