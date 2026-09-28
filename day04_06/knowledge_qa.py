"""
knowledge_qa.py — Day 04 作业：AI 个人知识问答助手
输入 txt 文档 → 问答（RAG+边界约束）/ 出题（Few-shot）/ 总结（CoT）
"""

import os
import sys

# 复用 day01_03 的 ask_glm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "day01_03"))
from hello_glm import ask_glm, save_output


# === 核心 RAG 函数 ===

def prepare_context(path: str) -> str:
    """启动时读文档，3 个功能共用"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def ask_with_context(context: str, question: str) -> str:
    """RAG 问答：只根据文档回答，没有就说没有（防幻觉）"""
    prompt = f"""请严格根据下面的文档内容回答问题。

规则：
1. 只使用文档中的信息回答，不要编造
2. 如果文档中找不到答案，就回答"文档中没有相关信息"
3. 回答时引用文档中的关键语句

【文档内容】
{context}

【问题】
{question}"""
    return ask_glm(prompt, "你是严谨的文档问答助手，只依据给定资料作答。")


# === 三个功能 ===

def qa(context: str, question: str) -> str:
    """功能1：RAG 问答 + 边界约束"""
    return ask_with_context(context, question)


def gen_quiz(context: str) -> str:
    """功能2：出 3 道选择题（Few-shot 保证格式）"""
    prompt = f"""请根据下面的文档内容出 3 道选择题。严格按照示例格式输出。

示例格式：
题目：Kafka 分区的作用是什么？
A. 提供容错
B. 实现并行处理
C. 存储元数据
D. 管理消费者
答案：B

现在请根据文档出 3 道题：

【文档内容】
{context}"""
    return ask_glm(prompt, "你是出题老师，题目基于文档、格式统一。")


def summarize(context: str) -> str:
    """功能3：CoT 总结（先梳理要点再总结）"""
    prompt = f"""请总结下面文档的核心内容。请一步一步思考：
1. 先列出文档涉及的主要知识点（3-5 个）
2. 再分别说明每个知识点的要点
3. 最后用一句话概括文档主旨

【文档内容】
{context}"""
    return ask_glm(prompt, "你是学习助手，擅长提炼文档核心内容。")


# === CLI ===

def main():
    print("╔══════════════════════════════════════╗")
    print("║     🤖 AI 个人知识问答助手           ║")
    print("╚══════════════════════════════════════╝")

    path = input("\n请输入文档路径: ").strip() or "/home/wuruiyang/724/考研笔记.txt"
    try:
        context = prepare_context(path)
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return

    print(f"✅ 文档已加载（共 {len(context)} 字）")

    while True:
        print("""
请选择功能：
  1. 问答（只根据文档回答）
  2. 出题（生成 3 道选择题）
  3. 总结（梳理核心内容）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            q = input("你的问题: ").strip()
            if not q:
                print("  ✗ 问题不能为空")
                continue
            print("\n思考中...\n")
            ans = qa(context, q)
            print(ans)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【问答】{q}\n{ans}")
        elif choice == "2":
            print("\n出题中...\n")
            ans = gen_quiz(context)
            print(ans)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【练习题】\n{ans}")
        elif choice == "3":
            print("\n总结中...\n")
            ans = summarize(context)
            print(ans)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【文档总结】\n{ans}")
        else:
            print("  ✗ 请输入 0-3")


if __name__ == "__main__":
    main()
