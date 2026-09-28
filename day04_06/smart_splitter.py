"""
smart_splitter.py — Day 06 作业：智能文本切分器
加载文档 → 固定/递归切分（可调参）→ 对比效果 → 切分后 RAG 问答
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_ROOT)
sys.path.insert(0, os.path.join(_PROJ, "day01_03"))
sys.path.insert(0, _ROOT)

from hello_glm import ask_glm, save_output
from doc_loader import load_any
from knowledge_qa import ask_with_context
from text_splitter import split_fixed, split_recursive, compare_splits, show_chunks, stats


def pick_relevant_chunks(chunks: list[str], question: str, top_n: int = 3) -> list[str]:
    """
    加分项：手动关键词匹配挑出相关 chunk（为向量检索做铺垫）
    简单做法：统计问题里的字在 chunk 中的出现次数，取最高的 top_n
    """
    q_chars = set(question) - set("的了是在和与及或什么怎样如何吗呢？?。，, ")
    scored = []
    for c in chunks:
        score = sum(1 for ch in q_chars if ch in c)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_n] if s > 0] or chunks[:top_n]


def main():
    print("╔══════════════════════════════════════╗")
    print("║     ✂️  智能文本切分器               ║")
    print("╚══════════════════════════════════════╝")

    paths_input = input("\n请输入文档路径（多个用逗号分隔）:\n> ").strip()
    if not paths_input:
        print("  ✗ 路径不能为空")
        return

    texts = []
    for p in [x.strip() for x in paths_input.split(",") if x.strip()]:
        try:
            doc = load_any(p)
            texts.append(doc["text"])
            print(f"✅ 已加载 {os.path.basename(p)}，共 {doc['chars']} 字")
        except Exception as e:
            print(f"  ✗ 跳过 {p}: {e}")

    if not texts:
        return
    text = "\n\n".join(texts)
    print(f"\n📊 合计 {len(text)} 字")

    current_chunks: list[str] = []

    while True:
        print("""
请选择功能：
  1. 固定长度切分（可调 chunk_size 和 overlap）
  2. 递归字符切分（推荐，可调参数）
  3. 对比两种切分效果
  4. 用切分结果做 RAG 问答
  5. 查看当前切分统计
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！")
            break

        elif choice in ("1", "2"):
            try:
                cs = int(input("chunk_size（回车默认 500）: ").strip() or "500")
                ov = int(input("overlap（回车默认 50）: ").strip() or "50")
            except ValueError:
                print("  ✗ 请输入数字")
                continue
            try:
                current_chunks = (split_fixed if choice == "1" else split_recursive)(text, cs, ov)
            except ValueError as e:
                print(f"  ✗ {e}")
                continue
            print(f"\n{'固定长度' if choice == '1' else '递归字符'}切分结果:")
            print(show_chunks(current_chunks))
            st = stats(current_chunks)
            print(f"\n统计: {st['count']} 块 | 平均 {st['avg_len']} 字 | "
                  f"最长 {st['max_len']} | 最短 {st['min_len']} | 总 {st['total_chars']} 字")
            if input("\n每块保存为单独文件？(y/n): ").strip().lower() == "y":
                outdir = os.path.join(_ROOT, "chunks_out")
                os.makedirs(outdir, exist_ok=True)
                for i, c in enumerate(current_chunks, 1):
                    with open(os.path.join(outdir, f"chunk_{i:03d}.txt"), "w", encoding="utf-8") as f:
                        f.write(c)
                print(f"  ✓ 已保存 {len(current_chunks)} 个文件到 {outdir}")

        elif choice == "3":
            try:
                cs = int(input("chunk_size（回车默认 500）: ").strip() or "500")
                ov = int(input("overlap（回车默认 50）: ").strip() or "50")
            except ValueError:
                cs, ov = 500, 50
            print()
            print(compare_splits(text, cs, ov))
            current_chunks = split_recursive(text, cs, ov)   # 默认用递归结果供后续问答

        elif choice == "4":
            if not current_chunks:
                print("  ⚠ 请先执行切分（功能1/2/3）")
                continue
            q = input("你的问题: ").strip()
            if not q:
                continue
            # 只把相关 chunk 塞进 prompt
            relevant = pick_relevant_chunks(current_chunks, q, top_n=3)
            print(f"\n（从 {len(current_chunks)} 个块中筛出 {len(relevant)} 个相关块，"
                  f"共 {sum(len(c) for c in relevant)} 字）\n")
            print(ask_with_context("\n\n".join(relevant), q))

        elif choice == "5":
            if not current_chunks:
                print("  ⚠ 还没有切分结果")
                continue
            st = stats(current_chunks)
            print(f"\n当前切分统计: {st}")

        else:
            print("  ✗ 请输入 0-5")


if __name__ == "__main__":
    main()
