"""
try_embedding.py — embedding_utils 演示脚本
1. 生成「我爱中国」的 1024 维向量，查看维度与前 5 个数值
2. 与另外 4 个句子做相似度对比
"""

import os
from embedding_utils import embed, cosine_similarity, similarity_between

if not os.getenv("ZHIPU_API_KEY"):
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")


def main():
    # ── 1. 基础向量 ──────────────────────────────────
    print("═" * 50)
    print("① 获取「我爱中国」的向量")
    print("═" * 50)

    text = "我爱中国"
    vec = embed(text)

    print(f"句子: {text}")
    print(f"向量维度: {len(vec)}")
    print(f"前 5 个数值: {[round(x, 4) for x in vec[:5]]}")
    print(f"后 5 个数值: {[round(x, 4) for x in vec[-5:]]}")

    # ── 2. 四句对比 ──────────────────────────────────
    print("\n" + "═" * 50)
    print("② 与其他 4 个句子的相似度对比")
    print("═" * 50)

    others = [
        "我爱北京天安门",
        "我是中国人，我热爱我的祖国",
        "今天天气不错，适合出门散步",
        "I love China",
    ]

    print(f"基准句: 「{text}」\n")
    for other in others:
        score = similarity_between(text, other)
        print(f"  {score:.4f}  「{other}」")

    # ── 3. 直观排序 ──────────────────────────────────
    print("\n" + "═" * 50)
    print("③ 按相似度降序排序")
    print("═" * 50)

    pairs = [(other, similarity_between(text, other)) for other in others]
    pairs.sort(key=lambda x: x[1], reverse=True)
    for i, (s, score) in enumerate(pairs, 1):
        print(f"  #{i}  {score:.4f}  「{s}」")


if __name__ == "__main__":
    main()
