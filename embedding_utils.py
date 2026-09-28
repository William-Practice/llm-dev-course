"""
Embedding 工具集
文字 → 1024 维向量 | 余弦相似度 | Top-N 检索
纯 HTTP/JSON 调用智谱 embedding-2，返回 [(文本, 分数), ...]
"""

import os
import json
import math
import urllib.request
import urllib.error

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBED_MODEL = "embedding-2"
EMBED_DIM = 1024


# ── 向量化 ──────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """将一段文字转为 1024 维向量"""
    body = json.dumps({
        "model": EMBED_MODEL,
        "input": text,
    }).encode("utf-8")

    req = urllib.request.Request(
        EMBED_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["data"][0]["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """批量向量化"""
    return [embed(t) for t in texts]


# ── 相似度 ──────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度，归一化到 [0, 1]
    （原始 cos 值域 [-1, 1]，线性映射到 [0, 1]）
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"向量维度不一致: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    na = math.sqrt(sum(x * x for x in vec_a))
    nb = math.sqrt(sum(x * x for x in vec_b))
    if na == 0 or nb == 0:
        return 0.0

    cos = dot / (na * nb)
    # 映射 [-1, 1] → [0, 1]
    return (cos + 1.0) / 2.0


def similarity_between(text_a: str, text_b: str) -> float:
    """直接计算两段文字的相似度"""
    return cosine_similarity(embed(text_a), embed(text_b))


# ── Top-N 检索 ──────────────────────────────────────────

def search(query: str, candidates: list[str], top_n: int = 5) -> list[tuple[str, float]]:
    """
    在候选文本池中检索与查询最相似的 top_n 条
    :param query: 查询文字
    :param candidates: 候选文本列表
    :param top_n: 返回条数
    :return: [(文本, 相似度0-1), ...] 按分数降序
    """
    if not candidates:
        return []

    top_n = max(1, min(top_n, len(candidates)))

    q_vec = embed(query)
    c_vecs = embed_batch(candidates)

    scored = [
        (text, cosine_similarity(q_vec, c_vec))
        for text, c_vec in zip(candidates, c_vecs)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def search_with_vectors(query_vec: list[float], candidates: list[tuple[str, list[float]]], top_n: int = 5) -> list[tuple[str, float]]:
    """
    向量化版本：candidates 已含向量，避免重复调用 API
    :param query_vec: 查询向量
    :param candidates: [(文本, 向量), ...]
    :return: [(文本, 相似度0-1), ...]
    """
    scored = [
        (text, cosine_similarity(query_vec, vec))
        for text, vec in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ── 演示 ────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Embedding 模型: {EMBED_MODEL} (维度 {EMBED_DIM})\n")

    # 测试 1: 向量维度
    v = embed("知识图谱是研究热点")
    print(f"单段文字向量维度: {len(v)}")

    # 测试 2: 相似度
    s = similarity_between("机器学习", "深度学习")
    print(f"「机器学习」vs「深度学习」相似度: {s:.4f}")

    # 测试 3: Top-N 检索
    pool = [
        "谱聚类是一种基于图的聚类算法",
        "变异测试通过注入故障评估测试用例质量",
        "知识图谱用三元组表示实体关系",
        "软考中级软件设计师每年举行两次考试",
        "神经网络是深度学习的核心模型",
        "突变体减少技术降低变异测试成本",
    ]
    results = search("如何减少变异测试的开销", pool, top_n=2)
    print("\nTop-2 检索「如何减少变异测试的开销」:")
    for text, score in results:
        print(f"  {score:.4f}  {text}")
