"""
hybrid_search.py — Day 09 作业核心：混合检索（向量 + BM25）
- tokenize()：中文 bigram + 英文单词（零依赖分词）
- bm25_search()：关键词稀疏检索
- hybrid_search()：向量 + BM25 分数融合 + 元数据过滤
- filter_search()：仅按元数据过滤
"""

import os
import sys
import math
import re
from collections import Counter

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from vector_store import Collection   # noqa: E402


# ── 分词（中文 bigram + 英文单词 + 数字） ──────────────

def tokenize(text: str) -> list[str]:
    """
    轻量分词，无需 jieba：
    - 中文：切成字符 bigram（"知识图谱" → 知识/识图/图谱）
    - 英文/数字：按单词
    """
    tokens = []
    # 英文单词和数字
    for w in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        tokens.append(w)
    # 中文字符 bigram
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chinese)):
        if i + 1 < len(chinese):
            tokens.append(chinese[i] + chinese[i + 1])
        else:
            tokens.append(chinese[i])
    return tokens


# ── BM25 实现 ────────────────────────────────────────────

def bm25_search(query: str, docs: list[dict], top_k: int = 10,
                k1: float = 1.5, b: float = 0.75) -> list[tuple[dict, float]]:
    """
    BM25 关键词检索
    :param docs: [{"id","document","metadata"}, ...]
    :return: [(doc, score), ...] 按分数降序
    """
    if not docs:
        return []

    # 建索引
    doc_tokens = [tokenize(d["document"]) for d in docs]
    doc_lens = [len(t) for t in doc_tokens]
    avg_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0

    # 文档频率
    df = Counter()
    for tokens in doc_tokens:
        for t in set(tokens):
            df[t] += 1

    N = len(docs)
    q_tokens = tokenize(query)
    scores = []

    for i, tokens in enumerate(doc_tokens):
        tf = Counter(tokens)
        score = 0.0
        for q in q_tokens:
            if q not in tf:
                continue
            # IDF
            n_q = df.get(q, 0)
            idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))
            # TF 归一化
            denom = tf[q] + k1 * (1 - b + b * doc_lens[i] / (avg_len or 1))
            score += idf * (tf[q] * (k1 + 1)) / denom
        scores.append((docs[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── 混合检索 ─────────────────────────────────────────────

def _normalize(scores: list[float]) -> list[float]:
    """min-max 归一化到 [0,1]"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_search(query: str, col: Collection, top_k: int = 3,
                  alpha: float = 0.5, where: dict | None = None,
                  threshold: float | None = None) -> list[dict]:
    """
    混合检索：alpha * 向量分 + (1-alpha) * BM25 分
    :param col: 向量库 Collection
    :param alpha: 向量权重（0=纯BM25, 1=纯向量）
    :param where: 元数据过滤 {"subject": "数学"} / {"year": 2023}
    :param threshold: 融合分阈值，低于此值的过滤掉
    :return: [{"id","document","metadata","vector_score","bm25_score","score"}, ...]
    """
    # 1. 候选（元数据过滤后）
    candidates = col.docs_for_bm25(where)
    if not candidates:
        return []

    # 2. 向量召回（用 Collection.query 拿距离 → 转相似度）
    vec_results = col.query(query, n=len(candidates), where=where)
    vec_map = {r["id"]: 1 - r["distance"] for r in vec_results}   # 距离→相似度

    # 3. BM25 打分
    bm25_raw = bm25_search(query, candidates, top_k=len(candidates))
    bm25_map = {d["id"]: s for d, s in bm25_raw}

    # 4. 归一化后融合
    ids = [c["id"] for c in candidates]
    vec_norm = _normalize([vec_map.get(i, 0) for i in ids])
    bm25_norm = _normalize([bm25_map.get(i, 0) for i in ids])
    vec_norm_map = dict(zip(ids, vec_norm))
    bm25_norm_map = dict(zip(ids, bm25_norm))

    results = []
    for c in candidates:
        vs = vec_norm_map.get(c["id"], 0)
        bs = bm25_norm_map.get(c["id"], 0)
        fused = alpha * vs + (1 - alpha) * bs
        results.append({
            "id": c["id"],
            "document": c["document"],
            "metadata": c["metadata"],
            "vector_score": round(vs, 4),
            "bm25_score": round(bs, 4),
            "score": round(fused, 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    if threshold is not None:
        results = [r for r in results if r["score"] >= threshold]
    return results[:top_k]


def filter_search(query: str, col: Collection, where: dict,
                  top_k: int = 3, alpha: float = 0.5) -> list[dict]:
    """仅按元数据过滤 + 混合检索（Day09 功能1/2）"""
    return hybrid_search(query, col, top_k=top_k, alpha=alpha, where=where)


def compare(query: str, col: Collection, top_k: int = 3,
            alpha: float = 0.5, where: dict | None = None) -> str:
    """对比模式：纯向量 vs 混合检索 并排输出"""
    vec_results = col.query(query, n=top_k, where=where)
    hyb_results = hybrid_search(query, col, top_k=top_k, alpha=alpha, where=where)

    W = 30   # 左右列宽
    lines = ["═" * 72,
             f"查询: {query}",
             f"参数: alpha={alpha}（向量权重）, top_k={top_k}" + (f", 过滤={where}" if where else ""),
             "═" * 72,
             f"{'【纯向量检索】':<{W}}{'【混合检索 (向量+BM25)】'}",
             "-" * 72]

    for i in range(max(len(vec_results), len(hyb_results))):
        if i < len(vec_results):
            r = vec_results[i]
            left = f"{i+1}.({1-r['distance']:.2f}) {r['document'][:W-12]}"
        else:
            left = ""
        if i < len(hyb_results):
            r = hyb_results[i]
            right = f"{i+1}.({r['score']:.2f}) {r['document'][:W-12]}"
        else:
            right = ""
        lines.append(f"{left:<{W}}{right}")

    lines += ["",
              "说明：混合检索融合语义相似度（向量）+ 关键词匹配（BM25），",
              "      分数 = alpha×向量分 + (1-alpha)×BM25分（均已归一化）。"]
    return "\n".join(lines)


if __name__ == "__main__":
    # 自测分词和 BM25
    print("分词示例:", tokenize("知识图谱嵌入Knowledge Graph")[:10])
    docs = [
        {"id": "1", "document": "Kafka 分区提供并行处理能力", "metadata": {}},
        {"id": "2", "document": "Kafka 副本机制保证数据可靠性", "metadata": {}},
        {"id": "3", "document": "知识图谱用三元组表示实体关系", "metadata": {}},
    ]
    for d, s in bm25_search("Kafka 分区", docs):
        print(f"  {s:.3f}  {d['document']}")
