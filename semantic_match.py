"""
semantic_match.py — 语义匹配工具
用户问题 vs 候选题目：向量化 + 余弦相似度，找出最匹配的题
复用 embedding_utils（search / search_with_vectors）
"""

import os
from pathlib import Path
from embedding_utils import embed, search, search_with_vectors

PROJECT_DIR = os.path.dirname(__file__)
QBANK_DIR = os.path.join(PROJECT_DIR, "qbank_split")  # 单题文件目录


# ── 候选加载 ──────────────────────────────────────────────

def load_questions(qbank_dir: str = QBANK_DIR) -> list[dict]:
    """
    加载题库目录下所有单题文件
    :return: [{"file": "q01.txt", "text": "题目全文"}, ...]
    """
    questions = []
    if not os.path.isdir(qbank_dir):
        return questions
    for f in sorted(Path(qbank_dir).glob("*.txt")):
        text = f.read_text(encoding="utf-8").strip()
        if text:
            questions.append({"file": f.name, "text": text})
    return questions


# ── 核心匹配逻辑（复用 embedding_utils） ──────────────────

def match(query: str, candidates: list[dict], top_n: int = 3) -> list[tuple[dict, float]]:
    """
    语义匹配：复用 embedding_utils.search（向量化 + 余弦 + 排序）
    :param query: 用户问题
    :param candidates: [{"file":..., "text":...}, ...]
    :param top_n: 返回条数
    :return: [(候选dict, 相似度0-1), ...] 按分数降序
    """
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    results = search(query, texts, top_n=top_n)  # [(text, score), ...]

    text_to_cand = {c["text"]: c for c in candidates}
    return [(text_to_cand[text], score) for text, score in results]


def best_match(query: str, candidates: list[dict]) -> tuple[dict, float] | None:
    """返回最匹配的一个（无候选时返回 None）"""
    results = match(query, candidates, top_n=1)
    return results[0] if results else None


def match_with_vectors(query_vec: list[float], candidates: list[dict], top_n: int = 3) -> list[tuple[dict, float]]:
    """
    向量版：候选向量已算好时复用，避免重复调用 embedding API
    复用 embedding_utils.search_with_vectors
    :param query_vec: 用户问题向量
    :param candidates: [{"file":..., "text":..., "vec": [...]}, ...]
    """
    cands = [(c["text"], c["vec"]) for c in candidates if "vec" in c]
    if not cands:
        return []

    results = search_with_vectors(query_vec, cands, top_n=top_n)  # [(text, score), ...]

    text_to_cand = {c["text"]: c for c in candidates}
    return [(text_to_cand[text], score) for text, score in results]


def precompute_vectors(candidates: list[dict]) -> list[dict]:
    """为候选批量计算向量（一次性开销，之后可复用 match_with_vectors）"""
    for cand in candidates:
        if "vec" not in cand:
            cand["vec"] = embed(cand["text"])
    return candidates


# ── CLI 演示 ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    questions = load_questions()
    print(f"题库加载: {len(questions)} 道题\n")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "什么是死锁？"

    print(f"用户问题: {query}\n")
    print("─" * 50)

    print("向量化候选中...")
    precompute_vectors(questions)

    print(f"匹配 Top-3：\n")
    for rank, (cand, score) in enumerate(match_with_vectors(embed(query), questions, top_n=3), 1):
        print(f"  #{rank}  相似度 {score:.4f}  ({cand['file']})")
        first_line = cand["text"].split("\n")[0]
        print(f"      {first_line}")
        print()
