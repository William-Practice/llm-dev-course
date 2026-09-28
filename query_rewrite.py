"""
query_rewrite.py — RAG Query 重写优化
四种策略：multi_query / hyde / stepback / decompose
重写后多路检索 + 合并去重，复用 embedding_utils
"""

import os
import json
import urllib.request
import urllib.error

from embedding_utils import embed, search_with_vectors

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def glm_chat(prompt: str, temperature: float = 0.3) -> str:
    """直接 HTTP POST JSON 调用 GLM"""
    body = json.dumps({
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        CHAT_URL, data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ── 四种重写策略 ──────────────────────────────────────────

def _clean_lines(reply: str, n: int) -> list[str]:
    """清理 GLM 输出：去空行、去行首编号/横线"""
    lines = []
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去掉 "1. " "1、" "- " "• " 等前缀
        import re
        line = re.sub(r"^[\d一二三四五六七八九十]+[.、)）]\s*", "", line)
        line = re.sub(r"^[-•*]\s*", "", line)
        if line:
            lines.append(line)
    return lines[:n]


def rewrite_multi_query(question: str, n: int = 3) -> list[str]:
    """Multi-Query：生成 n 个不同表述/角度的查询"""
    prompt = f"""请把下面的问题改写成 {n} 个不同的查询，要求：
1. 每个查询从不同角度/表述方式出发，便于检索到不同文档
2. 可以补充同义词、换说法、拆角度
3. 每行一个查询，不要编号，不要其他内容

原问题：{question}"""
    reply = glm_chat(prompt, temperature=0.5).strip()
    return _clean_lines(reply, n)


def rewrite_hyde(question: str) -> str:
    """HyDE：生成一个假设性回答文档（陈述句），用其向量检索"""
    prompt = f"""请针对下面的问题，写一段假设性的回答文档（80-150字，陈述句风格）。
要求：紧扣问题的主题和关键词，不要偏离主题，不要编造与主题无关的术语。
这段文档不需要非常准确，但语言风格要像一篇学术论文的段落。

问题：{question}

只输出回答内容，不要其他文字。"""
    return glm_chat(prompt, temperature=0.7).strip()


def rewrite_stepback(question: str) -> str:
    """Step-back：把具体问题抽象成更一般的提问"""
    prompt = f"""请把下面的具体问题抽象成一个更一般、更宽泛的问题，便于检索背景知识。
只输出抽象后的问题，不要其他内容。

原问题：{question}"""
    return glm_chat(prompt, temperature=0.3).strip()


def rewrite_decompose(question: str) -> list[str]:
    """Decompose：把复杂问题拆成多个独立子问题"""
    prompt = f"""下面的问题可能包含多个方面。请把它拆解成若干个独立的子问题，每个子问题可以单独检索。
每行一个子问题，不要编号，不要其他内容。如果问题足够简单，只输出原问题。

问题：{question}"""
    reply = glm_chat(prompt, temperature=0.3).strip()
    return _clean_lines(reply, 10)


def rewrite(question: str, style: str = "multi_query", n: int = 3) -> list[str]:
    """
    统一入口：按策略重写查询
    :return: 查询列表（hyde 返回单元素列表）
    """
    style = style.lower()
    if style == "multi_query":
        return rewrite_multi_query(question, n)
    elif style == "hyde":
        return [rewrite_hyde(question)]
    elif style == "stepback":
        return [rewrite_stepback(question)]
    elif style == "decompose":
        return rewrite_decompose(question)
    else:
        raise ValueError(f"未知策略: {style}，可选 multi_query/hyde/stepback/decompose")


# ── 重写后多路检索 + 合并 ─────────────────────────────────

def multi_route_search(question: str, index: list[dict], top_k: int = 3,
                       style: str = "multi_query", n: int = 3) -> list[dict]:
    """
    重写 → 多路检索 → 合并去重 → 返回 top_k
    :param index: [{id, text, vec}, ...]（minirag 索引格式）
    :return: [{id, text, vec, score}, ...] 按最高分降序
    """
    queries = rewrite(question, style, n)
    queries.insert(0, question)  # 原始问题也参与检索

    candidates = [(item["text"], item["vec"]) for item in index]
    merged: dict[str, dict] = {}

    for q in queries:
        q_vec = embed(q)
        results = search_with_vectors(q_vec, candidates, top_n=top_k)
        for text, score in results:
            # 同一文本取最高分
            if text not in merged or score > merged[text]["score"]:
                merged[text] = {"score": score, "text": text}

    # 回查原始 index 条目
    text_to_item = {item["text"]: item for item in index}
    ranked = []
    for text, info in merged.items():
        item = dict(text_to_item[text])
        item["score"] = info["score"]
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def compare(question: str, index: list[dict], top_k: int = 3,
            styles: list[str] | None = None) -> dict:
    """
    对比实验：原始检索 vs 各策略重写检索
    :return: {style: [(score, text_preview), ...], ...}
    """
    from embedding_utils import search

    if styles is None:
        styles = ["multi_query", "hyde", "stepback", "decompose"]

    result = {}

    # 原始检索（无重写）
    texts = [item["text"] for item in index]
    raw = search(question, texts, top_n=top_k)
    result["raw"] = [(round(s, 4), t[:60].replace("\n", " ")) for t, s in raw]

    # 各策略
    for style in styles:
        try:
            ranked = multi_route_search(question, index, top_k, style=style)
            result[style] = [(round(item["score"], 4), item["text"][:60].replace("\n", " "))
                             for item in ranked]
        except Exception as e:
            result[style] = [f"错误: {e}"]

    return result


if __name__ == "__main__":
    # 自测：四种策略各重写一次
    q = "论文中SCMT方法是怎么减少突变体数量的？"
    print(f"原问题: {q}\n")

    print("── Multi-Query ──")
    for i, rq in enumerate(rewrite_multi_query(q, 3), 1):
        print(f"  {i}. {rq}")

    print("\n── HyDE ──")
    print(" ", rewrite_hyde(q)[:120], "...")

    print("\n── Step-back ──")
    print(" ", rewrite_stepback(q))

    print("\n── Decompose ──")
    for i, rq in enumerate(rewrite_decompose(q), 1):
        print(f"  {i}. {rq}")
