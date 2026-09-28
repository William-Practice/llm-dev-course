"""
rag_tool.py — Day 22：RAG 知识库查询工具
把项目里的 RAG 能力封装成 Agent 可调用的 LangChain 工具 ask_kb

流程：docs/ 下文档 → 切块 → embedding-2 向量 → 余弦相似度检索 → GLM 生成答案
索引缓存在 kb_index.json，文档没变就不重复调用 embedding 接口。
"""

import os
import sys
import json
import hashlib

from langchain_core.tools import tool

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_ROOT)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from embedding_utils import embed_batch, cosine_similarity

DOCS_DIR = os.path.join(_ROOT, "docs")
INDEX_PATH = os.path.join(_ROOT, "kb_index.json")

_CACHE = {"chunks": None, "vectors": None}


# ── 加载与切块 ───────────────────────────────────────────

def load_docs() -> list[dict]:
    """读取 docs/ 下所有 md/txt 文档"""
    docs = []
    for name in sorted(os.listdir(DOCS_DIR)):
        if name.endswith((".md", ".txt")):
            path = os.path.join(DOCS_DIR, name)
            with open(path, encoding="utf-8") as f:
                docs.append({"name": name, "text": f.read()})
    return docs


def split_chunks(text: str, source: str, max_len: int = 400) -> list[dict]:
    """按标题+段落切块，每块带上来源"""
    chunks = []
    current_title = ""
    buf = ""
    for line in text.splitlines():
        if line.startswith("#"):
            if buf.strip():
                chunks.append({"text": f"[{current_title}] {buf.strip()}", "source": source})
                buf = ""
            current_title = line.lstrip("# ").strip()
        else:
            buf += line + "\n"
            if len(buf) >= max_len:
                chunks.append({"text": f"[{current_title}] {buf.strip()}", "source": source})
                buf = ""
    if buf.strip():
        chunks.append({"text": f"[{current_title}] {buf.strip()}", "source": source})
    return chunks


def build_index(force: bool = False) -> tuple[list[dict], list[list[float]]]:
    """建（或加载）向量索引"""
    if _CACHE["chunks"] is not None and not force:
        return _CACHE["chunks"], _CACHE["vectors"]

    docs = load_docs()
    chunks = []
    for d in docs:
        chunks.extend(split_chunks(d["text"], d["name"]))

    # 用内容 hash 判断索引是否需要重建
    fp = hashlib.md5("".join(c["text"] for c in chunks).encode()).hexdigest()
    if not force and os.path.exists(INDEX_PATH):
        saved = json.load(open(INDEX_PATH, encoding="utf-8"))
        if saved.get("fingerprint") == fp:
            _CACHE["chunks"], _CACHE["vectors"] = saved["chunks"], saved["vectors"]
            return _CACHE["chunks"], _CACHE["vectors"]

    print(f"  📚 索引 {len(chunks)} 个片段（{len(docs)} 篇文档），调用 embedding...")
    vectors = embed_batch([c["text"] for c in chunks])
    json.dump({"fingerprint": fp, "chunks": chunks, "vectors": vectors},
              open(INDEX_PATH, "w", encoding="utf-8"), ensure_ascii=False)

    _CACHE["chunks"], _CACHE["vectors"] = chunks, vectors
    return chunks, vectors


# ── 检索 ────────────────────────────────────────────────

def search_kb(query: str, top_n: int = 3) -> list[tuple[dict, float]]:
    """向量检索，返回 [(片段, 相似度)]"""
    chunks, vectors = build_index()
    qv = embed_batch([query])[0]
    scored = [(chunks[i], cosine_similarity(qv, vectors[i])) for i in range(len(chunks))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ── Agent 工具 ───────────────────────────────────────────

@tool
def ask_kb(question: str) -> str:
    """查询本地学习知识库。知识库包含：大数据技术栈（Kafka/Flink/Spark/HDFS/湖仓）、
    软考软件设计师考点、考研政治复习要点。
    当用户问到这些领域的概念、考点、知识点时，用这个工具检索资料后再回答。
    参数 question：要查询的问题"""
    hits = search_kb(question, top_n=3)
    if not hits or hits[0][1] < 0.30:
        return "知识库中没有找到相关信息（检索相似度过低）"

    context = "\n\n".join(
        f"【来源 {h['source']}，相似度 {s:.3f}】\n{h['text']}" for h, s in hits
    )
    return (f"检索到以下知识库内容（请基于这些内容回答，不要编造）：\n\n{context}")


if __name__ == "__main__":
    chunks, vectors = build_index()
    print(f"\n索引完成：{len(chunks)} 个片段")
    q = sys.argv[1] if len(sys.argv) > 1 else "Kafka 的分区有什么作用"
    print(f"\n查询：{q}\n")
    for c, s in search_kb(q, top_n=3):
        print(f"[{s:.3f}] {c['source']}: {c['text'][:90]}...")
