"""
vector_store.py — 极简持久化向量库
基于 embedding_utils（智谱 embedding-2）+ JSON 文件持久化
接口兼容 ChromaDB 风格：get_collection / add_docs / query_docs / col.delete / col.get
"""

import os
import json
import uuid

from embedding_utils import embed, cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "vector_store_data")


class Collection:
    """一个命名集合：文本 + 元数据 + 向量，持久化为 JSON"""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self._data = {"ids": [], "documents": [], "metadatas": [], "vectors": []}
        self._load()

    # ── 持久化 ──────────────────────────────────────

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    # ── 写入 ────────────────────────────────────────

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None):
        """添加文档（自动向量化并持久化）"""
        if metadatas is None:
            metadatas = [{}] * len(ids)

        for i, doc in enumerate(documents):
            did = ids[i] if i < len(ids) else f"doc_{uuid.uuid4().hex[:8]}"
            # 重复 id 则覆盖
            if did in self._data["ids"]:
                idx = self._data["ids"].index(did)
                self._data["documents"][idx] = doc
                self._data["metadatas"][idx] = metadatas[i] or {}
                self._data["vectors"][idx] = embed(doc)
            else:
                self._data["ids"].append(did)
                self._data["documents"].append(doc)
                self._data["metadatas"].append(metadatas[i] or {})
                self._data["vectors"].append(embed(doc))
        self._save()

    def delete(self, ids: list[str]) -> list[str]:
        """按 id 删除，返回实际删掉的 id 列表（不存在的 id 不会报错）"""
        deleted = []
        for did in ids:
            if did in self._data["ids"]:
                idx = self._data["ids"].index(did)
                for key in ("ids", "documents", "metadatas", "vectors"):
                    del self._data[key][idx]
                deleted.append(did)
        self._save()
        return deleted

    # ── 查询 ────────────────────────────────────────

    def query(self, query_text: str, n: int = 3, where: dict | None = None) -> list[dict]:
        """
        语义检索：返回 [{id, document, metadata, distance}, ...]
        distance = 1 - 相似度（越小越相关）
        :param where: 元数据过滤条件，如 {"tag": "研究"} 或 {"subject": "数学", "year": 2023}
        """
        if not self._data["ids"]:
            return []

        q_vec = embed(query_text)
        scored = []
        for i in range(len(self._data["ids"])):
            # 元数据过滤
            if where:
                meta = self._data["metadatas"][i] or {}
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue
            sim = cosine_similarity(q_vec, self._data["vectors"][i])
            scored.append((1 - sim, i))  # 距离
        scored.sort(key=lambda x: x[0])

        results = []
        for dist, idx in scored[:n]:
            results.append({
                "id": self._data["ids"][idx],
                "document": self._data["documents"][idx],
                "metadata": self._data["metadatas"][idx],
                "distance": dist,
            })
        return results

    def docs_for_bm25(self, where: dict | None = None) -> list[dict]:
        """取出（过滤后的）全部文档供 BM25 使用（不调 embedding API）"""
        out = []
        for i, did in enumerate(self._data["ids"]):
            meta = self._data["metadatas"][i] or {}
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            out.append({"id": did, "document": self._data["documents"][i], "metadata": meta})
        return out

    def tag_stats(self) -> dict:
        """加分项：统计每个 tag 下的文档数"""
        stats = {}
        for meta in self._data["metadatas"]:
            tag = (meta or {}).get("tag") or "未分类"
            stats[tag] = stats.get(tag, 0) + 1
        return stats

    def get(self) -> dict:
        """返回全部数据 {ids, documents, metadatas}"""
        return {
            "ids": list(self._data["ids"]),
            "documents": list(self._data["documents"]),
            "metadatas": [dict(m) for m in self._data["metadatas"]],
        }

    def count(self) -> int:
        return len(self._data["ids"])


# ── 顶层接口（兼容原代码调用方式） ─────────────────────────

_COLLECTIONS: dict[str, Collection] = {}


def get_collection(name: str) -> Collection:
    """获取（或创建）命名集合，数据持久化在 vector_store_data/ 下"""
    if name not in _COLLECTIONS:
        path = os.path.join(DATA_DIR, f"{name}.json")
        _COLLECTIONS[name] = Collection(name, path)
    return _COLLECTIONS[name]


def add_docs(col: Collection, documents: list[str], metadatas: list[dict] | None = None,
             ids: list[str] | None = None) -> list[str]:
    """添加文档，返回 id 列表"""
    if ids is None:
        ids = [f"doc_{col.count() + i}" for i in range(len(documents))]
    col.add(ids, documents, metadatas)
    return ids


def query_docs(col: Collection, query: str, n: int = 3, where: dict | None = None) -> list[dict]:
    """语义检索，返回 [{id, document, metadata, distance}, ...]，where 支持元数据过滤"""
    return col.query(query, n, where=where)


if __name__ == "__main__":
    # 自测
    col = get_collection("demo")
    col.add(["d1", "d2"], ["你好世界", "机器学习"], [{"title": "a"}, {"title": "b"}])
    print(col.query("深度学习", 2))
