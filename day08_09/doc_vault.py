"""
doc_vault.py — Day 08 作业：个人文档向量库
功能：搜索 / 查看 / 添加 / 删除 / 批量导入 / tag 过滤 / tag 统计
复用 vector_store.py（get_collection / add_docs / query_docs）
"""

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from vector_store import get_collection, add_docs, query_docs


def fmt_meta(meta: dict | None) -> str:
    """固定按 title、tag 顺序显示元数据"""
    meta = meta or {}
    return f"title={meta.get('title', '')} tag={meta.get('tag', '')}"


# ── 四个核心功能 ────────────────────────────────────────

def search_docs(col, query: str, n: int = 3, where: dict | None = None):
    """功能1：搜索文档（可加 tag 过滤 —— 加分项）"""
    print(f"\n◎ 查询: {query}" + (f"（过滤: {where}）" if where else ""))
    results = query_docs(col, query, n, where=where)
    if not results:
        print("  （无匹配结果）")
        return
    for i, r in enumerate(results, 1):
        print(f"  #{i} [{fmt_meta(r['metadata'])}] 距离 {r['distance']:.3f}")
        print(f"     {r['document'][:100]}")
        print(f"     (id: {r['id']})")


def list_all(col):
    """功能2：查看所有文档"""
    data = col.get()
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    print(f"\n◎ 文档列表（共 {len(ids)} 条）")
    print("─" * 46)
    if not ids:
        print("（暂无文档）")
        print("─" * 46)
        return
    for i, (did, text, meta) in enumerate(zip(ids, docs, metas), 1):
        meta = meta or {}
        title = meta.get("title") or "（无标题）"
        tag = meta.get("tag") or "未分类"
        preview = (text or "").replace("\n", " ")[:80]
        print(f"  [{i}] {title}")
        print(f"      id: {did} | tag: {tag}")
        print(f"      内容: {preview}")
        print("─" * 46)


def add_doc(col):
    """功能3：添加文档（带 metadata）"""
    title = input("标题: ").strip()
    content = input("内容: ").strip()
    tag = input("标签: ").strip() or "未分类"
    if not content:
        print("  ✗ 内容不能为空")
        return
    new_id = f"doc_{col.count()}"
    add_docs(col, [content], [{"title": title, "tag": tag}], [new_id])
    print(f"  ✓ 已添加 id={new_id}")


def delete_doc(col):
    """功能4：删除文档（先确认 id 存在）"""
    did = input("输入要删除的 id（如 doc_3）: ").strip()
    if not did:
        return
    existing = col.get()["ids"]
    if did not in existing:
        print(f"  ✗ id 不存在: {did}")
        return
    if input(f"确认删除 {did}？(y/n): ").strip().lower() != "y":
        print("  已取消")
        return
    deleted = col.delete(ids=[did])
    print(f"  ✓ 已删除 {deleted}")


def batch_import(col):
    """加分项：从 txt 文件批量导入（每行一条）"""
    path = input("txt 文件路径（每行一条文档）: ").strip()
    if not os.path.exists(path):
        print("  ✗ 文件不存在")
        return
    tag = input("统一标签（回车=未分类）: ").strip() or "未分类"
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        print("  ✗ 文件为空")
        return
    ids = [f"doc_{col.count() + i}" for i in range(len(lines))]
    metas = [{"title": ln[:20], "tag": tag} for ln in lines]
    add_docs(col, lines, metas, ids)
    print(f"  ✓ 已批量导入 {len(lines)} 条文档")


# ── CLI ─────────────────────────────────────────────────

def main():
    col = get_collection("doc vault")   # 启动时加载持久化库

    # tag 统计概览（加分项）
    stats = col.tag_stats()
    print("╔══════════════════════════════════════╗")
    print("║        📚 个人文档向量库             ║")
    print("╚══════════════════════════════════════╝")
    print(f"当前共 {col.count()} 条文档")
    if stats:
        print("标签概览: " + " | ".join(f"{k}:{v}" for k, v in stats.items()))

    while True:
        print("""
请选择功能：
  1. 搜索文档（找最像的 3 条）
  2. 查看所有文档
  3. 添加文档
  4. 删除文档
  5. 按标签过滤搜索
  6. 批量导入（txt 每行一条）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！（数据已持久化，下次启动还在）")
            break
        elif choice == "1":
            q = input("查询内容: ").strip()
            if q:
                search_docs(col, q)
        elif choice == "2":
            list_all(col)
        elif choice == "3":
            add_doc(col)
        elif choice == "4":
            delete_doc(col)
        elif choice == "5":
            tag = input("标签: ").strip()
            q = input("查询内容: ").strip()
            if tag and q:
                search_docs(col, q, where={"tag": tag})
        elif choice == "6":
            batch_import(col)
        else:
            print("  ✗ 请输入 0-6")


if __name__ == "__main__":
    main()
