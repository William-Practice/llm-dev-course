"""
persist_demo.py — 个人文档向量库 CLI（持久化演示）
基于 vector_store + embedding_utils，数据落盘在 vector_store_data/
功能：搜索 / 查看 / 添加 / 删除
"""

from vector_store import get_collection, add_docs, query_docs

col = get_collection("doc vault")


def fmt_meta(meta: dict | None) -> str:
    """固定按 title、tag 顺序显示元数据"""
    meta = meta or {}
    return f"title={meta.get('title', '')} tag={meta.get('tag', '')}"


def search_docs(query: str, n: int = 3):
    """搜索文档并打印结果"""
    print(f"\n◎ 查询: {query}\n")
    for i, r in enumerate(query_docs(col, query, n), 1):
        print(f"  #{i} [{fmt_meta(r['metadata'])}][距离 {r['distance']:.3f}]")
        print(f"     {r['document'][:100]}")
        print(f"     (id: {r['id']})")


def list_all():
    """列出所有文档"""
    data = col.get()
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    print(f"\n◎ 文档列表（共 {len(ids)} 条）")
    print("─" * 40)
    if not ids:
        print("（暂无文档）")
        print("─" * 40)
        return

    for i, (did, text, meta) in enumerate(zip(ids, docs, metas), 1):
        meta = meta or {}
        title = meta.get("title") or "（无标题）"
        tag = meta.get("tag") or "未分类"
        preview = (text or "").replace("\n", " ")[:80]
        print(f"  [{i}] {title}")
        print(f"      id: {did}")
        print(f"      tag: {tag}")
        print(f"      内容: {preview}")
        print("─" * 40)


def add_doc():
    """交互式添加文档"""
    title = input("标题: ").strip()
    content = input("内容: ").strip()
    tag = input("标签: ").strip() or "未分类"

    if not content:
        print("  ✗ 内容不能为空")
        return

    new_id = f"doc_{col.count()}"
    add_docs(col, [content], [{"title": title, "tag": tag}], [new_id])
    print(f"  ✓ 已添加 id={new_id}")


def delete_doc():
    """交互式删除文档"""
    did = input("输入要删除的 id（如 doc_3）: ").strip()
    if did:
        col.delete(ids=[did])
        print(f"  ✓ 已删除 {did}")


def main():
    while True:
        print("\n· 个人文档向量库")
        print("  1.搜索  2.查看  3.添加  4.删除  0.退出")
        c = input("选择: ").strip()

        if c == "0":
            break
        elif c == "1":
            q = input("输入查询: ").strip()
            if q:
                search_docs(q)
        elif c == "2":
            list_all()
        elif c == "3":
            add_doc()
        elif c == "4":
            delete_doc()
        else:
            print("  ✗ 请输入 0-4")


if __name__ == "__main__":
    main()
