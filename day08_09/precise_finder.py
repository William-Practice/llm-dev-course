"""
precise_finder.py — Day 09 作业：精准题库搜索器
题库带 subject + year 双元数据 → 按科目/年份过滤 + 混合检索 + 对比模式
复用 hybrid_search.py 的 hybrid_search / filter_search / compare
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_ROOT)
for p in (_PROJ, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from vector_store import get_collection, add_docs
from hybrid_search import hybrid_search, filter_search, compare

COLLECTION = "question_bank"

# ── 题库（10 道，带 subject + year） ─────────────────────

QUESTIONS = [
    ("二叉树中度为0的结点数比度为2的结点数多几个？", {"subject": "数据结构", "year": 2023}),
    ("折半查找长度为10的有序表最多比较几次？", {"subject": "数据结构", "year": 2023}),
    ("冒泡排序的最坏时间复杂度是多少？", {"subject": "数据结构", "year": 2022}),
    ("图的邻接矩阵存储结构适合什么场景？", {"subject": "数据结构", "year": 2022}),
    ("进程调度算法的作用是什么？", {"subject": "操作系统", "year": 2023}),
    ("死锁产生的四个必要条件是什么？", {"subject": "操作系统", "year": 2022}),
    ("虚拟存储器的作用是什么？", {"subject": "操作系统", "year": 2021}),
    ("关系数据库中主键的作用是什么？", {"subject": "数据库", "year": 2023}),
    ("SQL 中用于分组的子句是哪个？", {"subject": "数据库", "year": 2022}),
    ("E-R 图中不属于基本要素的是什么？", {"subject": "数据库", "year": 2021}),
]


def load_bank(col):
    """把题库灌进向量库（若已有同数量则跳过）"""
    if col.count() >= len(QUESTIONS):
        print(f"✅ 题库已加载（{col.count()} 道题）")
        return
    ids = [f"q{i}" for i in range(len(QUESTIONS))]
    docs = [q[0] for q in QUESTIONS]
    metas = [dict(q[1], title=q[0][:16]) for q in QUESTIONS]
    add_docs(col, docs, metas, ids)
    print(f"✅ 题库已初始化（{len(QUESTIONS)} 道题，含 subject + year）")


# ── 三个功能 ────────────────────────────────────────────

def search_by_subject(col):
    """功能1：按科目过滤 + 混合检索"""
    subjects = sorted({m["subject"] for _, m in QUESTIONS})
    print(f"可选科目: {' / '.join(subjects)}")
    subj = input("科目: ").strip()
    if subj not in subjects:
        print(f"  ✗ 无效科目（可选: {', '.join(subjects)}）")
        return
    q = input("问题: ").strip()
    if not q:
        return
    results = filter_search(q, col, {"subject": subj}, top_k=3)
    print(f"\n◎ 【{subj}】{q}")
    for i, r in enumerate(results, 1):
        print(f"  #{i} 融合分 {r['score']:.3f}（向量 {r['vector_score']:.2f} + BM25 {r['bm25_score']:.2f}）")
        print(f"     科目: {r['metadata']['subject']} | 年份: {r['metadata']['year']}")
        print(f"     {r['document']}")


def search_by_year(col):
    """功能2：按年份过滤 + 混合检索"""
    years = sorted({m["year"] for _, m in QUESTIONS}, reverse=True)
    print(f"可选年份: {' / '.join(map(str, years))}")
    y = input("年份: ").strip()
    try:
        year = int(y)
    except ValueError:
        print("  ✗ 请输入数字年份")
        return
    if year not in years:
        print(f"  ✗ 无效年份（可选: {', '.join(map(str, years))}）")
        return
    q = input("问题: ").strip()
    if not q:
        return
    results = filter_search(q, col, {"year": year}, top_k=3)
    print(f"\n◎ 【{year} 年】{q}")
    for i, r in enumerate(results, 1):
        print(f"  #{i} 融合分 {r['score']:.3f}  [{r['metadata']['subject']}/{r['metadata']['year']}]")
        print(f"     {r['document']}")


def compare_mode(col):
    """功能3：纯向量 vs 混合检索 并排对比（可调 alpha）"""
    q = input("问题: ").strip()
    if not q:
        return
    a = input("alpha（向量权重 0-1，回车默认 0.5）: ").strip()
    try:
        alpha = float(a) if a else 0.5
    except ValueError:
        alpha = 0.5
    print()
    print(compare(q, col, top_k=3, alpha=alpha))


def threshold_mode(col):
    """加分项：只显示融合分 > 阈值的题"""
    q = input("问题: ").strip()
    if not q:
        return
    t = input("相似度阈值（如 0.5）: ").strip()
    try:
        th = float(t)
    except ValueError:
        th = 0.5
    results = hybrid_search(q, col, top_k=10, threshold=th)
    print(f"\n◎ 融合分 ≥ {th} 的结果（{len(results)} 条）:")
    for i, r in enumerate(results, 1):
        print(f"  #{i} {r['score']:.3f}  [{r['metadata']['subject']}/{r['metadata']['year']}] {r['document']}")


def double_filter(col):
    """加分项：科目 + 年份双重过滤"""
    subj = input("科目: ").strip()
    y = input("年份: ").strip()
    try:
        year = int(y)
    except ValueError:
        print("  ✗ 年份需为数字")
        return
    q = input("问题: ").strip()
    if not q:
        return
    results = hybrid_search(q, col, top_k=3, where={"subject": subj, "year": year})
    print(f"\n◎ 【{subj} / {year}】{q}")
    if not results:
        print("  （无匹配，请检查科目或年份）")
        return
    for i, r in enumerate(results, 1):
        print(f"  #{i} 融合分 {r['score']:.3f}  {r['document']}")


def main():
    col = get_collection(COLLECTION)
    print("╔══════════════════════════════════════╗")
    print("║      🎯 精准题库搜索器               ║")
    print("╚══════════════════════════════════════╝")
    load_bank(col)

    while True:
        print("""
请选择功能：
  1. 搜索（选科目 + 问题，混合检索 top3）
  2. 按年份搜索
  3. 对比模式（纯向量 vs 混合检索）
  4. 阈值过滤（只显示高分结果）
  5. 科目+年份双重过滤
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            search_by_subject(col)
        elif choice == "2":
            search_by_year(col)
        elif choice == "3":
            compare_mode(col)
        elif choice == "4":
            threshold_mode(col)
        elif choice == "5":
            double_filter(col)
        else:
            print("  ✗ 请输入 0-5")


if __name__ == "__main__":
    main()
