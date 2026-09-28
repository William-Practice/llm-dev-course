"""
db_assistant.py — Day 14 作业：自然语言数据库助手
重建库 / 看结构 / 自然语言查询（生成SQL+执行+解读）/ 查询历史
"""

import os
import sys
import csv

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from build_db import build_sample_db, DB_DEFAULT
from text2sql import get_schema, ask_database

DB = DB_DEFAULT
HISTORY: list[tuple[str, str]] = []   # 最近 5 次查询


def rebuild_db():
    """功能1：重建数据库"""
    build_sample_db(DB)
    print("✅ 数据库已重建（6 个学生 × 3 科 + 班级信息）")


def show_schema():
    """功能2：查看表结构"""
    print("\n📋 数据库表结构：")
    print(get_schema(DB))


def query():
    """功能3：自然语言查询 + 加分项：导出 CSV"""
    if not os.path.exists(DB):
        print("  ⚠ 数据库不存在，请先执行功能1重建")
        return
    q = input("❓ 你的问题: ").strip()
    if not q:
        return

    print("\n生成 SQL 并执行...\n")
    r = ask_database(q, DB)

    print(f"🔧 生成的 SQL:\n   {r['sql']}")
    if r["error"]:
        print(f"✗ {r['answer']}")
        return

    # 打印表格
    if r["cols"]:
        print("\n📊 查询结果:")
        print("   " + " | ".join(r["cols"]))
        print("   " + "-" * 40)
        for row in r["rows"][:20]:
            print("   " + " | ".join(str(x) for x in row))
        if len(r["rows"]) > 20:
            print(f"   ... 共 {len(r['rows'])} 行")

    print(f"\n💬 {r['answer']}")

    # 加分项：导出 CSV
    if r["rows"] and input("\n导出结果为 CSV？(y/n): ").strip().lower() == "y":
        path = os.path.join(_ROOT, "query_result.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(r["cols"])
            w.writerows(r["rows"])
        print(f"  ✓ 已导出 {path}")

    # 记录历史（保留最近 5 次）
    HISTORY.append((q, f"{r['sql']} → {r['answer'][:60]}"))
    if len(HISTORY) > 5:
        HISTORY.pop(0)


def show_history():
    """功能4：查询历史"""
    if not HISTORY:
        print("暂无查询历史")
        return
    print(f"\n📜 最近 {len(HISTORY)} 次查询:")
    for i, (q, a) in enumerate(HISTORY, 1):
        print(f"\n[{i}] 问: {q}")
        print(f"    答: {a[:100]}...")


def main():
    print("╔══════════════════════════════════════╗")
    print("║   🤖 自然语言数据库助手              ║")
    print("╚══════════════════════════════════════╝")
    print("（用中文查学生成绩库，AI 自动生成 SQL 并解读）")

    while True:
        print("""
请选择功能：
  1. 重建数据库（6 学生 × 3 科）
  2. 查看表结构
  3. 自然语言查询
  4. 查询历史（最近 5 次）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            rebuild_db()
        elif choice == "2":
            show_schema()
        elif choice == "3":
            query()
        elif choice == "4":
            show_history()
        else:
            print("  ✗ 请输入 0-4")


if __name__ == "__main__":
    main()
