"""
text2sql.py — Day 14 作业核心：Text-to-SQL
get_schema()    提取数据库表结构
text_to_sql()   自然语言 → SQL（AI 生成）
execute_sql()   安全过滤后执行（拦截写操作）
ask_database()  完整流程：生成 → 执行 → 失败修正 → 解读
"""

import os
import re
import json
import sqlite3
import urllib.request

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4-flash"

# 危险 SQL 关键字（只允许只读查询）
FORBIDDEN = re.compile(
    r"\b(drop|delete|update|insert|alter|truncate|create|replace|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


def _ask(prompt: str, temperature: float = 0.1) -> str:
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature}).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


# ── Schema ──────────────────────────────────────────────

def get_schema(db_path: str) -> str:
    """提取表结构（建表语句 + 示例数据），供 AI 理解数据库"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

    parts = []
    for t in tables:
        ddl = cur.execute("SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()[0]
        sample = cur.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
        cols = [d[0] for d in cur.execute(f"SELECT * FROM {t} LIMIT 1").description]
        parts.append(f"表 {t}:\n{ddl}\n列: {', '.join(cols)}\n示例数据: {sample}")
    conn.close()
    return "\n\n".join(parts)


# ── Text to SQL ─────────────────────────────────────────

def text_to_sql(question: str, schema: str, error_hint: str = "") -> str:
    """把自然语言问题翻译成 SQL"""
    extra = f"\n\n上一次生成的 SQL 出错，错误信息：{error_hint}\n请修正后重新生成。" if error_hint else ""
    prompt = f"""你是 SQLite 专家。请根据数据库结构和用户问题，生成一条 SQL 查询语句。

规则：
1. 只输出 SQL 语句，不要任何解释、不要 markdown 代码块
2. 只能使用 SELECT 查询，禁止写操作
3. 需要时使用 JOIN 关联多表
4. 中文表名/字段用引号（如需要）

【数据库结构】
{schema}

【用户问题】
{question}{extra}

SQL:"""
    sql = _ask(prompt)
    # 清理可能的 markdown 包裹
    sql = re.sub(r"^```(sql)?|```$", "", sql.strip(), flags=re.MULTILINE).strip()
    return sql


# ── 执行（带安全过滤） ───────────────────────────────────

def execute_sql(db_path: str, sql: str) -> tuple[list[str], list[tuple]]:
    """
    安全执行 SQL（只读）
    :return: (列名, 结果行)
    :raises: ValueError（含写操作）/ sqlite3.Error（SQL 错误）
    """
    if FORBIDDEN.search(sql):
        raise ValueError(f"⛔ 检测到写操作，已拦截: {sql[:60]}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    conn.close()
    return cols, rows


# ── 完整流程 ────────────────────────────────────────────

def ask_database(question: str, db_path: str, max_retry: int = 2) -> dict:
    """
    完整 Text-to-SQL 流程：生成 → 执行 → （失败则让 AI 修正重试）→ 解读
    :return: {"sql", "cols", "rows", "answer", "error"}
    """
    schema = get_schema(db_path)
    sql = text_to_sql(question, schema)

    result = None
    for attempt in range(max_retry + 1):
        try:
            cols, rows = execute_sql(db_path, sql)
            result = (cols, rows)
            break
        except Exception as e:
            if attempt < max_retry:
                print(f"  ⚠ SQL 执行失败（{e}），让 AI 修正重试...")
                sql = text_to_sql(question, schema, error_hint=str(e))
            else:
                return {"sql": sql, "cols": [], "rows": [],
                        "answer": f"SQL 执行失败（已重试 {max_retry} 次）: {e}", "error": str(e)}

    cols, rows = result
    # AI 解读结果
    preview = "\n".join(f"{dict(zip(cols, r))}" for r in rows[:20])
    answer = _ask(f"""用户问题：{question}
执行的 SQL：{sql}
查询结果（{len(rows)} 行）：
{preview}

请用简洁的中文解读这个结果，直接回答问题。如果结果为空，说明没有符合条件的数据。""")

    return {"sql": sql, "cols": cols, "rows": rows, "answer": answer, "error": None}


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "school.db")
    print("=== Schema ===")
    print(get_schema(db))
    print("\n=== 测试查询 ===")
    r = ask_database("各科平均分是多少？", db)
    print(f"SQL: {r['sql']}")
    print(f"结果: {r['rows']}")
    print(f"解读: {r['answer']}")
