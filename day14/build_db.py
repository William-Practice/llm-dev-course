"""
build_db.py — Day 14：建学生成绩数据库（sqlite3 标准库）
students 表（学生+班级）+ grades 表（成绩），支持 JOIN 跨表查询
"""

import os
import sqlite3

DB_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "school.db")

# 学生：id, name, class
STUDENTS = [
    (1, "张三", "一班"),
    (2, "李四", "一班"),
    (3, "王五", "二班"),
    (4, "赵六", "二班"),
    (5, "钱七", "一班"),
    (6, "孙八", "二班"),
]

# 成绩：student_id, subject, score
GRADES = [
    (1, "语文", 88), (1, "数学", 95), (1, "英语", 79),
    (2, "语文", 76), (2, "数学", 82), (2, "英语", 91),
    (3, "语文", 92), (3, "数学", 78), (3, "英语", 85),
    (4, "语文", 68), (4, "数学", 90), (4, "英语", 73),
    (5, "语文", 85), (5, "数学", 88), (5, "英语", 94),
    (6, "语文", 79), (6, "数学", 65), (6, "英语", 88),
]


def build_sample_db(db_path: str = DB_DEFAULT) -> str:
    """重建数据库：清空并插入示例数据"""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            class TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            score REAL NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    cur.executemany("INSERT INTO students VALUES (?, ?, ?)", STUDENTS)
    cur.executemany("INSERT INTO grades (student_id, subject, score) VALUES (?, ?, ?)", GRADES)

    conn.commit()
    conn.close()
    return db_path


if __name__ == "__main__":
    p = build_sample_db()
    conn = sqlite3.connect(p)
    n_stu = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    n_grade = conn.execute("SELECT COUNT(*) FROM grades").fetchone()[0]
    print(f"✅ 数据库已重建: {p}")
    print(f"   students: {n_stu} 人 | grades: {n_grade} 条成绩")
    print("\n示例数据:")
    for row in conn.execute("""
        SELECT s.name, s.class, g.subject, g.score
        FROM grades g JOIN students s ON g.student_id = s.id
        LIMIT 5
    """):
        print(f"  {row[0]}（{row[1]}）{row[2]}: {row[3]}")
    conn.close()
