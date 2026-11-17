"""
split_qbank.py — 把题库 txt 按题拆成单独文件
输出到 qbank_split/ 目录，每道题一个 txt
"""

import os
import re

SRC = os.path.join(os.path.dirname(__file__), "软考软件设计师模拟题.txt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "qbank_split")


def split_by_question(text: str) -> list[tuple[str, str]]:
    """
    按"第X题"拆分，返回 [(题号, 题目文本), ...]
    """
    # 按 "第N题" 切分，保留分隔符
    parts = re.split(r"(第\d+题)", text)
    questions = []
    current_num = None
    current_text = []

    for part in parts:
        if re.fullmatch(r"第\d+题", part):
            if current_num is not None:
                questions.append((current_num, "".join(current_text).strip()))
            current_num = part
            current_text = []
        else:
            current_text.append(part)

    if current_num is not None:
        questions.append((current_num, "".join(current_text).strip()))

    return questions


def main():
    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    questions = split_by_question(text)
    print(f"共识别到 {len(questions)} 道题")

    os.makedirs(OUT_DIR, exist_ok=True)
    # 清空旧文件
    for old in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, old))

    for idx, (num, qtext) in enumerate(questions, 1):
        filename = f"q{idx:02d}.txt"
        path = os.path.join(OUT_DIR, filename)
        # 去掉题目文本开头的分隔符（如 "："）
        clean = qtext.lstrip("：:、.。 ")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{num}\n{clean}\n")
        # 预览前 30 字
        preview = clean.replace("\n", " ")[:30]
        print(f"  {filename}: {preview}...")

    print(f"\n完成！已输出到: {OUT_DIR}")


if __name__ == "__main__":
    main()
