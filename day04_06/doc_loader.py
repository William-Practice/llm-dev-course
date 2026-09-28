"""
doc_loader.py — Day 05 作业核心：多格式文档统一加载器
支持 txt / md / pdf / docx 四种格式，load_any() 自动识别
零第三方依赖：pdf 用 pdftotext（poppler），docx 用标准库解析 zip+XML
"""

import os
import re
import zipfile
import subprocess
import xml.etree.ElementTree as ET


# === 各格式解析器 ===

def _load_text(path: str) -> str:
    """txt / md：直接读"""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(path: str) -> tuple[str, int]:
    """pdf：pdftotext 提取文本，pdfinfo 取页数"""
    r = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True, timeout=60)
    text = r.stdout if r.returncode == 0 else ""

    pages = 0
    r2 = subprocess.run(["pdfinfo", path], capture_output=True, text=True, timeout=30)
    m = re.search(r"Pages:\s+(\d+)", r2.stdout)
    if m:
        pages = int(m.group(1))
    return text, pages


def _load_docx(path: str) -> str:
    """docx：docx 本质是 zip，直接解 word/document.xml（零依赖）"""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    for p in tree.getroot().iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        texts = [t.text for t in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if t.text]
        if texts:
            paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


# === 统一入口 ===

LOADERS = {
    ".txt": lambda p: (_load_text(p), 0),
    ".md": lambda p: (_load_text(p), 0),
    ".markdown": lambda p: (_load_text(p), 0),
    ".pdf": _load_pdf,
    ".docx": lambda p: (_load_docx(p), 0),
}


def load_any(path: str) -> dict:
    """
    自动识别格式并加载文档
    :return: {"path", "format", "text", "chars", "pages"}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if not loader:
        raise ValueError(f"不支持的格式: {ext}（支持 txt/md/pdf/docx）")

    text, pages = loader(path)
    text = text.strip()
    return {
        "path": path,
        "format": ext.lstrip("."),
        "text": text,
        "chars": len(text),
        "pages": pages,
    }


def describe(doc: dict) -> str:
    """生成文档描述信息"""
    parts = [f"已加载 {doc['format'].upper()}", f"共 {doc['chars']} 字"]
    if doc["pages"]:
        parts.insert(1, f"共 {doc['pages']} 页")
    return "，".join(parts)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "/home/wuruiyang/724/考研笔记.txt"
    doc = load_any(p)
    print(describe(doc))
    print(f"前 100 字: {doc['text'][:100]}")
