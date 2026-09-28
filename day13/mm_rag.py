"""
mm_rag.py — Day 13 作业核心：多模态 RAG
PDF 文字 + 图片（GLM-4V 描述）一起入库，检索时答案标注来源
零依赖抽取：pdftotext（文字）+ pdfimages（图片）
"""

import os
import re
import json
import base64
import subprocess
import urllib.request

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
CHAT_MODEL = "glm-4-flash"      # 文字生成
VL_MODEL = "glm-4v-flash"       # 图片理解

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_cache.json")


# ── GLM 调用 ─────────────────────────────────────────────

def ask_glm(prompt: str, model: str = CHAT_MODEL) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.3}).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def describe_image(img_path: str) -> str:
    """用 GLM-4V 把图片转成文字描述"""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = img_path.rsplit(".", 1)[-1].lower()
    mime = "png" if ext == "png" else "jpeg"

    body = json.dumps({
        "model": VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "请详细描述这张图片的内容。如果是图表，说明它展示了什么数据和趋势；如果是界面截图或流程图，说明结构和关键信息。50-200字。"},
            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        ]}],
    }).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def describe_page(img_path: str, page_no: int) -> str:
    """用 GLM-4V 描述整页内容，重点提取图表/公式/流程图信息"""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    body = json.dumps({
        "model": VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": f"""这是学术论文的第 {page_no} 页。请描述：
1. 页面中的图表（Figure/Table）：标题、展示了什么数据/流程/结构
2. 页面中的公式和算法伪代码要点
3. 本页的主要文字内容主题

请分点描述，如果本页没有图表就说明"本页无图表"，重点放在文字主题上。"""},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    }).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


# ── PDF 抽取 ─────────────────────────────────────────────

def _png_size(path: str) -> tuple[int, int]:
    """纯 Python 读 PNG 宽高（IHDR 块固定位置）"""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    w = int.from_bytes(head[16:20], "big")
    h = int.from_bytes(head[20:24], "big")
    return w, h


def extract_pdf(pdf_path: str, img_dir: str) -> tuple[str, list[str]]:
    """抽取 PDF 的文字和图片，返回 (文字, 图片路径列表)"""
    # 文字
    r = subprocess.run(["pdftotext", pdf_path, "-"], capture_output=True, text=True, timeout=60)
    text = r.stdout

    # 图片（-png 输出 PNG）
    os.makedirs(img_dir, exist_ok=True)
    subprocess.run(["pdfimages", "-png", pdf_path, os.path.join(img_dir, "img")],
                   capture_output=True, timeout=120)

    images = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    )
    # 过滤小图（图标/装饰，宽高 < 100px）——纯 Python 读 PNG 头
    kept = []
    for p in images:
        try:
            w, h = _png_size(p)
            if w >= 100 and h >= 100:
                kept.append(p)
            else:
                os.remove(p)
        except Exception:
            kept.append(p)
    return text, kept


# ── 建库（文字 chunk + 图片描述） ─────────────────────────

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def render_pages(pdf_path: str, img_dir: str, dpi: int = 130) -> list[str]:
    """
    加分项：整页渲染为 PNG（处理矢量绘制的图表——pdfimages 抽不到）
    用 pdftoppm 把每页转成图片，再交给 GLM-4V 描述
    """
    os.makedirs(img_dir, exist_ok=True)
    prefix = os.path.join(img_dir, "page")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf_path, prefix],
                   capture_output=True, timeout=300)
    return sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                  if f.startswith("page") and f.endswith(".png"))


def load_pdf(pdf_path: str, chunk_size: int = 600, full_page: bool = False) -> list[dict]:
    """
    加载 PDF → 文字 chunk + 图片描述，返回知识库条目
    :param full_page: True 时额外做整页渲染描述（捕获矢量图表），较慢但更全
    每条: {"type": "text"|"image"|"page", "content", "source", "image_path"(可选)}
    """
    name = os.path.basename(pdf_path)
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_images",
                           os.path.splitext(name)[0])
    print(f"  抽取文字和图片...")
    text, images = extract_pdf(pdf_path, img_dir)
    print(f"  文字 {len(text)} 字，位图 {len(images)} 张")

    kb = []
    cache = _load_cache()

    # 文字分块
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) < chunk_size:
            buf += p + "\n"
        else:
            if buf:
                kb.append({"type": "text", "content": buf.strip(),
                           "source": f"文字-{name}", "image_path": None})
            buf = p + "\n"
    if buf.strip():
        kb.append({"type": "text", "content": buf.strip(),
                   "source": f"文字-{name}", "image_path": None})

    # 位图描述（过滤掉小 logo：内容太短或纯色描述）
    for i, img in enumerate(images, 1):
        key = os.path.basename(img)
        if key in cache:
            desc = cache[key]
        else:
            print(f"  [{i}/{len(images)}] {key} → GLM-4V...")
            try:
                desc = describe_image(img)
            except Exception as e:
                desc = f"(图片描述失败: {e})"
            cache[key] = desc
            _save_cache(cache)
        kb.append({"type": "image", "content": desc,
                   "source": f"图片-{key}", "image_path": img})

    # 加分项：整页渲染（矢量图表）
    if full_page:
        page_dir = os.path.join(img_dir, "pages")
        pages = render_pages(pdf_path, page_dir)
        print(f"  整页渲染 {len(pages)} 页 → GLM-4V 描述（较慢）...")
        for i, pg in enumerate(pages, 1):
            key = "page_" + os.path.basename(pg)
            if key in cache:
                desc = cache[key]
            else:
                print(f"  [{i}/{len(pages)}] 第 {i} 页...")
                try:
                    desc = describe_page(pg, i)
                except Exception as e:
                    desc = f"(整页描述失败: {e})"
                cache[key] = desc
                _save_cache(cache)
            kb.append({"type": "page", "content": desc,
                       "source": f"整页-第{i}页", "image_path": pg})

    return kb


# ── 检索 + 问答 ──────────────────────────────────────────

def _tokenize(s: str) -> set:
    """中文 bigram + 英文词（关键词粗筛）"""
    toks = set(re.findall(r"[a-zA-Z0-9_]+", s.lower()))
    ch = re.findall(r"[\u4e00-\u9fff]", s)
    toks |= {ch[i] + ch[i + 1] for i in range(len(ch) - 1)}
    return toks


def retrieve(kb: list[dict], question: str, top_k: int = 4) -> list[dict]:
    """关键词打分检索（零依赖；生产环境应换向量检索）"""
    q = _tokenize(question)
    # 问"图/图表"类问题时，给视觉条目（图片/整页）加权
    visual_query = any(kw in question for kw in
                       ["图", "图片", "图表", "figure", "table", "截图", "流程图", "表格"])

    scored = []
    for item in kb:
        c = _tokenize(item["content"])
        score = len(q & c) / (len(q) + 1)
        if visual_query:
            if item["type"] == "page":
                score += 0.35          # 整页描述信息最全
            elif item["type"] == "image":
                score += 0.20
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for s, it in scored[:top_k]]


def ask_question(kb: list[dict], question: str) -> str:
    """检索 + 生成，答案标注来源（文字/图片）"""
    hits = retrieve(kb, question, top_k=4)
    context_parts = []
    for i, h in enumerate(hits, 1):
        tag = "【文字片段】" if h["type"] == "text" else "【图片描述】"
        context_parts.append(f"[{i}] {tag}{h['source']}\n{h['content']}")
    context = "\n\n".join(context_parts)

    prompt = f"""请根据下面的资料回答问题。资料包含文字片段和图片描述。

要求：
1. 只根据资料回答，资料里没有就说"资料中没有相关信息"
2. 回答最后标注信息来源，格式：来源：[1] 文字-xxx.md / [2] 图片-img-000.png

【资料】
{context}

【问题】{question}"""
    return ask_glm(prompt)


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    if pdf:
        kb = load_pdf(pdf)
        print(f"\n知识库条目: {len(kb)}")
        for item in kb:
            print(f"  [{item['type']}] {item['source']}: {item['content'][:50]}...")
