import os
import json
import subprocess
import urllib.request
import urllib.error

from embedding_utils import embed, cosine_similarity, search_with_vectors

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

PAPER_PATH = (
    "/home/wuruiyang/文档/xwechat_files/wxid_zuvhk4hfske022_6e3a/msg/file/2026-07/"
    "2-Spectral clustering based mutant reduction for mutation testing.pdf"
)
INDEX_PATH = os.path.join(os.path.dirname(__file__), "paper_index.json")

def glm_chat(prompt: str, temperature: float = 0.2) -> str:
    body = json.dumps({
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]

def extract_text() -> str:
    r = subprocess.run(["pdftotext", PAPER_PATH, "-"], capture_output=True, text=True, timeout=30)
    return r.stdout.strip()

def clean_text(text: str) -> str:
    """
    清洗论文文本噪声：
    - 页眉（期刊名/卷期行）、页脚（纯数字页码）
    - 作者贡献声明、利益冲突声明
    - 参考文献列表（对检索价值低且占 token）
    """
    lines = text.split("\n")
    cleaned = []
    skip_section = False  

    for line in lines:
        s = line.strip()

        if s == "References":
            skip_section = True
            continue
        if skip_section:
            continue

        # 页眉：期刊信息行
        if s.startswith("Information and Software Technology"):
            continue
        if s.startswith("C. Wei et al."):
            continue
        # DOI / 版权行
        if s.startswith("https://doi.org/") or s.startswith("0950-5849/"):
            continue
        # 作者贡献 / 利益冲突声明
        if s.startswith("CRediT authorship") or s.startswith("Declaration of competing"):
            continue
        # 页脚：孤立数字页码
        if s.isdigit() and len(s) <= 3:
            continue
        # 页脚：文章编号行（如 "106502" 带前缀）
        if s.endswith("106502") and len(s) < 40:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """按段落 + 字符数分块，带重叠"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) < chunk_size:
            buf += p + "\n"
        else:
            if buf:
                chunks.append(buf.strip())
            overlap_text = buf[-overlap:] if len(buf) > overlap else buf
            buf = overlap_text + "\n" + p + "\n"
    if buf:
        chunks.append(buf.strip())
    return chunks


def build_index(force: bool = False) -> list[dict]:
    """构建/加载索引（复用 embedding_utils.embed）"""
    if not force and os.path.exists(INDEX_PATH):
        print(f"加载缓存索引 ({INDEX_PATH})")
        with open(INDEX_PATH) as f:
            return json.load(f)

    print("提取论文文本...")
    text = extract_text()
    raw_len = len(text)
    text = clean_text(text)
    print(f"清洗: {raw_len} → {len(text)} 字符（去除 {raw_len - len(text)} 噪声）")
    print(f"分块中...")
    chunks = chunk_text(text)
    print(f"共 {len(chunks)} 个块，生成向量...")

    index = []
    for i, chunk in enumerate(chunks):
        vec = embed(chunk)
        index.append({"id": i, "text": chunk, "vec": vec})
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            print(f"  向量化进度: {i+1}/{len(chunks)}")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"索引已保存 ({len(index)} 块)")
    return index


def retrieve(query: str, index: list[dict], top_k: int = 5) -> list[dict]:
    """语义检索 top-K 相关块（复用 search_with_vectors）"""
    q_vec = embed(query)
    candidates = [(item["text"], item["vec"]) for item in index]
    results = search_with_vectors(q_vec, candidates, top_n=top_k)
    # 按文本回查原始 index 条目
    text_to_item = {item["text"]: item for item in index}
    return [text_to_item[text] for text, _ in results]


def ask(question: str, index: list[dict] | None = None) -> str:
    """基于论文知识回答问题"""
    if index is None:
        index = build_index()

    top_chunks = retrieve(question, index)

    context = "\n\n---\n\n".join(
        f"[片段 {c['id']+1}]: {c['text'][:600]}" for c in top_chunks
    )

    prompt = f"""你是一位科研助手。以下内容来自一篇软件工程论文（Spectral clustering based mutant reduction for mutation testing, IST 2021）。

请基于给定的论文片段，回答用户的问题。如果片段信息不足以回答，如实说明。

论文片段：
{context}

用户问题：{question}

请用中文回答，引用相关片段支撑你的回答。"""
    return glm_chat(prompt, temperature=0.3)


def interactive():
    """交互式问答"""
    print("╔════════════════════════════════════════════╗")
    print("║      Mini RAG · 导师论文知识问答           ║")
    print("╚════════════════════════════════════════════╝")
    print(f"论文: {os.path.basename(PAPER_PATH)}")
    print("输入问题（输入 q 退出）\n")

    index = build_index()

    while True:
        q = input(">>> ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        print("思考中...")
        answer = ask(q, index)
        print(f"\n{answer}\n")

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    force = "--rebuild" in args
    args = [a for a in args if a != "--rebuild"]

    if args:
        q = " ".join(args)
        idx = build_index(force=force)
        print(ask(q, idx))
    else:
        interactive()
