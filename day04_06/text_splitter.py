"""
text_splitter.py — Day 06 作业核心：三种文本切分策略
split_fixed()      固定长度切分
split_recursive()  递归字符切分（推荐）
split_token()      按 token 近似切分
compare_splits()   对比不同策略的效果
"""

import re


def split_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """固定长度切分：每 chunk_size 字符一刀，相邻块重叠 overlap 字符"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def split_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    递归字符切分：按 段落 → 换行 → 句子 → 逗号 → 字符 的优先级递归切
    尽量保持语义完整（在自然边界处切），比固定长度更懂内容
    """
    separators = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " "]

    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size or not seps:
            return [t] if t.strip() else []

        sep = seps[0]
        parts = t.split(sep)
        if len(parts) == 1:      # 该分隔符切不开，换下一个
            return _split(t, seps[1:])

        # 把切开的片段重新组装成不超过 chunk_size 的块
        result = []
        buf = ""
        for i, part in enumerate(parts):
            piece = part + (sep if i < len(parts) - 1 else "")
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf.strip():
                    result.append(buf.strip())
                if len(piece) > chunk_size:
                    result.extend(_split(piece, seps[1:]))
                    buf = ""
                else:
                    buf = piece
        if buf.strip():
            result.append(buf.strip())
        return result

    chunks = _split(text, separators)

    # 加 overlap：把上一块尾部若干字符拼到下一块开头
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + chunks[i]).strip())
        return overlapped
    return chunks


def split_token(text: str, max_tokens: int = 400, overlap_tokens: int = 40) -> list[str]:
    """
    按 token 近似切分（中文 1 字 ≈ 1.5 token，英文 1 词 ≈ 1.3 token）
    精确 token 需要 tokenizer，这里用近似估算，够 RAG 场景用
    """
    def est_tokens(s: str) -> float:
        chinese = len(re.findall(r"[\u4e00-\u9fff]", s))
        others = len(s) - chinese
        return chinese * 1.5 + others * 0.3

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        # 逐步扩展直到接近 max_tokens
        end = start
        while end < n and est_tokens(text[start:end]) < max_tokens:
            end += 10
        end = min(end, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # 计算 overlap 对应的字符数（近似）
        back = int(overlap_tokens / 1.5) if overlap_tokens else 0
        start = max(end - back, start + 1)
    return chunks


def stats(chunks: list[str]) -> dict:
    """统计切分质量指标"""
    if not chunks:
        return {"count": 0}
    lens = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "total_chars": sum(lens),
        "avg_len": round(sum(lens) / len(lens), 1),
        "max_len": max(lens),
        "min_len": min(lens),
    }


def show_chunks(chunks: list[str], preview: int = 80, limit: int = 10) -> str:
    """展示切分结果：数量、每块前 N 字、总字数"""
    lines = [f"共 {len(chunks)} 个 chunk，总字数 {sum(len(c) for c in chunks)}"]
    for i, c in enumerate(chunks[:limit], 1):
        one_line = c.replace("\n", " ")
        lines.append(f"  [{i}] ({len(c)}字) {one_line[:preview]}{'...' if len(c) > preview else ''}")
    if len(chunks) > limit:
        lines.append(f"  ... 还有 {len(chunks) - limit} 个块")
    return "\n".join(lines)


def compare_splits(text: str, chunk_size: int = 500, overlap: int = 50) -> str:
    """对比固定长度 vs 递归切分的效果差异"""
    fixed = split_fixed(text, chunk_size, overlap)
    recur = split_recursive(text, chunk_size, overlap)

    sf, sr = stats(fixed), stats(recur)

    lines = [
        "═" * 56,
        f"切分对比（chunk_size={chunk_size}, overlap={overlap}）",
        "═" * 56,
        f"{'指标':<12}{'固定长度':<14}{'递归字符':<14}",
        f"{'-'*44}",
        f"{'chunk 数量':<12}{sf['count']:<16}{sr['count']:<16}",
        f"{'平均长度':<12}{sf.get('avg_len','-'):<16}{sr.get('avg_len','-'):<16}",
        f"{'最大长度':<12}{sf.get('max_len','-'):<16}{sr.get('max_len','-'):<16}",
        f"{'最小长度':<12}{sf.get('min_len','-'):<16}{sr.get('min_len','-'):<16}",
        "",
        "边界质量对比：",
    ]

    # 展示同一位置的切分边界（看哪个切得更自然）
    f0 = fixed[0].replace("\n", " ")[-30:] if fixed else ""
    r0 = recur[0].replace("\n", " ")[-30:] if recur else ""
    lines.append(f"  固定长度块1结尾: ...{f0}")
    lines.append(f"  递归字符块1结尾: ...{r0}")
    lines.append("")
    lines.append("结论：递归切分在段落/句子边界处断开，语义更完整；")
    lines.append("      固定长度可能把一句话拦腰截断，但块大小最均匀。")
    return "\n".join(lines)


if __name__ == "__main__":
    sample = "第一段内容。" * 20 + "\n\n" + "第二段内容，讲另一个话题。" * 20
    print("固定长度:", stats(split_fixed(sample, 200, 20)))
    print("递归字符:", stats(split_recursive(sample, 200, 20)))
    print()
    print(compare_splits(sample, 200, 20))
