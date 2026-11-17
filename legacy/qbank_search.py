"""
qbank_search.py — 题库知识库检索
输入知识库编号 + 查询内容 → 返回库内最相似的 top_n 道题
纯 HTTP/JSON 调用智谱知识库检索 API
"""

import os
import json
import sys
import urllib.request
import urllib.error

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

RETRIEVE_URL = "https://open.bigmodel.cn/api/llm-application/open/knowledge/retrieve"
LIST_URL = "https://open.bigmodel.cn/api/llm-application/open/knowledge"
CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def glm_chat(prompt: str, temperature: float = 0.2) -> str:
    """直接 HTTP POST JSON 调用 GLM 对话"""
    body = json.dumps({
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        CHAT_URL, data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def list_knowledge_bases(page: int = 1, size: int = 20) -> list[dict]:
    """
    获取知识库列表（默认 20 条）
    :return: [{id, name, description, document_size, length, word_num, ...}, ...]
    """
    url = f"{LIST_URL}?page={page}&size={size}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if data.get("code") != 200:
        raise RuntimeError(f"获取列表失败: code={data.get('code')}, message={data.get('message')}")

    return data.get("data", {}).get("list", [])


def retrieve(query: str, knowledge_ids: list[str], top_k: int = 3,
             recall_method: str = "mixed", recall_ratio: int = 80) -> list[dict]:
    """
    知识库检索
    :param query: 查询内容（≤1000字）
    :param knowledge_ids: 知识库ID列表
    :param top_k: 最终召回数量 1-20
    :param recall_method: embedding / keyword / mixed
    :param recall_ratio: 混合检索时向量权重 0-100
    :return: data 列表 [{text, score, metadata}, ...]
    """
    body = json.dumps({
        "query": query,
        "knowledge_ids": knowledge_ids,
        "top_k": top_k,
        "top_n": max(top_k * 2, 10),
        "recall_method": recall_method,
        "recall_ratio": recall_ratio,
    }).encode("utf-8")

    req = urllib.request.Request(
        RETRIEVE_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if data.get("code") != 200:
        raise RuntimeError(f"检索失败: code={data.get('code')}, message={data.get('message')}")

    return data.get("data", [])


def format_result(item: dict, index: int) -> str:
    """格式化单条检索结果"""
    text = item.get("text", "").strip()
    score = item.get("score", 0)
    meta = item.get("metadata", {})
    doc_name = meta.get("doc_name", "")
    doc_url = meta.get("doc_url", "")

    lines = [f"  #{index}  相似度 {score:.4f}"]
    if doc_name:
        lines.append(f"      来源: {doc_name}")
    lines.append(f"      内容: {text[:200]}")
    if doc_url:
        lines.append(f"      链接: {doc_url}")
    return "\n".join(lines)


def ask(query: str, knowledge_ids: list[str], top_k: int = 3) -> tuple[list[dict], str]:
    """
    检索 + GLM 综合回答
    :return: (检索结果列表, GLM 回答文本)
    """
    results = retrieve(query, knowledge_ids, top_k=top_k)

    if not results:
        return [], "知识库中未找到相关题目。"

    # 拼检索到的题目上下文
    context = "\n\n".join(
        f"【题目{i+1}】\n{item.get('text', '').strip()}" for i, item in enumerate(results)
    )

    prompt = f"""你是软考软件设计师（中级）备考助手。用户问了一道题，以下是题库中检索到的最相似的 {top_k} 道题（含答案与解析）。

请判断检索到的题目中，哪一道与用户的问题最匹配，然后：
1. 给出该题的答案选项和完整解析
2. 如果检索到的题都不完全匹配，说明最接近的是哪道，并给出你自己的解答

题库检索结果：
{context}

用户问题：{query}

请用中文回答，格式清晰。"""
    answer = glm_chat(prompt, temperature=0.3)
    return results, answer


def main():
    """CLI 入口：输入知识库编号，交互式查询"""
    print("╔════════════════════════════════════════════╗")
    print("║        题库知识库检索 · Top-N 相似题       ║")
    print("╚════════════════════════════════════════════╝")

    # --list 模式：列出知识库
    if len(sys.argv) >= 2 and sys.argv[1] in ("--list", "-l"):
        size = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
        try:
            kbs = list_knowledge_bases(page=1, size=size)
        except Exception as e:
            print(f"  ✗ 获取失败: {e}")
            return
        print(f"\n共 {len(kbs)} 个知识库：\n")
        for i, kb in enumerate(kbs, 1):
            print(f"  #{i}  ID: {kb.get('id')}")
            print(f"      名称: {kb.get('name')}")
            if kb.get("description"):
                print(f"      描述: {kb.get('description')}")
            print(f"      文档数: {kb.get('document_size', 0)} | 切片数: {kb.get('length', 0)} | 字数: {kb.get('word_num', 0)}")
            print()
        return

    # 支持命令行传参: python qbank_search.py <知识库ID> [top_k]
    if len(sys.argv) >= 2:
        kb_id = sys.argv[1]
        top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else 3
    else:
        kb_id = input("请输入知识库编号: ").strip()
        try:
            top_k = int(input("请输入返回条数（默认3）: ").strip() or "3")
        except ValueError:
            top_k = 3

    print(f"\n知识库: {kb_id} | 返回 Top-{top_k}\n")

    while True:
        try:
            q = input("查询题目（输入 q 退出）>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue

        print("检索 + 解答中...")
        try:
            results, answer = ask(q, [kb_id], top_k=top_k)
        except Exception as e:
            print(f"  ✗ 出错: {e}")
            continue

        if not results:
            print(f"  {answer}\n")
            continue

        print(f"\n  ── 检索到的 {len(results)} 条相似题 ──")
        for i, item in enumerate(results, 1):
            print(format_result(item, i))

        print(f"\n  ── GLM 解答 ──\n{answer}\n")


if __name__ == "__main__":
    main()
