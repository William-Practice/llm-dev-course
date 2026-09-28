"""
导师论文解析器
提取 PDF 全文 → 纯 HTTP/JSON 发给 GLM → CLI 输出启示
"""

import os
import json
import subprocess
import urllib.request
import urllib.error


API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

PAPER_PATH = (
    "/home/wuruiyang/文档/xwechat_files/wxid_zuvhk4hfske022_6e3a/msg/file/2026-07/"
    "2-Spectral clustering based mutant reduction for mutation testing.pdf"
)


def glm_chat(prompt: str, temperature: float = 0.3) -> str:
    """直接 HTTP POST JSON 调用 GLM"""
    body = json.dumps({
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_pdf_text(pdf_path: str) -> str:
    """用 pdftotext 提取 PDF 全文"""
    out = subprocess.run(
        ["pdftotext", pdf_path, "-"],
        capture_output=True, text=True, timeout=30
    )
    return out.stdout.strip()


def analyze() -> str:
    """提取论文 → 发给 GLM 分析 → 返回启示"""
    today = subprocess.run(["date", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip()

    print("正在提取论文文本...")
    full_text = extract_pdf_text(PAPER_PATH)
    print(f"论文提取完成（{len(full_text)} 字符）\n")

    # 截取关键部分（正文约 25K 字符，控制 token 消耗）
    # 取标题 + 摘要 + 引言 + 方法 + 实验结论 + 结论
    key_sections = _extract_key_sections(full_text)

    prompt = f"""你是一位学术导师。今天是 {today}。

以下是我导师在读研期间发表的一篇期刊论文（Information and Software Technology, 2021）的全文关键内容。请认真阅读，然后从以下几个方面给出分析：

## 要求分析框架：

### 1. 论文核心贡献（一句话概括）
### 2. 研究方法论亮点（导师做研究的方法、思路、技巧 — 哪些值得我学习）
### 3. 论文与我的关联
   - 我目前研究方向：知识图谱
   - 这篇论文用了谱聚类（spectral clustering），与图算法有交集
   - 可以从中借鉴什么思路到知识图谱研究中？
### 4. 对投稿 IST 的启示
   - 这篇就发在 IST，分析它的写作风格、实验设计、论文结构
   - 我该注意什么？
### 5. 导师的学术风格
   - 从选题、方法选择、实验设计、写作风格推断导师的科研品味
   - 作为他的学生，我应该延续什么、避开什么？

请以 Markdown 格式输出，每部分用 ### 分隔。

论文内容（摘录）：
---
{key_sections}
"""

    return glm_chat(prompt, temperature=0.5)


def _extract_key_sections(text: str, max_chars: int = 28000) -> str:
    """从全文提取关键部分"""
    # 先尝试提取标题到引言、方法、实验、结论
    lines = text.split("\n")
    result_lines = []
    current_size = 0
    in_header = True

    for line in lines:
        stripped = line.strip()
        # 跳过页眉页脚行
        if stripped.startswith("Information and Software Technology"):
            continue
        if stripped.startswith("C. Wei et al."):
            continue
        if stripped.startswith("https://doi.org/"):
            continue

        result_lines.append(line)
        current_size += len(line) + 1
        if current_size >= max_chars:
            result_lines.append("\n[... 以下内容因长度限制已截断 ...]")
            break

    return "\n".join(result_lines)


def main():
    print("╔════════════════════════════════════════════╗")
    print("║        导师论文解析 · 启示报告              ║")
    print("╚════════════════════════════════════════════╝")
    print()

    insight = analyze()
    print(insight)

    print()
    print("─" * 55)
    print("以上分析由 GLM 生成，供参考学习。")


if __name__ == "__main__":
    main()
