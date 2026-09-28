"""
gen_qbank.py — 用 GLM 生成软考软件设计师模拟题，保存为 txt 题库
生成后到智谱控制台知识库页面上传该文件即可
"""

import os
import json
import urllib.request
import urllib.error

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

OUTPUT = os.path.join(os.path.dirname(__file__), "软考软件设计师模拟题.txt")


def glm_chat(prompt: str, temperature: float = 0.7) -> str:
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
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def generate_batch(topic: str, count: int) -> str:
    """生成一批题目"""
    prompt = f"""请出 {count} 道软考中级软件设计师考试的选择题，主题：{topic}。

要求：
1. 每道题格式如下，题与题之间空一行：
第X题：题干
A. 选项A
B. 选项B
C. 选项C
D. 选项D
答案：X
解析：简要解析

2. 难度接近真实考试，考查核心知识点
3. 不要输出其他任何内容"""
    return glm_chat(prompt, temperature=0.7)


def main():
    topics = [
        ("数据结构与算法", 6),
        ("操作系统", 4),
        ("数据库系统", 4),
        ("软件工程", 3),
        ("计算机网络", 3),
    ]

    print("正在生成题库（分 5 批调用 GLM）...")
    all_text = "软考软件设计师（中级）模拟题\n" + "=" * 40 + "\n\n"
    total = 0

    for topic, count in topics:
        print(f"  生成中: {topic} ({count}题)...")
        batch = generate_batch(topic, count)
        all_text += f"\n【{topic}】\n" + batch.strip() + "\n\n"
        total += count

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"\n完成！共 {total} 道题，已保存到: {OUTPUT}")
    print(f"文件大小: {os.path.getsize(OUTPUT)} 字节")


if __name__ == "__main__":
    main()
