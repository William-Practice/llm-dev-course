"""
hello_glm.py — 基础模块
提供 ask_glm()：调用智谱 GLM 的最简封装（纯 HTTP/JSON，无第三方 SDK）
Day 01-03 的所有作业都复用这个函数
"""

import os
import json
import urllib.request

API_KEY = os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY（export ZHIPU_API_KEY='你的key'）")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4-flash"          # 支持 Function Calling


def ask_glm(question: str, system: str | None = None,
            model: str = DEFAULT_MODEL, temperature: float = 0.7) -> str:
    """
    问 GLM 一个问题，返回回答文本
    :param question: 用户问题
    :param system: 系统角色设定（可选）
    :param model: 模型名
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def save_output(text: str, filename: str = "output.txt"):
    """加分项：把结果保存到 txt 文件"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n" + "=" * 50 + "\n")
    print(f"  ✓ 已保存到 {path}")


if __name__ == "__main__":
    print(ask_glm("用一句话介绍你自己"))
