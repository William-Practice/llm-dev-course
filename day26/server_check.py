"""
server_check.py — Day 26：Ollama 本地服务检查与调用封装
- check_service()      检查 Ollama 服务是否在运行
- list_models()        列出本地已有模型
- call_local_model()   OpenAI 兼容接口调用（切到任何 OpenAI SDK 都能用）

零第三方依赖：纯 urllib + json
"""

import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:11434"
API_KEY = "ollama"                      # Ollama 不校验，占位即可


# ── 服务检查 ─────────────────────────────────────────────

def check_service(timeout: int = 5) -> tuple[bool, str]:
    """检查 Ollama 服务是否在跑"""
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/version")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return True, f"Ollama 服务运行中，版本 {data.get('version', '?')}"
    except urllib.error.URLError as e:
        return False, f"无法连接 Ollama（{e.reason}）—— 先运行 `ollama serve`"
    except Exception as e:
        return False, f"检查失败：{type(e).__name__}: {e}"


def list_models(timeout: int = 10) -> list[dict]:
    """列出本地模型（名称/大小/量化）"""
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception:
        return []

    models = []
    for m in data.get("models", []):
        size_gb = m.get("size", 0) / 1024**3
        models.append({
            "name": m.get("name", "?"),
            "size_gb": round(size_gb, 2),
            "family": (m.get("details") or {}).get("family", "?"),
            "params": (m.get("details") or {}).get("parameter_size", "?"),
            "quant": (m.get("details") or {}).get("quantization_level", "?"),
        })
    return sorted(models, key=lambda x: x["name"])


# ── 调用封装（OpenAI 兼容） ──────────────────────────────

def call_local_model(prompt: str, model: str, system: str = "",
                     messages: list | None = None,
                     temperature: float = 0.7, max_tokens: int = 512,
                     timeout: int = 300) -> tuple[str, float]:
    """
    调用本地模型（走 OpenAI 兼容接口 /v1/chat/completions）

    :param messages: 传入完整消息历史则忽略 prompt/system（多轮对话用）
    :return: (回答文本, 耗时秒)
    """
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    elapsed = time.time() - t0

    reply = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return reply.strip(), elapsed


def call_openai_compatible(prompt: str, model: str, base_url: str = BASE_URL,
                           api_key: str = API_KEY, **kw) -> tuple[str, float]:
    """通用 OpenAI 兼容调用（改 base_url 就能指向别的服务，如 vLLM / LM Studio）"""
    global BASE_URL, API_KEY
    old = (BASE_URL, API_KEY)
    BASE_URL, API_KEY = base_url.rstrip("/"), api_key
    try:
        return call_local_model(prompt, model, **kw)
    finally:
        BASE_URL, API_KEY = old


# ── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║   🔍 Ollama 服务检查（Day 26）               ║")
    print("╚══════════════════════════════════════════════╝")

    ok, msg = check_service()
    print(f"\n服务状态: {'✅' if ok else '❌'} {msg}")

    if ok:
        models = list_models()
        if models:
            print(f"\n本地模型（{len(models)} 个）:")
            print(f"  {'名称':<38} {'大小':>8} {'参数':>8} {'量化':>8}")
            for m in models:
                print(f"  {m['name']:<38} {m['size_gb']:>6.2f}GB {m['params']:>8} {m['quant']:>8}")
        else:
            print("\n本地没有模型 —— 运行 `ollama pull qwen2.5:1.5b` 拉一个")
