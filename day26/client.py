"""
client.py — Day 26：本地模型 API 客户端
菜单：1.单轮问答  2.多轮对话  3.批量测试  0.退出
全部走 OpenAI 兼容接口（localhost:11434/v1），断网可用

用法: python client.py [模型名]
"""

import os
import sys
import json
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server_check import check_service, list_models, call_local_model

DEFAULT_MODEL = "qwen2.5:1.5b"
BATCH_QUESTIONS = [
    "什么是机器学习？",
    "用一句话解释什么是递归。",
    "水的化学式是什么？",
    "简述 TCP 三次握手。",
    "考研数学一考哪些内容？",
]


def pick_model(preferred: str) -> str:
    """选模型：优先用命令行指定的，否则用本地第一个"""
    models = list_models()
    if not models:
        return preferred
    names = [m["name"] for m in models]
    if preferred in names:
        return preferred
    # 名字模糊匹配（如 qwen2.5:1.5b 在列表里可能是 qwen2.5:1.5b-instruct-q4_K_M）
    for n in names:
        if preferred.split(":")[0] in n:
            print(f"（{preferred} 未找到，改用 {n}）")
            return n
    print(f"（未找到 {preferred}，改用 {names[0]}）")
    return names[0]


# ── 功能 1：单轮问答 ─────────────────────────────────────

def single_ask(model: str):
    q = input("请输入问题: ").strip()
    if not q:
        return
    print("思考中...")
    try:
        reply, elapsed = call_local_model(q, model, system="你是一个乐于助人的中文助手。")
        print(f"\n回答（{elapsed:.2f}s，{len(reply)} 字）:\n{reply}\n")
    except Exception as e:
        print(f"✗ 调用失败: {type(e).__name__}: {e}\n")


# ── 功能 2：多轮对话 ─────────────────────────────────────

def multi_turn_chat(model: str):
    print("进入多轮对话（输入 q 结束本轮，输入 /clear 清空历史）")
    history = [{"role": "system", "content": "你是一个乐于助人的中文助手。"}]
    turn = 0
    while True:
        q = input(f"\n[第 {turn+1} 轮] 你: ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if q == "/clear":
            history = [history[0]]
            turn = 0
            print("（历史已清空）")
            continue
        if not q:
            continue

        history.append({"role": "user", "content": q})
        try:
            reply, elapsed = call_local_model("", model, messages=history)
            history.append({"role": "assistant", "content": reply})
            turn += 1
            print(f"助手（{elapsed:.2f}s）: {reply}")
        except Exception as e:
            print(f"✗ 调用失败: {type(e).__name__}: {e}")
            history.pop()
    print(f"（本轮共 {turn} 轮对话，历史 {len(history)} 条消息）\n")


# ── 功能 3：批量测试 ─────────────────────────────────────

def batch_test(model: str):
    print(f"批量测试 {len(BATCH_QUESTIONS)} 个问题（同一模型、同一参数）\n")
    total_time = 0
    results = []
    for i, q in enumerate(BATCH_QUESTIONS, 1):
        try:
            reply, elapsed = call_local_model(q, model, max_tokens=256)
            total_time += elapsed
            results.append({"q": q, "a": reply, "t": elapsed})
            print(f"[{i}/{len(BATCH_QUESTIONS)}] {q}")
            print(f"  → {reply[:100]}{'...' if len(reply) > 100 else ''}")
            print(f"  耗时 {elapsed:.2f}s，{len(reply)} 字\n")
        except Exception as e:
            print(f"[{i}] ✗ 失败: {type(e).__name__}: {e}\n")

    if results:
        avg = total_time / len(results)
        print(f"{'─'*55}")
        print(f"总计 {len(results)} 条，总耗时 {total_time:.2f}s，平均 {avg:.2f}s/条")
        print(f"平均回答 {sum(len(r['a']) for r in results)/len(results):.1f} 字")
        out = os.path.join(_ROOT, "batch_results.json")
        json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"结果已保存: {out}\n")


# ── 主菜单 ──────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║   🚀 本地模型 API 客户端（Day 26）               ║")
    print("╚══════════════════════════════════════════════════╝")

    # 启动先检查服务
    ok, msg = check_service()
    print(f"\n服务检查: {'✅' if ok else '❌'} {msg}")
    if not ok:
        print("\n请先启动 Ollama：\n   ollama serve          # 前台启动\n   systemctl start ollama  # 或用 systemd")
        return

    model = pick_model(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    print(f"使用模型: {model}\n")

    while True:
        print("请选择：")
        print("  1. 单轮问答")
        print("  2. 多轮对话")
        print("  3. 批量测试（5 个问题 + 计时）")
        print("  0. 退出")
        c = input("请输入数字: ").strip()

        if c == "0":
            print("👋 再见！")
            break
        elif c == "1":
            single_ask(model)
        elif c == "2":
            multi_turn_chat(model)
        elif c == "3":
            batch_test(model)
        else:
            print("✗ 请输入 0-3\n")


if __name__ == "__main__":
    main()
