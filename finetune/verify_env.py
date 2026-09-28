"""
verify_env.py — Day 23：微调环境三项检查
① GPU 可用（CUDA + 显存）
② LLaMA-Factory 可导入
③ Qwen2.5-1.5B 模型已下载
用法: venv/bin/python verify_env.py
"""

import os
import sys

MODEL_DIR = "/data/models/Qwen2.5-1.5B-Instruct"
results = []


def check(title: str, fn):
    print(f"\n{'─'*55}\n【{title}】")
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
    print(f"  {'✅ 通过' if ok else '❌ 失败'}")
    for line in str(detail).splitlines():
        print(f"     {line}")
    results.append((title, ok))
    return ok


# ── 检查 1：GPU ──────────────────────────────────────────

def check_gpu():
    import torch
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False（无 CUDA 设备）"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    info = (f"PyTorch {torch.__version__}（CUDA {torch.version.cuda}）\n"
            f"设备: {name}\n显存: {total:.1f} GB（当前空闲 {free_mem:.1f} GB）\n"
            f"算力: {torch.cuda.get_device_capability(0)}\n"
            f"bf16 支持: {torch.cuda.is_bf16_supported()}")
    # 4GB 显存提示
    if total < 6:
        info += "\n⚠️ 显存 < 6GB，训练需用 4bit QLoRA 量化"
    return True, info


# ── 检查 2：LLaMA-Factory ───────────────────────────────

def check_llamafactory():
    import llamafactory
    ver = getattr(llamafactory, "__version__", "未知")
    # 关键子模块也验一下
    from llamafactory.train.tuner import run_exp          # noqa: F401
    from llamafactory.model import load_model             # noqa: F401
    return True, (f"llamafactory {ver}\n"
                  f"路径: {os.path.dirname(llamafactory.__file__)}\n"
                  f"关键模块可导入: train.tuner / model")


# ── 检查 3：模型 ─────────────────────────────────────────

def check_model():
    if not os.path.isdir(MODEL_DIR):
        return False, f"目录不存在: {MODEL_DIR}（先运行 download_model.py）"
    must = ["config.json"]
    missing = [f for f in must if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return False, f"缺少文件: {missing}"
    files = os.listdir(MODEL_DIR)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    total = sum(os.path.getsize(os.path.join(MODEL_DIR, f))
                for f in files if os.path.isfile(os.path.join(MODEL_DIR, f)))
    import json
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8"))
    info = (f"路径: {MODEL_DIR}\n"
            f"架构: {cfg.get('architectures', ['?'])[0]}  "
            f"层数 {cfg.get('num_hidden_layers','?')}  隐层 {cfg.get('hidden_size','?')}\n"
            f"权重分片: {len(safetensors)} 个 safetensors\n"
            f"总大小: {total/1024**3:.2f} GB")
    if not safetensors:
        return False, info + "\n✗ 没有 safetensors 权重文件"
    return True, info


print("╔══════════════════════════════════════════════════════╗")
print("║   🔍 微调环境验证（Day 23）                          ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"Python: {sys.version.split()[0]}  可执行: {sys.executable}")

ok1 = check("检查 1/3：GPU / CUDA", check_gpu)
ok2 = check("检查 2/3：LLaMA-Factory", check_llamafactory)
ok3 = check("检查 3/3：Qwen2.5-1.5B 模型", check_model)

print(f"\n{'═'*55}")
passed = sum(results[i][1] for i in range(3))
for title, ok in results:
    print(f"  {'✅' if ok else '❌'} {title}")
print(f"\n结果: {passed}/3 项通过" + ("  🎉 环境就绪！" if passed == 3 else "  ⚠️ 请修复失败项"))
sys.exit(0 if passed == 3 else 1)
