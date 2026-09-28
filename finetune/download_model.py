"""
download_model.py — Day 23：下载 Qwen2.5-1.5B-Instruct
优先走魔搭（modelscope，国内直连快），失败则回退 hf-mirror
用法: venv/bin/python download_model.py [模型ID]
"""

import os
import sys
import subprocess

TARGET_DIR = "/data/models"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HF_MIRROR = "https://hf-mirror.com"


def via_modelscope(model_id: str) -> str:
    """魔搭下载（推荐，国内快）"""
    from modelscope import snapshot_download
    local = os.path.join(TARGET_DIR, model_id.split("/")[-1])
    print(f"📥 魔搭下载 {model_id} → {local}")
    path = snapshot_download(model_id, local_dir=local)
    return path


def via_hf_mirror(model_id: str) -> str:
    """hf-mirror 回退方案"""
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    os.environ.pop("http_proxy", None)          # 直连镜像，不走代理
    os.environ.pop("https_proxy", None)
    from huggingface_hub import snapshot_download
    local = os.path.join(TARGET_DIR, model_id.split("/")[-1])
    print(f"📥 hf-mirror 下载 {model_id} → {local}")
    path = snapshot_download(model_id, local_dir=local,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    return path


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    os.makedirs(TARGET_DIR, exist_ok=True)
    local = os.path.join(TARGET_DIR, model_id.split("/")[-1])

    if os.path.exists(os.path.join(local, "config.json")):
        size = subprocess.run(["du", "-sh", local], capture_output=True, text=True).stdout.split()[0]
        print(f"✓ 模型已存在：{local}（{size}），跳过下载")
        return

    for name, fn in [("modelscope", via_modelscope), ("hf-mirror", via_hf_mirror)]:
        try:
            path = fn(model_id)
            size = subprocess.run(["du", "-sh", path], capture_output=True, text=True).stdout.split()[0]
            print(f"✓ 下载完成（{name}）：{path}（{size}）")
            return
        except Exception as e:
            print(f"✗ {name} 失败：{type(e).__name__}: {str(e)[:150]}")

    print("✗ 两种方式都失败，请检查网络")
    sys.exit(1)


if __name__ == "__main__":
    main()
