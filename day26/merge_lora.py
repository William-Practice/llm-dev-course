"""
merge_lora.py — Day 26：把 LoRA 权重合并进基座模型，导出完整 HF 模型
Ollama 0.34+ 不再支持 `ADAPTER` 指令，所以要先合并再导入。

用法: python merge_lora.py
输出: /data/models/Qwen2.5-1.5B-kaoyan-merged
"""

import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/data/models/Qwen2.5-1.5B-Instruct"
ADAPTER = "/data/Projects/GLMModelsDemo/day25/saves/qwen2.5-1.5b-lora"
OUT = "/data/models/Qwen2.5-1.5B-kaoyan-merged"


def main():
    print("=" * 60)
    print("LoRA 合并（CPU 执行，避免 4GB 显存不足）")
    print("=" * 60)

    print(f"\n1) 加载基座模型（bf16, CPU）: {BASE}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    print(f"2) 挂载 LoRA: {ADAPTER}")
    model = PeftModel.from_pretrained(model, ADAPTER)

    print("3) 合并权重（merge_and_unload）...")
    model = model.merge_and_unload()
    print("   ✓ 合并完成")

    print(f"4) 保存到 {OUT}")
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(OUT)

    size = sum(os.path.getsize(os.path.join(OUT, f))
               for f in os.listdir(OUT) if os.path.isfile(os.path.join(OUT, f)))
    print(f"   ✓ 已保存（{size/1024**3:.2f} GB，{len(os.listdir(OUT))} 个文件）")
    print("\n文件清单:")
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        if os.path.isfile(p):
            print(f"   {os.path.getsize(p)/1024**2:>9.1f} MB  {f}")


if __name__ == "__main__":
    main()
