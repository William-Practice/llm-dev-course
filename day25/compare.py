"""
compare.py — Day 25：对比原模型 vs 微调模型
用 5 个训练集外的测试问题，各生成一次回答，打印对照

用法: python compare.py
"""

import os
import sys
import time
import json

import torch

BASE_MODEL = "/data/models/Qwen2.5-1.5B-Instruct"
ADAPTER = "/data/Projects/GLMModelsDemo/day25/saves/qwen2.5-1.5b-lora"

# 5 个测试问题：覆盖 5 个子主题，均不在训练集里
TEST_QUESTIONS = [
    "辩证唯物主义中，为什么说实践是检验真理的唯一标准？",
    "考研英语阅读里，作者态度题（attitude）怎么判断？有哪些常见干扰项？",
    "求极限 lim(x→0) (sin x - x)/x³，说说解题思路。",
    "操作系统里进程和线程的区别是什么？为什么说线程是调度单位？",
    "考研冲刺阶段（最后一个月）应该怎么安排各科复习时间？",
]


def load_model(adapter_path: str | None = None):
    """加载模型；adapter_path 为 None 时加载原模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def generate(model, tok, question: str, max_new_tokens: int = 256) -> str:
    """单轮问答生成"""
    messages = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip(), time.time() - t0


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   🔬 原模型 vs 微调模型 对比（Day 25）               ║")
    print("╚══════════════════════════════════════════════════════╝")

    if not os.path.exists(ADAPTER):
        print(f"✗ 找不到 LoRA 权重: {ADAPTER}")
        sys.exit(1)

    # 只加载一份 base，用 adapter 开关切换 —— 4GB 显存放不下两份模型
    from peft import PeftModel
    print(f"\n加载模型: {BASE_MODEL}")
    base, tok = load_model(None)
    print("挂载 LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    results = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'═'*70}\n【问题 {i}/{len(TEST_QUESTIONS)}】{q}\n{'═'*70}")
        # 原模型 = 临时关掉 adapter
        with model.disable_adapter():
            a_base, t1 = generate(model, tok, q)
        # 微调模型 = adapter 生效
        a_ft, t2 = generate(model, tok, q)
        print(f"\n🔵 原模型（{t1:.1f}s，{len(a_base)} 字）:\n{a_base}")
        print(f"\n🟢 微调模型（{t2:.1f}s，{len(a_ft)} 字）:\n{a_ft}")
        results.append({"question": q, "base": a_base, "finetuned": a_ft,
                        "base_len": len(a_base), "ft_len": len(a_ft)})

    out = "/data/Projects/GLMModelsDemo/day25/compare_results.json"
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n💾 对比结果已保存: {out}")


if __name__ == "__main__":
    main()
