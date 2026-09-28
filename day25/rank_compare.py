"""
rank_compare.py — Day 25 加分项：对比 rank=4 / 8 / 16 三个 LoRA 权重
用同一批问题，同一份基座模型，只切换 adapter
用法: python rank_compare.py
"""

import os
import torch

BASE = "/data/models/Qwen2.5-1.5B-Instruct"
SAVES = "/data/Projects/GLMModelsDemo/day25/saves"
ADAPTERS = {
    "rank=4": f"{SAVES}/qwen2.5-1.5b-lora-r4",
    "rank=8": f"{SAVES}/qwen2.5-1.5b-lora",
    "rank=16": f"{SAVES}/qwen2.5-1.5b-lora-r16",
}
QUESTIONS = [
    "简述马克思主义认识论中实践和认识的关系。",
    "求极限 lim(x→0) (sin x - x)/x³，说说解题思路。",
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("加载基座模型（4bit）...")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True)

    for q in QUESTIONS:
        print(f"\n{'═'*72}\n【问题】{q}\n{'═'*72}")
        for label, path in ADAPTERS.items():
            if not os.path.exists(path):
                print(f"\n{label}: 权重不存在，跳过")
                continue
            model = PeftModel.from_pretrained(base, path)
            model.eval()
            messages = [{"role": "user", "content": q}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                     temperature=0.7, top_p=0.9, repetition_penalty=1.05,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n🔸 {label}（{len(reply.strip())} 字）:\n{reply.strip()}")
            del model                       # 释放，为下一个 adapter 腾显存
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
