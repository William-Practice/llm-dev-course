"""
eval_model.py — Day 27：微调效果评估核心模块
prepare_test_questions()  准备 15 道微调方向的测试题（训练集外）
compare_models()          原模型 vs 微调模型跑同一套题（同 temperature / max_tokens）
score_answer()            自动评分（LLM 裁判，rubric 三维度）+ 成对比较
generate_report()         生成 eval_report.txt + eval_report.json

用法: 被 run_eval.py 调用
"""

import os
import re
import sys
import json
import time
import urllib.request
import urllib.error

DAY_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 裁判模型调用（纯 urllib，不依赖 openai SDK） ──────────

ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def ask_glm(question: str, system: str = "", model: str = "glm-4-flash",
            temperature: float = 0.3, timeout: int = 120) -> str:
    """调用智谱 API（OpenAI 兼容），返回回答文本"""
    key = os.environ.get("ZHIPU_API_KEY")
    if not key:
        raise ValueError("请设置环境变量 ZHIPU_API_KEY")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": temperature}).encode()
    req = urllib.request.Request(
        ZHIPU_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]

BASE_MODEL = "/data/models/Qwen2.5-1.5B-Instruct"
ADAPTER = "/data/Projects/GLMModelsDemo/day25/saves/qwen2.5-1.5b-lora"

# ── 1. 测试题（15 道，覆盖 5 个子主题，均在训练集之外） ──

TEST_QUESTIONS = [
    # 考研政治（3）
    {"q": "为什么说实践是检验真理的唯一标准？", "subject": "考研政治"},
    {"q": "简述矛盾的普遍性和特殊性的关系。", "subject": "考研政治"},
    {"q": "为什么要坚持全面深化改革？", "subject": "考研政治"},
    # 考研数学（3）
    {"q": "求极限 lim(x→0) (sin x - x)/x³。", "subject": "考研数学"},
    {"q": "简述拉格朗日中值定理的条件和结论。", "subject": "考研数学"},
    {"q": "如何判断一个矩阵是否可对角化？", "subject": "考研数学"},
    # 考研英语（3）
    {"q": "考研英语阅读中，如何判断作者的态度？", "subject": "考研英语"},
    {"q": "翻译时遇到长难句该怎么处理？", "subject": "考研英语"},
    {"q": "小作文（应用文）有哪些常见类型和套路？", "subject": "考研英语"},
    # 计算机专业课 408（3）
    {"q": "进程和线程的区别是什么？为什么说线程是调度单位？", "subject": "计算机专业课408"},
    {"q": "TCP 三次握手的过程是什么？为什么需要三次？", "subject": "计算机专业课408"},
    {"q": "二叉排序树和平衡二叉树的区别是什么？", "subject": "计算机专业课408"},
    # 备考方法（3）
    {"q": "考研冲刺阶段最后一个月应该怎么安排复习？", "subject": "备考方法"},
    {"q": "真题应该怎么用才最有效？", "subject": "备考方法"},
    {"q": "复习中总是焦虑怎么办？", "subject": "备考方法"},
]

# ── 2. 模型生成 ─────────────────────────────────────────

_MODELS = {}


def load_models():
    """加载基座模型 + 挂 LoRA（4bit，4GB 显存友好）"""
    if _MODELS:
        return _MODELS
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True)

    ft = base
    if os.path.exists(ADAPTER):
        ft = PeftModel.from_pretrained(base, ADAPTER)
    ft.eval()

    _MODELS.update({"base": base, "ft": ft, "tok": tok})
    return _MODELS


def _gen(model, tok, question: str, max_new_tokens: int = 256) -> tuple[str, float]:
    """同一套采样参数（保证对比公平）"""
    import torch
    messages = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, repetition_penalty=1.05,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip(), time.time() - t0


def compare_models(questions: list[dict], verbose: bool = True) -> list[dict]:
    """两个模型跑同一套题"""
    m = load_models()
    tok, ft, base = m["tok"], m["ft"], m["base"]
    results = []
    for i, item in enumerate(questions, 1):
        q = item["q"]
        if verbose:
            print(f"  [{i}/{len(questions)}] {q[:34]}...")
        with ft.disable_adapter():                 # 原模型
            a_base, t_base = _gen(base, tok, q)
        a_ft, t_ft = _gen(ft, tok, q)              # 微调模型
        results.append({
            "question": q, "subject": item["subject"],
            "base_answer": a_base, "ft_answer": a_ft,
            "base_len": len(a_base), "ft_len": len(a_ft),
            "base_time": round(t_base, 2), "ft_time": round(t_ft, 2),
        })
    return results


# ── 3. 自动评分（LLM 裁判） ──────────────────────────────

SCORE_PROMPT = """你是严格的考研辅导答案评审。请给下面两个回答打分。

问题：{q}

【回答A】
{a}

【回答B】
{b}

评分维度（各 1-5 分，5 分最好）：
1. 准确性：知识点/公式/时间是否正确
2. 完整性：是否答全了问题的要点
3. 简洁性：是否有可说可不说的废话（越精炼越高）

请**只输出 JSON**，不要 markdown 代码块：
{{"A": {{"accuracy": 分, "completeness": 分, "conciseness": 分}},
  "B": {{"accuracy": 分, "completeness": 分, "conciseness": 分}},
  "winner": "A 或 B 或 tie", "reason": "20字内的理由"}}"""


def score_answer(question: str, a: str, b: str) -> dict:
    """用 LLM 裁判给两个回答打分并选出更好的"""
    prompt = SCORE_PROMPT.format(q=question, a=a[:800], b=b[:800])
    try:
        reply = ask_glm(prompt, "你是严格的评审，只输出 JSON。", model="glm-4-flash")
        reply = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
        m = re.search(r"\{.*\}", reply, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception as e:
        return {"error": str(e)[:80], "winner": "?", "total_a": 0, "total_b": 0}

    def tot(d):
        return sum(d.get(k, 0) for k in ("accuracy", "completeness", "conciseness"))

    return {
        "score_base": data.get("A", {}), "score_ft": data.get("B", {}),
        "total_base": tot(data.get("A", {})), "total_ft": tot(data.get("B", {})),
        "winner": data.get("winner", "?"), "reason": data.get("reason", ""),
    }


def evaluate_all(results: list[dict], verbose: bool = True) -> list[dict]:
    """对每条结果做 LLM 评分"""
    for i, r in enumerate(results, 1):
        if verbose:
            print(f"  评分 [{i}/{len(results)}] {r['question'][:28]}...")
        s = score_answer(r["question"], r["base_answer"], r["ft_answer"])
        r.update(s)
        if verbose and "reason" in s:
            print(f"     原 {s['total_base']}/15  微调 {s['total_ft']}/15  → {s['winner']}（{s['reason'][:24]}）")
    return results


# ── 4. 报告生成 ─────────────────────────────────────────

def generate_report(results: list[dict], model_info: dict) -> tuple[str, str]:
    """生成 eval_report.txt + eval_report.json"""
    n = len(results)
    cnt = {"A": 0, "B": 0, "tie": 0, "?": 0}
    for r in results:
        cnt[r.get("winner", "?")] = cnt.get(r.get("winner", "?"), 0) + 1

    avg_base = sum(r.get("total_base", 0) for r in results) / n
    avg_ft = sum(r.get("total_ft", 0) for r in results) / n
    avg_len_base = sum(r["base_len"] for r in results) / n
    avg_len_ft = sum(r["ft_len"] for r in results) / n
    avg_t_base = sum(r["base_time"] for r in results) / n
    avg_t_ft = sum(r["ft_time"] for r in results) / n

    lines = []
    ap = lines.append
    ap("=" * 72)
    ap("微调效果评估报告（自动生成）")
    ap("=" * 72)
    ap(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ap(f"测试题数: {n}")
    ap("")
    ap("【模型信息】")
    for k, v in model_info.items():
        ap(f"  {k}: {v}")
    ap("")
    ap("【总体指标】")
    ap(f"  自动评分总分（满分 15）:  原模型 {avg_base:.2f}  |  微调模型 {avg_ft:.2f}  "
       f"（{'提升' if avg_ft >= avg_base else '下降'} {abs(avg_ft-avg_base):.2f}）")
    ap(f"  平均回答长度:            原模型 {avg_len_base:.0f} 字  |  微调模型 {avg_len_ft:.0f} 字  "
       f"（{((avg_len_ft-avg_len_base)/avg_len_base*100):+.0f}%）")
    ap(f"  平均生成耗时:            原模型 {avg_t_base:.2f}s  |  微调模型 {avg_t_ft:.2f}s  "
       f"（{((avg_t_ft-avg_t_base)/avg_t_base*100):+.0f}%）")
    ap("")
    ap("【胜负统计（LLM 裁判）】")
    ap(f"  微调模型胜: {cnt.get('B', 0)} 题  ({cnt.get('B',0)/n*100:.0f}%)")
    ap(f"  原模型胜  : {cnt.get('A', 0)} 题  ({cnt.get('A',0)/n*100:.0f}%)")
    ap(f"  平手      : {cnt.get('tie', 0)} 题")
    ap("")
    ap("【分科目表现】")
    subjects = sorted({r["subject"] for r in results})
    ap(f"  {'科目':<20} {'微调胜':>6} {'原模型胜':>8} {'平手':>6} {'平均分差':>9}")
    for s in subjects:
        sub = [r for r in results if r["subject"] == s]
        w = sum(1 for r in sub if r.get("winner") == "B")
        l = sum(1 for r in sub if r.get("winner") == "A")
        t = sum(1 for r in sub if r.get("winner") == "tie")
        diff = sum(r.get("total_ft", 0) - r.get("total_base", 0) for r in sub) / len(sub)
        ap(f"  {s:<20} {w:>6} {l:>8} {t:>6} {diff:>+9.2f}")
    ap("")
    ap("【逐题明细】")
    for i, r in enumerate(results, 1):
        ap("")
        ap("-" * 72)
        ap(f"[{i}] 【{r['subject']}】{r['question']}")
        ap(f"  原模型  ({r['base_len']}字/{r['base_time']}s) 得分 {r.get('total_base','?')}/15")
        ap(f"    {r['base_answer'][:200]}")
        ap(f"  微调模型({r['ft_len']}字/{r['ft_time']}s) 得分 {r.get('total_ft','?')}/15")
        ap(f"    {r['ft_answer'][:200]}")
        ap(f"  裁判: {r.get('winner','?')} —— {r.get('reason','')}")
    ap("")
    ap("=" * 72)

    txt = "\n".join(lines)
    txt_path = os.path.join(DAY_DIR, "eval_report.txt")
    json_path = os.path.join(DAY_DIR, "eval_report.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)
    json.dump({"model_info": model_info,
               "summary": {"n": n, "avg_score_base": avg_base, "avg_score_ft": avg_ft,
                           "wins_ft": cnt.get("B", 0), "wins_base": cnt.get("A", 0),
                           "ties": cnt.get("tie", 0),
                           "avg_len_base": avg_len_base, "avg_len_ft": avg_len_ft,
                           "avg_time_base": avg_t_base, "avg_time_ft": avg_t_ft},
               "details": results},
              open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return txt_path, json_path


def prepare_test_questions() -> list[dict]:
    """返回测试题（课程要求的入口函数名）"""
    return TEST_QUESTIONS


if __name__ == "__main__":
    print(f"测试题 {len(TEST_QUESTIONS)} 道:")
    for i, t in enumerate(TEST_QUESTIONS, 1):
        print(f"  {i:>2}. 【{t['subject']}】{t['q']}")
