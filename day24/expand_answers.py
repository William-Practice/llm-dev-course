"""
expand_answers.py — Day 24 补丁：把过短的答案扩写到 120-200 字
保留原答案的事实内容，补充背景/例子/分点，不改问题

用法: python expand_answers.py
"""

import os
import re
import sys
import json
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_PROJ, "day01_03") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJ, "day01_03"))
from hello_glm import ask_glm
from generate_data import parse_json_array

DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DIR, "train.jsonl")
BACKUP = os.path.join(DIR, "train_v1_short.jsonl")
BATCH = 10
MIN_LEN = 100          # 短于这个长度就要扩写


def expand_batch(items: list[dict]) -> list[str]:
    """把一批问答的答案扩写，返回扩写后的答案列表（与输入同序）"""
    numbered = "\n".join(
        f"{i}. 问题：{it['instruction']}\n   现有答案（{len(it['output'])} 字）：{it['output']}"
        for i, it in enumerate(items, 1)
    )
    prompt = f"""下面 {len(items)} 条考研问答的答案太短了，请把每条答案扩写到 **120-180 字**。

{numbered}

扩写要求：
1. **严格控制在 120-180 字之间，绝对不要超过 200 字**（这是硬性要求）
2. **保留原答案的全部信息，不能改变事实**（考点、时间、公式必须准确）
3. 补充内容的方向（挑 1-2 个即可，不要全都写）：背景解释、一个具体例子、分点说明、易错点提示
4. 不要空话套话，每一句都要有信息量
5. 问题（instruction）保持原样不变

**只输出 JSON 数组**，不要 markdown 代码块，不要解释。格式：
[
  {{"instruction": "原问题", "output": "扩写后的答案"}},
  ...
]"""
    reply = ask_glm(prompt, "你是考研辅导专家，只输出 JSON 数组。")
    parsed = parse_json_array(reply)
    # 按顺序取 output
    outs = []
    for i in range(len(items)):
        if i < len(parsed) and parsed[i].get("output"):
            outs.append(str(parsed[i]["output"]).strip())
        else:
            outs.append("")            # 缺失标记
    return outs


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    print(f"读取 {len(rows)} 条，其中 <{MIN_LEN} 字的需扩写")

    # 备份原始
    with open(BACKUP, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已备份原文件 → {os.path.basename(BACKUP)}")

    todo = [i for i, r in enumerate(rows) if len(r["output"]) < MIN_LEN]
    print(f"待扩写: {len(todo)} 条，分 {(len(todo)+BATCH-1)//BATCH} 批\n")

    ok, failed = 0, 0
    t0 = time.time()
    for bi in range(0, len(todo), BATCH):
        idxs = todo[bi:bi + BATCH]
        batch = [rows[i] for i in idxs]
        try:
            outs = expand_batch(batch)
        except Exception as e:
            print(f"  批 {bi//BATCH+1} 失败: {type(e).__name__}: {str(e)[:80]}")
            failed += len(idxs)
            time.sleep(2)
            continue

        for idx, new_out in zip(idxs, outs):
            if new_out and 100 <= len(new_out) <= 300:
                rows[idx]["output"] = new_out
                ok += 1
            else:
                failed += 1

        done = bi + len(idxs)
        el = time.time() - t0
        eta = el / done * (len(todo) - done) if done else 0
        print(f"  批 {bi//BATCH+1}/{(len(todo)+BATCH-1)//BATCH}  已处理 {done}/{len(todo)}  "
              f"成功 {ok}  失败 {failed}  用时 {el:.0f}s  预计剩余 {eta:.0f}s")

    # 写回
    with open(SRC, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["output"]) for r in rows]
    print(f"\n✅ 扩写完成: 成功 {ok} / 失败 {failed}")
    print(f"新字数: 最小 {min(lens)} / 平均 {sum(lens)/len(lens):.1f} / 最大 {max(lens)}")
    print(f"已写回 {SRC}")


if __name__ == "__main__":
    main()
