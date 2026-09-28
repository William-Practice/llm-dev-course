"""
投递期刊 + 软考规划生成器
纯 HTTP + JSON 调用 GLM，CLI 输出
"""

import os
import json
import subprocess
import urllib.request
import urllib.error


API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def glm_chat(prompt: str, temperature: float = 0.1) -> str:
    """直接 HTTP POST JSON 调用 GLM"""
    body = json.dumps({
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def generate_plan() -> str:
    """生成投期刊 + 软考双线规划，返回 Markdown 文本"""
    today = subprocess.run(["date", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip()

    prompt = f"""你是科研 + 职业规划导师。今天是 {today}。

用户背景：
- 身份：中国研究生，知识图谱方向
- 目标期刊：Information and Software Technology (IST, ISSN: 0950-5849)
- 目标证书：软考中级 - 软件设计师（下半年考试，11月第一个周六）

请制定一份详细的、可执行的「双线并行规划」，包含以下两部分：

## 一、IST 期刊投稿规划
- 审稿周期：初审约 4 个月，修回后约 2-3 个月
- 从今天起倒推时间节点（投初稿 → 初审结果 → 修回 → 录用/拒稿）
- 每个节点的截止日期和目标

## 二、软考·软件设计师备考规划
- 考试时间：{today[:4]} 年 11 月第一个周六
- 考试科目：基础知识（上午选择题）+ 应用技术（下午案例题）
- 从今天到考试日的分阶段计划（基础→强化→冲刺）
- 每周建议投入时间

## 三、时间冲突协调建议
- 两项任务如何穿插安排
- 优先级建议

格式要求：用 Markdown 输出，日期标注清晰（YYYY-MM-DD），每个阶段带 checklist 风格的任务项。一、二、三部分用 --- 分隔。
"""

    return glm_chat(prompt, temperature=0.7)


def print_plan():
    """生成规划并打印到 CLI"""
    print("╔════════════════════════════════════════════╗")
    print("║    IST 投稿 + 软考软件设计师 · 双线规划    ║")
    print("╚════════════════════════════════════════════╝")
    print()
    plan = generate_plan()
    print(plan)
    print()
    print("─" * 50)
    print("规划由 GLM 生成，仅供参考，请结合实际情况调整。")


if __name__ == "__main__":
    print_plan()
