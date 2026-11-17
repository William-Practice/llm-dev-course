# glmtools.py

import os
import json
import subprocess
import urllib.request
import urllib.error


API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def glm_chat(prompt: str) -> str:
    """直接 HTTP POST JSON 调用 GLM，返回文本回复"""
    body = json.dumps({
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def report(extra_events: list[tuple[str, str]] | None = None) -> str:
    """
    让 GLM 计算今日日期 + 所有事件天数，返回完整报告字符串
    extra_events: [(事件名, 描述), ...]
    """
    # 第一步：从系统获取今日日期（纯字符串，不涉及 Python 日期计算）
    today = subprocess.run(["date", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip()

    # 第二步：构造事件列表
    events = [
        ("国庆节", "今年10月1日"),
        ("下半年软考", "今年11月第一个周六"),
        ("IST投稿通知", "按审稿周期4个月估算"),
        ("明年春节", ""),
        ("毕业", "2028年6月10日"),
    ]
    if extra_events:
        events.extend(extra_events)

    lines = [f"{i+1}. {name}（{desc}）" for i, (name, desc) in enumerate(events)]

    # 第三步：让 GLM 一次性算出所有天数，输出 JSON
    prompt = f"""今天是 {today}。
请计算距离以下每个事件还有多少天（正数）或已经过去多少天（负数）。

{chr(10).join(lines)}

严格按照以下 JSON 格式输出，不要 markdown 代码块，不要其他文字：
{{"today":"{today}","events":[{{"name":"国庆节","date":"YYYY-MM-DD","days_left":数字}},...]}}
"""

    reply = glm_chat(prompt).strip()
    # 清理可能的 markdown 包裹
    reply = reply.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(reply)

    # 组装可读报告
    out = [f"今天日期: {data['today']}", "─" * 48]
    for e in data["events"]:
        d = e["days_left"]
        if d > 0:
            suffix = f"距离还有 {d} 天"
        elif d == 0:
            suffix = "就是今天！"
        else:
            suffix = f"已经过去 {-d} 天"
        out.append(f"[{e['name']}] {e['date']}  {suffix}")

    return "\n".join(out)


def custom(name: str, desc: str) -> str:
    """快捷：计算任意事件天数，desc 里写日期或描述即可"""
    today = subprocess.run(["date", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip()
    data = json.loads(glm_chat(
        f"今天是 {today}。"
        f"请计算距离 {name}（{desc}）还有多少天。"
        f"只输出 JSON：{{'name':'{name}','date':'YYYY-MM-DD','days_left':数字}}"
    ).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    d = data["days_left"]
    if d > 0:
        return f"[{data['name']}] {data['date']}  距离还有 {d} 天"
    elif d == 0:
        return f"[{data['name']}] {data['date']}  就是今天！"
    else:
        return f"[{data['name']}] {data['date']}  已经过去 {-d} 天"


if __name__ == "__main__":
    print(report())
