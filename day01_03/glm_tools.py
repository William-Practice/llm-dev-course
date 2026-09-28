"""
glm_tools.py — Day 03 工具集：让 GLM 能"算天数、查日期、算 BMI"
提供 ask_with_tools()：手写 Function Calling 循环（纯 HTTP/JSON，无第三方 SDK）
"""

import os
import json
import urllib.request
from datetime import datetime, timedelta

API_KEY = os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4-flash"          # 支持 Function Calling


# === 三个工具函数 ===

def get_today() -> str:
    """获取今天日期"""
    return datetime.now().strftime("%Y-%m-%d")


def days_until(target_date: str) -> int:
    """算距离某天还有几天（负数=已过去）"""
    t = datetime.strptime(target_date, "%Y-%m-%d").date()
    return (t - datetime.now().date()).days


def calculate_bmi(height_cm: float, weight_kg: float) -> dict:
    """算 BMI 并给出健康分类"""
    h = float(height_cm) / 100
    bmi = float(weight_kg) / (h * h)
    if bmi < 18.5:
        level = "偏瘦"
    elif bmi < 24:
        level = "正常"
    elif bmi < 28:
        level = "偏胖"
    else:
        level = "肥胖"
    return {"bmi": round(bmi, 2), "level": level}


# === 工具定义（告诉 AI 有哪些工具可用）===

TOOLS = [
    {"type": "function", "function": {
        "name": "get_today",
        "description": "获取今天的日期，格式 YYYY-MM-DD",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "days_until",
        "description": "计算距离某个日期还有多少天，负数表示已经过去多少天",
        "parameters": {
            "type": "object",
            "properties": {"target_date": {"type": "string", "description": "目标日期，格式 YYYY-MM-DD"}},
            "required": ["target_date"],
        },
    }},
    {"type": "function", "function": {
        "name": "calculate_bmi",
        "description": "根据身高体重计算 BMI 数值和健康分类",
        "parameters": {
            "type": "object",
            "properties": {
                "height_cm": {"type": "number", "description": "身高，单位厘米"},
                "weight_kg": {"type": "number", "description": "体重，单位千克"},
            },
            "required": ["height_cm", "weight_kg"],
        },
    }},
]

FUNC_MAP = {
    "get_today": get_today,
    "days_until": days_until,
    "calculate_bmi": calculate_bmi,
}


# === Function Calling 循环 ===

def _post(body: dict) -> dict:
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask_with_tools(question: str, max_rounds: int = 5) -> str:
    """
    带工具的问答：AI 自主决定是否调工具，拿到真实数据后再回答
    :param question: 用户问题
    :param max_rounds: 最多工具调用轮数（防死循环）
    """
    messages = [{"role": "user", "content": question}]

    for _ in range(max_rounds):
        data = _post({"model": MODEL, "messages": messages, "tools": TOOLS})
        msg = data["choices"][0]["message"]

        # 不需要工具 → 直接返回答案
        if not msg.get("tool_calls"):
            return (msg.get("content") or "").strip()

        # 记录 assistant 的工具调用
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": msg["tool_calls"],
        })

        # 依次执行工具，结果塞回对话
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            fn = FUNC_MAP.get(name)
            try:
                result = fn(**args) if fn else f"未知工具: {name}"
            except Exception as e:
                result = f"工具执行错误: {e}"
            print(f"  [工具调用] {name}({args}) → {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "（达到最大工具调用轮数）"


if __name__ == "__main__":
    print(ask_with_tools("今天几月几号？"))
    print(ask_with_tools("距离2026年12月20日还有几天？"))
    print(ask_with_tools("我身高175cm体重70kg，BMI是多少？"))
