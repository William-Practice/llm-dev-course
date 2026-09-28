"""
study_planner.py — Day 03 作业：AI 学习规划管家
功能：生成复习计划（CoT+FC）/ 倒计时提醒（FC）/ 健康建议（Few-shot+FC）
复用 glm_tools.ask_with_tools()
"""

from hello_glm import save_output
from glm_tools import ask_with_tools


def generate_plan(goal: str, exam_date: str) -> str:
    """功能1：CoT + Function Calling 生成复习计划"""
    prompt = f"""你是学习规划专家。我要准备：{goal}，考试日期：{exam_date}。

请一步一步思考（CoT）：
1. 先用工具查出距离考试还有几天
2. 分析这个目标需要掌握哪些内容
3. 按内容重要程度分配天数
4. 用 Markdown 表格输出按天复习计划

表格列：阶段 | 天数范围 | 复习内容 | 每日建议时长 | 重点提示"""
    return ask_with_tools(prompt)


def countdown(target_date: str) -> str:
    """功能2：倒计时提醒（根据剩余天数调整语气 —— 加分项）"""
    prompt = f"""距离 {target_date} 还有几天？请先用工具算出准确天数。

然后根据剩余天数给我一句加油语（要求）：
- 剩 100 天以上：温和鼓励，强调打基础
- 剩 30-100 天：提醒抓紧，强调计划执行
- 剩 10-30 天：紧张感，强调查漏补缺
- 剩 10 天以内：冲刺语气，强调心态和作息
"""
    return ask_with_tools(prompt)


def health_advice(height_cm: str, weight_kg: str) -> str:
    """功能3：Few-shot + Function Calling 健康建议"""
    prompt = f"""你是健康顾问。我身高 {height_cm} cm，体重 {weight_kg} kg。

请先用工具算出我的 BMI，然后给作息建议。按以下格式输出（参考例子）：

例子1：BMI=22.5，正常。建议：每天睡 7 小时，学习 50 分钟休息 10 分钟，每周运动 3 次。
例子2：BMI=26.8，偏胖。建议：每天运动 30 分钟，少吃夜宵，注意久坐后起身活动。

请按此格式给出我的建议。"""
    return ask_with_tools(prompt)


def todo_countdown(todos: str) -> str:
    """加分项：输入待办事项，AI 自动算每个截止日期还剩几天"""
    prompt = f"""以下是我的待办事项（含截止日期），请先用工具逐个算出每个截止日期还剩几天，然后按紧急程度排序给出提醒：

{todos}"""
    return ask_with_tools(prompt)


def main():
    print("╔══════════════════════════════════════╗")
    print("║      🤖 AI 学习规划管家              ║")
    print("╚══════════════════════════════════════╝")

    while True:
        print("""
请选择功能：
  1. 生成复习计划（目标 + 考试日期）
  2. 倒计时提醒（输入日期）
  3. 健康建议（身高体重 → BMI + 作息）
  4. 待办倒计时（多个截止日期）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见，祝你学习高效！")
            break
        elif choice == "1":
            goal = input("你的目标（如：软考软件设计师冲刺）: ").strip()
            date = input("考试日期（如：2026-11-07）: ").strip()
            if not goal or not date:
                print("  ✗ 目标和日期都不能为空")
                continue
            print("\n正在规划...\n")
            result = generate_plan(goal, date)
            print(result)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【复习计划 - {goal}】\n{result}")
        elif choice == "2":
            date = input("目标日期（如：2026-10-01）: ").strip()
            if not date:
                print("  ✗ 日期不能为空")
                continue
            print()
            print(countdown(date))
        elif choice == "3":
            h = input("身高（cm）: ").strip()
            w = input("体重（kg）: ").strip()
            if not h or not w:
                print("  ✗ 身高体重都不能为空")
                continue
            print()
            print(health_advice(h, w))
        elif choice == "4":
            print("请输入待办事项（每行一条，含截止日期，输入 END 结束）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            if not lines:
                print("  ✗ 不能为空")
                continue
            print()
            print(todo_countdown("\n".join(lines)))
        else:
            print("  ✗ 请输入 0-4")


if __name__ == "__main__":
    main()
