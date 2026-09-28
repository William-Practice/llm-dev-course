"""
study_assistant.py — Day 02 作业：AI 学习助手
输入学习笔记 → 总结考点（Markdown）/ 出练习题 / 生成复习计划（表格）
用 CRISPE 框架写提示词 + 输出格式控制 + 边界约束
"""

from hello_glm import ask_glm, save_output


# === CRISPE 框架说明 ===
# Capacity(角色) Role(身份) Insight(背景) Statement(要求) Personality(风格) Experiment(格式)

def summarize(notes: str) -> str:
    """功能1：总结考点 —— 分3-5点 + 记忆口诀，Markdown 格式"""
    system = "你是资深考点总结师，擅长从学习笔记中提炼核心考点，语言精炼、条理清晰。"
    prompt = f"""请根据下面的学习笔记，总结核心考点。

要求（CRISPE）：
- 角色：考点总结师
- 背景：这是学生的考试复习笔记
- 任务：提炼 3-5 个核心考点
- 格式：Markdown（每个考点一个标题 + 要点说明）+ 最后附一句记忆口诀
- 约束：只根据笔记内容总结，不要编造笔记里没有的内容

学习笔记：
{notes}"""
    return ask_glm(prompt, system)


def make_quiz(notes: str) -> str:
    """功能2：出练习题 —— 5道简答题 + 参考答案"""
    system = "你是出题老师，擅长根据知识点设计考查核心理解的练习题。"
    prompt = f"""请根据下面的学习笔记，出 5 道简答题，并给出参考答案。

要求：
- 题目要考查对笔记核心内容的理解，不要考死记硬背的细节
- 格式：先列 5 道题，再统一给参考答案，题号对应
- 约束：题目和答案都必须来源于下面的笔记内容

学习笔记：
{notes}"""
    return ask_glm(prompt, system)


def make_plan(notes: str, days: int = 7) -> str:
    """功能3：生成复习计划 —— 按天安排，表格格式"""
    system = "你是学习规划师，擅长把复习内容拆解成可执行的每日计划。"
    prompt = f"""请根据下面的学习笔记，制定一份 {days} 天复习计划。

要求：
- 先分析笔记涉及的知识板块
- 按重要性把内容分配到 {days} 天里
- 用 Markdown 表格输出，列：天数 | 复习内容 | 建议时长 | 重点提示
- 约束：计划内容必须基于笔记，不要加入笔记外的知识点

学习笔记：
{notes}"""
    return ask_glm(prompt, system)


# === 菜单主程序 ===

def main():
    print("╔══════════════════════════════════════╗")
    print("║        🤖 AI 学习助手                ║")
    print("╚══════════════════════════════════════╝")

    print("\n请粘贴你的学习笔记（输入 END 结束）:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    notes = "\n".join(lines).strip()

    if not notes:
        print("  ✗ 笔记内容不能为空，退出")
        return

    print(f"\n✅ 笔记已加载（共 {len(notes)} 字）")

    while True:
        print("""
请选择功能：
  1. 总结考点（Markdown + 记忆口诀）
  2. 出练习题（5 道简答题 + 参考答案）
  3. 生成复习计划（表格格式）
  0. 退出""")
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见，学习加油！")
            break
        elif choice == "1":
            print("\n正在总结考点...\n")
            result = summarize(notes)
            print(result)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【考点总结】\n{result}")
        elif choice == "2":
            print("\n正在出题...\n")
            result = make_quiz(notes)
            print(result)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【练习题】\n{result}")
        elif choice == "3":
            try:
                days = int(input("计划天数（回车默认 7）: ").strip() or "7")
            except ValueError:
                days = 7
            print(f"\n正在生成 {days} 天复习计划...\n")
            result = make_plan(notes, days)
            print(result)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【{days}天复习计划】\n{result}")
        else:
            print("  ✗ 请输入 0-3")


if __name__ == "__main__":
    main()
