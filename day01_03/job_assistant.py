"""
job_assistant.py — Day 01 作业：AI 求职小助手
命令行程序：优化简历 / 写自我介绍 / 生成面试题 / 模拟面试
复用 hello_glm.ask_glm()，每个功能一个独立函数 + 不同 system 角色
"""

from hello_glm import ask_glm, save_output


# === 3 个核心功能 ===

def optimize_resume(job: str, resume: str) -> str:
    """功能1：优化简历"""
    system = f"你是资深{job}岗位的简历顾问，擅长把学生简历改得专业、有吸引力、突出重点。"
    prompt = f"""请优化这份求职{job}岗位的简历，要求：
1. 给出优化后的完整简历
2. 简要说明主要改动点

原简历：
{resume}"""
    return ask_glm(prompt, system)


def write_intro(job: str) -> str:
    """功能2：写自我介绍"""
    system = f"你是面试辅导专家，擅长为求职{job}岗位的应届生写自然、不生硬的自我介绍。"
    prompt = f"请写一段求职{job}岗位的自我介绍，时长约 2 分钟（300 字左右），口语化、真诚、有条理。"
    return ask_glm(prompt, system)


def gen_interview(job: str) -> str:
    """功能3：生成面试题（5 道题 + 参考答案）"""
    system = f"你是{job}岗位的资深面试官，出题贴近真实面试。"
    prompt = f"""请为{job}岗位生成 5 道常见的面试问题，每题附参考答案要点。

格式：
【问题1】xxx
参考答案：xxx
..."""
    return ask_glm(prompt, system)


# === 加分项：模拟面试 ===

def mock_interview(job: str, rounds: int = 3):
    """加分项：AI 当面试官，连续提问，用户回答后点评"""
    print(f"\n=== 模拟面试开始（{job}岗位，共 {rounds} 轮）===")
    history = []
    for i in range(1, rounds + 1):
        # AI 出题
        if i == 1:
            q = ask_glm(f"你是{job}岗位面试官。请提第一个面试问题，只输出问题本身。",
                        f"你是{job}岗位的面试官，问题专业且贴近实战。")
        else:
            ctx = "\n".join(f"问：{h['q']}\n答：{h['a']}" for h in history)
            q = ask_glm(f"""你是{job}岗位面试官。以下是你和候选人的对话：
{ctx}

请根据候选人的回答，提出第 {i} 个追问，只输出问题本身。""",
                        f"你是{job}岗位的面试官。")
        print(f"\n【面试官 - 第{i}题】{q}")
        a = input("你的回答（直接回车跳过点评）: ").strip()
        history.append({"q": q, "a": a or "(未回答)"})
        if a:
            fb = ask_glm(f"候选人回答这道题：「{q}」\n候选人的回答：{a}\n请简短点评（优点+改进建议，3句以内）。",
                         f"你是{job}岗位面试官，点评中肯简洁。")
            print(f"\n【点评】{fb}")

    # 总结
    summary = ask_glm("以下是模拟面试记录：\n" +
                      "\n".join(f"问：{h['q']}\n答：{h['a']}" for h in history) +
                      "\n请给出整体表现评价和提升建议。", f"你是{job}岗位面试官。")
    print(f"\n=== 面试总结 ===\n{summary}")


# === 菜单主程序 ===

MENU = """
╔══════════════════════════════════════╗
║        🤖 AI 求职小助手              ║
╠══════════════════════════════════════╣
║  1. 优化简历（输入岗位+简历）        ║
║  2. 写自我介绍（输入岗位）           ║
║  3. 生成面试题（输入岗位）           ║
║  4. 模拟面试（AI 当面试官）          ║
║  0. 退出                             ║
╚══════════════════════════════════════╝"""


def main():
    print("欢迎使用 AI 求职小助手！")
    while True:
        print(MENU)
        choice = input("请输入数字: ").strip()

        if choice == "0":
            print("👋 再见，祝你求职顺利！")
            break

        elif choice == "1":
            job = input("目标岗位: ").strip()
            if not job:
                print("  ✗ 岗位不能为空")
                continue
            print("请粘贴你的简历内容（输入 END 结束）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            resume = "\n".join(lines)
            if not resume.strip():
                print("  ✗ 简历内容不能为空")
                continue
            print("\n正在优化...\n")
            result = optimize_resume(job, resume)
            print(result)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【简历优化 - {job}】\n{result}")

        elif choice == "2":
            job = input("目标岗位: ").strip()
            if not job:
                print("  ✗ 岗位不能为空")
                continue
            print("\n正在撰写...\n")
            result = write_intro(job)
            print(result)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【自我介绍 - {job}】\n{result}")

        elif choice == "3":
            job = input("目标岗位: ").strip()
            if not job:
                print("  ✗ 岗位不能为空")
                continue
            print("\n正在出题...\n")
            result = gen_interview(job)
            print(result)
            if input("\n保存到文件？(y/n): ").strip().lower() == "y":
                save_output(f"【面试题 - {job}】\n{result}")

        elif choice == "4":
            job = input("目标岗位: ").strip()
            if not job:
                print("  ✗ 岗位不能为空")
                continue
            mock_interview(job)

        else:
            print("  ✗ 请输入 0-4")


if __name__ == "__main__":
    main()
