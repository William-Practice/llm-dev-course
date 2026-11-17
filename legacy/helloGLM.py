import os
import base64
from pathlib import Path
from openai import OpenAI

# 从环境变量读取 API Key
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY，例如：export ZHIPU_API_KEY='your_key_here'")


def _image_to_data_uri(image_path: str) -> str:
    """将本地图片转为 data URI (base64)"""
    path = Path(image_path)
    ext = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


class HelloGLM:
    """智谱 GLM API 封装类（文本对话 + 图像理解）"""

    def __init__(self, model: str = "glm-4v-flash", system_prompt: str | None = None):
        self.client = OpenAI(
            api_key=ZHIPU_API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    # ── 文本对话 ──────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """单次对话（无记忆），不修改历史"""
        messages = self.messages.copy()
        messages.append({"role": "user", "content": user_input})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return resp.choices[0].message.content

    def chat_turn(self, user_input: str) -> str:
        """多轮对话（有记忆），自动维护对话上下文"""
        self.messages.append({"role": "user", "content": user_input})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )

        reply = resp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """清空对话历史（保留 system prompt）"""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    @property
    def history(self) -> list:
        """查看当前对话历史"""
        return self.messages.copy()

    # ── 图像理解 ──────────────────────────────────────────

    def analyze_image(self, image_path: str, prompt: str = "请详细描述这张图片的内容") -> str:
        """单张图片理解（无记忆）"""
        uri = _image_to_data_uri(image_path)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content

    def analyze_images(self, image_paths: list[str], prompt: str = "请对比分析这些图片") -> str:
        """多张图片理解（无记忆）"""
        content = [{"type": "text", "text": prompt}]
        for p in image_paths:
            uri = _image_to_data_uri(p)
            content.append({"type": "image_url", "image_url": {"url": uri}})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content


if __name__ == "__main__":
    # ── 文本对话测试 ──
    bot = HelloGLM(
        model="glm-4v-flash",
        system_prompt="你是一个友好的AI助手，回答简洁准确。"
    )

    r1 = bot.chat_turn("你好，我叫小明")
    print("GLM:", r1)

    r2 = bot.chat_turn("你还记得我叫什么名字吗？")
    print("GLM:", r2)

    # ── 图像理解测试 ──
    pic_dir = Path("/data/Projects/GLMModelsDemo/pic_knlg")
    images = sorted(pic_dir.glob("*"))
    print(f"\n=== 图像理解测试 ({len(images)}张图片) ===\n")

    for img in images:
        print(f"--- {img.name} ---")
        desc = bot.analyze_image(str(img), "请用一句话描述这张图片")
        print(desc + "\n")
