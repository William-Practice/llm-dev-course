# 大模型开发实战 · 27 天课程作业集

30 天大模型开发技能培训的完整实操作业，覆盖 **Prompt 工程 → RAG → Agent → 微调部署** 全链路。
每一课都跑通了实际代码，不是笔记堆砌。

> 环境：Ubuntu 24.04 / RTX 3050 Laptop 4GB / Python 3.12
> 模型：智谱 GLM 系列（API）+ Qwen2.5-1.5B-Instruct（本地微调）

---

## 目录导航

```
├── day01_03/           基础用法        GLM 调用封装、求职/学习助手、Function Calling 工具集
├── day04_06/           文档处理        四格式加载器、三种切分策略、防幻觉问答、知识问答
├── day08_09/           检索进阶        零依赖 BM25、混合检索、元数据过滤检索器
├── day13/              多模态 RAG      GLM-4V 图像理解、矢量图表整页渲染
├── day14/              Text-to-SQL     SQL 生成 + 写操作拦截 + 失败重试
├── day17/              手写 ReAct      Thought/Action/Observation 循环（含 4 大坑修复）
├── day18/              LangChain        @tool 工具集 + Agent（新旧 API 兼容）
├── day19/              自动调研 Agent   Plan-and-Execute + Reflexion
├── day20/              MCP 文件助手     FastMCP Server 7 工具 + 客户端封装
├── day21/              LangGraph 工作流 7 节点 + 2 条件分支 + 人工确认 + 回退循环
├── smart_study_agent/  RAG+Agent 综合   Phase 3 收官项目
├── finetune/           微调环境        一键脚本 + 模型下载 + 环境验证
├── day24/              训练数据集      蒸馏 604 条 → 清洗扩写 → 492 条
├── day25/              QLoRA 微调      4GB 显存配置 + rank 对比实验 + LoRA 权重
├── day26/              本地 API 服务   Ollama 部署 + OpenAI 兼容客户端
├── day27/              效果评估        15 题对比 + LLM 裁判 + 自动报告
└── Phase4_交付清单.md   Phase 4 汇总
```

另有一个独立仓库：[knowledge-base-rag](https://github.com/William-Practice/knowledge-base-rag)
（本项目课程中做的 5 层 RAG 架构 + FastAPI 服务，已单独发布）

---

## 四个阶段

### Phase 1-2 基础与 RAG（day01-16）

- **Function Calling**：手写 FC 循环；发现 `glm-4v-flash` 不支持工具调用，必须换 `glm-4-flash`
- **文档处理**：docx 用标准库 `zipfile`+`xml.etree` 解析（零依赖）；PDF 用 poppler 命令行
- **切分策略对比**：固定长度 / 递归字符 / 按 token，实测对比切分质量
- **混合检索**：向量 + BM25 分数融合（中文用 bigram 分词，无需 jieba），
  把"冒泡排序"题目的检索区分度从 0.95/0.68/0.64 拉到 **1.00/0.13/0.07**
- **防幻觉**：检索相似度低于阈值直接回"没找到"，不让模型硬答
- **多模态**：学术 PDF 的矢量图表用 `pdfimages` 抽不到，必须 `pdftoppm` 整页渲染

### Phase 3 Agent（day17-22）

- **ReAct 手写循环**：踩全了 4 个坑 —— scratchpad 污染导致死循环、Action 解析取错块、
  模型跳过工具心算算错、缺专用工具撞上限
- **Plan-and-Execute + Reflexion**：规划 → 执行 → 反思 → 重试 → 汇总报告
  （发现反思机制**只审形式不审事实**：8 步全通过，报告却把 LangGraph 说成 NLP 框架）
- **MCP 协议**：FastMCP Server + LangChain 客户端；踩了 `asyncio.run` 断连、
  返回嵌套格式导致小模型幻觉两个坑，"整理文件夹"实测自动串联 **10 次工具调用**
- **LangGraph**：StateGraph + 条件分支 + 人工确认 + 回退循环（含 `recursion_limit` 逃生口）
- **核心工程原则**：**程序层强制约束 >>> 提示词约束** —— Agent 会谎称"已完成"，
  必须自动续跑 + 校验产物 + 输入清洗

### Phase 4 微调部署评估（day23-27）

- **环境**：独立 venv + torch CUDA + LLaMA-Factory；实测踩坑记录在文档里
- **数据**：glm-4-flash 蒸馏 604 条 → 去重 108 条 → 扩写补长度（59.6 → 138.2 字）→ 492 条
- **训练**：QLoRA 4bit，rank 8/alpha 16/dropout 0.05，**7 分 14 秒**跑完 186 步，
  显存 2980/4096 MiB，loss 1.255 → 1.0111
- **rank 实验**：4/8/16 三档的 loss 差异 < 0.2%（参数翻 4 倍）→ **瓶颈是数据不是容量**
- **部署**：Ollama 0.34 用户级安装 + OpenAI 兼容接口（纯 urllib 客户端）
- **评估**：15 道手写测试题（训练集外），微调模型 8 胜 / 原模型 6 胜，
  平均回答 393 → 138 字（-65%），理工科题提升明显（数学 +2.33）

---

## 三个诚实的结论

1. **Agent 的"完成"要靠程序校验**。反思机制抓不住事实错误，提示词约束挡不住"谎称完成"。
2. **LLM 裁判不可靠**。实测它偏好短答案（总分相同时几乎总选短的），
   且**不验算数学** —— 给一个算错的极限题打了满分并判胜。关键题必须人工复核。
3. **小数据量下加大模型容量是无用功**。rank 从 4 提到 16 参数量翻 4 倍，
   loss 几乎不动，实际问题在数据质量和数量。

---

## 快速开始

```bash
# 1) 基础用法（需要 ZHIPU_API_KEY）
export ZHIPU_API_KEY=你的key
cd day01_03 && python study_assistant.py

# 2) MCP 文件助手
cd day20 && python mcp_file_assistant.py

# 3) 微调环境（详见 finetune/环境配置文档.md）
cd finetune && bash prepare_env.sh

# 4) 微调训练
llamafactory-cli train day25/train_qlora.yaml

# 5) 评估（一键出报告）
cd day27 && python run_eval.py
```

所有 API key 均从环境变量读取，仓库内**不含任何密钥**。

---

## 说明

- `day25/saves/` 保留了 rank 4/8/16 三套 LoRA 权重（各 17-70MB），可直接加载复现实验
- `knowledge_base/` 目录未包含在本仓库（已独立发布为 knowledge-base-rag）
- 训练中间态（checkpoint）和虚拟环境已通过 `.gitignore` 排除
