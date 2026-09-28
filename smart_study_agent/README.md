# 智能学习助手（Phase 3 收官项目）

RAG + Agent 融合的多工具协作系统：能查资料、能算日期、能记笔记。

## 文件结构

```
smart_study_agent/
├── rag_tool.py       RAG 知识库查询工具（ask_kb）
├── study_tools.py    日期/笔记工具（get_today / days_until / write_note / read_note / list_notes）
├── study_agent.py    Agent 主程序（多轮对话 + 工具调用日志）
├── docs/             3 个知识文档（大数据技术栈 / 软考软件设计师 / 考研政治）
├── notes/            Agent 自动写的笔记
└── kb_index.json     向量索引缓存（首次运行自动生成，提交时可删）
```

## 运行

```bash
export ZHIPU_API_KEY=你的key        # 绝不硬编码
python study_agent.py
```

示例对话：

```
你: Kafka 的分区有什么作用？          → 调 ask_kb，基于知识库回答
你: 软考还有多少天？                  → 调 days_until，41 天
你: 查一下 Kafka 分区，记成笔记       → ask_kb + write_note 两步
你: 读一下我记的 Kafka核心 笔记        → 调 read_note
```

## 用了什么工具

- **LangChain + LangGraph**：`create_react_agent` 编排 ReAct 循环，`MemorySaver` 做多轮对话记忆
- **RAG**：docs/ 文档切块 → embedding-2 向量 → 余弦相似度检索，索引按内容 hash 缓存
- **自定义工具**：5 个日期/笔记工具，全部 `@tool` 装饰
- **LLM**：智谱 glm-4-flash（Function Calling）

## Agent 怎么决策

system prompt 里给了明确的工具选择规则：知识类问题先 `ask_kb` 检索（不靠模型记忆，
避免编造），日期问题先 `get_today`/`days_until`，组合任务必须把每一步都执行完。

程序层还有**两道兜底**（Day 17 学到的经验）：
1. Agent 只调工具不给回答 → 自动追加"请继续完成剩余步骤"续跑（最多 2 次）
2. `write_note` 自动清洗内容，去掉检索工具返回的"相似度/来源"等元信息

## 亮点

- **RAG 与 Agent 真正融合**：检索结果以工具返回值进入 ReAct 循环，Agent 自主决定何时检索
- **完整工具调用日志**：每次调用打印 `🔧 工具名(参数)` 和 `👀 返回内容`，便于调试
- **程序层强制约束**：不依赖模型自觉，用代码保证"说要写就一定写了"
- **零脏数据**：笔记内容经过清洗，只留整理后的正文

## 注意

- 提交时**不要**带 `.env`、`kb_index.json`、`notes/`
- 首次运行会调 embedding 接口建索引（3 篇文档 / 17 个片段，约 3 秒）
