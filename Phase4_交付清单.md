# Phase 4 交付清单（Day 23-27 · 微调）

> 更新于 2026-09-27 23:20　项目：/data/Projects/GLMModelsDemo/

---

## 完成状态总览

| Day | 主题 | 状态 | 关键产出 |
|---|---|---|---|
| 23 | 微调环境就绪 | ✅ 完成 | 环境配置文档.md（3/3 检查通过） |
| 24 | 训练数据集 | ✅ 完成 | train.jsonl 492 条 + 数据集说明.md |
| 25 | LoRA 微调 | ✅ 完成 | LoRA 权重 35.3MB + 微调实验报告.md |
| 26 | 本地模型 API 服务 | ✅ 完成 | server_check.py + client.py + 三项功能实测通过 |
| 27 | 微调评估报告 | ✅ 完成 | eval_report.txt/json/md（15 题对比 + LLM 裁判） |

**Phase 4 收官。** 五个实验日全部跑通，含 4 个加分项（rank 对比、多子主题数据、
常见报错文档、模型候选清单）。

---

## Day 23 —— 微调环境

```
finetune/
├── 环境配置文档.md      8 节：魔搭流程 + 本机等价方案 + 5 条真实踩坑
├── prepare_env.sh      一键装环境（venv + torch + 依赖 + LLaMA-Factory）
├── download_model.sh   curl 直下模型（实测 2.1MB/s，比 hf-mirror 快 6000 倍）
├── download_model.py   modelscope SDK 版（带 hf-mirror 回退）
├── verify_env.py       三项检查脚本
└── venv/               6.7GB，torch 2.14.0+cu130
```

验证结果：**3/3 通过**
- GPU：RTX 3050 Laptop 3.68GB / sm_86 / bf16 ✓
- 框架：llamafactory 0.9.5 ✓
- 模型：Qwen2.5-1.5B-Instruct 2.89GB（sha256 校验通过）✓

---

## Day 24 —— 训练数据集

```
day24/
├── generate_data.py      生成 + 清洗 + 统计 + 抽查
├── expand_answers.py     答案扩写（59.6 → 137.9 字）
├── train.jsonl           492 条，严格 Alpaca 三字段（提交用，238KB）
├── train_with_tags.jsonl 492 条 + subject 分类标签
├── 数据集说明.md          9 节：来源/清洗/扩写/质量/抽查/用法/局限
└── gen_full.log / expand.log
```

数据统计：
- **492 条**（要求 500±50）✓
- output 平均 **138.2 字**（要求 100-200）✓
- 5 个子主题全覆盖：数学 111 / 408 108 / 政治 102 / 备考 87 / 英语 84
- 清洗：604 生成 → 去重 108 + 乱码 3 → 493 → 抽查删改 → 492
- 抽查 30 条发现 3 处错误，已修正 2 条、删除 1 条

---

## Day 25 —— LoRA 微调

```
day25/
├── train_qlora.yaml      主实验（rank=8, alpha=16, dropout=0.05）
├── train_rank4.yaml      rank=4 对比
├── train_rank16.yaml     rank=16 对比
├── compare.py            原模型 vs 微调模型（5 个问题）
├── rank_compare.py       三个 rank 权重效果对比
├── compare_results.json
├── 微调实验报告.md        7 节，含 loss 曲线、对比表、rank 实验
└── saves/
    ├── qwen2.5-1.5b-lora/       adapter_model.safetensors 35.3MB ★提交用
    ├── qwen2.5-1.5b-lora-r4/    17.7MB
    └── qwen2.5-1.5b-lora-r16/   70.5MB
```

训练结果（4GB 显存 QLoRA 4bit）：
| 指标 | 数值 |
|---|---|
| train_loss | 1.255 → **1.0111**（末段 0.81，降幅 35%） |
| 耗时 | 7 分 14 秒（186 步，2.3 s/it） |
| 显存 | 2980 / 4096 MiB（无 OOM） |
| 可训练参数 | 9,232,384（0.595%） |

对比结果：回答长度 443→141 字（-68%），速度 4.98→2.86s（-43%），质量 3/5 微调更好。

rank 对比：三者 loss 几乎相同（1.0095/1.0111/1.0102）→ **瓶颈是数据不是容量**。

---

## Day 26 —— 本地模型 API 服务（待跑）

```
day26/
├── server_check.py   服务检查 + 模型列表 + call_local_model()（纯 urllib 零依赖）
├── client.py         菜单：单轮 / 多轮 / 批量测试（含计时）
└── Modelfile         注册微调模型到 Ollama（FROM 基座 + ADAPTER LoRA）
```

待执行的命令：
```bash
# 方案 A：Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
ollama create kaoyan-qwen -f day26/Modelfile
python day26/client.py

# 方案 B：复用已有 llama.cpp（零下载）
llama-server -m /data/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
# 然后把 client.py 的 BASE_URL 改为 http://localhost:8080
```

---

## Day 27 —— 微调评估报告（待跑）

```
day27/
├── eval_model.py   15 道测试题 + 双模型对比 + LLM 裁判评分 + 报告生成
└── run_eval.py     一键跑
```

待执行的命令：
```bash
cd day27 && ../finetune/venv/bin/python run_eval.py
```
产出：eval_report.txt + eval_report.json，再据此写 eval_report.md（手写部分）。

---

## 环境速查

```
微调 venv:  /data/Projects/GLMModelsDemo/finetune/venv
            torch 2.14.0+cu130 / llamafactory 0.9.5 / transformers 5.6.0
基座模型:   /data/models/Qwen2.5-1.5B-Instruct (2.89GB)
数据集:     /data/Projects/GLMModelsDemo/day24/train.jsonl (492 条)
LoRA 权重:  /data/Projects/GLMModelsDemo/day25/saves/qwen2.5-1.5b-lora
磁盘:       /data 剩 96GB
```
