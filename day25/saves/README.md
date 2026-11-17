# 微调产物说明

此处仅保留训练日志、评估结果与曲线（`*.json`、`*.jsonl`、`training_loss.png`），用于复现实验结论。

**权重文件不再入库**：`adapter_model.safetensors` 与 `tokenizer.json` 体积较大（合计约 87MB），已移出 git 跟踪，改由仓库 Release 的 `day25-lora-adapters.zip` 附件提供。需要推理时下载解压到对应目录即可恢复原结构。

被移出的文件：

| 目录 | 文件 |
| --- | --- |
| `qwen2.5-1.5b-lora/` | `adapter_model.safetensors`（36MB）、`tokenizer.json`（11MB） |
| `qwen2.5-1.5b-lora-r4/` | `adapter_model.safetensors`（18MB）、`tokenizer.json`（11MB） |
| `qwen2.5-1.5b-lora-r16/` | `tokenizer.json`（11MB） |

训练配置与超参见 `../train_qlora.yaml`、`../train_rank4.yaml`、`../train_rank16.yaml`，实验结论见 `../微调实验报告.md`。
