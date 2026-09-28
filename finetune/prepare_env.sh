#!/usr/bin/env bash
# prepare_env.sh — Day 23 微调环境准备（本机等价方案，已实测通过）
# 用法: bash prepare_env.sh
# 环境: Ubuntu 24.04 / Python 3.12 / RTX 3050 Laptop 4GB / 驱动 595.84
# 实测耗时: torch 段约 33 分钟（PyPI 版带 ~3GB nvidia-cu13 运行时），其余约 5 分钟
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/venv"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"      # 清华 PyPI 镜像

echo "═══ 1/5 创建独立虚拟环境（不要用系统 python，避免依赖污染）═══"
[ -d "$VENV" ] || python3 -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install -U pip setuptools wheel -q -i "$MIRROR"

echo "═══ 2/5 安装 PyTorch（CUDA 版，约 33 分钟）═══"
# 注意：PyPI 上的 Linux wheel 即 CUDA 版，但会连带下载 nvidia-cu13 全家桶（~3GB）
"$PIP" install torch -i "$MIRROR"

echo "═══ 3/5 安装微调依赖 ═══"
"$PIP" install -q numpy "transformers>=4.45" "datasets>=2.20" \
    "accelerate>=0.34" "peft>=0.12" "trl>=0.9" "bitsandbytes>=0.43" \
    sentencepiece protobuf modelscope huggingface_hub -i "$MIRROR"

echo "═══ 4/5 安装 LLaMA-Factory ═══"
# 让它自行解析依赖版本（它会自动把 transformers 5.x 降级到兼容的 5.6.0）
"$PIP" install llamafactory -i "$MIRROR"

echo "═══ 5/5 验证环境 ═══"
"$VENV/bin/python" "$ROOT/verify_env.py"
