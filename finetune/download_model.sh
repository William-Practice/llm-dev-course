#!/usr/bin/env bash
# download_model.sh — 从魔搭下载 Qwen2.5-1.5B-Instruct（curl 直下，支持断点续传）
# 比 modelscope SDK 更快更可控；中断后重跑会自动续传
set -u

DIR=/data/models/Qwen2.5-1.5B-Instruct
BASE="https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/master"
FILES="config.json configuration.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors"

mkdir -p "$DIR"
unset http_proxy https_proxy all_proxy          # 镜像直连，不走代理

echo "目标目录: $DIR"
for f in $FILES; do
    if [ -f "$DIR/$f" ]; then
        want=$(curl -sI -L --max-time 20 "$BASE/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
        have=$(stat -c%s "$DIR/$f" 2>/dev/null || echo 0)
        if [ -n "$want" ] && [ "$have" = "$want" ]; then
            echo "✓ $f 已完成（$((have/1024/1024)) MB）"
            continue
        fi
    fi
    echo "↓ 下载 $f ..."
    curl -L -C - --retry 3 --retry-delay 5 --max-time 3600 -o "$DIR/$f" "$BASE/$f" \
        --progress-bar 2>&1 | tail -1
done

echo
echo "═══ 下载结果 ═══"
du -sh "$DIR"
ls -la "$DIR" | awk '{print $5, $9}'
