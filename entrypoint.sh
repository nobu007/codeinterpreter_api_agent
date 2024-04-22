#!/bin/bash

# ログファイルのパス
LOG_FILE="/app/work/all.log"

# 所有者変更
sudo chown -R $USER:$USER /app/work

# ログファイルが存在しない場合は作成
touch "$LOG_FILE"

# ログをtailして出力しつづける
tail -f "$LOG_FILE" 
