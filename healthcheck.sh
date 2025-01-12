#!/bin/bash
# ログファイルのパス
LOG_FILE="/app/work/all.log"

# ログを出力する関数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}
APP_START_PY="/app/server.py"

# 所有者変更
mkdir -p /app/work
sudo chown -R $USER:$USER /app/work

cp -p .env /app/work/
cd /app/work
export PYTHONPATH=.

# /app/server.pyが起動しているかチェック
if ! pgrep -f "python ${APP_START_PY}" >/dev/null; then
    log "${APP_START_PY} is not running. Starting it now..."
    # open_interpreterを起動
    if bash -c "eval $(pyenv init -) && python ${APP_START_PY} 2>&1" & then
        log "${APP_START_PY} started successfully."
        sleep 3600
        exit 0
    else
        log "Failed to start ${APP_START_PY}."
        exit 1
    fi
else
    log "${APP_START_PY} is already running."
    exit 0
fi
