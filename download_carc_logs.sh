#!/bin/bash
SERVER="jingxil@discovery.usc.edu"
REMOTE_DIR="/scratch1/jingxil/logs/rl_games/Forge"
LOCAL_DIR="./carc_logs"

rsync -avz --progress \
    "$SERVER:$REMOTE_DIR/2026-04-02_09-01-17" \
    "$SERVER:$REMOTE_DIR/2026-04-02_09-04-50" \
    "$LOCAL_DIR/"
