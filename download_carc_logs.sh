#!/bin/bash
SERVER="jingxil@discovery.usc.edu"
REMOTE_DIR="/scratch1/jingxil/logs/rl_games/Forge"
REMOTE_DIR2="/scratch1/jingxil/logs/rl_games/Assembly"
LOCAL_DIR="./carc_logs"

rsync -avz --progress \
    "$SERVER:$REMOTE_DIR2/2026-04-17_16-00-58" \
    "$SERVER:$REMOTE_DIR/2026-04-17_16-08-07" \
    "$LOCAL_DIR/"
