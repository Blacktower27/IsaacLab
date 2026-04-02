#!/bin/bash
SERVER="jingxil@discovery.usc.edu"
REMOTE_DIR="/scratch1/jingxil/logs/rl_games/Forge"
LOCAL_DIR="./carc_logs"

rsync -avz --progress \
    "$SERVER:$REMOTE_DIR/2026-04-01_10-01-00" \
    "$SERVER:$REMOTE_DIR/2026-04-01_10-02-00" \
    "$SERVER:$REMOTE_DIR/2026-04-01_11-08-39" \
    "$LOCAL_DIR/"
