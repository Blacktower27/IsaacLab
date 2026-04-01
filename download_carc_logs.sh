#!/bin/bash
SERVER="jingxil@discovery.usc.edu"
REMOTE_DIR="/scratch1/jingxil/logs/rl_games/Forge"
LOCAL_DIR="./carc_logs"

rsync -avz --progress \
    "$SERVER:$REMOTE_DIR/2026-03-30_06-17-25" \
    "$SERVER:$REMOTE_DIR/2026-03-30_06-54-21" \
    "$SERVER:$REMOTE_DIR/2026-03-30_07-24-29" \
    "$SERVER:$REMOTE_DIR/2026-03-30_07-57-08" \
    "$SERVER:$REMOTE_DIR/2026-03-30_08-16-47" \
    "$SERVER:$REMOTE_DIR/2026-03-30_08-35-47" \
    "$LOCAL_DIR/"
