#!/usr/bin/env bash
set -euo pipefail

session="pazzle_rl_v2_20260726"
project="/home/kva/pazzle_rl_v2"

cd "$project"
mkdir -p outputs

if pgrep -f '[t]rain_rl_swap_actor_critic_v2.py' >/dev/null; then
  echo "training process already exists; refusing duplicate" >> outputs/keepalive.log
  exit 1
fi

tmux new-session -d -s "$session" \
  "bash -lc '$project/run_training_rl_v2.sh >> $project/outputs/tmux_wrapper.log 2>&1'"
echo "$(date --iso-8601=seconds) started $session" >> outputs/keepalive.log

# Keeping this command in the foreground keeps a Windows-side wsl.exe handle
# alive.  The tmux job remains inspectable and survives SSH disconnects.
while tmux has-session -t "$session" 2>/dev/null; do
  sleep 30
done

echo "$(date --iso-8601=seconds) session ended" >> outputs/keepalive.log
