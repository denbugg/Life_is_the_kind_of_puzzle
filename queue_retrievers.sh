#!/usr/bin/env bash
# Retriever variants, one change per run so an effect can be attributed.
#   v2  continuation-prediction auxiliary loss: InfoNCE gives the trunk about
#       one bit per row, predicting the pixels past the edge gives 180 numbers.
#   v3  adds the severity sweep (M81) and twin-tolerant targets (M83).
# The re-ranker runs alongside, started by hand.
#
# Two rules learned the hard way:
#   * wait on the LOG, never pgrep -- Git Bash cannot see Windows processes, so
#     a pgrep guard fires instantly and starts the whole queue at once on one
#     8 GB card;
#   * never edit a script while bash is running it.  Bash reads by byte offset,
#     so an overwrite resumes mid-token: rewriting run_chain.sh under itself
#     produced "bed_v2.pt: command not found" and skipped a whole run.  Write a
#     new filename instead.
set -u
LOG=E:/pazzle_work/logs

wait_for() {           # $1 = logfile, $2 = marker
  local last=x now miss=0
  while ! grep -q "$2" "$1" 2>/dev/null; do
    now=$(stat -c %Y "$1" 2>/dev/null || echo 0)
    if [ "$now" = "$last" ]; then
      miss=$((miss + 1))
      # one quiet interval can be an eval pass or a slow checkpoint write;
      # three in a row means the writer is gone
      if [ "$miss" -ge 3 ]; then
        echo "=== $1 silent for 15 min, abandoning '$2' ===" >> $LOG/chain.log
        return 1
      fi
    else
      miss=0
    fi
    last=$now
    sleep 300
  done
}

wait_for $LOG/seam_v1.log "eval @ 30000"
echo "=== v1 done -> v2 (prediction aux) ===" >> $LOG/chain.log
python src/train_seam_embed.py --ch 96 --blocks 6 --dim 192 --batch 1 \
  --steps 16000 --lr 4e-4 --predict-weight 0.3 --eval-every 1000 \
  --eval-boards 6 --out seam_embed_v2.pt > $LOG/seam_v2.log 2>&1

echo "=== v2 done -> v3 (prediction + severity sweep + twins) ===" >> $LOG/chain.log
python src/train_seam_embed.py --ch 96 --blocks 6 --dim 192 --batch 1 \
  --steps 16000 --lr 4e-4 --predict-weight 0.3 --mix 0.5 --twin-thr 10 \
  --eval-every 1000 --eval-boards 6 --out seam_embed_v3.pt > $LOG/seam_v3.log 2>&1
echo "=== v3 done ===" >> $LOG/chain.log
