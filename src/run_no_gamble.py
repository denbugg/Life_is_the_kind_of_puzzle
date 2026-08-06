"""Run the no-gamble PAZZLE pipeline with live status JSON.

The runner writes:
  E:/pazzle_work/no_gamble_status.json
  E:/pazzle_work/logs/no_gamble/*.log

Use `python src/no_gamble_monitor.py --port 8010` to watch progress.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque

from config import WORK_ROOT, CACHE_DIR


STATUS_PATH = os.path.join(WORK_ROOT, "no_gamble_status.json")
LOG_DIR = os.path.join(WORK_ROOT, "logs", "no_gamble")

STEP_RE = re.compile(r"step\s+(\d+)/(\d+).*?([\d.]+)s/it")
MINED_RE = re.compile(r"mined\s+(\d+)/(\d+)")
IMG_RE = re.compile(r"img_\d+\.png")
SUMMARY_RE = re.compile(r"(R@1|R@5|R@25|bb_prec|place_acc|neigh_acc|SSIM_solve)\s*[=:]\s*([0-9.]+)")


def now():
    return time.time()


def write_status(data):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    data["updated"] = now()
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATUS_PATH)


def stage_defs(args):
    hard_path = os.path.join(CACHE_DIR, f"hardneg_train_{args.preprocess}_K{args.K}.npz")
    py = sys.executable
    return [
        dict(name="baseline_score_raw_norm", total=args.score_n, log="01_score_raw_norm.log",
             cmd=[py, "-u", "src/score_with_preprocess.py", "--n", str(args.score_n),
                  "--modes", "raw,norm", "--pair_tag", args.base_pair_tag]),
        dict(name="baseline_buddies_raw", total=args.solve_n, log="02_buddies_raw.log",
             cmd=[py, "-u", "src/eval_neighbour.py", "--n", str(args.solve_n),
                  "--solver", "buddies", "--preprocess", "raw", "--pair_tag", args.base_pair_tag]),
        dict(name="train_match_denoiser", total=args.matchden_steps, log="03_train_matchden.log",
             cmd=[py, "-u", "src/train_match_denoiser.py", "--steps", str(args.matchden_steps),
                  "--bs", str(args.matchden_bs), "--workers", str(args.workers),
                  "--tag", args.denoise_tag]),
        dict(name="score_denoised", total=args.score_n, log="04_score_denoised.log",
             cmd=[py, "-u", "src/score_with_preprocess.py", "--n", str(args.score_n),
                  "--modes", "raw,norm,denoise,denoise_norm", "--denoise_tag", args.denoise_tag,
                  "--pair_tag", args.base_pair_tag]),
        dict(name="mine_hard_negatives", total=args.mine_n, log="05_mine_hard.log",
             cmd=[py, "-u", "src/mine_hard_negatives.py", "--n", str(args.mine_n),
                  "--K", str(args.K), "--preprocess", args.preprocess,
                  "--denoise_tag", args.denoise_tag, "--pair_tag", args.base_pair_tag]),
        dict(name="train_pair_hard", total=args.hard_steps, log="06_train_pair_hard.log",
             cmd=[py, "-u", "src/train_pair_hard.py", "--hard", hard_path,
                  "--steps", str(args.hard_steps), "--bs", str(args.hard_bs),
                  "--M", str(args.hard_M), "--workers", str(args.workers),
                  "--preprocess", args.preprocess, "--denoise_tag", args.denoise_tag,
                  "--init_tag", args.base_pair_tag, "--tag", args.hard_pair_tag]),
        dict(name="score_pair_hard", total=args.score_n, log="07_score_pair_hard.log",
             cmd=[py, "-u", "src/score_with_preprocess.py", "--n", str(args.score_n),
                  "--modes", args.preprocess, "--denoise_tag", args.denoise_tag,
                  "--pair_tag", args.hard_pair_tag]),
        dict(name="buddies_pair_hard", total=args.solve_n, log="08_buddies_pair_hard.log",
             cmd=[py, "-u", "src/solve_buddies.py", "--n", str(args.solve_n),
                  "--preprocess", args.preprocess, "--denoise_tag", args.denoise_tag,
                  "--pair_tag", args.hard_pair_tag]),
    ]


def parse_line(stage, line, seen_imgs):
    progress = None
    eta_sec = None
    metrics = {}
    m = STEP_RE.search(line)
    if m:
        step, total, sit = int(m.group(1)), int(m.group(2)), float(m.group(3))
        progress = dict(done=step, total=total, pct=100.0 * step / max(1, total))
        eta_sec = max(0.0, (total - step) * sit)
    m = MINED_RE.search(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        progress = dict(done=done, total=total, pct=100.0 * done / max(1, total))
    if IMG_RE.search(line):
        seen_imgs += 1
        total = stage.get("total") or 1
        progress = dict(done=min(seen_imgs, total), total=total,
                        pct=100.0 * min(seen_imgs, total) / max(1, total))
    for k, v in SUMMARY_RE.findall(line):
        metrics[k] = float(v)
    return progress, eta_sec, metrics, seen_imgs


def run_stage(stage, idx, total, status):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, stage["log"])
    tail = deque(maxlen=30)
    status.update(dict(stage_index=idx, stage_total=total, stage=stage["name"],
                       command=stage["cmd"], log_path=log_path, state="running",
                       progress=dict(done=0, total=stage.get("total"), pct=0.0),
                       eta_sec=None, started=now(), returncode=None,
                       last_lines=[], metrics={}))
    write_status(status)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("RUN: " + " ".join(stage["cmd"]) + "\n")
        log.flush()
        p = subprocess.Popen(stage["cmd"], cwd=os.getcwd(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        seen_imgs = 0
        last_write = 0.0
        for line in p.stdout:
            log.write(line)
            log.flush()
            tail.append(line.rstrip())
            progress, eta_sec, metrics, seen_imgs = parse_line(stage, line, seen_imgs)
            if progress is not None:
                status["progress"] = progress
            if eta_sec is not None:
                status["eta_sec"] = eta_sec
            if metrics:
                status.setdefault("metrics", {}).update(metrics)
            status["last_lines"] = list(tail)
            if now() - last_write > 1.0:
                write_status(status)
                last_write = now()
        rc = p.wait()
    status["returncode"] = rc
    status["state"] = "done" if rc == 0 else "error"
    status["eta_sec"] = 0
    status["last_lines"] = list(tail)
    if rc == 0 and status.get("progress", {}).get("total"):
        total_items = status["progress"]["total"]
        status["progress"] = dict(done=total_items, total=total_items, pct=100.0)
    write_status(status)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_n", type=int, default=12)
    ap.add_argument("--solve_n", type=int, default=20)
    ap.add_argument("--matchden_steps", type=int, default=8000)
    ap.add_argument("--matchden_bs", type=int, default=256)
    ap.add_argument("--mine_n", type=int, default=400)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--hard_steps", type=int, default=6000)
    ap.add_argument("--hard_bs", type=int, default=2)
    ap.add_argument("--hard_M", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--preprocess", default="denoise_norm",
                    choices=("raw", "norm", "denoise", "denoise_norm"))
    ap.add_argument("--denoise_tag", default="matchden")
    ap.add_argument("--base_pair_tag", default="pair")
    ap.add_argument("--hard_pair_tag", default="pair_hard")
    ap.add_argument("--start_at", default="")
    args = ap.parse_args()

    stages = stage_defs(args)
    if args.start_at:
        names = [s["name"] for s in stages]
        if args.start_at not in names:
            raise SystemExit(f"unknown --start_at {args.start_at}; choices={names}")
        stages = stages[names.index(args.start_at):]

    status = dict(run_started=now(), state="starting", stages=[s["name"] for s in stages])
    write_status(status)
    for i, stage in enumerate(stages, 1):
        rc = run_stage(stage, i, len(stages), status)
        if rc != 0:
            raise SystemExit(rc)
    status["state"] = "complete"
    status["stage"] = "complete"
    status["eta_sec"] = 0
    write_status(status)


if __name__ == "__main__":
    main()

