"""Rebuild kaggle_kernel/pazzle_kaggle_train.ipynb for a new run.
Regenerates cell 1 (embeds current src/*.py) and cell 5 (orchestration), and
keeps cells 0/2/3/4 verbatim -- so the W&B cell (with its embedded key) and the
data-detection logic are preserved untouched. Never prints the key."""
import json, os

REPO = os.path.dirname(os.path.abspath(__file__))
SRCDIR = os.path.join(REPO, "src")
NB = os.path.join(REPO, "kaggle_kernel", "pazzle_kaggle_train.ipynb")

EMBED = ["config.py", "imgio.py", "distort.py", "recover.py", "datasets.py",
         "models.py", "solve.py", "pipeline.py", "train_pair.py",
         "train_restore.py", "eval_place.py", "eval_full.py", "infer.py",
         "diag_scores.py"]


def cell1_source():
    files = {}
    for name in EMBED:
        with open(os.path.join(SRCDIR, name), encoding="utf-8") as f:
            files[name] = f.read()
    body = ",\n".join(f"    {json.dumps(n)}: {json.dumps(t)}" for n, t in files.items())
    return (
        "from pathlib import Path\n"
        "import sys\n"
        "SRC = Path('/kaggle/working/src')\n"
        "SRC.mkdir(parents=True, exist_ok=True)\n"
        "SRC_FILES = {\n" + body + "\n}\n"
        "for name, text in SRC_FILES.items():\n"
        "    (SRC / name).write_text(text, encoding='utf-8')\n"
        "if str(SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(SRC))\n"
        "print('wrote src modules:', len(SRC_FILES), '->', SRC)\n"
    )


# run-#2 orchestration (ensemble): resume perms.npz, train TWO v2 scorers CONCURRENTLY
# (one per T4, different seeds -> ensembled at eval; avoids DataParallel overhead),
# then the solver-independent score diagnostic, placement, and end-to-end SSIM w/ NLM.
NEW_TRY = '''# --- run #2 (ensemble): resume perms.npz, two scorers concurrently on both T4s ---
import threading
RESUME = Path('/kaggle/input/vsos-ai-pazzle-resume-v7')
cache = PWORK / 'cache' / 'perms.npz'
if not cache.exists() and (RESUME / 'perms.npz').exists():
    import shutil
    shutil.copy(RESUME / 'perms.npz', cache)
    print('restored perms.npz from resume dataset ->', cache)


def run_parallel(jobs):
    """jobs: list of dict(cmd, log, cvd). Run concurrently, each pinned to one GPU
    via CUDA_VISIBLE_DEVICES so every process sees a single device (no DataParallel)."""
    codes = {}
    def worker(job):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = job['cvd']
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        lp = PWORK / 'logs' / job['log']
        print('RUN[gpu%s]:' % job['cvd'], ' '.join(job['cmd']))
        with open(lp, 'w', encoding='utf-8') as f:
            p = subprocess.Popen(job['cmd'], cwd=SRC, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in p.stdout:
                print('[%s] %s' % (job['log'], line), end='')
                f.write(line)
                wandb_log_line('pair', line)
            codes[job['log']] = p.wait()
        if WANDB_RUN is not None:
            import wandb; wandb.save(str(lp))
    ts = [threading.Thread(target=worker, args=(j,)) for j in jobs]
    for t in ts: t.start()
    for t in ts: t.join()
    for name, code in codes.items():
        if code != 0:
            raise RuntimeError('%s failed rc=%s' % (name, code))


try:
    if RUN_TRAIN:
        if cache.exists():
            print('skip recover: existing cache/perms.npz')
        else:
            run([sys.executable, '-u', 'recover.py'], 'recover.log', stage='recover')
        # bs=1 -> 3072 pairs/step (fits one T4; the DataParallel run split 6144->3072/GPU).
        # GroupNorm is batch-independent so bs=1 is fine; M=32 keeps the extra negatives.
        common = ['--steps', '8000', '--bs', '1', '--nA', '48', '--M', '32',
                  '--real_prob', '0.6', '--workers', '2', '--lr', '1e-3']
        run_parallel([
            dict(cmd=[sys.executable, '-u', 'train_pair.py', *common, '--seed', '1234', '--tag', 'pair0'], log='pair0.log', cvd='0'),
            dict(cmd=[sys.executable, '-u', 'train_pair.py', *common, '--seed', '5678', '--tag', 'pair1'], log='pair1.log', cvd='1'),
        ])
        run([sys.executable, '-u', 'diag_scores.py', '--n', '8'], 'diag_scores.log', stage='diag')
        run([sys.executable, '-u', 'eval_place.py', '--n', '20', '--full_pair',
             '--iters', '3000000', '--restarts', '3'], 'eval_place.log', stage='eval_place')
        run([sys.executable, '-u', 'eval_full.py', '--n', '30', '--full_pair', '--nlm',
             '--iters', '4000000', '--restarts', '3'], 'eval_full.log', stage='eval_full')
        save_final_outputs_to_wandb()
    else:
        print('RUN_TRAIN=False: bootstrap only.')
finally:
    if WANDB_RUN is not None:
        import wandb
        save_final_outputs_to_wandb()
        wandb.finish()
'''


def cell5_source(orig):
    prefix = orig.split("\ntry:", 1)[0].rstrip("\n") + "\n\n"
    return prefix + NEW_TRY


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    assert len(nb["cells"]) == 6, f"expected 6 cells, got {len(nb['cells'])}"
    orig5 = "".join(nb["cells"][5]["source"])
    nb["cells"][1]["source"] = cell1_source().splitlines(keepends=True)
    nb["cells"][5]["source"] = cell5_source(orig5).splitlines(keepends=True)
    for c in nb["cells"]:
        c.setdefault("outputs", []) if c["cell_type"] == "code" else None
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("rebuilt", NB)
    print("embedded src files:", len(EMBED))
    # sanity: byte size + cell 5 uses new stages, does NOT retrain restore or full infer
    c5 = "".join(nb["cells"][5]["source"])
    print("cell5 has diag_scores:", "diag_scores.py" in c5,
          "| nlm eval_full:", "'--nlm'" in c5,
          "| no train_restore:", "train_restore.py" not in c5,
          "| no full infer:", "infer.py" not in c5)


if __name__ == "__main__":
    main()
