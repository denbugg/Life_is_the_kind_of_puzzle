"""Rigorous check: is the shuffle permutation reproducible from a guessable seed?

Uses the highest-confidence recover.py entries (near-exact recovered ground
truth) and tries many candidate seed-derivation schemes across every common
Python/numpy PRNG API. A hit would mean perfect placement for all 700 test
images with zero ML -- worth checking thoroughly before giving up on it.
"""
import re
import hashlib
import numpy as np

NFRAG = 576

z = np.load(r"E:/pazzle_work/cache/perms.npz", allow_pickle=True)
names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["names"]]
perms = z["perm"].astype(np.int64)
conf = z["conf"].astype(np.float32)

mean_conf = conf.mean(axis=1)
min_conf = conf.min(axis=1)
order = np.argsort(-mean_conf)
print("top-10 by mean confidence:")
for i in order[:10]:
    print(f"  {names[i]} mean_conf={mean_conf[i]:.4f} min_conf={min_conf[i]:.4f}")

# Use the 5 best-recovered images (highest mean AND min confidence) as ground truth.
candidates_idx = [i for i in order if min_conf[i] > 0.85][:5]
if not candidates_idx:
    candidates_idx = order[:5].tolist()
print("\nusing as ground truth:", [names[i] for i in candidates_idx])

def numbers_from_name(name: str) -> list[int]:
    digits = re.findall(r"\d+", name)
    out = []
    for d in digits:
        out.append(int(d))
    return out

def candidate_seeds(name: str, index: int) -> dict[str, int]:
    nums = numbers_from_name(name)
    n = nums[-1] if nums else index
    seeds = {
        "file_number": n,
        "file_number_x1000003": (n * 1_000_003) % (2**32),
        "sorted_index": index,
        "sorted_index_plus_1234": index + 1234,
        "file_number_plus_1234": n + 1234,
        "hash_md5_mod32": int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**32),
        "hash_sha256_mod32": int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**32),
        "python_hash_name": abs(hash(name)) % (2**32),
        "n_only": n,
        "n_times_2": n * 2,
        "n_times_7": n * 7,
    }
    return seeds

def perm_variants(seed: int) -> dict[str, np.ndarray]:
    variants = {}
    try:
        variants["default_rng.permutation"] = np.random.default_rng(seed).permutation(NFRAG)
    except Exception:
        pass
    try:
        arr = np.arange(NFRAG)
        np.random.default_rng(seed).shuffle(arr)
        variants["default_rng.shuffle"] = arr
    except Exception:
        pass
    try:
        variants["RandomState.permutation"] = np.random.RandomState(seed).permutation(NFRAG)
    except Exception:
        pass
    try:
        arr = np.arange(NFRAG)
        np.random.RandomState(seed).shuffle(arr)
        variants["RandomState.shuffle"] = arr
    except Exception:
        pass
    try:
        import random as pyrandom
        r = pyrandom.Random(seed)
        arr = list(range(NFRAG))
        r.shuffle(arr)
        variants["python_random.shuffle"] = np.array(arr)
    except Exception:
        pass
    try:
        import random as pyrandom
        r = pyrandom.Random(seed)
        variants["python_random.sample"] = np.array(r.sample(range(NFRAG), NFRAG))
    except Exception:
        pass
    return variants

best_overall = (0.0, None)
for idx in candidates_idx:
    name = names[idx]
    true_perm = perms[idx]
    weight = (conf[idx] > 0.7)  # only compare on confidently-recovered positions
    seeds = candidate_seeds(name, idx)
    local_best = (0.0, None)
    for seed_label, seed in seeds.items():
        for variant_label, arr in perm_variants(seed).items():
            agree = np.mean(arr[weight] == true_perm[weight]) if weight.sum() else 0.0
            if agree > local_best[0]:
                local_best = (agree, f"{seed_label}={seed} via {variant_label}")
            if agree > best_overall[0]:
                best_overall = (agree, f"{name}: {seed_label}={seed} via {variant_label}")
    print(f"{name}: best agreement {local_best[0]:.4f} ({local_best[1]}) chance~{1/NFRAG:.5f}")

print(f"\nBEST OVERALL: {best_overall}")
