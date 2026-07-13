#!/usr/bin/env bash
set -u

ROOT="/Users/rusyalain/Documents/test"
ENV_PREFIX="$ROOT/.conda"

echo "root=$ROOT"
echo "resolved_root=$(cd "$ROOT" && pwd -P)"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "ERROR: missing project Python at $ENV_PREFIX/bin/python"
  exit 1
fi

"$ENV_PREFIX/bin/python" -c 'import platform; print("python", platform.python_version())'
"$ENV_PREFIX/bin/python" -c 'import numpy, PIL, scipy, skimage; print("numpy", numpy.__version__); print("pillow", PIL.__version__); print("scipy", scipy.__version__); print("skimage", skimage.__version__)'
"$ENV_PREFIX/bin/python" -c 'import torch; print("torch", torch.__version__); print("mps_available", bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())); print("cuda_available", torch.cuda.is_available())'

"$ENV_PREFIX/bin/kaggle" --version || true
"$ENV_PREFIX/bin/kaggle" kernels list --mine --page-size 1 >/dev/null 2>&1 && echo "kaggle_auth=ok" || echo "kaggle_auth=unavailable"

for d in "$ROOT/puzzle/train/inputs" "$ROOT/puzzle/train/targets" "$ROOT/puzzle/test"; do
  if [[ -d "$d" ]]; then
    count=$(find "$d" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
    echo "$d png_count=$count"
  else
    echo "ERROR: missing $d"
  fi
done

echo "preflight=complete"
