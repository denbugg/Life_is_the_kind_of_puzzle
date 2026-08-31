"""Small end-to-end check for the local scientific and ML environment."""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

PACKAGES = (
    "albumentations",
    "jupyterlab",
    "matplotlib",
    "networkx",
    "numpy",
    "opencv-python-headless",
    "optuna",
    "ortools",
    "pandas",
    "pillow",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "timm",
    "torch",
    "torchvision",
)


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def run_smoke_check() -> dict[str, Any]:
    """Exercise image, optimization, classical ML, and PyTorch code paths."""
    import albumentations as a
    import cv2
    import networkx as nx
    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    from ortools.sat.python import cp_model
    from scipy.optimize import linear_sum_assignment
    from skimage.color import rgb2gray
    from sklearn.metrics.pairwise import cosine_similarity

    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    augmented = a.HorizontalFlip(p=1.0)(image=image)["image"]
    skimage_gray = rgb2gray(image)

    costs = np.array([[3.0, 1.0], [1.0, 3.0]])
    rows, columns = linear_sum_assignment(costs)
    similarity = cosine_similarity(costs)

    graph = nx.Graph()
    graph.add_edge("piece_0", "piece_1", cost=float(costs[0, 1]))

    model = cp_model.CpModel()
    selected = model.new_bool_var("selected")
    model.add(selected == 1)
    solver_status = cp_model.CpSolver().solve(model)

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    device = torch.device("mps" if mps_available else "cpu")
    tensor = torch.ones((32, 32), device=device)
    torch_checksum = float((tensor @ tensor).sum().cpu())

    frame = pd.DataFrame({"piece": rows, "slot": columns})
    assert gray.shape == (8, 8)
    assert augmented.shape == image.shape
    assert skimage_gray.shape == (8, 8)
    assert frame["slot"].tolist() == [1, 0]
    assert similarity.shape == (2, 2)
    assert graph.number_of_edges() == 1
    assert solver_status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    assert torch_checksum == 32**3
    assert torchvision.__version__

    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_device": str(device),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "packages": _versions(),
    }


def main() -> None:
    print(json.dumps(run_smoke_check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
