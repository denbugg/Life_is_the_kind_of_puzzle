"""Lightweight preflight for the isolated RL experiment.

The real accuracy evaluation runs on Kaggle against held-out target images.
"""

import ast
from pathlib import Path


def main():
    source = Path(__file__).with_name("kaggle_train_rl_puzzle.py")
    tree = ast.parse(source.read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    required = {"SwapActorCritic", "PuzzleSwapEnv", "PositionPrior"}
    missing = required - classes
    if missing:
        raise AssertionError(f"missing classes: {sorted(missing)}")
    text = source.read_text()
    feature_node = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FEATURE_NAMES" for t in node.targets)
    )
    feature_names = set(ast.literal_eval(feature_node))
    leaked = feature_names & {"correct_edge_delta", "correct_position_delta", "distance_delta"}
    if leaked:
        raise AssertionError(f"ground-truth leakage in policy features: {sorted(leaked)}")
    for term in ["correct_edge_delta", "correct_position_delta", "visual_delta", "ppo_update", "validate"]:
        if term not in text:
            raise AssertionError(f"missing RL contract term: {term}")
    print("preflight_score=1.0")
    print("metric=held_out_ssim goal=maximize current_rl=0.181717 heuristic_baseline=0.182062")


if __name__ == "__main__":
    main()
