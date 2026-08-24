from __future__ import annotations

import json
import hashlib
import os
import random
import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e26_contextual_edge_net import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    ContextualDirectionalEdgeNet,
    ContextualEdgeConfig,
    model_from_checkpoint_payload,
)
from train_e26_contextual_edge import (  # noqa: E402
    CAL_SOURCE_COUNT,
    DEFAULT_OUTPUT_DIR,
    DEV_SOURCE_COUNT,
    FIT_SOURCE_COUNT,
    FROZEN_ACCUMULATE,
    FROZEN_EPOCHS,
    FROZEN_SEED,
    RECOVERY_SCHEMA,
    SOURCE_MAPPING_SERIALIZATION,
    TOTAL_OPTIMIZER_STEPS,
    LossConfig,
    _loader,
    _artifact_paths,
    build_split_manifest,
    capture_rng_state,
    checkpoint_payload,
    clean_canvas_to_tiles,
    clean_tiles_in_input_order,
    compute_loss,
    deterministic_seed,
    epoch_source_order,
    load_checkpoint,
    load_training_source_groups,
    move_optimizer_state_to_device,
    restore_rng_state,
    save_checkpoint_atomic,
    source_mapping_sha256,
    split_manifest_sha256,
    split_source_groups,
    validate_e_only_runtime,
    validate_recovery_checkpoint,
)


def tiny_config() -> ContextualEdgeConfig:
    return ContextualEdgeConfig(
        grid_height=2,
        grid_width=3,
        cnn_width=16,
        d_model=32,
        local_dim=24,
        match_dim=16,
        transformer_layers=1,
        attention_heads=4,
        ff_multiplier=1.5,
        dropout=0.0,
        boundary_band=1,
        boundary_bins=3,
        reconstruction_samples=4,
        encoder_chunk_size=6,
    )


class SourceGroupProtocolTests(unittest.TestCase):
    @staticmethod
    def frozen_synthetic_mapping() -> tuple[tuple[str, ...], dict[str, str], str]:
        names = tuple(f"img_{index:06d}.png" for index in range(6_700))
        groups = {name: f"synthetic-group-{index:06d}" for index, name in enumerate(names)}
        return names, groups, source_mapping_sha256(names, groups)

    def test_split_is_exact_deterministic_and_group_disjoint(self) -> None:
        names, groups, mapping_sha = self.frozen_synthetic_mapping()
        first = split_source_groups(names, groups, mapping_sha256=mapping_sha)
        second = split_source_groups(names, dict(reversed(list(groups.items()))), mapping_sha256=mapping_sha)
        self.assertEqual(first, second)
        self.assertEqual(len(first.fit_names), FIT_SOURCE_COUNT)
        self.assertEqual(len(first.calibration_names), CAL_SOURCE_COUNT)
        self.assertEqual(len(first.development_names), DEV_SOURCE_COUNT)
        partitions = (first.fit_names, first.calibration_names, first.development_names)
        self.assertEqual(set().union(*map(set, partitions)), set(names))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(set(partitions[i]) & set(partitions[j]))
        manifest = build_split_manifest(first)
        self.assertEqual(len(split_manifest_sha256(manifest)), 64)
        self.assertEqual(manifest["counts"], {"FIT": 5360, "CAL": 670, "DEV": 670})

    def test_manifest_loader_authenticates_exact_namespace_and_mapping(self) -> None:
        records = {
            f"img_{index:06d}.png": {"source_group": f"g{index}"}
            for index in range(6)
        }
        parent = Path(os.environ.get("E26_TEST_TMP", tempfile.gettempdir()))
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            path = Path(directory) / "groups.json"
            path.write_text(json.dumps({"files": records}), encoding="utf-8")
            raw_sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            expected_names = tuple(f"img_{index:06d}.png" for index in range(4))
            expected_groups = {name: records[name]["source_group"] for name in expected_names}
            mapping_sha = source_mapping_sha256(expected_names, expected_groups)
            names, groups, digest = load_training_source_groups(
                path,
                train_source_count=4,
                expected_manifest_sha256=raw_sha,
                expected_mapping_sha256=mapping_sha,
            )
        self.assertEqual(names, expected_names)
        self.assertEqual(set(groups), set(names))
        self.assertEqual(len(digest), 64)
        self.assertIn("tab-source_group-lf", SOURCE_MAPPING_SERIALIZATION)

    def test_manifest_loader_rejects_alias_namespace_and_shared_groups(self) -> None:
        records = {
            f"{index:04d}.png": {"source_group": "same"}
            for index in range(4)
        }
        parent = Path(os.environ.get("E26_TEST_TMP", tempfile.gettempdir()))
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            path = Path(directory) / "groups.json"
            path.write_text(json.dumps({"files": records}), encoding="utf-8")
            raw_sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "namespace"):
                load_training_source_groups(
                    path,
                    train_source_count=4,
                    expected_manifest_sha256=raw_sha,
                    expected_mapping_sha256="0" * 64,
                )

    def test_rng_and_epoch_order_are_stateless(self) -> None:
        names, groups, mapping_sha = self.frozen_synthetic_mapping()
        split = split_source_groups(names, groups, mapping_sha256=mapping_sha)
        split_sha = split_manifest_sha256(build_split_manifest(split))
        first = epoch_source_order(split.fit_names, split_sha, 3)
        second = epoch_source_order(tuple(reversed(split.fit_names)), split_sha, 3)
        self.assertEqual(first, second)
        self.assertNotEqual(first, epoch_source_order(split.fit_names, split_sha, 4))
        corrupt = deterministic_seed(split_sha, "FIT", 3, first[0], "corrupt")
        permutation = deterministic_seed(split_sha, "FIT", 3, first[0], "perm")
        self.assertEqual(corrupt, deterministic_seed(split_sha, "FIT", 3, first[0], "corrupt"))
        self.assertNotEqual(corrupt, permutation)
        with self.assertRaisesRegex(ValueError, "epoch zero"):
            deterministic_seed(split_sha, "DEV", 1, first[0], "corrupt")

    def test_large_outputs_default_to_e_drive(self) -> None:
        self.assertEqual(DEFAULT_OUTPUT_DIR.drive.upper(), "E:")

    def test_runtime_guard_rejects_c_and_accepts_complete_e_contract(self) -> None:
        environment = {
            name: f"E:/pazzle_work/runtime/{name.lower()}"
            for name in (
                "TEMP",
                "TMP",
                "PYTHONPYCACHEPREFIX",
                "TORCH_EXTENSIONS_DIR",
                "CUDA_CACHE_PATH",
                "PAZZLE_DATA",
                "PAZZLE_WORK",
            )
        }
        environment["PYTHONHASHSEED"] = str(FROZEN_SEED)
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        validate_e_only_runtime(
            source_manifest=Path("E:/manifest.json"),
            output_dir=Path("E:/run"),
            resume=Path("E:/run/recovery.pt"),
            environment=environment,
        )
        with self.assertRaisesRegex(ValueError, "output_dir"):
            validate_e_only_runtime(
                source_manifest=Path("E:/manifest.json"),
                output_dir=Path("C:/run"),
                resume=None,
                environment=environment,
            )


class TrainingAndCheckpointTests(unittest.TestCase):
    def test_clean_canvas_tile_conversion_and_input_alignment(self) -> None:
        canonical = torch.arange(6.0).reshape(1, 1, 2, 3).expand(1, 3, 2, 3)
        tiles = clean_canvas_to_tiles(canonical, 2, 3)
        self.assertEqual(tuple(tiles.shape), (1, 6, 3, 1, 1))
        self.assertEqual(tiles[0, :, 0, 0, 0].tolist(), list(range(6)))
        permutation = torch.tensor([[2, 0, 5, 1, 4, 3]])
        aligned = clean_tiles_in_input_order(canonical, permutation, 2, 3)
        self.assertEqual(aligned[0, :, 0, 0, 0].tolist(), permutation[0].float().tolist())

    def test_combined_loss_backpropagates_through_all_branches(self) -> None:
        torch.manual_seed(126)
        config = tiny_config()
        model = ContextualDirectionalEdgeNet(config).train()
        tiles = torch.rand(1, 6, 3, 8, 8)
        clean = torch.rand(1, 3, 16, 24)
        permutation = torch.tensor([[2, 0, 5, 1, 4, 3]])
        output = model(tiles)
        loss, terms = compute_loss(
            output,
            {"perm": permutation, "clean": clean},
            config,
            LossConfig(),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(terms["listwise_ce"].detach()), 0.0)
        self.assertGreater(float(terms["boundary"].detach()), 0.0)
        parameters = (
            model.tile_encoder.stem[0].weight,
            model.set_blocks[0].attention.in_proj_weight,
            model.right_query.weight,
            model.none_heads[0][-1].weight,
            model.boundary_decoder[-1].weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_checkpoint_round_trip_restores_exact_outputs(self) -> None:
        torch.manual_seed(226)
        model = ContextualDirectionalEdgeNet(tiny_config()).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        tiles = torch.rand(1, 6, 3, 8, 8)
        with torch.no_grad():
            expected = model(tiles)["logits"]
        payload = checkpoint_payload(
            model,
            optimizer,
            step=7,
            training_config={"seed": 226},
            metrics={"neighbour_r1": 0.5},
            split_contract={"sha256": "a" * 64},
        )
        parent = Path(os.environ.get("E26_TEST_TMP", tempfile.gettempdir()))
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint_atomic(path, payload)
            loaded = load_checkpoint(path)
        self.assertEqual(loaded["schema"], CHECKPOINT_SCHEMA)
        self.assertEqual(loaded["step"], 7)
        restored = model_from_checkpoint_payload(loaded).eval()
        with torch.no_grad():
            actual = restored(tiles)["logits"]
        self.assertTrue(torch.equal(actual, expected))

    def test_recovery_contains_and_validates_full_optimizer_boundary_state(self) -> None:
        torch.manual_seed(326)
        model = ContextualDirectionalEdgeNet(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        split_sha = "b" * 64
        source_sha = "c" * 64
        split_contract = {
            "split_manifest_sha256": split_sha,
            "source_manifest_sha256": source_sha,
        }
        dependencies = {"source_sha256": {"trainer": "e" * 64}}
        training_config = {"seed": FROZEN_SEED}
        run_contract = {
            "training_schema": "test",
            "training_config": training_config,
            "split_contract": split_contract,
            "dependencies": dependencies,
        }
        run_sha = hashlib.sha256(
            json.dumps(
                run_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        run_contract["sha256"] = run_sha
        progress = {
            "next_epoch": 2,
            "next_sample_cursor": 16,
            "optimizer_steps_completed": 2 * (FIT_SOURCE_COUNT // FROZEN_ACCUMULATE) + 2,
        }
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=progress["optimizer_steps_completed"],
            progress=progress,
            rng_state=capture_rng_state(),
            training_config=training_config,
            split_contract=split_contract,
            run_contract=run_contract,
            dependencies=dependencies,
            history={
                "started_utc": "synthetic",
                "loss_log": [],
                "recovery_commits": [],
                "epoch_end": [],
            },
            checkpoint_kind="recovery",
        )
        self.assertEqual(payload["recovery_schema"], RECOVERY_SCHEMA)
        self.assertIsNotNone(payload["optimizer"])
        self.assertIsNotNone(payload["scheduler"])
        self.assertIsNotNone(payload["scaler"])
        self.assertEqual(
            validate_recovery_checkpoint(
                payload,
                run_contract_sha256=run_sha,
                split_manifest_sha256_value=split_sha,
                source_manifest_sha256=source_sha,
            ),
            progress,
        )
        tampered = dict(payload)
        tampered["step"] = payload["step"] + 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_recovery_checkpoint(
                tampered,
                run_contract_sha256=run_sha,
                split_manifest_sha256_value=split_sha,
                source_manifest_sha256=source_sha,
            )

    def test_rng_capture_restore_is_exact(self) -> None:
        random.seed(426)
        np.random.seed(426)
        torch.manual_seed(426)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.random()), torch.rand(4))
        _ = (random.random(), np.random.random(), torch.rand(7))
        restore_rng_state(state)
        actual = (random.random(), float(np.random.random()), torch.rand(4))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_dataloader_iterator_does_not_consume_model_rng(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(8))
        torch.manual_seed(486)
        before = torch.get_rng_state().clone()
        loader = _loader(dataset, workers=0, device=torch.device("cpu"))
        iterator = iter(loader)
        _ = next(iterator)
        after = torch.get_rng_state()
        self.assertTrue(torch.equal(after, before))

    def test_optimizer_state_move_visits_nested_tensors(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        move_optimizer_state_to_device(optimizer, torch.device("cpu"))
        tensors = [
            value
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ]
        self.assertTrue(tensors)
        self.assertTrue(all(value.device.type == "cpu" for value in tensors))

    def test_optimizer_boundary_resume_matches_uninterrupted_training_exactly(self) -> None:
        config = replace(tiny_config(), dropout=0.20)
        data_generator = torch.Generator().manual_seed(526)
        batches = [torch.rand(1, 6, 3, 8, 8, generator=data_generator) for _ in range(3)]

        def make_training_state() -> tuple[
            ContextualDirectionalEdgeNet,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler,
            torch.amp.GradScaler,
        ]:
            model = ContextualDirectionalEdgeNet(config).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 1.0 - 0.01 * step
            )
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            return model, optimizer, scheduler, scaler

        def take_step(
            state: tuple[
                ContextualDirectionalEdgeNet,
                torch.optim.Optimizer,
                torch.optim.lr_scheduler.LRScheduler,
                torch.amp.GradScaler,
            ],
            tiles: torch.Tensor,
        ) -> None:
            model, optimizer, scheduler, scaler = state
            optimizer.zero_grad(set_to_none=True)
            loss = model(tiles)["logits"].float().square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        torch.manual_seed(526)
        uninterrupted = make_training_state()
        take_step(uninterrupted, batches[0])
        boundary = {
            "model": {name: value.detach().clone() for name, value in uninterrupted[0].state_dict().items()},
            "optimizer": copy.deepcopy(uninterrupted[1].state_dict()),
            "scheduler": copy.deepcopy(uninterrupted[2].state_dict()),
            "scaler": copy.deepcopy(uninterrupted[3].state_dict()),
            "rng": capture_rng_state(),
        }
        take_step(uninterrupted, batches[1])
        take_step(uninterrupted, batches[2])

        # Construction deliberately consumes global Torch RNG.  Restoring the
        # boundary RNG after all objects/state have been rebuilt is essential.
        resumed = make_training_state()
        resumed[0].load_state_dict(boundary["model"])
        resumed[1].load_state_dict(boundary["optimizer"])
        resumed[2].load_state_dict(boundary["scheduler"])
        resumed[3].load_state_dict(boundary["scaler"])
        restore_rng_state(boundary["rng"])
        take_step(resumed, batches[1])
        take_step(resumed, batches[2])
        for name, expected in uninterrupted[0].state_dict().items():
            self.assertTrue(torch.equal(resumed[0].state_dict()[name], expected), name)
        self.assertEqual(resumed[2].state_dict(), uninterrupted[2].state_dict())
        self.assertEqual(resumed[3].state_dict(), uninterrupted[3].state_dict())

    def test_artifacts_have_recovery_and_fixed_final_but_no_best(self) -> None:
        paths = _artifact_paths(Path("E:/run"), "edge")
        self.assertIn("recovery", paths)
        self.assertEqual(paths["final"].name, "edge_final_epoch08.pt")
        self.assertNotIn("best", paths)
        self.assertEqual(TOTAL_OPTIMIZER_STEPS, FROZEN_EPOCHS * 670)


if __name__ == "__main__":
    unittest.main()
