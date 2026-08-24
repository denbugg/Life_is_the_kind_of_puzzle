from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import run_e26_autonomous as runner


STUB = (REPO_ROOT / "tests" / "e26_stage_stub.py").resolve()
RUNNER_SOURCE = (SRC_ROOT / "run_e26_autonomous.py").resolve()
BOOTSTRAP_SOURCE = (SRC_ROOT / "e26_stage_bootstrap.py").resolve()
PYTHON = str(Path(sys.executable).resolve())


class AutonomousRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = Path(os.environ.get("E26_TEST_TMP", tempfile.gettempdir())).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="e26_autonomous_", dir=parent)).resolve()
        self.spec_path = self.root / "spec.json"
        self.plan_path = self.root / "preflight" / "plan.json"

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def command(
        self,
        *,
        output: Path,
        payload: str,
        decision: str = "PASS",
        counter: Path | None = None,
        fail_first: Path | None = None,
        sleep: float = 0.0,
    ) -> list[str]:
        argv = [
            PYTHON, "-B", str(STUB), "write",
            "--output", str(output), "--payload", payload,
            "--decision", decision,
        ]
        if counter is not None:
            argv += ["--counter", str(counter)]
        if fail_first is not None:
            argv += ["--fail-first", str(fail_first)]
        if sleep:
            argv += ["--sleep", str(sleep)]
        return argv

    def stage(
        self,
        name: str,
        *,
        payload: str | None = None,
        decision: str = "PASS",
        dependencies: list[str] | None = None,
        counter: Path | None = None,
        fail_first: Path | None = None,
        max_attempts: int = 1,
        sleep: float = 0.0,
        timeout_seconds: int = 30,
        inputs: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        output = self.root / "artifacts" / f"{name}.json"
        payload = payload or name
        command = self.command(
            output=output,
            payload=payload,
            decision=decision,
            counter=counter,
            fail_first=fail_first,
            sleep=sleep,
        )
        return {
            "name": name,
            "argv": command,
            "resume_argv": command if max_attempts > 1 else None,
            "dependencies": dependencies or [],
            "inputs": inputs or [],
            "outputs": [{"path": str(output), "min_bytes": 10, "max_bytes": 100_000}],
            "verifier_argv": [
                PYTHON, "-B", str(STUB), "verify",
                "--path", str(output), "--expected", payload,
            ],
            "working_directory": str(self.root),
            "timeout_seconds": timeout_seconds,
            "max_rss_bytes": 2 << 30,
            "min_free_bytes": 0,
            "max_attempts": max_attempts,
            "progress_regex": r"step=(?P<done>\d+)/(?P<total>\d+)",
            "resource_class": "cpu",
            "gate": {
                "path": str(output),
                "pointer": ["decision"],
                "pass_value": "PASS",
            },
        }

    def spec(self, stages: list[dict[str, object]]) -> dict[str, object]:
        return {
            "schema": runner.SPEC_SCHEMA,
            "pipeline_id": "e26_test_pipeline",
            "repo_root": str(REPO_ROOT),
            "work_root": str(self.root),
            "python_executable": PYTHON,
            "source_files": [str(RUNNER_SOURCE), str(BOOTSTRAP_SOURCE), str(STUB)],
            "environment": {},
            "global_caps": {
                "max_cpu_seconds": 3_600,
                "max_gpu_seconds": 3_600,
                "max_wall_seconds": 3_600,
                "max_artifact_bytes": 100 << 20,
            },
            "stages": stages,
        }

    def freeze(self, stages: list[dict[str, object]]) -> dict[str, object]:
        self.spec_path.write_bytes(runner.canonical_json(self.spec(stages)))
        return runner.freeze_plan(
            self.spec_path, self.plan_path, allow_non_e=True
        )

    def make_runner(self, plan: dict[str, object], **kwargs: object) -> runner.AutonomousRunner:
        return runner.AutonomousRunner(
            self.plan_path,
            str(plan["plan_sha256"]),
            allow_non_e=True,
            poll_seconds=0.05,
            **kwargs,
        )

    def test_freeze_roundtrip_and_nested_e_environment(self) -> None:
        plan = self.freeze([self.stage("alpha")])
        loaded = runner.load_and_verify_plan(
            self.plan_path, str(plan["plan_sha256"]), allow_non_e=True
        )
        self.assertEqual(plan, loaded)
        self.assertEqual(plan["schema"], runner.PLAN_SCHEMA)
        on_disk = runner._read_canonical_json(self.plan_path)
        self.assertNotIn("plan_sha256", on_disk)
        self.assertEqual(plan["plan_sha256"], runner.canonical_digest(on_disk))
        self.assertEqual(plan["plan_sha256"], runner.sha256_file(self.plan_path))
        for name in runner.REQUIRED_E_ENV:
            self.assertTrue(runner._is_within(Path(plan["environment"][name]), self.root))

    def test_plan_path_must_be_under_reserved_preflight(self) -> None:
        self.spec_path.write_bytes(runner.canonical_json(self.spec([self.stage("alpha")])))
        with self.assertRaisesRegex(runner.ContractError, "preflight"):
            runner.freeze_plan(
                self.spec_path, self.root / "orchestrator" / "status.json",
                allow_non_e=True,
            )
        plan = self.freeze([self.stage("alpha")])
        relocated = self.root / "orchestrator" / "status.json"
        relocated.parent.mkdir(parents=True, exist_ok=True)
        relocated.write_bytes(self.plan_path.read_bytes())
        with self.assertRaisesRegex(runner.ContractError, "preflight"):
            runner.load_and_verify_plan(
                relocated, str(plan["plan_sha256"]), allow_non_e=True
            )

    def test_cli_contract_rejects_non_e_production_root(self) -> None:
        spec = self.spec([self.stage("alpha")])
        spec["work_root"] = "C:/e26_forbidden_test_root"
        self.spec_path.write_bytes(runner.canonical_json(spec))
        with self.assertRaisesRegex(runner.ContractError, "must be on E"):
            runner.freeze_plan(self.spec_path, Path("C:/e26_forbidden_test_root/plan.json"))

    def test_nonfinite_caps_poll_and_attempt_usage_fail_closed(self) -> None:
        spec = self.spec([self.stage("alpha")])
        spec["global_caps"]["max_cpu_seconds"] = "nan"
        self.spec_path.write_bytes(runner.canonical_json(spec))
        with self.assertRaisesRegex(runner.ContractError, "global caps"):
            runner.freeze_plan(self.spec_path, self.plan_path, allow_non_e=True)

        plan = self.freeze([self.stage("alpha")])
        with self.assertRaisesRegex(runner.ContractError, "poll_seconds"):
            runner.AutonomousRunner(
                self.plan_path, str(plan["plan_sha256"]),
                allow_non_e=True, poll_seconds=float("nan"),
            )
        instance = self.make_runner(plan)
        instance._ensure_runtime_dirs()
        attempt_dir = instance.attempt_root / "alpha" / "attempt_0001"
        attempt_dir.mkdir(parents=True)
        payload = runner._self_digest_payload({
            "schema": runner.ATTEMPT_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "stage": "alpha",
            "resource_class": "cpu",
            "state": "failed",
            "cpu_seconds": "nan",
            "elapsed_seconds": 1.0,
            "accounting_poll_seconds": 0.05,
        }, "attempt_sha256")
        runner._atomic_write_canonical(attempt_dir / "attempt.json", payload)
        with self.assertRaisesRegex(runner.ContractError, "account"):
            instance._cumulative_attempt_usage()

    def test_process_gone_is_false_but_access_denied_fails_closed(self) -> None:
        import psutil

        identity = {"pid": 12345, "create_time": 1.0, "executable": PYTHON}
        with mock.patch.object(psutil, "Process", side_effect=psutil.NoSuchProcess(12345)):
            self.assertFalse(runner._process_identity_is_alive(identity))
        with mock.patch.object(psutil, "Process", side_effect=psutil.AccessDenied(12345)):
            with self.assertRaisesRegex(runner.ContractError, "access denied"):
                runner._process_identity_is_alive(identity)

    def test_command_must_use_exact_frozen_python_and_dash_b(self) -> None:
        stage = self.stage("alpha")
        stage["argv"] = ["python", str(STUB), "--help"]
        with self.assertRaisesRegex(runner.ContractError, "exact frozen Python and -B"):
            self.freeze([stage])

    def test_entry_script_must_be_frozen_and_cwd_must_be_e26_root(self) -> None:
        stage = self.stage("alpha")
        unfrozen = self.root / "unfrozen.py"
        unfrozen.write_text("raise SystemExit(0)\n", encoding="utf-8")
        stage["argv"][2] = str(unfrozen)
        with self.assertRaisesRegex(runner.ContractError, "not a frozen source"):
            self.freeze([stage])
        stage = self.stage("alpha")
        stage["working_directory"] = str(REPO_ROOT)
        with self.assertRaisesRegex(runner.ContractError, "inside E26 work root"):
            self.freeze([stage])

    def test_local_import_closure_must_be_frozen(self) -> None:
        helper = self.root / "local_helper.py"
        helper.write_text("VALUE = 1\n", encoding="utf-8")
        script = self.root / "entry.py"
        script.write_text("import local_helper\n", encoding="utf-8")
        stage = self.stage("alpha")
        stage["argv"][2] = str(script)
        stage["verifier_argv"][2] = str(script)
        spec = self.spec([stage])
        spec["repo_root"] = str(self.root)
        spec["source_files"] = [str(script)]
        self.spec_path.write_bytes(runner.canonical_json(spec))
        with self.assertRaisesRegex(runner.ContractError, "import closure is not frozen"):
            runner.freeze_plan(self.spec_path, self.plan_path, allow_non_e=True)

    def test_outputs_cannot_overlap_reserved_orchestrator_runtime_preflight(self) -> None:
        for reserved in ("orchestrator", "runtime", "preflight"):
            with self.subTest(reserved=reserved):
                stage = self.stage("alpha")
                path = self.root / reserved / "forbidden.json"
                stage["outputs"] = [{"path": str(path), "min_bytes": 1}]
                stage["gate"]["path"] = str(path)
                with self.assertRaisesRegex(runner.ContractError, "reserved"):
                    self.freeze([stage])

    def test_two_stage_run_receipts_status_final_report_and_resume_skip(self) -> None:
        counter_a = self.root / "counter_a.txt"
        counter_b = self.root / "counter_b.txt"
        stages = [
            self.stage("alpha", counter=counter_a),
            self.stage("beta", dependencies=["alpha"], counter=counter_b),
        ]
        plan = self.freeze(stages)
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 0)
        self.assertEqual(counter_a.read_text("ascii"), "1")
        self.assertEqual(counter_b.read_text("ascii"), "1")
        final = runner._read_canonical_json(instance.final_path)
        final_bytes = instance.final_path.read_bytes()
        self.assertEqual(final["state"], "complete")
        self.assertEqual(len(final["completed"]), 2)
        status = runner._read_canonical_json(instance.status_path)
        self.assertEqual(status["state"], "complete")
        for stage in stages:
            receipt = instance._verify_receipt(stage, instance._receipt_path(str(stage["name"])))
            self.assertTrue(receipt["scientific_pass"])
        # A second invocation authenticates and skips both stages.
        second = self.make_runner(plan)
        self.assertEqual(second.run(), 0)
        self.assertEqual(instance.final_path.read_bytes(), final_bytes)
        self.assertEqual(counter_a.read_text("ascii"), "1")
        self.assertEqual(counter_b.read_text("ascii"), "1")

    def test_subprocess_receives_only_nested_runtime_paths(self) -> None:
        plan = self.freeze([self.stage("alpha")])
        instance = self.make_runner(plan)
        prior = os.environ.get("E26_AMBIENT_SENTINEL")
        os.environ["E26_AMBIENT_SENTINEL"] = "must-not-leak"
        try:
            self.assertEqual(instance.run(), 0)
        finally:
            if prior is None:
                os.environ.pop("E26_AMBIENT_SENTINEL", None)
            else:
                os.environ["E26_AMBIENT_SENTINEL"] = prior
        artifact = runner._read_canonical_json(self.root / "artifacts" / "alpha.json")
        for path in artifact["environment"].values():
            self.assertTrue(runner._is_within(Path(path), self.root))
        self.assertIsNone(artifact["ambient_sentinel"])
        self.assertEqual(artifact["environment_sha256"], plan["environment_sha256"])
        self.assertEqual(Path(artifact["working_directory"]), self.root)

    def test_failed_first_attempt_uses_frozen_resume_and_then_commits(self) -> None:
        counter = self.root / "counter.txt"
        marker = self.root / "mutable_checkpoint" / "failed_once.txt"
        stage = self.stage(
            "alpha", counter=counter, fail_first=marker, max_attempts=2
        )
        plan = self.freeze([stage])
        instance = self.make_runner(plan)
        rc = instance.run()
        if rc != 0:
            diagnostic = runner._read_canonical_json(instance.recovery_path)
            self.fail(f"retry run failed unexpectedly: {diagnostic['failure']!r}")
        self.assertEqual(counter.read_text("ascii"), "2")
        receipt = instance._verify_receipt(stage, instance._receipt_path("alpha"))
        self.assertEqual(receipt["attempt"], 2)
        first = runner._read_canonical_json(
            instance.attempt_root / "alpha" / "attempt_0001" / "attempt.json"
        )
        second = runner._read_canonical_json(
            instance.attempt_root / "alpha" / "attempt_0002" / "attempt.json"
        )
        self.assertEqual(first["state"], "failed")
        self.assertEqual(second["state"], "process_complete")

    def test_scientific_fail_commits_evidence_and_seals_downstream(self) -> None:
        counter = self.root / "downstream_counter.txt"
        stages = [
            self.stage("gate", decision="FAIL"),
            self.stage("downstream", dependencies=["gate"], counter=counter),
        ]
        plan = self.freeze(stages)
        instance = self.make_runner(plan)
        rc = instance.run()
        if rc != 20:
            path = instance.final_path if instance.final_path.exists() else instance.recovery_path
            diagnostic = runner._read_canonical_json(path)
            self.fail(f"scientific-fail route returned {rc}: {diagnostic['failure']!r}")
        self.assertFalse(counter.exists())
        self.assertFalse(instance._receipt_path("downstream").exists())
        report = runner._read_canonical_json(instance.final_path)
        self.assertEqual(report["state"], "scientific_fail")
        self.assertIsNone(report["next_stage"])
        final_bytes = instance.final_path.read_bytes()
        self.assertEqual(self.make_runner(plan).run(), 20)
        self.assertEqual(instance.final_path.read_bytes(), final_bytes)

    def test_output_tamper_blocks_resume_and_preserves_diagnostic(self) -> None:
        stage = self.stage("alpha")
        plan = self.freeze([stage])
        instance = self.make_runner(plan)
        first_rc = instance.run()
        if first_rc != 0:
            diagnostic = runner._read_canonical_json(instance.recovery_path)
            self.fail(f"initial run failed unexpectedly: {diagnostic['failure']!r}")
        output = self.root / "artifacts" / "alpha.json"
        output.write_bytes(output.read_bytes() + b"\n")
        resumed = self.make_runner(plan)
        self.assertEqual(resumed.run(), 1)
        report = runner._read_canonical_json(resumed.recovery_path)
        self.assertEqual(report["state"], "blocked")
        self.assertIn("drift", report["failure"]["message"])

    def test_orphan_final_output_without_receipt_fails_closed(self) -> None:
        stage = self.stage("alpha")
        plan = self.freeze([stage])
        output = Path(stage["outputs"][0]["path"])
        output.parent.mkdir(parents=True)
        output.write_bytes(b"orphan")
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 1)
        self.assertFalse(instance._receipt_path("alpha").exists())
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertIn("without any attempt", report["failure"]["message"])

    def test_completed_main_attempt_without_receipt_is_recovered_not_reexecuted(self) -> None:
        counter = self.root / "counter.txt"
        stage = self.stage("alpha", counter=counter)
        plan = self.freeze([stage])
        instance = self.make_runner(plan)
        instance._ensure_runtime_dirs()
        inputs = instance._verify_explicit_inputs(stage)
        dependencies = instance._dependency_records(stage)
        attempt, _ = instance._run_process(
            stage=stage,
            argv=stage["argv"],
            attempt_number=1,
            inputs=inputs,
            dependencies=dependencies,
        )
        self.assertEqual(attempt["state"], "process_complete")
        self.assertEqual(counter.read_text("ascii"), "1")
        self.assertFalse(instance._receipt_path("alpha").exists())
        self.assertEqual(instance.run(), 0)
        self.assertEqual(counter.read_text("ascii"), "1")
        self.assertTrue(instance._receipt_path("alpha").exists())

    def test_explicit_input_hash_drift_is_rejected(self) -> None:
        input_path = self.root / "input.bin"
        input_path.write_bytes(b"original")
        record = runner._path_record(input_path)
        stage = self.stage("alpha", inputs=[record])
        plan = self.freeze([stage])
        input_path.write_bytes(b"changed")
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 1)
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertIn("input", report["failure"]["message"])

    def test_verifier_cannot_mutate_and_bless_main_output(self) -> None:
        stage = self.stage("alpha")
        output = Path(stage["outputs"][0]["path"])
        stage["verifier_argv"] = self.command(
            output=output, payload="mutated-by-verifier", decision="PASS"
        )
        plan = self.freeze([stage])
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 1)
        self.assertFalse(instance._receipt_path("alpha").exists())
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertIn("verifier mutated", report["failure"]["message"])

    def test_verifier_cannot_mutate_frozen_source(self) -> None:
        private_stub = self.root / "private_stage.py"
        private_stub.write_bytes(STUB.read_bytes())
        output = self.root / "artifacts" / "alpha.json"
        stage = self.stage("alpha")
        stage["argv"][2] = str(private_stub)
        stage["verifier_argv"] = [
            PYTHON, "-B", str(private_stub), "mutate",
            "--path", str(private_stub), "--payload", "source-drift",
        ]
        spec = self.spec([stage])
        spec["repo_root"] = str(self.root)
        spec["source_files"] = [str(private_stub)]
        self.spec_path.write_bytes(runner.canonical_json(spec))
        plan = runner.freeze_plan(self.spec_path, self.plan_path, allow_non_e=True)
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 1)
        self.assertFalse(instance._receipt_path("alpha").exists())
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertIn("hash/size drift", report["failure"]["message"])

    def test_verifier_cannot_mutate_explicit_input(self) -> None:
        input_path = self.root / "immutable_input.bin"
        input_path.write_bytes(b"original-input")
        input_record = runner._path_record(input_path)
        stage = self.stage("alpha", inputs=[input_record])
        stage["verifier_argv"] = [
            PYTHON, "-B", str(STUB), "mutate",
            "--path", str(input_path), "--payload", "input-drift",
        ]
        plan = self.freeze([stage])
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 1)
        self.assertFalse(instance._receipt_path("alpha").exists())
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertIn("input", report["failure"]["message"])

    def test_source_drift_rejected_before_runner_construction(self) -> None:
        private_source = self.root / "source.py"
        private_source.write_text("VALUE = 1\n", encoding="utf-8")
        private_stub = self.root / "e26_stage_stub.py"
        private_stub.write_bytes(STUB.read_bytes())
        stage = self.stage("alpha")
        stage["argv"][2] = str(private_stub)
        stage["verifier_argv"][2] = str(private_stub)
        spec = self.spec([stage])
        spec["repo_root"] = str(self.root)
        spec["source_files"] = [str(private_source), str(private_stub)]
        self.spec_path.write_bytes(runner.canonical_json(spec))
        plan = runner.freeze_plan(self.spec_path, self.plan_path, allow_non_e=True)
        private_source.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(runner.ContractError, "drift"):
            self.make_runner(plan)

    def test_timeout_kills_stage_and_writes_recovery_report(self) -> None:
        stage = self.stage("slow", sleep=2.0, timeout_seconds=1)
        plan = self.freeze([stage])
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 1)
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertEqual(report["failure"]["type"], "ResourceLimitError")
        self.assertIn("timeout", report["failure"]["message"])
        attempt = runner._read_canonical_json(
            instance.attempt_root / "slow" / "attempt_0001" / "attempt.json"
        )
        self.assertEqual(attempt["state"], "failed")

    def test_live_lock_rejected_and_stale_lock_needs_explicit_recovery(self) -> None:
        lock_path = self.root / "runner.lock"
        live = runner._self_digest_payload({
            "schema": runner.LOCK_SCHEMA,
            "pid": os.getpid(),
            "process_identity": runner._process_identity(os.getpid()),
            "host": runner.socket.gethostname(),
            "nonce": "live",
            "plan_sha256": "a" * 64,
            "started_utc": runner.utc_now(),
        }, "lock_sha256")
        runner._create_once_canonical(lock_path, live)
        with self.assertRaisesRegex(runner.ContractError, "another autonomous runner"):
            runner.PipelineLock(lock_path, plan_sha256="a" * 64).acquire()
        lock_path.unlink()
        stale = runner._self_digest_payload({
            "schema": runner.LOCK_SCHEMA,
            "pid": 2_147_483_647,
            "process_identity": {
                "pid": 2_147_483_647,
            "create_time": 1.0,
                "executable": PYTHON,
            },
            "host": runner.socket.gethostname(),
            "nonce": "stale",
            "plan_sha256": "a" * 64,
            "started_utc": runner.utc_now(),
        }, "lock_sha256")
        runner._create_once_canonical(lock_path, stale)
        with self.assertRaisesRegex(runner.ContractError, "--recover-stale-lock"):
            runner.PipelineLock(lock_path, plan_sha256="a" * 64).acquire()
        recovered = runner.PipelineLock(
            lock_path, plan_sha256="a" * 64, recover_stale=True
        )
        recovered.acquire()
        recovered.release()
        self.assertTrue(any(self.root.glob("runner.lock.stale.stale.*")))

    def test_prior_live_child_attempt_blocks_duplicate_stage(self) -> None:
        stage = self.stage("alpha")
        plan = self.freeze([stage])
        instance = self.make_runner(plan)
        instance._ensure_runtime_dirs()
        attempt_dir = instance.attempt_root / "alpha" / "attempt_0001"
        attempt_dir.mkdir(parents=True)
        attempt = runner._self_digest_payload({
            "schema": runner.ATTEMPT_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "stage": "alpha",
            "pid": os.getpid(),
            "process_identity": runner._process_identity(os.getpid()),
            "state": "running",
        }, "attempt_sha256")
        runner._atomic_write_canonical(attempt_dir / "attempt.json", attempt)
        self.assertEqual(instance.run(), 1)
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertIn("still alive", report["failure"]["message"])

    def test_durable_caps_count_prior_interrupted_attempts(self) -> None:
        stage = self.stage("alpha")
        spec = self.spec([stage])
        spec["global_caps"]["max_cpu_seconds"] = 1.0
        self.spec_path.write_bytes(runner.canonical_json(spec))
        plan = runner.freeze_plan(self.spec_path, self.plan_path, allow_non_e=True)
        instance = self.make_runner(plan)
        instance._ensure_runtime_dirs()
        attempt_dir = instance.attempt_root / "alpha" / "attempt_0001"
        attempt_dir.mkdir(parents=True)
        attempt = runner._self_digest_payload({
            "schema": runner.ATTEMPT_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "stage": "alpha",
            "resource_class": "cpu",
            "pid": None,
            "state": "failed",
            "elapsed_seconds": 12.0,
            "cpu_seconds": 2.0,
        }, "attempt_sha256")
        runner._atomic_write_canonical(attempt_dir / "attempt.json", attempt)
        self.assertGreater(instance._cumulative_cpu_seconds(), 1.0)
        # The original stage has no resume command, so it cannot hide or reset
        # the previously consumed budget by starting a fresh attempt.
        self.assertEqual(instance.run(), 1)
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertEqual(report["cumulative_cpu_seconds"], 2.0)
        self.assertIn("cap exhausted before stage", report["failure"]["message"])
        self.assertFalse((self.root / "artifacts" / "alpha.json").exists())

    def test_corrupt_attempt_still_produces_recovery_report(self) -> None:
        plan = self.freeze([self.stage("alpha")])
        instance = self.make_runner(plan)
        instance._ensure_runtime_dirs()
        attempt_dir = instance.attempt_root / "alpha" / "attempt_0001"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "attempt.json").write_bytes(b'{"not":"canonical", "x":1}')
        self.assertEqual(instance.run(), 1)
        report = runner._read_canonical_json(instance.recovery_path)
        self.assertEqual(report["state"], "blocked")
        self.assertIsNotNone(report["resource_accounting_error"])
        self.assertIn("canonical", report["failure"]["message"])

    def test_final_report_self_digest_is_authenticated(self) -> None:
        plan = self.freeze([self.stage("alpha")])
        instance = self.make_runner(plan)
        self.assertEqual(instance.run(), 0)
        payload = runner._read_canonical_json(instance.final_path)
        payload["message"] = "tampered"
        instance.final_path.write_bytes(runner.canonical_json(payload))
        resumed = self.make_runner(plan)
        self.assertEqual(resumed.run(), 1)
        report = runner._read_canonical_json(resumed.recovery_path)
        self.assertIn("digest", report["failure"]["message"])

    def test_verify_snapshot_is_byte_for_byte_read_only(self) -> None:
        plan = self.freeze([self.stage("alpha")])
        instance = self.make_runner(plan)
        before = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }
        snapshot = instance.verify_snapshot()
        after = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertEqual(snapshot["receipts"], [])
        self.assertIsNone(snapshot["final"])


if __name__ == "__main__":
    unittest.main()
