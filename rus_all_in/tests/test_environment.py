from aiijc_puzzle.environment import run_smoke_check


def test_environment_smoke() -> None:
    report = run_smoke_check()
    assert report["status"] == "ok"
    assert report["python"].startswith("3.11.")
