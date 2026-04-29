import os
import pytest
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--h5-path",
        default=None,
        help="Absolute path to an H5 match file used by integration tests.",
    )


@pytest.fixture(scope="session")
def h5_path(request):
    value = request.config.getoption("--h5-path")
    if value is not None:
        p = Path(value)
        if p.exists():
            return p

    env_value = os.environ.get("ORBIT_TEST_H5")
    if env_value is not None:
        p = Path(env_value)
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parents[2]
    matches_dir = repo_root / "data" / "matches"
    if matches_dir.exists():
        found = sorted(matches_dir.rglob("*.h5"))
        if found:
            return found[0]

    pytest.skip("No H5 test data found — set ORBIT_TEST_H5 or pass --h5-path")
