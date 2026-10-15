import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_every_runtime_dependency_has_a_lower_and_an_upper_bound():
    # Audit SUP-1: a later install must not pull in a new major (or a new 0.x minor) of a
    # dependency the tests never ran against.
    dependencies = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["dependencies"]
    assert dependencies
    for requirement in dependencies:
        match = re.fullmatch(r"[a-z0-9-]+>=([0-9.]+),<([0-9.]+)", requirement)
        assert match, requirement
        lower = tuple(int(part) for part in match.group(1).split("."))
        upper = tuple(int(part) for part in match.group(2).split("."))
        assert lower < upper, requirement
        # Within one major, or one minor for a 0.x release.
        assert upper[0] == lower[0] + 1 or (lower[0] == 0 and upper[:2] == (0, lower[1] + 1))
