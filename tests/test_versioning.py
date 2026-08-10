"""The version comes from `v*` tags and from nothing else.

Data artefacts are published under their own tag prefix (`data-v1`), and
setuptools-scm's default `tag_regex` would read one as a code release — its
optional `(?:[\\w-]+-)?` prefix group strips `data-` and parses the `v1` that
is left. Measured on this repository, adding a `data-v1` tag took the reported
version from ``0.1.0.post1.dev173`` to ``1``.

`pyproject.toml` constrains both the describe command and the parse. These
tests pin the parse, which is the half that can be checked without a git
repository to hand.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _raw_options() -> dict:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return pyproject["tool"]["hatch"]["version"]["raw-options"]


@pytest.fixture(scope="module")
def tag_regex() -> re.Pattern[str]:
    return re.compile(_raw_options()["tag_regex"])


class TestTagRegex:
    @pytest.mark.parametrize(
        ("tag", "version"),
        [("v0.1.0", "0.1.0"), ("v1.2.3", "1.2.3"), ("v0.2.0rc1", "0.2.0rc1")],
    )
    def test_a_code_release_tag_parses(
        self, tag_regex: re.Pattern[str], tag: str, version: str
    ) -> None:
        match = tag_regex.match(tag)
        assert match is not None
        assert match.group("version") == version

    @pytest.mark.parametrize(
        "tag", ["data-v1", "data-v2", "data-v1.1", "magna-v1", "nightly"]
    )
    def test_a_non_code_tag_does_not_parse(
        self, tag_regex: re.Pattern[str], tag: str
    ) -> None:
        # The whole point: `data-v1` must not reach a version of `1`.
        assert tag_regex.match(tag) is None


def test_describe_only_matches_v_tags() -> None:
    """The first line of defence: such a tag never reaches the parse."""
    command = _raw_options()["git_describe_command"]
    assert "--match" in command
    assert command[command.index("--match") + 1] == "v[0-9]*"
