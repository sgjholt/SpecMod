#!/usr/bin/env python
"""Check that the distributions in ``dist/`` carry the version of the tag.

Run as::

    python tools/check_built_version.py v0.2.0 dist

The version is never written down: ``hatch-vcs`` derives it from
``git describe``, filtered by the ``tag_regex`` in ``pyproject.toml``. That
makes one silent failure possible — a tag the regex does not match leaves the
build falling back to ``<last-tag>.postN.devN``, which is a perfectly valid
version string and would be uploaded under that name. PyPI does not allow a
filename to be reused, so a wrong upload is permanent.

This runs between ``uv build`` and the upload, on the artefacts themselves
rather than on the configuration that produced them. Standard library only:
the publish job has no virtualenv.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: ``specmod-0.2.0-py3-none-any.whl`` and ``specmod-0.2.0.tar.gz``. Both put
#: the version in the second ``-``/``.``-delimited field of the stem, and both
#: are normalised by the build backend, so the comparison is exact.
_WHEEL = re.compile(r"^(?P<name>[^-]+)-(?P<version>[^-]+)-.*\.whl$")
_SDIST = re.compile(r"^(?P<name>[^-]+)-(?P<version>.+)\.tar\.gz$")


def versions_in(dist: Path) -> dict[str, str]:
    """Map each distribution filename in ``dist`` to the version it declares."""
    found: dict[str, str] = {}
    for path in sorted(dist.iterdir()):
        for pattern in (_WHEEL, _SDIST):
            match = pattern.match(path.name)
            if match:
                found[path.name] = match.group("version")
                break
    return found


def check(tag: str, dist: Path) -> list[str]:
    """Return the problems found; an empty list means the artefacts are good."""
    if not tag.startswith("v"):
        return [f"tag {tag!r} does not start with 'v', which pyproject requires"]
    expected = tag[1:]

    found = versions_in(dist)
    if not found:
        return [f"no wheel or sdist in {dist}"]

    return [
        f"{filename} is version {version}, expected {expected} from tag {tag}"
        for filename, version in found.items()
        if version != expected
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2

    tag, dist = argv[1], Path(argv[2])
    problems = check(tag, dist)
    if problems:
        for problem in problems:
            print(f"ERROR  {problem}")
        print(
            "\nThe usual cause is a tag pyproject.toml's tag_regex does not "
            "match, which leaves hatch-vcs on its fallback version."
        )
        return 1

    for filename, version in versions_in(dist).items():
        print(f"ok  {filename} is {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
