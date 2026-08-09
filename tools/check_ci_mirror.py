#!/usr/bin/env python
"""Report whether the staged copies in ``ci/`` match the live workflows.

``ci/workflows/*.yml`` holds complete copies of ``.github/workflows/*.yml``,
because a Claude Code session's token cannot push to ``.github/workflows/``.
See ``ci/README.md``.

Exits non-zero when any pair differs, so it can be wired into CI once the pair
is in sync. Until the copy is made the mirror is ahead on purpose, and a
difference here is the reminder rather than a fault.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGED = ROOT / "ci" / "workflows"
LIVE = ROOT / ".github" / "workflows"


def main() -> int:
    if not STAGED.is_dir():
        print(f"no staged workflows at {STAGED.relative_to(ROOT)}")
        return 0

    differing = 0
    for staged in sorted(STAGED.glob("*.yml")):
        live = LIVE / staged.name
        if not live.is_file():
            print(f"MISSING  {live.relative_to(ROOT)} — copy {staged.name} into place")
            differing += 1
            continue

        want, have = staged.read_text(), live.read_text()
        if want == have:
            print(f"in sync  {live.relative_to(ROOT)}")
            continue

        differing += 1
        print(f"DIFFERS  {live.relative_to(ROOT)} — copy the staged file over it")
        sys.stdout.writelines(
            difflib.unified_diff(
                have.splitlines(keepends=True),
                want.splitlines(keepends=True),
                fromfile=str(live.relative_to(ROOT)),
                tofile=str(staged.relative_to(ROOT)),
            )
        )

    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
