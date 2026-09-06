"""Vendor the KaTeX stylesheet and fonts into ``docs/_static/katex/``.

    python tools/vendor_katex.py

``sphinxcontrib-katex`` ships the KaTeX *JavaScript* itself and serves it from
``_static/``, but its stylesheet defaults to a jsdelivr URL — so out of the box
the site still depends on a CDN, just for a smaller file. This fetches the
matching CSS and its woff2 fonts so nothing is requested from outside.

Fetched from the npm registry rather than from a CDN: it is where the CDN gets
it, it is the release the version number actually names, and it carries the
licence in the same tarball.

**The version is read from the extension, never typed here.** KaTeX's rendered
markup and its stylesheet are coupled: CSS from a different release than the
JavaScript that produced the markup mis-sizes delimiters and radicals, and
does it quietly rather than failing. The extension bundles one specific
version, so that is the only correct one to vendor, and asking it is the only
way to stay correct when it is upgraded. A hand-typed version here would be a
number that silently rots — the prototype for this work vendored 0.18.7
against the bundled 0.16.22 before anyone checked.

Only woff2 is kept. KaTeX also ships woff and ttf for browsers that predate
these docs by a decade, and together they triple the payload.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "_static" / "katex"


def katex_version() -> str:
    """The KaTeX release ``sphinxcontrib-katex`` bundles the JavaScript for."""
    import sphinxcontrib.katex as extension  # noqa: PLC0415

    version = getattr(extension, "katex_version", None)
    if not version:
        raise SystemExit(
            "sphinxcontrib-katex no longer exposes `katex_version`; find where "
            "it records the version it bundles and read it from there rather "
            "than hard-coding one."
        )
    return str(version)


def fetch(url: str) -> bytes:
    done = subprocess.run(
        ["curl", "-sSfL", url],
        check=True,
        capture_output=True,
    )
    return done.stdout


def main() -> int:
    version = katex_version()
    tar = fetch(f"https://registry.npmjs.org/katex/-/katex-{version}.tgz")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fonts").mkdir(exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:gz") as archive:
        members = {m.name: m for m in archive.getmembers()}

        def read(name: str) -> bytes:
            handle = archive.extractfile(members[f"package/{name}"])
            if handle is None:  # pragma: no cover - a directory, never here
                raise SystemExit(f"{name} is not a file in the katex tarball")
            return handle.read()

        css = read("dist/katex.min.css").decode()
        licence = read("LICENSE")
        shipped = {
            name.removeprefix("package/dist/")
            for name in members
            if "/dist/fonts/" in name
        }

        # Drop the woff and ttf sources before resolving what to extract, so
        # the stylesheet and the files beside it cannot disagree about which
        # formats exist.
        css = re.sub(
            r"src:[^;]*;",
            lambda m: re.sub(
                r",?\s*url\([^)]*\.(?:woff|ttf)\)\s*format\([^)]*\)", "", m.group(0)
            ),
            css,
        )

        fonts = sorted(set(re.findall(r"url\((fonts/[^)]+\.woff2)\)", css)))
        missing = [name for name in fonts if name not in shipped]
        if missing:
            raise SystemExit(f"stylesheet names fonts not in the tarball: {missing}")
        for name in fonts:
            (OUT / name).write_bytes(read(f"dist/{name}"))

    (OUT / "katex.min.css").write_text(css)
    (OUT / "LICENSE").write_bytes(licence)
    (OUT / "README.md").write_text(
        f"# Vendored KaTeX {version}\n\n"
        f"Stylesheet and woff2 fonts for KaTeX {version}, taken from the npm\n"
        "release by `tools/vendor_katex.py`. The version is whatever\n"
        "`sphinxcontrib-katex` bundles the JavaScript for — the two are\n"
        "coupled, and a mismatch mis-sizes delimiters silently. Re-run the\n"
        "script after upgrading the extension.\n\n"
        "KaTeX is MIT licensed; `LICENSE` is the text as published.\n"
    )

    print(f"vendored KaTeX {version}: katex.min.css and {len(fonts)} woff2 fonts")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
