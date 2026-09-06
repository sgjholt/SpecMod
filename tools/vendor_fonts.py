"""Vendor the branding fonts into ``docs/_static/fonts/``.

    python tools/vendor_fonts.py

Downloads the woff2 files Google Fonts serves for the three families in
``docs/branding.md`` §2.1, and writes ``docs/_static/fonts.css`` with
``@font-face`` rules pointing at them. Run it to add a family, change a
weight, or refresh the files; it is not part of the docs build.

**Why self-hosted rather than the ``@import`` the manual specifies.** A
stylesheet that fetches from ``fonts.googleapis.com`` puts the site's
typography behind a third party being reachable *from the reader's browser* —
and it fails silently, falling back to Georgia and a system sans with nothing
in the build to notice. That is not hypothetical: it is what the sandbox this
was developed in did, and it is what any reader behind a corporate firewall
gets. Self-hosting also stops the site telling Google who is reading it.

The cost is eight files, 322 KB with the licences, and this script to
regenerate them.

**Subsets are chosen, not taken wholesale.** Google serves each family split
by ``unicode-range``; taking every subset would add Cyrillic and Vietnamese
that no page uses. Scanning the built site for non-ASCII characters gives 33
distinct ones: punctuation and mathematical symbols, and eight Greek letters
in the prose of ``processing.md`` and ``choosing-a-transform.md`` (sigma, tau,
omega, pi, lambda, delta, in both cases). So ``latin``, ``latin-ext`` and
``greek``, and nothing else.

Characters outside those subsets — subscript letters, arrows, ``almost equal
to`` — fall back to a system font, as they already did when the fonts came
from the CDN.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "docs" / "_static"
FONTS = STATIC / "fonts"

#: What `academic.css` asks for, per the typography table in the manual.
CSS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Fira+Code"
    "&family=Merriweather:wght@400;700"
    "&family=Open+Sans:wght@400;600"
    "&display=swap"
)

#: Only these. See the module docstring for the measurement behind the choice.
SUBSETS = ("latin", "latin-ext", "greek")

#: A modern browser, or the API serves the ttf stylesheet instead of woff2.
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

#: Every family here is under the SIL Open Font License 1.1, which requires
#: the licence to travel with the files.
LICENCES = {
    "Merriweather": "https://raw.githubusercontent.com/SorkinType/Merriweather/master/OFL.txt",
    "Open Sans": "https://raw.githubusercontent.com/googlefonts/opensans/main/OFL.txt",
    "Fira Code": "https://raw.githubusercontent.com/tonsky/FiraCode/master/LICENSE",
}


def fetch(url: str) -> bytes:
    """GET ``url`` as a browser would, or raise."""
    done = subprocess.run(
        ["curl", "-sSfL", "-A", UA, url],
        check=True,
        capture_output=True,
    )
    return done.stdout


def blocks(css: str) -> list[tuple[str, str]]:
    """``(subset, @font-face block)`` pairs, in the order Google wrote them.

    The subset is only knowable from the comment Google puts above each block
    — there is nothing in the rule itself that names it.
    """
    out = []
    for match in re.finditer(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", css):
        out.append((match.group(1), match.group(2)))
    return out


def describe(block: str) -> tuple[str, str, str]:
    """``(family, weight, style)`` for one ``@font-face`` block."""
    family = re.search(r"font-family:\s*'([^']+)'", block)
    weight = re.search(r"font-weight:\s*(\d+)", block)
    style = re.search(r"font-style:\s*(\w+)", block)
    if family is None or weight is None or style is None:
        raise ValueError(f"cannot name this face:\n{block}")
    return (
        family.group(1).lower().replace(" ", "-"),
        weight.group(1),
        style.group(1),
    )


def main() -> int:
    FONTS.mkdir(parents=True, exist_ok=True)
    css = fetch(CSS_URL).decode()

    # Fetch first, name second. Merriweather and Open Sans are *variable*
    # fonts, so Google hands out the same file for every weight of a family
    # and subset — five of the thirteen faces here are byte-identical pairs.
    # Naming each after its own weight would commit the same bytes twice and
    # make a reader download them twice.
    faces = []
    for subset, block in blocks(css):
        if subset not in SUBSETS:
            continue
        url = re.search(r"url\((https://[^)]+)\)", block)
        if url is None:  # pragma: no cover - Google always supplies one
            continue
        family, weight, _style = describe(block)
        faces.append((block, url.group(1), fetch(url.group(1)), family, weight, subset))

    #: Which weights each distinct file serves, so a shared one can be named
    #: for what it is rather than for whichever weight happened to fetch it.
    weights: dict[bytes, set[str]] = {}
    for _, _, data, _, weight, _ in faces:
        weights.setdefault(data, set()).add(weight)

    kept, rules, written = 0, [], set()
    for block, url, data, family, weight, subset in faces:
        shared = len(weights[data]) > 1
        stem = f"{family}-{subset}" if shared else f"{family}-{weight}-{subset}"
        name = f"{stem}.woff2"
        if name not in written:
            (FONTS / name).write_bytes(data)
            written.add(name)
            kept += 1
        # `fonts.css` sits in `_static/`, so the path is relative to it.
        rules.append(block.replace(url, f"fonts/{name}"))

    header = (
        "/* Generated by tools/vendor_fonts.py — do not edit.\n"
        " *\n"
        " * The branding fonts, served from this site rather than from\n"
        " * fonts.googleapis.com, so the typography does not depend on a third\n"
        " * party being reachable from the reader's browser. Licences and\n"
        " * provenance are in _static/fonts/README.md.\n"
        " */\n\n"
    )
    (STATIC / "fonts.css").write_text(header + "\n\n".join(rules) + "\n")

    notice = ["# Vendored fonts\n"]
    notice.append(
        "Downloaded from the Google Fonts API by `tools/vendor_fonts.py`, "
        f"subsets {', '.join(SUBSETS)}. Each family is under the SIL Open "
        "Font License 1.1; the full text of each licence sits beside the "
        "files it covers.\n"
    )
    for family, url in LICENCES.items():
        name = f"OFL-{family.lower().replace(' ', '-')}.txt"
        (FONTS / name).write_bytes(fetch(url))
        notice.append(f"- **{family}** — `{name}`, from <{url}>")
    (FONTS / "README.md").write_text("\n".join(notice) + "\n")

    print(f"wrote {kept} font files and 3 licences into {FONTS}")
    print(f"wrote {STATIC / 'fonts.css'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
