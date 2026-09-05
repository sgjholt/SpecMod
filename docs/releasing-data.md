# Publishing a dataset

How to take an event from an FDSN archive to a hash-pinned entry in
`specmod.datasets.REGISTRY`. Written to be followed end to end in one sitting.

Datasets are versioned **by registry name** — `magna_2020_v1` and
`magna_2020_v2` are separate entries — so a published result pinned to v1 keeps
fetching v1 after v2 exists. Nothing here ever revises an entry in place.

## Before you start

- Network access to the data centres named in the config (IRIS for waveforms,
  USGS for the catalogue, in Magna's case).
- Push access to the repository, and permission to create releases.
- `specmod` installed from the working tree: `uv pip install -e '.[dev]'`.
- **`pyproject.toml`'s `tag_regex` in place** — it has been on `main` since
  v0.2.0. Step 3 creates a `data-v1` tag, and without that constraint the tag
  silently becomes the package's version number. See "Why the tag prefix
  matters" below.

## 1. Fetch

```bash
specmod fetch datasets/magna_2020.toml -o build/magna_2020
```

This is the only command in the project that touches the network. It resolves
the hypocentre from the catalogue service named in the config, expands the
station wildcards against the data centre, cuts each channel to the window
relative to origin, and writes an `EventDirectory` with a `manifest.json`
beside it.

Expect it to take a few minutes — Magna is 88 stations across seven networks.

**Check the manifest before going further.** It records what came back, not
just what was asked for, and this is the moment to read it:

```bash
python -m json.tool build/magna_2020/manifest.json | head -40
```

- `event.origin`, `event.latitude/longitude/depth_km` — check these against the
  USGS event page for `uu60363602` rather than against memory. The config
  resolves the hypocentre from ComCat by id, so a wrong id gives a confidently
  wrong event rather than an error, and nothing downstream would notice.
- `channels` — the list *after* wildcard expansion. `HH*` does not tell you
  what you got; this does. A much shorter list than expected usually means a
  network was not open at that time, not that the request was malformed.
- `data_centre`, `obspy_version`, `specmod_version`, `fetched_at` — the
  provenance that makes the response reproducible-ish. It cannot be fully
  reproducible: FDSN is not content-addressed, archives get backfilled and
  catalogue solutions revised.

If any of that is wrong, fix the config and re-fetch. The artefact is about to
become immutable.

## 2. Archive and hash

```bash
tar -czf magna_2020_v1.tar.gz -C build magna_2020
shasum -a 256 magna_2020_v1.tar.gz
```

Two things that matter about this command:

- **`-C build magna_2020`** so the tarball contains a single top-level
  directory. `pooch.Untar` unpacks it into the cache and `datasets.load()`
  expects that one directory; a tarball of loose files lands them in the cache
  root.
- **The digest is the pin.** Keep the output of `shasum` — it goes in the
  registry entry, and it is what makes a corrupted or substituted download fail
  loudly instead of being analysed quietly.

Sanity-check the size before uploading. GitHub release assets are capped at
2 GB, and anything over a few hundred MB is worth reconsidering as a dataset.

## 3. Publish

Create a release tagged **`data-v1`** — not `v1`, not `1.0` — and attach the
tarball.

```bash
gh release create data-v1 magna_2020_v1.tar.gz \
  --title "Data release 1" \
  --notes "Magna, Utah, Mw 5.7, 2020-03-18. 88 stations, raw counts plus
response. Built with: specmod fetch datasets/magna_2020.toml"
```

Or through the web UI: Releases → Draft a new release → tag `data-v1` →
attach → publish.

Then copy the asset's download URL. It looks like:

```
https://github.com/sgjholt/SpecMod/releases/download/data-v1/magna_2020_v1.tar.gz
```

### Why the tag prefix matters

`data-` is not decoration. Two things in this repository read tags, and both
have to be told that data tags are not code releases:

- **hatch-vcs**, which derives `specmod.__version__` from `git describe`.
  setuptools-scm's default `tag_regex` has an optional `(?:[\w-]+-)?` prefix
  group, so it strips `data-`, reads the remaining `v1` as a version, and the
  package reports **`1`**. Measured on this repository: `0.1.0.post1.dev173`
  before the tag, `1` after it. `pyproject.toml` now constrains both
  `git_describe_command` (`--match "v[0-9]*"`) and `tag_regex`, and
  `tests/test_versioning.py` pins it.
- **release-please**, which is wired up and has cut three releases. It is
  configured manifest-first — `.release-please-manifest.json` is the source of
  truth — rather than inferring the last release by scanning tags, which is
  what keeps a `data-v1` tag from being read as a code release and offered as
  `2` next. `include-component-in-tag` is `false`, so its tags are plain `v*`.
  See [Releasing the software](releasing.md).

If the release feed gets noisy with data tags, the fallback in §5.2.3 of the
plan is a sibling `sgjholt/specmod-data` repository holding assets only — a
repo, not a package, so no extra release cycle for code.

## 4. Register

Two additions to `src/specmod/datasets.py`, landing in the same change as the
asset they point at — never before. A `load_magna_2020()` that can only 404 is
worse than its absence.

First the event itself. Only `PNR_2019` exists today, and `DatasetSpec` needs
an `Event`. **Take the values from the manifest**, not from the paper or a web
page, so that the constant and the artefact cannot disagree:

```python
MAGNA_2020 = Event(
    origin="2020-03-18T13:09:31.000000Z",   # from manifest.json
    latitude=...,
    longitude=...,
    depth_km=...,
    catalogue_magnitude=5.7,
    catalogue_magnitude_type="Mw",
)
```

Then the registry entry:

```python
REGISTRY: dict[str, DatasetSpec] = {
    "magna_2020_v1": DatasetSpec(
        name="magna_2020_v1",
        url="https://github.com/sgjholt/SpecMod/releases/download/data-v1/magna_2020_v1.tar.gz",
        sha256="sha256:<the digest from step 2>",
        event=MAGNA_2020,
        # Only if the event directory is nested inside the tarball rather than
        # being its single top-level directory. Following step 2, it is not.
        # member="",
    ),
}
```

Then verify it end to end from a cold cache:

```bash
SPECMOD_DATA_DIR=$(mktemp -d) python -c "
from specmod.datasets import load
d = load('magna_2020_v1')
print(len(d.stream()), 'traces')
print(d.inventory())
print(d.catalog())
"
```

A second run against the same `SPECMOD_DATA_DIR` should print the same thing
without downloading — that is pooch's cache working, and it is what CI relies
on.

## 5. Test

Tests needing a fetched dataset are marked, so that `pytest -m "not dataset"`
stays a complete offline run:

```python
@pytest.mark.dataset
def test_magna_loads() -> None:
    dataset = load("magna_2020_v1")
    ...
```

Run them once locally (`pytest -m dataset`), then confirm the offline suite is
still complete (`pytest -m "not dataset"`).

## If something goes wrong

**The digest does not match on download.** Re-hash the local tarball. If it
matches locally but not through pooch, the upload was truncated — delete the
asset and re-attach it. Never "fix" this by updating the hash to whatever
arrived.

**You need to change a published artefact.** You do not change it; you publish
`data-v2` and add a `magna_2020_v2` entry. The old entry stays and keeps
working, which is the entire reason versioning is by registry name.

**The tag went out without the `data-` prefix.** Delete the tag and the release
(`gh release delete v1 --cleanup-tag`), confirm `hatchling version` reports a
`0.1.0.post*` string again, and re-publish under the right name. Anyone who
built a wheel in between got a wrongly-versioned one.

**`load()` says "Unknown dataset".** The registry entry is not in the installed
copy of the package — likely an editable install pointing elsewhere, or the
entry was added but not saved.
