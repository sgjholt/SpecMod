"""The release configuration must agree with how the version is derived.

Two files decide a release between them, and nothing makes them talk:
``release-please-config.json`` decides what the tag is *called*, and
``pyproject.toml`` decides which tags ``hatch-vcs`` will read (see
``test_versioning.py``). If they disagree the failure is quiet and expensive —
the tag is created, the release is published, and the wheel built from it
carries the fallback version ``0.1.1.postN.devN`` instead of ``0.2.0``. PyPI
will accept that upload and will not let the filename be reused.

So the coupling is asserted here rather than left to be discovered once.
``tools/check_built_version.py`` is the same check made again at publish time,
on the artefact instead of the configuration, and its logic is tested below.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "release-please-config.json"
MANIFEST = ROOT / ".release-please-manifest.json"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def package_config() -> dict:
    """The root package's entry — this is a single-package manifest."""
    packages = json.loads(CONFIG.read_text())["packages"]
    assert list(packages) == ["."], "one package, at the repository root"
    return packages["."]


@pytest.fixture(scope="module")
def tag_regex() -> re.Pattern[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return re.compile(pyproject["tool"]["hatch"]["version"]["raw-options"]["tag_regex"])


def _load_check_built_version() -> ModuleType:
    """Import ``tools/check_built_version.py``, which is a script rather than
    a package member and so is not importable by name."""
    path = ROOT / "tools" / "check_built_version.py"
    spec = importlib.util.spec_from_file_location("check_built_version", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_built_version"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load_check_built_version()


class TestTheTagFormat:
    def test_the_component_stays_out_of_the_tag(self, package_config: dict) -> None:
        """With it left on, release-please tags ``specmod-v0.2.0``, which
        pyproject's ``--match v[0-9]*`` does not describe and its ``tag_regex``
        does not parse. The default differs between release-please's own
        modes, so it is set explicitly rather than relied on."""
        assert package_config["include-component-in-tag"] is False

    @pytest.mark.parametrize("version", ["0.2.0", "0.10.0", "1.0.0", "0.2.0-rc1"])
    def test_the_tag_release_please_will_create_is_one_hatch_vcs_reads(
        self, tag_regex: re.Pattern[str], version: str
    ) -> None:
        # `include-component-in-tag: false` means the tag is exactly this.
        tag = f"v{version}"
        match = tag_regex.match(tag)
        assert match is not None, f"hatch-vcs would not parse the tag {tag}"
        assert match.group("version") == version


class TestTheVersionItWillPropose:
    def test_the_manifest_names_the_same_single_package(self) -> None:
        assert list(json.loads(MANIFEST.read_text())) == ["."]

    def test_the_recorded_version_is_a_release_version(self) -> None:
        """release-please computes the next version from this one, so it has
        to be a version and not a description of one."""
        version = json.loads(MANIFEST.read_text())["."]
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), version

    def test_a_breaking_change_cannot_mint_1_0_0_by_itself(
        self, package_config: dict
    ) -> None:
        """The plan's §6.5 keeps the project on 0.x for the whole refactor.
        There are already two `!` commits in the history, and the default
        behaviour would read either as a 1.0.0 — a version that says the API
        has stopped moving, minted with a DOI that cannot be retracted.

        This only matters below 1.0, which is exactly where the manifest is.
        """
        major = int(json.loads(MANIFEST.read_text())["."].split(".")[0])
        if major >= 1:
            pytest.skip("past 1.0; a breaking change should bump the major")
        assert package_config["bump-minor-pre-major"] is True


class TestTheWorkflowUsesTheseFiles:
    """A rename here is silent: release-please falls back to its defaults and
    releases with none of the settings above.

    Reads the live workflow. It used to read a staged copy under ``ci/``,
    which existed because an agent token could not push ``.github/workflows/``;
    that copy is gone, and this reads the file that actually runs.
    """

    @pytest.mark.parametrize("path", [CONFIG, MANIFEST])
    def test_the_workflow_names_the_config_files(self, path: Path) -> None:
        assert path.name in WORKFLOW.read_text()

    def test_the_publish_step_runs_the_built_version_check(self) -> None:
        assert "tools/check_built_version.py" in WORKFLOW.read_text()


class TestTheBuiltVersionCheck:
    """``tools/check_built_version.py`` runs between `uv build` and the
    upload, where a wrong answer is permanent."""

    def _dist(self, tmp_path: Path, *names: str) -> Path:
        dist = tmp_path / "dist"
        dist.mkdir()
        for name in names:
            (dist / name).touch()
        return dist

    def test_a_matching_pair_passes(self, checker: ModuleType, tmp_path: Path) -> None:
        dist = self._dist(
            tmp_path, "specmod-0.2.0-py3-none-any.whl", "specmod-0.2.0.tar.gz"
        )
        assert checker.check("v0.2.0", dist) == []

    def test_the_fallback_version_is_caught(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        """What a tag `tag_regex` cannot parse actually produces."""
        dist = self._dist(tmp_path, "specmod-0.1.1.post1.dev4-py3-none-any.whl")
        problems = checker.check("v0.2.0", dist)
        assert len(problems) == 1
        assert "0.1.1.post1.dev4" in problems[0]

    def test_one_bad_artefact_among_good_ones_is_caught(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        dist = self._dist(
            tmp_path, "specmod-0.2.0-py3-none-any.whl", "specmod-0.1.1.tar.gz"
        )
        assert len(checker.check("v0.2.0", dist)) == 1

    def test_an_empty_dist_is_a_failure_not_a_pass(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        """Nothing to compare must not read as nothing wrong."""
        assert checker.check("v0.2.0", self._dist(tmp_path)) != []

    def test_a_tag_without_the_v_is_refused(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        dist = self._dist(tmp_path, "specmod-0.2.0-py3-none-any.whl")
        assert checker.check("0.2.0", dist) != []


class TestTheRoadmapKeepsUpWithTheReleases:
    """``docs/roadmap.md`` says what shipped in which version. Nothing moved it.

    release-please writes ``CHANGELOG.md`` and the manifest on merge; the
    roadmap is hand-written and was not in its path. So v0.3.0 went out with
    its own contents still filed on that page under *In progress — not yet
    released*, where they sat for a week: the page most likely to be read as
    "what is done" was the one page saying the done work was not.

    The rule was already written down — ``docs/releasing.md`` distinguishes
    merged from shipped, and the roadmap's own closing section says an entry
    gets its version when a release goes out. Both were true and neither was
    checked, which is §6.6's shape exactly.
    """

    #: Every version the roadmap names as carrying shipped work.
    SHIPPED = re.compile(r"^##+ Shipped in v(\d+)\.(\d+)\.(\d+)", re.MULTILINE)

    def _roadmap_versions(self) -> list[tuple[int, int, int]]:
        text = (ROOT / "docs" / "roadmap.md").read_text()
        return [tuple(int(p) for p in m.groups()) for m in self.SHIPPED.finditer(text)]

    def _released(self) -> tuple[int, int, int]:
        version = json.loads(MANIFEST.read_text())["."]
        return tuple(int(p) for p in version.split("."))

    def test_the_roadmap_names_some_shipped_version(self) -> None:
        """A heading rename would leave the check below matching nothing."""
        assert self._roadmap_versions()

    def test_every_version_the_roadmap_claims_has_actually_been_released(
        self,
    ) -> None:
        """A section written ahead of its release is the same defect the other
        way round: the page claims a reader can install it and check."""
        changelog = (ROOT / "CHANGELOG.md").read_text()
        missing = [
            f"{major}.{minor}.{patch}"
            for major, minor, patch in self._roadmap_versions()
            if f"## [{major}.{minor}.{patch}]" not in changelog
        ]
        assert not missing, (
            f"docs/roadmap.md files work under {missing}, which CHANGELOG.md "
            "has no release heading for. Unreleased work belongs under "
            "*In progress — not yet released* until the release goes out."
        )

    def test_the_roadmap_is_not_behind_the_released_minor(self) -> None:
        """Feature and breaking work bumps the minor while this is 0.x, and
        that is the work the roadmap tracks. A patch release is deliberately
        not enough to fail this: a bug fix does not owe the page a section, and
        a check that fires on every release teaches people to add empty ones.
        """
        released = self._released()
        newest = max(self._roadmap_versions())
        assert newest[:2] >= released[:2], (
            f"v{'.'.join(map(str, released))} is released and the newest "
            f"version docs/roadmap.md names is v{'.'.join(map(str, newest))}. "
            "Move what shipped out of *In progress — not yet released* and "
            "give it a `## Shipped in` section."
        )
