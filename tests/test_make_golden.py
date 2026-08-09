"""``tools/make_golden.py`` must not destroy what it cannot recapture.

``bsnr_legacy`` and ``band_legacy`` record what the pre-refactor lineage
produced. The code paths that produced them are gone, so the committed
reference files are the only copies, and ``test_golden_reference.py`` and
``test_pipeline.py`` each assert the record is still present.

The generator did not write them — they were added to the reference files by
hand — so the documented ``python tools/make_golden.py`` step silently dropped
them. These tests pin the carry-forward that fixes it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_make_golden() -> ModuleType:
    """Import ``tools/make_golden.py``, a script rather than a package member."""
    pytest.importorskip("obspy")
    path = ROOT / "tools" / "make_golden.py"
    spec = importlib.util.spec_from_file_location("make_golden", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_golden"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def make_golden() -> ModuleType:
    return _load_make_golden()


class TestTheCommittedReferencesStillHoldTheLegacyRecord:
    """The reason the carry-forward exists, asserted on the real files."""

    def test_every_pipeline_window_has_bsnr_legacy(self) -> None:
        reference = json.loads(
            (ROOT / "tests" / "golden" / "pipeline_reference.json").read_text()
        )
        missing = [
            f"{estimator}/{name}"
            for estimator, windows in reference.items()
            if estimator != "_environment"
            for name, record in windows.items()
            if "bsnr_legacy" not in record
        ]
        assert not missing, missing

    def test_every_motion_pair_has_band_legacy(self) -> None:
        reference = json.loads(
            (ROOT / "tests" / "golden" / "motion_reference.json").read_text()
        )
        missing = [
            name
            for name, record in reference["displacement"].items()
            if "band_legacy" not in record
        ]
        assert not missing, missing


class TestCarryForward:
    def test_it_copies_the_legacy_record_into_a_fresh_capture(
        self, make_golden: ModuleType, tmp_path: Path
    ) -> None:
        previous = {
            "_environment": {"system": "Linux"},
            "fft": {"LV.L001..HHE": {"bsnr": {"n": 3}, "bsnr_legacy": {"n": 3}}},
        }
        path = tmp_path / "pipeline_reference.json"
        path.write_text(json.dumps(previous))

        fresh = {
            "_environment": {"system": "Linux"},
            "fft": {"LV.L001..HHE": {"bsnr": {"n": 3}}},
        }
        make_golden._carry_forward_legacy(fresh, None, "bsnr_legacy", path)
        assert fresh["fft"]["LV.L001..HHE"]["bsnr_legacy"] == {"n": 3}

    def test_a_sectioned_reference_carries_too(
        self, make_golden: ModuleType, tmp_path: Path
    ) -> None:
        """``motion_reference.json`` nests its records under ``displacement``."""
        path = tmp_path / "motion_reference.json"
        path.write_text(
            json.dumps({"displacement": {"LV.L001..HHE": {"band_legacy": [1.0, 2.0]}}})
        )
        fresh = {
            "_environment": {},
            "displacement": {"LV.L001..HHE": {"band": [1.0, 3.0]}},
        }
        make_golden._carry_forward_legacy(fresh, "displacement", "band_legacy", path)
        assert fresh["displacement"]["LV.L001..HHE"]["band_legacy"] == [1.0, 2.0]

    def test_a_missing_record_is_reported_rather_than_passed_over(
        self,
        make_golden: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A silent gap is the failure the carry-forward exists to prevent."""
        path = tmp_path / "pipeline_reference.json"
        path.write_text(json.dumps({"fft": {"LV.L001..HHE": {"bsnr": {"n": 3}}}}))
        fresh = {"fft": {"NEW.STATION..HHE": {"bsnr": {"n": 3}}}}
        make_golden._carry_forward_legacy(fresh, None, "bsnr_legacy", path)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "NEW.STATION..HHE" in out

    def test_no_previous_file_is_not_an_error(
        self, make_golden: ModuleType, tmp_path: Path
    ) -> None:
        fresh = {"fft": {"LV.L001..HHE": {"bsnr": {"n": 3}}}}
        make_golden._carry_forward_legacy(
            fresh, None, "bsnr_legacy", tmp_path / "absent.json"
        )
        assert "bsnr_legacy" not in fresh["fft"]["LV.L001..HHE"]
