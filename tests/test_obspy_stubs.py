"""The hand-written ObsPy stubs must describe the ObsPy that is installed.

ObsPy ships no ``py.typed`` and neither ``obspy-stubs`` nor ``types-obspy``
exists on PyPI, so ``stubs/obspy`` in this repository is what makes
``st: Stream`` mean anything. That is worth having — wiring it in immediately
found two latent defects that ``Any`` had hidden, an unguarded ``None``
inventory in ``set_stream_distance`` and an unguarded ``None`` signal stream in
``plot_traces``.

It is also a liability. A stub that has drifted from the library is worse than
no stub at all, because a type checker believes it and reports nothing. mypy
cannot detect the drift: it reads the stub *instead of* the library, so a
method we renamed in the stub and a method ObsPy deleted look identical to it.

These tests are the only thing that can tell the difference, because they load
the real ObsPy. They check that every name declared exists and that function
parameters match. They do **not** check return types — those were read off the
running library when the stubs were written and are noted in the stubs where
they are surprising.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

obspy = pytest.importorskip("obspy")

STUBS = Path(__file__).resolve().parent.parent / "stubs" / "obspy"

#: Declared in the stub for the type checker's benefit but not present as a
#: real attribute, so a runtime lookup is not the right check.
#:
#: `Stats` is an `AttribDict`: `__getattr__`, `__getitem__` and friends come
#: from the base class or from its dynamic behaviour, and asking whether the
#: class "has" them answers a different question than the stub is making a
#: claim about.
NOT_RUNTIME_ATTRIBUTES = frozenset(
    {
        "__getattr__",
        "__setattr__",
        "__getitem__",
        "__setitem__",
        "__contains__",
        "__iter__",
        "__len__",
        "__add__",
        "__iadd__",
        "__sub__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__float__",
        "__init__",
    }
)


def _stub_classes(path: Path) -> dict[str, list[ast.FunctionDef]]:
    """Class name to its declared methods, for one ``.pyi``."""
    tree = ast.parse(path.read_text())
    found: dict[str, list[ast.FunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found[node.name] = [
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef)
                and child.name not in NOT_RUNTIME_ATTRIBUTES
            ]
    return found


def _stub_functions(path: Path) -> list[ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def _declared_parameters(node: ast.FunctionDef) -> list[str]:
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return [n for n in names if n != "self"]


def test_the_stub_directory_is_where_the_configuration_says() -> None:
    """A stub tree mypy is not reading silences nothing and checks nothing."""
    import tomllib  # noqa: PLC0415

    root = STUBS.parent.parent
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert config["tool"]["mypy"]["mypy_path"] == "stubs"
    assert STUBS.is_dir()
    # And obspy must NOT be in the ignore-missing-imports list, or the stubs
    # are redundant and every ObsPy object is `Any` again.
    ignored: set[str] = set()
    for override in config["tool"]["mypy"]["overrides"]:
        if override.get("ignore_missing_imports"):
            ignored.update(override["module"])
    assert "obspy.*" not in ignored


@pytest.mark.parametrize(
    ("stub", "module"),
    [
        ("core/trace.pyi", "obspy.core.trace"),
        ("core/stream.pyi", "obspy.core.stream"),
        ("core/utcdatetime.pyi", "obspy.core.utcdatetime"),
        ("core/inventory.pyi", "obspy.core.inventory"),
    ],
)
def test_every_declared_class_and_method_exists(stub: str, module: str) -> None:
    import importlib  # noqa: PLC0415

    real = importlib.import_module(module)
    for name, methods in _stub_classes(STUBS / stub).items():
        cls = getattr(real, name, None)
        assert cls is not None, f"{module}.{name} is declared but does not exist"
        for method in methods:
            assert hasattr(cls, method.name), (
                f"{module}.{name}.{method.name} is declared in {stub} but "
                "does not exist on the installed ObsPy"
            )


@pytest.mark.parametrize(
    ("stub", "module"),
    [
        ("__init__.pyi", "obspy"),
        ("geodetics/__init__.pyi", "obspy.geodetics"),
        ("signal/konnoohmachismoothing.pyi", "obspy.signal.konnoohmachismoothing"),
    ],
)
def test_every_declared_function_exists_with_the_same_parameters(
    stub: str, module: str
) -> None:
    """Names *and* parameter names.

    A keyword argument renamed upstream is the drift most likely to go
    unnoticed, because every call in this package that uses it still reads
    correctly.
    """
    import importlib  # noqa: PLC0415

    real = importlib.import_module(module)
    for node in _stub_functions(STUBS / stub):
        function = getattr(real, node.name, None)
        assert function is not None, (
            f"{module}.{node.name} is declared in {stub} but does not exist"
        )
        actual = list(inspect.signature(function).parameters)
        declared = _declared_parameters(node)
        # Positional prefix must line up; the stub may stop early (`**kwargs`
        # upstream) but must not invent or reorder.
        assert declared[: len(actual)] == actual[: len(declared)], (
            f"{module}.{node.name}: stub declares {declared}, ObsPy has {actual}"
        )


def test_the_class_methods_take_the_parameters_the_stub_claims() -> None:
    """The same check for methods, where the risk is identical."""
    checks: list[tuple[str, Any]] = [
        ("core/trace.pyi", obspy.core.trace),
        ("core/stream.pyi", obspy.core.stream),
    ]
    for stub, real in checks:
        for name, methods in _stub_classes(STUBS / stub).items():
            cls = getattr(real, name)
            for node in methods:
                function = getattr(cls, node.name)
                try:
                    actual = [
                        p for p in inspect.signature(function).parameters if p != "self"
                    ]
                except (TypeError, ValueError):  # pragma: no cover
                    continue  # a C-level or property object; nothing to compare
                declared = _declared_parameters(node)
                if "args" in actual or "kwargs" in actual:
                    continue  # upstream forwards; the stub says so too
                assert declared[: len(actual)] == actual[: len(declared)], (
                    f"{name}.{node.name}: stub declares {declared}, ObsPy has {actual}"
                )


def test_the_open_half_of_stats_is_really_open() -> None:
    """The stub lets any unknown field through as ``Any``, and it must.

    SpecMod sets fourteen of its own fields on ``Stats``. If ObsPy ever made
    the class reject unknown keys, the stub would be describing something
    permissive that no longer is, and every one of those writes would be a
    runtime failure the type checker had blessed.
    """
    trace = obspy.Trace()
    trace.stats["p_time"] = obspy.UTCDateTime(0)
    trace.stats["repi"] = 1.5
    assert trace.stats["p_time"] == obspy.UTCDateTime(0)
    assert trace.stats.repi == 1.5


def test_the_arithmetic_the_stub_overloads_behaves_as_declared() -> None:
    """``UTCDateTime`` minus ``UTCDateTime`` is seconds; minus a float is a time.

    Declared as an overload, which is the one place the stub says something a
    reader might doubt. It is also the asymmetry that produced a real bug in
    ``cut_c`` — ``float + UTCDateTime`` raises, because there is no
    ``__radd__``.
    """
    t0 = obspy.UTCDateTime("2019-08-26T07:49:24.2")
    assert isinstance(t0 - t0, float)
    assert isinstance(t0 - 1.0, obspy.UTCDateTime)
    assert isinstance(t0 + 1.0, obspy.UTCDateTime)
    with pytest.raises(TypeError):
        _ = 1.0 + t0  # the raise is the assertion
