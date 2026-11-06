from typing import Any

import pytest

from sbmlsim.run import ConfigurationError, Kind, PlanMismatch, RunContext, Spec
from sbmlsim.run import kinds as kinds_module
from sbmlsim.run.kinds import KINDS, get_kind, register
from tests.run.kinds_dummy import DUMMY, DummySpec


class _Plain(Spec):
    x: int = 1


def _run(spec: _Plain, ctx: RunContext) -> dict[str, Any]:
    return {"x": spec.x}


#: not a kind, for the reference of a module attribute which is none
NOT_A_KIND = 1


def test_reference_to_a_kind() -> None:
    assert get_kind("tests.run.kinds_dummy:DUMMY") is DUMMY


@pytest.mark.parametrize(
    "reference",
    [
        "tests.run.test_kinds:NOT_A_KIND",
        "tests.run.test_kinds:MISSING",
        "tests.run.missing_module:DUMMY",
    ],
)
def test_reference_to_no_kind_is_a_configuration_error(reference: str) -> None:
    with pytest.raises(ConfigurationError):
        get_kind(reference)


def test_unknown_kind_lists_the_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(KINDS, "plain", Kind(name="plain", spec=_Plain, run=_run))
    with pytest.raises(KeyError, match="plain") as info:
        get_kind("unknown")
    assert isinstance(info.value, ConfigurationError)
    assert "unknown" in str(info.value)


def test_register_twice_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kinds_module, "KINDS", {})
    kind = Kind(name="plain", spec=_Plain, run=_run)
    register(kind)
    assert kinds_module.KINDS["plain"] is kind
    with pytest.raises(ValueError, match="plain"):
        register(kind)


def test_a_name_with_a_colon_is_refused() -> None:
    with pytest.raises(ValueError, match="':'"):
        register(Kind(name="a:b", spec=_Plain, run=_run))


def test_splittable() -> None:
    assert DUMMY.splittable
    plain = Kind(name="plain", spec=_Plain, run=_run)
    assert not plain.splittable
    assert plain.plan(_Plain(x=3)) == _Plain(x=3)


def test_a_kind_is_checked() -> None:
    with pytest.raises(ValueError, match="input"):
        Kind(name="plain", spec=_Plain, run=_run, inputs=("missing",))
    with pytest.raises(ValueError, match="merge"):
        Kind(
            name="plain",
            spec=DummySpec,
            run=DUMMY.run,
            compute_unit=DUMMY.compute_unit,
        )
    with pytest.raises(ValueError, match="units"):
        Kind[Any](
            name="plain",
            spec=_Plain,
            run=_run,
            compute_unit=DUMMY.compute_unit,
            merge=DUMMY.merge,
        )


def test_spec_forbids_extra_fields_and_is_frozen() -> None:
    with pytest.raises(ValueError, match="extra"):
        _Plain.model_validate({"x": 1, "y": 2})
    spec = _Plain()
    with pytest.raises(ValueError, match="frozen"):
        spec.x = 2  # ty: ignore[invalid-assignment]


def test_plan_mismatch_is_a_configuration_error() -> None:
    assert issubclass(PlanMismatch, ConfigurationError)
