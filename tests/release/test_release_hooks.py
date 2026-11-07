"""Tests of the release settings of a framework and of the release hook."""

import dataclasses
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from sbmlsim import project_checks
from sbmlsim.release.hooks import CREDIBILITY_PARTS, ReleaseHook
from sbmlsim.release.settings import ReleaseSettings
from tests.project_toy import PROJECT

assert PROJECT.release is not None
SETTINGS: ReleaseSettings = PROJECT.release

#: the release hook of a framework, which the tests reference
HOOK = ReleaseHook()
#: no release hook, which the tests reference
NO_HOOK = object()


def test_the_framework_of_a_release_is_sbmlsim_by_default() -> None:
    assert SETTINGS.framework == "sbmlsim"
    assert SETTINGS.core_libraries == ("sbmlsim", "sbmlutils")
    assert SETTINGS.archive is None
    assert SETTINGS.hook is None
    framework = dataclasses.replace(
        SETTINGS, framework="pkpdsim", core_libraries=["pkpdsim", "sbmlsim"]
    )
    assert framework.core_libraries == ("pkpdsim", "sbmlsim")
    assert hash(framework) == hash(dataclasses.replace(framework))
    assert framework.missing() == []
    assert dataclasses.replace(SETTINGS, core_libraries=()).missing() == []


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"framework": "pkpd sim"}, "no name of a package"),
        ({"core_libraries": ["sbmlsim", "a/b"]}, "no name of a package"),
        ({"hook": "pkpdsim.release"}, "module:attribute"),
    ],
)
def test_the_settings_of_a_framework_are_checked(changes: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(SETTINGS, **changes)


def test_a_project_without_a_hook_has_the_hook_of_sbmlsim() -> None:
    hook = PROJECT.release_hook()
    assert type(hook) is ReleaseHook
    assert type(PROJECT.replace(release=None).release_hook()) is ReleaseHook


def test_the_hook_of_a_project_is_resolved() -> None:
    settings = dataclasses.replace(SETTINGS, hook=f"{__name__}:HOOK")
    assert PROJECT.replace(release=settings).release_hook() is HOOK
    settings = dataclasses.replace(SETTINGS, hook=HOOK)
    assert PROJECT.replace(release=settings).release_hook() is HOOK
    wrong = PROJECT.replace(
        release=dataclasses.replace(SETTINGS, hook=f"{__name__}:NO_HOOK")
    )
    with pytest.raises(ValueError, match=r"is no sbmlsim\.release\.hooks\.ReleaseHook"):
        wrong.release_hook()
    with pytest.raises(AssertionError, match=r"is no sbmlsim\.release\.hooks"):
        project_checks.check_model(wrong)
    missing = PROJECT.replace(
        release=dataclasses.replace(SETTINGS, hook=f"{__name__}:MISSING")
    )
    with pytest.raises(AssertionError, match=r"release hook .* does not resolve"):
        project_checks.check_model(missing)


def test_the_hook_adds_nothing_by_default(tmp_path: Path) -> None:
    hook = ReleaseHook()
    nothing: Any = None
    assert hook.templates() == ()
    assert hook.values(nothing, tmp_path) == {}
    assert hook.pages(nothing) == ()
    assert hook.notes_sections(nothing, nothing, nothing) == ()
    data: dict[str, Any] = {"contexts": []}
    assert hook.credibility(nothing, data) is data
    assert hook.data_study(PurePosixPath("drug/Study2000/a.tsv")) == "drug/Study2000"
    assert hook.data_study(PurePosixPath("collect_data.py")) is None
    assert set(CREDIBILITY_PARTS) == {"context", "sections"}
