"""The credibility settings of a project: its metadata, its domain fields and its extensions."""

import pydantic
import pytest

from sbmlsim.credibility import DeclarationError, EvidenceExtension
from sbmlsim.fit.objects import MappingMetaData
from sbmlsim.project import CredibilitySettings
from tests.credibility_toy import PROJECT
from tests.credibility_toy.extensions import ARMS
from tests.credibility_toy.metadata import Health, Route, ToyMetaData


def test_the_metadata_of_the_project() -> None:
    assert PROJECT.metadata_type() is ToyMetaData
    assert PROJECT.domain_fields() == {"route": Route, "health": Health}
    plain = PROJECT.replace(credibility_settings=CredibilitySettings())
    assert plain.metadata_type() is MappingMetaData
    assert plain.domain_fields() == {}
    every = PROJECT.replace(
        credibility_settings=CredibilitySettings(metadata_type=ToyMetaData)
    )
    assert every.domain_fields() == {"route": Route, "health": Health}


def test_the_project_checks_the_domains() -> None:
    assert PROJECT.load_credibility().context("dose").domain.metadata
    outside = PROJECT.replace(credibility=".credibility:OUTSIDE")
    with pytest.raises(DeclarationError) as err:
        outside.load_credibility()
    first, second = err.value.problems
    assert "the domain field 'label' is not a field of the metadata" in first
    assert "'route' is not a member of Route" in second
    # without the fields of the metadata no domain claims a value
    plain = PROJECT.replace(credibility_settings=CredibilitySettings())
    with pytest.raises(DeclarationError, match="it has no fields"):
        plain.load_credibility()


def test_a_declaration_of_another_kind_is_not_checked() -> None:
    other = PROJECT.replace(credibility={"context": "a plain object"})
    assert other.load_credibility() == {"context": "a plain object"}


def test_the_extensions_of_the_project() -> None:
    assert PROJECT.evidence_extensions() == []
    by_reference = PROJECT.replace(
        credibility_settings=CredibilitySettings(extensions=(".extensions:ARMS",))
    )
    assert by_reference.evidence_extensions() == [ARMS]
    by_object = PROJECT.replace(
        credibility_settings=CredibilitySettings(extensions=(ARMS,))
    )
    assert by_object.evidence_extensions() == [ARMS]
    twice = PROJECT.replace(
        credibility_settings=CredibilitySettings(extensions=(ARMS, ARMS))
    )
    with pytest.raises(ValueError, match=r"a name of their own, not \['arms'\]"):
        twice.evidence_extensions()
    nameless = PROJECT.replace(
        credibility_settings=CredibilitySettings(extensions=(EvidenceExtension(),))
    )
    with pytest.raises(ValueError, match="a name of their own"):
        nameless.evidence_extensions()
    wrong = PROJECT.replace(
        credibility_settings=CredibilitySettings(extensions=(".metadata:Route",))
    )
    with pytest.raises(ValueError, match=r"are no sbmlsim\.credibility\.extensions"):
        wrong.evidence_extensions()


def test_the_settings_are_checked() -> None:
    with pytest.raises(pydantic.ValidationError, match="a class or a reference"):
        CredibilitySettings(metadata_type=1)
    with pytest.raises(pydantic.ValidationError):
        CredibilitySettings(extensions=("no reference",))
    not_a_class = PROJECT.replace(
        credibility_settings=CredibilitySettings(metadata_type=".fitting:DOSE")
    )
    with pytest.raises(ValueError, match="is no class"):
        not_a_class.metadata_type()
