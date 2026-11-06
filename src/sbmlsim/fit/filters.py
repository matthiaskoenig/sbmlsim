"""The filters of the fit mappings a fit selects its data with.

A filter is called with the key of a fit mapping and the mapping and says
whether the mapping passes (`sbmlsim.fit.helpers.MappingFilter`), see
`sbmlsim.fit.helpers.FitMappings.select` for what a fit does with the mappings
which pass. The filters here are frozen dataclasses, so they compare, pickle
(into the workers of a parallel fit) and describe themselves:

- `metadata_filter` by the values of the fields of the metadata of a mapping,
  a dataclass derived from `MappingMetaData`, e.g. its route and dosing; the
  value of a field is `MappingMetaData.filter_value`, which a metadata class
  overrides for a field it compares in another form;
- `observable_filter` by the observable of a mapping (`observable_id`);
- `named_filter` by the key of a mapping together with its study (`named`),
  since two studies can share a key;
- `filter_keys` by the key of a mapping alone, what a selection by keys, e.g.
  the validation data or a problem read from PEtab, uses.
"""

import dataclasses
from collections.abc import Collection, Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sbmlsim.fit.objects import MappingMetaData

if TYPE_CHECKING:
    from sbmlsim.fit.objects import FitMapping

__all__ = [
    "KeyFilter",
    "MetadataFilter",
    "NamedFilter",
    "NamedMappings",
    "ObservableFilter",
    "filter_keys",
    "metadata_filter",
    "named",
    "named_filter",
    "observable_filter",
    "observable_id",
    "study",
]


def observable_id(fit_mapping: FitMapping) -> str:
    """Get the observable of a fit mapping, the index of its data.

    The index of the y data of the observable (`sbmlsim.data.Data.index`): the
    selection without the brackets of a concentration, `Cve_hctz` for
    `[Cve_hctz]`, `Aurine_hctz` for `Aurine_hctz`.

    Args:
        fit_mapping: the fit mapping.

    Returns:
        The id of the observable, e.g. `Cve_hctz`.
    """
    return fit_mapping.observable.y.index


def study(fit_mapping: FitMapping) -> str:
    """Get the study of a fit mapping, the id of its experiment."""
    return fit_mapping.reference.experiment.sid


def _show(values: frozenset[Hashable]) -> str:
    """Show values sorted, enum members by their name."""
    names = sorted(v.name if isinstance(v, Enum) else repr(v) for v in values)
    return f"[{', '.join(names)}]"


def _values(name: str, values: object) -> frozenset[Hashable]:
    """Check that the values of a clause are a collection and make a set of them.

    A single value would silently become a set of its characters when it is a
    string (every `StrEnum` member is one).

    Raises:
        TypeError: if the values are a string or not a collection.
    """
    if isinstance(values, str | bytes) or not isinstance(values, Collection):
        raise TypeError(
            f"{name} needs a collection of values, e.g. a set, not {values!r}"
        )
    return frozenset(values)


@dataclass(frozen=True)
class MetadataFilter:
    """A filter of fit mappings by the values of their metadata.

    A mapping passes when every allowed field has one of its values and no
    excluded field has one of its values; the value of a field is the one of
    `MappingMetaData.filter_value`. Build it with `metadata_filter`.

    Attributes:
        metadata_type: the metadata class of the mappings.
        allowed: the fields with the values a mapping must have.
        excluded: the fields with the values a mapping must not have.
    """

    metadata_type: type[MappingMetaData]
    allowed: tuple[tuple[str, frozenset[Hashable]], ...]
    excluded: tuple[tuple[str, frozenset[Hashable]], ...]

    def __call__(self, fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
        """Check a fit mapping, see `sbmlsim.fit.helpers.MappingFilter`.

        Raises:
            TypeError: if the mapping has no metadata of the type of the filter.
        """
        metadata = fit_mapping.metadata
        if not isinstance(metadata, self.metadata_type):
            raise TypeError(
                f"{self.metadata_type.__name__} required on: '{fit_mapping}'"
            )
        return all(
            metadata.filter_value(name) in values for name, values in self.allowed
        ) and not any(
            metadata.filter_value(name) in values for name, values in self.excluded
        )

    def __repr__(self) -> str:
        """Say what the filter selects."""
        clauses = [f"{name} in {_show(values)}" for name, values in self.allowed]
        clauses += [f"{name} not in {_show(values)}" for name, values in self.excluded]
        return f"MetadataFilter({self.metadata_type.__name__}: {' and '.join(clauses)})"


def metadata_filter(
    metadata_type: type[MappingMetaData],
    /,
    *,
    excluded: Mapping[str, Collection[Hashable]] | None = None,
    **allowed: Collection[Hashable],
) -> MetadataFilter:
    """Build a filter of fit mappings by the values of their metadata.

    ```python
    # not executed
    filter_control = metadata_filter(
        StudyMetaData,
        route={Route.PO, Route.IV},
        excluded={"dosing": {Dosing.MULTIPLE}},
    )
    ```

    Args:
        metadata_type: the metadata class of the mappings, a dataclass derived
            from `MappingMetaData`.
        excluded: the fields with the values a mapping must not have.
        **allowed: the fields with the values a mapping must have.

    Returns:
        The filter, called by the selection of the mappings of a fit.

    Raises:
        TypeError: if the metadata class is no dataclass (or an instance of
            one), has no field of a given name, or the values of a field are not
            a collection (a single value, also a string or `None`), i.e. when
            the module of the filters is imported.
        ValueError: for a filter without clauses, with an empty set of allowed
            values, which would select nothing, or with an empty set of excluded
            values, which would exclude nothing.
    """
    if not isinstance(metadata_type, type):
        raise TypeError(
            f"metadata_type must be the class, not an instance: {metadata_type!r}"
        )
    if not dataclasses.is_dataclass(metadata_type):
        raise TypeError(f"{metadata_type.__name__} is not a dataclass")
    excluded = excluded or {}
    names = {field.name for field in dataclasses.fields(metadata_type)}
    unknown = sorted((set(allowed) | set(excluded)) - names)
    if unknown:
        raise TypeError(f"{metadata_type.__name__} has no fields {unknown}")
    if not allowed and not excluded:
        raise ValueError("a metadata filter needs at least one clause")
    allowed_sets = {name: _values(name, values) for name, values in allowed.items()}
    excluded_sets = {name: _values(name, values) for name, values in excluded.items()}
    empty = sorted(name for name, values in allowed_sets.items() if not values)
    if empty:
        raise ValueError(f"an empty set of allowed values selects nothing: {empty}")
    empty = sorted(name for name, values in excluded_sets.items() if not values)
    if empty:
        raise ValueError(f"an empty set of excluded values excludes nothing: {empty}")
    return MetadataFilter(
        metadata_type=metadata_type,
        allowed=tuple(allowed_sets.items()),
        excluded=tuple(excluded_sets.items()),
    )


@dataclass(frozen=True)
class ObservableFilter:
    """A filter of fit mappings by their observable, see `observable_id`.

    Attributes:
        observables: the ids of the observables a mapping may have, the index
            of the data (`Data.index`).
    """

    observables: frozenset[str]

    def __call__(self, fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
        """Check a fit mapping, see `sbmlsim.fit.helpers.MappingFilter`."""
        return observable_id(fit_mapping) in self.observables

    def __repr__(self) -> str:
        """Say what the filter selects."""
        return f"ObservableFilter({', '.join(sorted(self.observables))})"


def observable_filter(observables: Collection[str]) -> ObservableFilter:
    """Build a filter of fit mappings by their observable, e.g. the PK data.

    Args:
        observables: the ids of the observables, the index of the data
            (`Data.index`), e.g. `Cve_hctz`.

    Returns:
        The filter.

    Raises:
        TypeError: if the observables are a single id or not a collection.
        ValueError: for no observables, which would select nothing.
    """
    ids = _values("observables", observables)
    if not ids:
        raise ValueError("an empty set of observables selects nothing")
    return ObservableFilter(frozenset(str(i) for i in ids))


#: the keys of fit mappings by study: the keys of the mappings are SIds of their
#: experiment, which two studies can share, so a named mapping is named with its
#: study
type NamedMappings = dict[str, set[str]]


def named(
    mappings: NamedMappings, fit_mapping_key: str, fit_mapping: FitMapping
) -> bool:
    """Check whether a fit mapping is one of the named mappings of its study.

    The key alone is not enough: two studies can share a key, so it counts only
    for the study it is named with.
    """
    return fit_mapping_key in mappings.get(study(fit_mapping), set())


@dataclass(frozen=True)
class NamedFilter:
    """A filter of fit mappings by their key and study, see `named`.

    Attributes:
        mappings: the study ids with their mapping keys, sorted, so that the
            filter compares, pickles and describes the same for equal mappings.
    """

    mappings: tuple[tuple[str, frozenset[str]], ...]

    def __call__(self, fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
        """Check a fit mapping, see `sbmlsim.fit.helpers.MappingFilter`."""
        return fit_mapping_key in dict(self.mappings).get(study(fit_mapping), ())


def named_filter(mappings: NamedMappings) -> NamedFilter:
    """Build a filter of the fit mappings named with their study.

    Args:
        mappings: the mapping keys by study id.

    Returns:
        The filter.
    """
    return NamedFilter(
        tuple((sid, frozenset(keys)) for sid, keys in sorted(mappings.items()))
    )


@dataclass(frozen=True)
class KeyFilter:
    """A filter of fit mappings by their key alone, in whichever study.

    Attributes:
        keys: the keys of the mappings which pass.
    """

    keys: frozenset[str]

    def __call__(self, fit_mapping_key: str, fit_mapping: FitMapping) -> bool:
        """Check a fit mapping, see `sbmlsim.fit.helpers.MappingFilter`."""
        return fit_mapping_key in self.keys

    def __repr__(self) -> str:
        """Say what the filter selects."""
        return f"KeyFilter({', '.join(sorted(self.keys))})"


def filter_keys(keys: Iterable[str]) -> KeyFilter:
    """Build a filter which selects the fit mappings of the given keys.

    A selector written by hand says what it means, e.g. "the tablets"; this
    one says which mappings it resolved to and is what a problem read from
    PEtab uses, because a condition stores the resolution and not the rule. A
    key selects the mapping of that key in every study, `named_filter` names
    the study as well.

    Args:
        keys: the keys of the fit mappings to select.

    Returns:
        A filter which passes exactly those mappings.
    """
    return KeyFilter(frozenset(keys))
