"""The credibility of a model for its contexts of use (ASME V&V 40, ICH M15).

A model declares what it is used for (`ContextOfUse`), how risky a wrong answer
is, which evidence supports it and when it is accepted (`Credibility`); its
project references the declaration (`sbmlsim.project.Project.credibility`)
and gives the metadata of its fit mappings, the fields of it a domain can
claim and its evidence extensions (`sbmlsim.project.CredibilitySettings`). The
evidence is selected from the data points of the adopted fits
(`evaluate_fits`, `mapping_table`) with the kinds their records store
(`select`, `check_declaration`) and fixed in advance by a lock
(`write_lock`, `lock_differences`). The parameter uncertainty of the adopted
fits is propagated to the decisive predictions (`propagate_fits`,
`PropagationSettings`), the decisive points are checked with the declared
variability of the subjects (`predictive_checks`, `VariabilitySettings`), the
visual predictive check, and an adopted fit is cross-validated by leaving out
one study at a time (`cross_validate`). The factors of V&V 40 (`FACTORS`) list
what an assessment assesses. A domain adds evidence of its own with an
`EvidenceExtension`, e.g. the ratios between the arms of a study.

`import sbmlsim` does not import this package.
"""

from sbmlsim.credibility.check import check_declaration
from sbmlsim.credibility.cross_validation import (
    CrossValidation,
    HeldOutStudy,
    cross_validate,
    read_cross_validations,
    write_cross_validation,
)
from sbmlsim.credibility.declaration import (
    DEFAULT_COVARIATES,
    UNCLAIMED,
    Acceptance,
    AcceptanceResult,
    AcceptanceRule,
    AfeRange,
    Consequence,
    ContextOfUse,
    CouLevel,
    Credibility,
    DeclarationError,
    Domain,
    Evidence,
    EvidenceRole,
    Influence,
    Level,
    MaxGmfe,
    QuantityOfInterest,
    Risk,
    Selection,
    Unclaimed,
    acceptance_result,
    metadata_fields,
)
from sbmlsim.credibility.extensions import (
    ContextContribution,
    EvidenceExtension,
    ExtensionSetting,
    Pairing,
    read_sections,
)
from sbmlsim.credibility.factors import FACTORS, Factor, FactorRecord, FactorStatus
from sbmlsim.credibility.lock import LockMismatchError, lock_differences, write_lock
from sbmlsim.credibility.records import (
    FitTable,
    KindMismatchError,
    MappingLookup,
    adopted_ids,
    evaluate_fits,
    kind_differences,
    mapping_table,
    recorded_kinds,
)
from sbmlsim.credibility.risk import (
    DEFAULT_ACCEPTANCE,
    DEFAULT_MIN_POINTS,
    RISK_TABLE,
    default_acceptance,
    model_risk,
)
from sbmlsim.credibility.selection import (
    EvidenceSelection,
    SelectionError,
    Selections,
    normalize_observable,
    select,
)
from sbmlsim.credibility.uncertainty import (
    FitPropagation,
    PropagationSettings,
    UncertaintyRecord,
    propagate_fits,
)
from sbmlsim.credibility.variability import (
    PredictiveChecks,
    VariabilityRecord,
    VariabilitySettings,
    predictive_checks,
)

__all__ = [
    "DEFAULT_ACCEPTANCE",
    "DEFAULT_COVARIATES",
    "DEFAULT_MIN_POINTS",
    "FACTORS",
    "RISK_TABLE",
    "UNCLAIMED",
    "Acceptance",
    "AcceptanceResult",
    "AcceptanceRule",
    "AfeRange",
    "Consequence",
    "ContextContribution",
    "ContextOfUse",
    "CouLevel",
    "Credibility",
    "CrossValidation",
    "DeclarationError",
    "Domain",
    "Evidence",
    "EvidenceExtension",
    "EvidenceRole",
    "EvidenceSelection",
    "ExtensionSetting",
    "Factor",
    "FactorRecord",
    "FactorStatus",
    "FitPropagation",
    "FitTable",
    "HeldOutStudy",
    "Influence",
    "KindMismatchError",
    "Level",
    "LockMismatchError",
    "MappingLookup",
    "MaxGmfe",
    "Pairing",
    "PredictiveChecks",
    "PropagationSettings",
    "QuantityOfInterest",
    "Risk",
    "Selection",
    "SelectionError",
    "Selections",
    "UncertaintyRecord",
    "Unclaimed",
    "VariabilityRecord",
    "VariabilitySettings",
    "acceptance_result",
    "adopted_ids",
    "check_declaration",
    "cross_validate",
    "default_acceptance",
    "evaluate_fits",
    "kind_differences",
    "lock_differences",
    "mapping_table",
    "metadata_fields",
    "model_risk",
    "normalize_observable",
    "predictive_checks",
    "propagate_fits",
    "read_cross_validations",
    "read_sections",
    "recorded_kinds",
    "select",
    "write_cross_validation",
    "write_lock",
]
