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

`assess` assesses the contexts of use on the adopted fits into a
`CredibilityAssessment`: the acceptance, the coverage, the verdict and the
factors of every context, verified on tightened tolerances and a second
integration method (`verify`) and the battery of the checks of the variants
of the model (`run_battery`), which a project extends with a `BatteryHook`;
the checks are open (`Check`, `register_check`). `write_outputs` writes the
record, the tables and the report, `scorecard` and `compare` read the records
of many models.

`import sbmlsim` does not import this package.
"""

from sbmlsim.credibility.assessment import (
    AcceptanceRecord,
    ContextResult,
    Coverage,
    CoverageEntry,
    CredibilityAssessment,
    EvidenceResult,
    Provenance,
    QuantityRecord,
    Verdict,
    applicable,
    assess,
    assess_table,
    describe_selection,
    provenance,
    verdict,
)
from sbmlsim.credibility.battery import (
    Battery,
    BatteryBuild,
    BatteryCase,
    BatteryCollector,
    BatteryContribution,
    BatteryHook,
    BatteryResult,
    BatteryRun,
    BatterySettings,
    BatteryStatus,
    run_battery,
)
from sbmlsim.credibility.check import check_declaration
from sbmlsim.credibility.cross_validation import (
    CrossValidation,
    HeldOutStudy,
    cross_validate,
    read_cross_validations,
    write_cross_validation,
)
from sbmlsim.credibility.declaration import (
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
    ContextBlock,
    ContextContribution,
    EvidenceExtension,
    ExtensionSetting,
    FactorStatement,
    Pairing,
    ReportSection,
    ScorecardChange,
    read_acceptance,
    read_sections,
)
from sbmlsim.credibility.factors import (
    FACTORS,
    Factor,
    FactorRecord,
    FactorStatus,
    combine_factor,
)
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
from sbmlsim.credibility.report import print_summary, render_html, write_outputs
from sbmlsim.credibility.risk import (
    DEFAULT_ACCEPTANCE,
    DEFAULT_MIN_POINTS,
    RISK_TABLE,
    default_acceptance,
    model_risk,
)
from sbmlsim.credibility.scorecard import Regression, compare, load_records, scorecard
from sbmlsim.credibility.selection import (
    EvidenceSelection,
    SelectionError,
    Selections,
    in_domain,
    metadata_values,
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
from sbmlsim.credibility.verification import (
    BATTERY_CHECKS,
    CHECKS,
    NUMERICAL_CHECKS,
    PREDICTION_CHECKS,
    Check,
    CheckRecord,
    Verification,
    VerificationSummary,
    check_of,
    register_check,
    verify,
)

__all__ = [
    "BATTERY_CHECKS",
    "CHECKS",
    "DEFAULT_ACCEPTANCE",
    "DEFAULT_MIN_POINTS",
    "FACTORS",
    "NUMERICAL_CHECKS",
    "PREDICTION_CHECKS",
    "RISK_TABLE",
    "UNCLAIMED",
    "Acceptance",
    "AcceptanceRecord",
    "AcceptanceResult",
    "AcceptanceRule",
    "AfeRange",
    "Battery",
    "BatteryBuild",
    "BatteryCase",
    "BatteryCollector",
    "BatteryContribution",
    "BatteryHook",
    "BatteryResult",
    "BatteryRun",
    "BatterySettings",
    "BatteryStatus",
    "Check",
    "CheckRecord",
    "Consequence",
    "ContextBlock",
    "ContextContribution",
    "ContextOfUse",
    "ContextResult",
    "CouLevel",
    "Coverage",
    "CoverageEntry",
    "Credibility",
    "CredibilityAssessment",
    "CrossValidation",
    "DeclarationError",
    "Domain",
    "Evidence",
    "EvidenceExtension",
    "EvidenceResult",
    "EvidenceRole",
    "EvidenceSelection",
    "ExtensionSetting",
    "Factor",
    "FactorRecord",
    "FactorStatement",
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
    "Provenance",
    "QuantityOfInterest",
    "QuantityRecord",
    "Regression",
    "ReportSection",
    "Risk",
    "ScorecardChange",
    "Selection",
    "SelectionError",
    "Selections",
    "UncertaintyRecord",
    "Unclaimed",
    "VariabilityRecord",
    "VariabilitySettings",
    "Verdict",
    "Verification",
    "VerificationSummary",
    "acceptance_result",
    "adopted_ids",
    "applicable",
    "assess",
    "assess_table",
    "check_declaration",
    "check_of",
    "combine_factor",
    "compare",
    "cross_validate",
    "default_acceptance",
    "describe_selection",
    "evaluate_fits",
    "in_domain",
    "kind_differences",
    "load_records",
    "lock_differences",
    "mapping_table",
    "metadata_fields",
    "metadata_values",
    "model_risk",
    "normalize_observable",
    "predictive_checks",
    "print_summary",
    "propagate_fits",
    "provenance",
    "read_acceptance",
    "read_cross_validations",
    "read_sections",
    "recorded_kinds",
    "register_check",
    "render_html",
    "run_battery",
    "scorecard",
    "select",
    "verdict",
    "verify",
    "write_cross_validation",
    "write_lock",
    "write_outputs",
]
