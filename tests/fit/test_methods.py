"""Tests of the methods of a fit."""

import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.fit import methods
from sbmlsim.fit.derived import HookSummary
from sbmlsim.fit.fisher import DEFAULT_STEP, FisherInformation
from sbmlsim.fit.identifiability import IdentifiabilityResult, ProfileSettings
from sbmlsim.fit.methods import (
    ReportContext,
    UnsupportedSetting,
    describe_parameter,
    methods_section,
)
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem, _optimizer_metadata
from sbmlsim.fit.options import (
    DataAtChangeType,
    FitSettings,
    LossFunctionType,
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.report.document import Document, Section, p
from sbmlsim.report.references import (
    bibliography,
    cited_keys,
    load_references,
    write_bibliography,
)
from sbmlsim.report.render import (
    HTMLRenderer,
    LaTeXRenderer,
    TypstRenderer,
    compile_pdf,
)

Fitted = tuple[OptimizationProblem, OptimizationResult]

SETTINGS = [
    FitSettings(
        residual=r, loss_function=loss, weighting_curves=curves, weighting_points=points
    )
    for r, loss, curves, points in itertools.product(
        list(ResidualType),
        list(LossFunctionType),
        [
            (),
            (WeightingCurvesType.MAPPING,),
            (WeightingCurvesType.POINTS,),
            (WeightingCurvesType.MAPPING, WeightingCurvesType.POINTS),
        ],
        list(WeightingPointsType),
    )
]


def _id(settings: FitSettings) -> str:
    """Get the id of a test of the settings."""
    curves = "+".join(w.name for w in settings.weighting_curves) or "NONE"
    return (
        f"{settings.residual.name}-{settings.loss_function.name}-{curves}-"
        f"{settings.weighting_points.name}"
    )


def _keys(section: Section) -> list[str]:
    return [b.key for b in section.blocks if isinstance(b, Section)]


def _html(section: Section) -> str:
    return HTMLRenderer().render(Document("t", (section,)))


def _section(
    fitted: Fitted,
    settings: FitSettings | None = None,
    identifiability: IdentifiabilityResult | None = None,
    fisher: FisherInformation | None = None,
    context: ReportContext | None = None,
) -> Section:
    problem, result = fitted
    return methods_section(
        problem=problem,
        settings=settings if settings is not None else FitSettings(),
        result=result,
        identifiability=identifiability,
        fisher=fisher,
        context=context if context is not None else ReportContext(),
    )


def _identifiability(
    problem: OptimizationProblem, reoptimize: bool = True
) -> IdentifiabilityResult:
    """Get an analysis without profiles, its settings are what the methods say."""
    return IdentifiabilityResult(
        opid=problem.opid,
        parameter_set=problem.parameter_set_model(),
        parameters=problem.parameters,
        settings=ProfileSettings(reoptimize=reoptimize),
        fit_settings=FitSettings(),
        cost=1.0,
        profiles={},
    )


def _fisher(
    problem: OptimizationProblem, step: float = DEFAULT_STEP
) -> FisherInformation:
    """Get a Fisher information of rank 2 of the three parameters."""
    k = len(problem.pids)
    matrix = np.diag([10.0**-j for j in range(k)])
    matrix[-1, -1] = 0.0
    return FisherInformation(
        opid=problem.opid,
        sid="fit",
        pids=list(problem.pids),
        values=np.ones(k),
        scale=ParameterScaleType.LOG10,
        matrix=matrix,
        cost=1.0,
        n=40,
        step=step,
    )


@pytest.mark.parametrize("settings", SETTINGS, ids=_id)
def test_every_setting_has_methods(fitted: Fitted, settings: FitSettings) -> None:
    section = _section(fitted, settings)
    assert _keys(section) == [
        "methods:parameters",
        "methods:objective",
        "methods:optimization",
        "methods:metrics",
        "methods:provenance",
    ]
    doc = Document("t", (section,))
    assert set(cited_keys(doc)) <= set(load_references())
    TypstRenderer().render(doc)
    HTMLRenderer().render(doc)
    LaTeXRenderer().render(doc)


def test_text_follows_the_settings(fitted: Fitted) -> None:
    settings = FitSettings(
        residual=ResidualType.NORMALIZED,
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    )
    html = _html(
        _section(fitted, settings, context=ReportContext(description="PK of HCTZ."))
    )
    assert "PK of HCTZ." in html
    assert "coefficient of variation" in html
    assert "trust region reflective" in html
    assert "seed 1234" in html


def test_only_what_the_fit_used(fitted: Fitted) -> None:
    html = _html(_section(fitted, FitSettings()))
    assert "standard least squares problem" in html
    assert "coefficient of variation" not in html
    assert "baseline" not in html
    assert "all curves have the same weight" in html
    assert "differential evolution" not in html
    assert "hybridization" not in html
    assert "the parameter set with the lowest cost was used" in html

    html = _html(
        _section(
            fitted,
            FitSettings(
                residual=ResidualType.NORMALIZED_TO_BASELINE,
                loss_function=LossFunctionType.CAUCHY,
                weighting_curves=(
                    WeightingCurvesType.POINTS,
                    WeightingCurvesType.MAPPING,
                ),
                weighting_points=WeightingPointsType.ERROR_WEIGHTING,
                parameter_scale=ParameterScaleType.LOG,
                variable_step_size=False,
                absolute_tolerance=AbsoluteTolerance(amount=1e-8, ids={"x": 1e-12}),
                data_at_change=DataAtChangeType.AFTER,
            ),
        )
    )
    assert "standard least squares problem" not in html
    assert "first data point" in html
    assert "weight 0" in html
    assert "natural log space" in html
    assert "without the variable step size" in html
    assert "1 state of its own" in html
    assert "state after the change" in html


def test_parameters_and_data(fitted: Fitted) -> None:
    problem, _ = fitted
    context = ReportContext(
        parameter_descriptions={"GU__HCTZABS_k": "absorption rate"},
        notes=("The dissolution is fitted on the oral data",),
    )
    html = _html(_section(fitted, context=context))
    assert "3 parameters were optimized" in html
    assert "absorption rate (<code>GU__HCTZABS_k</code>)" in html
    assert "The dissolution is fitted on the oral data." in html
    counts = problem.mapping_counts()
    n_training = counts[MappingKind.TRAINING]
    points = sum(len(problem.y_references[k]) for k in problem.training_indices)
    assert (
        f"{n_training} fit mappings of training data with {points} data points" in html
    )
    assert "validation data" in html and "outlier" in html


def test_a_kind_without_mappings_is_not_described(
    fitted: Fitted, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, _ = fitted
    counts = {MappingKind.TRAINING: problem.mapping_counts()[MappingKind.TRAINING]}
    monkeypatch.setattr(problem, "mapping_counts", lambda: counts)
    html = _html(_section(fitted))
    assert "validation" not in html
    assert "outlier" not in html


def test_describe_parameter(fitted: Fitted) -> None:
    problem, _ = fitted
    pid = "Ka_dis_hctz"
    sbml_name = describe_parameter(problem, pid, ReportContext())
    assert sbml_name, "the name of the target in the model"
    parameter = next(p for p in problem.parameters if p.pid == pid)
    original = parameter.description
    try:
        parameter.description = "dissolution"
        assert describe_parameter(problem, pid, ReportContext()) == "dissolution"
        context = ReportContext(parameter_descriptions={pid: "context"})
        assert describe_parameter(problem, pid, context) == "context"
    finally:
        parameter.description = original
    assert describe_parameter(problem, "no_such_parameter", ReportContext()) == ""


def test_without_provenance(fitted: Fitted) -> None:
    problem, result = fitted
    result.provenance = None
    html = _html(
        methods_section(
            problem=problem,
            settings=FitSettings(),
            result=result,
            identifiability=None,
            fisher=None,
            context=ReportContext(),
        )
    )
    assert "was not recorded" in html
    assert "<table" not in html
    assert "seed" not in html


def test_provenance(fitted: Fitted) -> None:
    _, result = fitted
    assert result.provenance is not None
    html = _html(_section(fitted))
    assert f"The fit {result.sid} was run on" in html
    assert "SHA-256 checksums are listed in" in html
    assert "<table" in html

    result.provenance = replace(result.provenance, git=None)
    assert "not under version control" in _html(_section(fitted))


def test_without_result(fitted: Fitted) -> None:
    problem, _ = fitted
    section = methods_section(
        problem=problem,
        settings=FitSettings(),
        result=None,
        identifiability=None,
        fisher=None,
        context=ReportContext(),
    )
    html = _html(section)
    assert "was not recorded" in html
    assert "libRoadRunner" in html


def test_differential_evolution(fitted: Fitted) -> None:
    _, result = fitted
    metadata = _optimizer_metadata(
        OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION, {"rng": 1}
    )
    for fit in result.fits:
        fit.algorithm = OptimizationAlgorithmType.DIFFERENTIAL_EVOLUTION.name
        fit.optimizer_kwargs = metadata
    html = _html(_section(fitted))
    assert "differential evolution" in html
    assert "best1bin" in html
    assert "trust region reflective" not in html
    assert "L-BFGS-B" in html


def test_identifiability(fitted: Fitted) -> None:
    problem, _ = fitted
    section = _section(
        fitted,
        identifiability=_identifiability(problem),
        fisher=_fisher(problem, step=5e-4),
    )
    assert _keys(section) == [
        "methods:parameters",
        "methods:objective",
        "methods:optimization",
        "methods:metrics",
        "methods:identifiability",
        "methods:provenance",
    ]
    html = _html(section)
    assert "profile likelihood" in html
    assert "re-optimized" in html
    assert "1.92" in html
    assert "structurally non-identifiable" in html
    assert "Gauss-Newton" in html
    assert "rank 2 of 3" in html
    assert "reaches the threshold above the optimum but not below it" in html
    assert "ends at the lower bound or after 50 points" in html
    typst = TypstRenderer().render(Document("t", (section,)))
    assert "epsilon = 5 dot 10^(-4)$" in typst
    # one symbol for the parameters and one for their number
    assert "theta" not in typst and "n - k" not in typst and "n-k" not in typst
    assert "2 n_p" in typst and "(2 F) / (n - n_p)" in typst

    scan = _html(
        _section(fitted, identifiability=_identifiability(problem, reoptimize=False))
    )
    assert "re-optimized" not in scan and "scan of the cost" in scan
    assert "Gauss-Newton" not in scan

    fisher_only = _section(fitted, fisher=_fisher(problem))
    assert "methods:identifiability" in _keys(fisher_only)
    assert "profile likelihood" not in _html(fisher_only)


def test_an_unsupported_setting_raises(
    fitted: Fitted, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(methods.LOSS_TEXT, LossFunctionType.SOFT_L1)
    _section(fitted, FitSettings())
    with pytest.raises(UnsupportedSetting, match="SOFT_L1"):
        _section(fitted, FitSettings(loss_function=LossFunctionType.SOFT_L1))


def test_the_pdf_builds(fitted: Fitted, tmp_path: Path) -> None:
    problem, _ = fitted
    section = _section(
        fitted,
        FitSettings(
            residual=ResidualType.NORMALIZED_TO_BASELINE,
            loss_function=LossFunctionType.SOFT_L1,
            weighting_curves=(WeightingCurvesType.MAPPING, WeightingCurvesType.POINTS),
            weighting_points=WeightingPointsType.ERROR_WEIGHTING,
        ),
        identifiability=_identifiability(problem),
        fisher=_fisher(problem),
        context=ReportContext(description="PK of HCTZ.", notes=("A note",)),
    )
    doc = Document("Methods", (section,))
    doc = replace(doc, references=bibliography(doc))
    write_bibliography(doc.references, tmp_path)
    typ = tmp_path / "methods.typ"
    typ.write_text(TypstRenderer().render(doc))
    assert compile_pdf(typ).stat().st_size > 0


def test_a_parameter_with_a_scale_of_its_own(
    fitted: Fitted, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, _ = fitted
    parameter = problem.parameters[0]
    monkeypatch.setattr(parameter, "scale", ParameterScaleType.LINEAR)
    html = _html(_section(fitted))
    assert f"except <code>{parameter.pid}</code> on the linear scale" in html
    assert f"uniformly for <code>{parameter.pid}</code>" in html


class _Hook:
    """A hybridization which only describes itself."""

    def summary(self) -> HookSummary:
        return HookSummary(
            name="net",
            kind="rate",
            description="2 layers",
            targets=("k",),
            groups=(),
        )


def test_hybridizations(fitted: Fitted, monkeypatch: pytest.MonkeyPatch) -> None:
    problem, _ = fitted
    monkeypatch.setattr(problem, "hybridizations", [_Hook()])
    html = _html(_section(fitted))
    assert "the hybridizations <code>net</code> (rate: 2 layers)" in html


@pytest.mark.parametrize(
    ("value", "typst"),
    [
        (0, "0"),
        (0.0, "0"),
        (1e-12, "$10^(-12)$"),
        (2.5e-7, "$2.5 dot 10^(-7)$"),
        (-2.5e-7, "$-2.5 dot 10^(-7)$"),
        (-1e-6, "$-10^(-6)$"),
        (1e-4, "$10^(-4)$"),
        (5e-4, "$5 dot 10^(-4)$"),
        (-5e-4, "$-5 dot 10^(-4)$"),
        (9.99e-4, "$9.99 dot 10^(-4)$"),
        (1e-3, "0.001"),
        (0.05, "0.05"),
        (-0.5, "-0.5"),
        (2, "2"),
        (12345, "12345"),
        (99999, "99999"),
        (1e5, "$10^(5)$"),
        (123456789, "$1.23 dot 10^(8)$"),
        (1e8, "$10^(8)$"),
        (-3e6, "$-3 dot 10^(6)$"),
        (float("inf"), "$oo$"),
    ],
)
def test_numbers_of_every_decade(value: float, typst: str) -> None:
    assert TypstRenderer().block(p(methods._number(value))) == typst
