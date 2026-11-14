"""The interactive panel of a model: its knobs, curves, study data and simulation.

A model declares its interactive surface in an annotation of its SBML: the
knobs a reader turns, the curves which are drawn, the study data under them and
the simulation they run. This module is the declaration itself, frozen and
picklable data, and the versioned wire contract `panel/1` which a renderer
sees: `panel_payload` gives the JSON of a panel and `panel_from_payload` reads
it back, each the inverse of the other for every field.

The `App` of a model is the surface above these panels: a nav of `AppPage`s,
each a list of panel ids with the `Scenario`s a reader picks from, named
states of their knobs, the `AppVariant`s a reader switches between and the
citation, the DOI and the licenses of the provenance footer, so a model app is
configuration and no code of its own. It is the declaration of the project
(`Project.app`) and no part of the SBML, since pages and navigation are no
model semantics, and it carries the contract `APP_CONTRACT` of its own
(`app_payload`, `app_from_payload`): a renderer of an app reads its version and
the one of every panel it places. A page names the panels of the model, which
the declaration cannot check itself, since a project is imported without its
SBML; it is checked where the models of a service are registered
(`sbmlsim.serve.runner.ModelRegistry.of`), before any request.

The module is pure data: it imports pydantic and the standard library only, no
simulator, no libsbml and no project, so `sbmlsim.project` imports it for the
fields `Project.panels` and `Project.app`.

The hard limits `MAX_END`, `MAX_POINTS`, `MAX_KNOBS`, `MAX_GRID` and
`MAX_CURVES` bound what a panel declares and what a request of a grid over its
knobs may ask for. The declaration of a panel carries them, so a renderer
knows the limits of the endpoint it talks to without a contract of its own.
"""

import re
from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "APP_CONTRACT",
    "CONTRACT",
    "MAX_CURVES",
    "MAX_END",
    "MAX_GRID",
    "MAX_KNOBS",
    "MAX_POINTS",
    "App",
    "AppPage",
    "AppVariant",
    "Choice",
    "ContractError",
    "Knob",
    "Panel",
    "PanelCurve",
    "PanelData",
    "PanelSimulation",
    "Scenario",
    "app_from_payload",
    "app_payload",
    "panel_from_payload",
    "panel_payload",
]

#: the version of the wire contract of a panel, in every payload
CONTRACT: Final = "panel/1"

#: the version of the wire contract of the app of a model, in every payload of
#: it, see `app_payload`
APP_CONTRACT: Final = "app/1"

#: the largest end of the simulation of a panel, in the time unit of its model
MAX_END: Final = 10000.0
#: the most output points of the simulation of a panel
MAX_POINTS: Final = 2000
#: the most knobs of a panel, which are the most a request may set or grid
#: over; the work of a request is bounded by `MAX_GRID` and `MAX_POINTS`, and a
#: knob which is not gridded is one value of the pre-initialization
MAX_KNOBS: Final = 8
#: the largest product of the sizes of the grid of a request
MAX_GRID: Final = 64
#: the most curves of a panel
MAX_CURVES: Final = 16

#: a panel and its parts forbid unknown fields and do not change
_CONFIG = ConfigDict(extra="forbid", frozen=True)

#: the shape of the dataset of a series of the study data: the id of a figure
#: or a table of a study, e.g. `Beermann1976_Fig2`, which names the study and
#: the file of the data of the project (`sbmlsim.data.load_pkdb_dataframe`).
#: It is no path: a separator, a dot and a drive are refused where a panel is
#: read, so a declaration can never name a file outside of that data and no
#: reader of a panel has to resolve a path to find out, see `PanelData`.
_DATASET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _has_text(value: str) -> str:
    """Check that a text of a declaration is not empty.

    An empty text is written into the annotation of a model as no attribute
    (`sbmlsim.panel_sbml`) and read back as the default of its field: a
    required text would then be missing and an optional one `None`, so the
    declaration would not survive its own SBML. A text which says nothing is
    therefore refused where it is declared, and an optional one which is not
    given is `None`.

    Args:
        value: the text.

    Returns:
        The text as it is.

    Raises:
        ValueError: for a text which is empty or only white space.
    """
    if not value.strip():
        raise ValueError("a text of a panel is not empty; leave out one which is none")
    return value


#: a text of a declaration which is not empty, see `_has_text`
type _Text = Annotated[str, AfterValidator(_has_text)]
#: a number of a declaration, which is finite: a bound, a default or a value
#: of infinity or `NaN` has no JSON and no point of a grid
type _Number = Annotated[float, Field(allow_inf_nan=False)]

#: the shape of the id of a page and of a variant of an app: an SBML `SId`,
#: a letter or `_` and then letters, digits and `_`. The id of a page is the
#: one a link of the app names, so it is no path, no separator and no dot.
_SID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ContractError(ValueError):
    """A payload which is of none of the contracts of this package.

    The contracts are `CONTRACT`, the one of a panel, and `APP_CONTRACT`, the
    one of the app of a model.
    """


class Choice(BaseModel):
    """One choice of a knob, e.g. a dose of a tablet.

    It is a choice of a knob of the kind `choice` and a named stop of one of
    the kind `range`, see `Knob.choices`.

    Attributes:
        label: the choice as it is shown, e.g. `"12.5 mg"`.
        value: the value the knob sets, in its unit.
    """

    model_config = _CONFIG

    label: _Text
    value: _Number


class Knob(BaseModel):
    """A knob of a panel: an entity of the model a reader sets, or an input of one.

    A knob of the kind `range` has the bounds `min` and `max` and a grid of
    `steps` points over them, one of the kind `choice` its `choices`; see
    `values`. A range knob may carry `choices` as well, its named stops: points
    of its range with a label, e.g. the stages of a renal impairment on a
    slider of the creatinine clearance, which a renderer offers as a select of
    stops next to the slider. They are no grid and no kind of their own: the
    grid of a range knob is its `steps` whatever its stops are.

    Attributes:
        id: the id of the knob, the key of its value in a request and the id
            of its axis in a grid.
        label: the knob as it is shown, e.g. `"renal function"`.
        target: the entity of the model the knob sets, a selection of the
            simulation, e.g. `"PODOSE_hctz"`; `None` for a knob which sets no
            entity itself and is an input of the formulas of the derived
            targets of the panel (`PanelSimulation.derived`), e.g. a
            creatinine clearance from which the renal function is computed.
            Such a knob is read by a formula, which the service checks when it
            starts (`sbmlsim.serve.runner.ModelRegistry.of`).
        unit: the unit of the values, the unit of the target without; a knob
            without a target and without a unit carries values of no unit.
        kind: `range` for a knob of bounds, `choice` for one of choices.
        min: the smallest value of a range knob.
        max: the largest value of a range knob.
        steps: the points of the grid of a range knob, at most `MAX_GRID`,
            see `values`.
        choices: the choices of a choice knob, in their order, or the named
            stops of a range knob, each inside its bounds.
        default: the value the knob starts at, inside the bounds of a range
            knob, which need not be a stop, and one of the values of a choice
            knob.
    """

    model_config = _CONFIG

    id: _Text
    label: _Text
    target: _Text | None = None
    unit: _Text | None = None
    kind: Literal["range", "choice"] = "range"
    min: _Number | None = None
    max: _Number | None = None
    steps: int = Field(default=5, ge=2, le=MAX_GRID)
    choices: tuple[Choice, ...] = ()
    default: _Number

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        """Check the knob against its kind.

        Raises:
            ValueError: for a range knob without its bounds, with bounds which
                are no interval, with a default outside them or with a stop
                outside them, and for a choice knob without choices, with more
                choices than `MAX_GRID`, the points a grid over it runs, or with
                a default which is no choice.
        """
        if self.kind == "range":
            if self.min is None or self.max is None:
                raise ValueError(
                    f"The range knob '{self.id}' needs the bounds 'min' and 'max'."
                )
            if self.min >= self.max:
                raise ValueError(
                    f"The range knob '{self.id}' needs min < max, not "
                    f"min={self.min} and max={self.max}."
                )
            if not self.min <= self.default <= self.max:
                raise ValueError(
                    f"The default {self.default} of the knob '{self.id}' is outside "
                    f"its bounds [{self.min}, {self.max}]."
                )
            outside = [
                choice.value
                for choice in self.choices
                if not self.min <= choice.value <= self.max
            ]
            if outside:
                raise ValueError(
                    f"The stops {outside} of the range knob '{self.id}' are outside "
                    f"its bounds [{self.min}, {self.max}]; a stop is a point of the "
                    f"range of its knob."
                )
        elif not self.choices:
            raise ValueError(f"The choice knob '{self.id}' needs its choices.")
        elif len(self.choices) > MAX_GRID:
            # a grid over a choice knob runs its choices, so they are the points
            # of a request, which `MAX_GRID` bounds: a knob of more could never run
            raise ValueError(
                f"The choice knob '{self.id}' has {len(self.choices)} choices, more "
                f"than the {MAX_GRID} points of a grid over it (`MAX_GRID`)."
            )
        elif self.default not in [choice.value for choice in self.choices]:
            raise ValueError(
                f"The default {self.default} of the knob '{self.id}' is none of its "
                f"choices {[choice.value for choice in self.choices]}."
            )
        return self

    def values(self, n: int | None = None) -> tuple[float, ...]:
        """The values of the knob which a grid over it runs.

        Args:
            n: the number of points of a range knob, its `steps` without; the
                values of a choice knob are its choices whatever `n` is.

        Returns:
            The points from `min` to `max`, both included, for a range knob,
            whatever its stops are, the values of the choices for a choice
            knob.

        Raises:
            ValueError: for fewer than one point, and for a range knob without
                bounds, which `_check_kind` refuses.
        """
        if self.kind == "choice":
            return tuple(choice.value for choice in self.choices)
        count = self.steps if n is None else n
        if count < 1:
            raise ValueError(
                f"A grid over the knob '{self.id}' needs a point, not {count}."
            )
        low, high = self.min, self.max
        if low is None or high is None:  # a range knob has both, see `_check_kind`
            raise ValueError(f"The range knob '{self.id}' has no bounds.")
        if count == 1:
            return (low,)
        # the weighted form keeps a decimal grid exact, where the affine one
        # does not: `0.2 + 2 * 0.2` is 0.6000000000000001, `(0.2 * 2 + 1.0 * 2)
        # / 4` is 0.6, and the ends are the bounds themselves
        last = count - 1
        inner = [(low * (last - i) + high * i) / last for i in range(1, last)]
        return (low, *inner, high)


class PanelCurve(BaseModel):
    """A curve of a panel: one output of the simulation over the time.

    Attributes:
        id: the id of the curve, a selection of the model, e.g. `"[Cve_hctz]"`,
            and the key of its values in a result.
        label: the curve as it is shown, its entry in the legend.
        unit: the unit of the values, the unit of the selection without.
        scale: the scale the value axis of the panel opens on, `linear` or
            `log`; the curves of a panel share that axis and declare one
            scale, see `Panel._check_scales`.
    """

    model_config = _CONFIG

    id: _Text
    label: _Text
    unit: _Text | None = None
    scale: Literal["linear", "log"] = "linear"


class PanelData(BaseModel):
    """The study data of a panel, drawn under its curves.

    The series is read from the data of the project the model belongs to, see
    `sbmlsim.serve.data`.

    Attributes:
        id: the id of the series, the key of its values in a result; the ids
            of the series of a panel are distinct, also of two series of one
            curve, e.g. of two studies.
        label: the series as it is shown, e.g. the study it comes from.
        curve: the curve of the panel the series belongs under, whose unit it
            is delivered in; its own `id` without, see `curve_id`. A curve
            which is set is a curve the panel declares, see
            `Panel._check_curves`.
        dataset: the dataset of the series, the id of a figure or a table of a
            study, e.g. `"Beermann1976_Fig2"`; it is no path, see `_DATASET`.
        x: the column of the times.
        y: the column of the values.
        yerr: the column of the error of the values, none without.
        sel: the rows of the dataset the series is, by the values of the
            columns which select them, e.g. `{"label": "hctz5po_4"}`; one
            dataset holds several series, and a series without a selection is
            every row of its dataset.
        at: the knob state the series was measured under, by the id of the
            knob and the value it takes, e.g. `{"impairment": 0.08,
            "dose": 50.0}`; a series without it is drawn whatever the knobs
            are set to, which is right for data of a condition the panel does
            not vary. A knob the mapping does not name is not constrained, and
            a knob it names is a knob of the panel whose choices carry the
            value: a choice knob, or a range knob whose named stops do, see
            `Panel._check_data_at`.
        conversion: a quantity the values and their error are multiplied by
            before they are delivered, e.g. `"1/(297.7 g/mole)"` for the mass
            of a study which the model carries as a substance; none without.
        unit: the unit of the values, the one of its curve without, and the
            one of the dataset for a series whose own `id` is no curve of the
            panel, see `Panel.series_unit`. A unit the series declares itself
            is the unit its values are delivered in as it is and is not
            checked against the unit of its curve, so a series may be drawn on
            a scale of its own.
    """

    model_config = _CONFIG

    id: _Text
    label: _Text
    curve: _Text | None = None
    dataset: str
    x: _Text
    y: _Text
    yerr: _Text | None = None
    sel: Mapping[str, str] | None = None
    at: Mapping[str, _Number] | None = None
    conversion: _Text | None = None
    unit: _Text | None = None

    @property
    def curve_id(self) -> str:
        """The id of the curve the series belongs under, its own `id` without.

        It is the one place the default of `curve` lives: a series which
        declares none belongs under the curve of its own id, so a panel of one
        series per curve names it once, and a panel of two studies under one
        curve names the curve in both of them.
        """
        return self.curve if self.curve is not None else self.id

    @model_validator(mode="after")
    def _check_dataset(self) -> Self:
        """Check that the dataset is the id of a dataset and no path.

        Raises:
            ValueError: naming the dataset and the series, for a dataset which
                is no id of the shape of `_DATASET`, e.g. a path, a name of a
                path separator or a name of a dot.
        """
        if _DATASET.fullmatch(self.dataset) is None:
            raise ValueError(
                f"The dataset '{self.dataset}' of the series '{self.id}' is no "
                f"dataset id: a dataset is the id of a figure or a table of a "
                f"study, e.g. 'Beermann1976_Fig2', of letters, digits, '_' and "
                f"'-', and never a path."
            )
        return self


class PanelSimulation(BaseModel):
    """The simulation a panel runs, the same for every point of its grid.

    Attributes:
        variant: the variant of the model, the default variant of the project
            without.
        end: the end of the simulation, in the time unit of the model; at most
            `MAX_END`.
        points: the output times of the simulation, at most `MAX_POINTS`.
        x_unit: the unit the time axis of a result is delivered in, e.g.
            `"hr"` for a model whose time unit is minutes; `None` delivers it
            in the time unit of the model. The `end` and the `points` are in
            the time unit of the model whatever it is.
        preinit: the values of the pre-initialization which are no knob, e.g.
            the body weight, by the entity of the model they set.
        derived: the targets of the model which are computed from the knobs,
            by the target (an SBML `SId`) and its formula, a formula of the
            math of PEtab over the ids of the knobs of the panel and numbers,
            e.g. `{"f_renal_function": "min(crcl / 110, 1)"}`. A formula sees
            the value of each knob as the number the knob carries, in the unit
            the knob declares and without a conversion, and its result is the
            value of the target in the unit of the model, so a formula states
            its conversion itself. A derived target is set by its formula
            alone: it is no target of a knob and no key of `preinit`. The
            formulas are compiled and evaluated by the service, which checks
            that they read knobs of the panel when it starts
            (`sbmlsim.serve.runner.ModelRegistry.of`), and a renderer never
            evaluates one: this module is pure data and has no parser.
    """

    model_config = _CONFIG

    variant: _Text | None = None
    end: float = Field(gt=0.0, le=MAX_END)
    points: int = Field(default=200, ge=2, le=MAX_POINTS)
    x_unit: _Text | None = None
    preinit: Mapping[str, _Number] = {}
    derived: Mapping[str, str] = {}

    @model_validator(mode="after")
    def _check_derived(self) -> Self:
        """Check the derived targets of the simulation.

        Raises:
            ValueError: naming the target, for a derived target which is no
                `SId`, for one which is a key of `preinit`, which would set it
                twice, and for a formula which is empty.
        """
        for target, formula in self.derived.items():
            if _SID.fullmatch(target) is None:
                raise ValueError(
                    f"The derived target '{target}' is no SId: a derived target is "
                    f"an entity of the model, a letter or '_' and then letters, "
                    f"digits and '_'."
                )
            if target in self.preinit:
                raise ValueError(
                    f"The derived target '{target}' is set by the pre-initialization "
                    f"of the simulation as well; a derived target is set by its "
                    f"formula alone."
                )
            if not formula.strip():
                raise ValueError(f"The derived target '{target}' needs its formula.")
        return self


class Panel(BaseModel):
    """The interactive panel of a model, see the module.

    Attributes:
        id: the id of the panel, the one a request names.
        title: the title of the panel.
        subtitle: the question the panel answers, shown below the title.
        x_label: the label of the time axis, e.g. `"time (hr)"`.
        y_label: the label of the value axis, e.g. `"plasma concentration"`;
            empty for a panel whose renderer names that axis after its curves,
            which is what every panel does today. The unit is no part of it: a
            renderer appends the unit its curves share.
        knobs: the knobs of the panel, in the order they are shown; they are
            the axes of the grid a request runs, so there are at most
            `MAX_KNOBS` of them, the knobs one request drives.
        curves: the curves of the panel, in the order they are drawn; at most
            `MAX_CURVES`. They share one value axis and declare one scale of
            it, see `_check_scales`.
        data: the study data of the panel, drawn under its curves; several
            series may belong under one curve, e.g. two studies, and their
            ids are distinct whether they do or not. A series which names a
            curve names one of `curves`, see `_check_curves`, and one which
            names the state it was measured under names choices of the knobs
            of the panel, see `_check_data_at`.
        simulation: the simulation the panel runs.
        y_min: the lower limit of the value axis, in the unit the curves of
            the panel are delivered in, which they then share; `None` leaves
            it to the renderer, which fits the axis to the curves. A fixed
            axis keeps two settings of the knobs comparable.
        y_max: the upper limit of the value axis, as `y_min`; both are
            positive on a panel whose curves declare the `log` scale.
        x_max: the upper limit of the time axis, in the unit it is delivered
            in (`PanelSimulation.x_unit`, the time unit of the model without),
            at most the end of the simulation; `None` is the end.
    """

    model_config = _CONFIG

    id: _Text
    title: _Text
    subtitle: str = ""
    x_label: _Text = "time"
    y_label: str = ""
    y_min: float | None = Field(default=None, allow_inf_nan=False)
    y_max: float | None = Field(default=None, allow_inf_nan=False)
    x_max: float | None = Field(default=None, allow_inf_nan=False)
    knobs: tuple[Knob, ...] = Field(default=(), max_length=MAX_KNOBS)
    curves: tuple[PanelCurve, ...] = Field(max_length=MAX_CURVES)
    data: tuple[PanelData, ...] = ()
    simulation: PanelSimulation

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        """Check that the knobs, the curves and the data have distinct ids.

        Raises:
            ValueError: naming the ids which are declared twice.
        """
        parts: tuple[tuple[str, list[str]], ...] = (
            ("knobs", [knob.id for knob in self.knobs]),
            ("curves", [curve.id for curve in self.curves]),
            ("data", [data.id for data in self.data]),
        )
        for kind, ids in parts:
            twice = sorted({id_ for id_ in ids if ids.count(id_) > 1})
            if twice:
                raise ValueError(
                    f"The {kind} of the panel '{self.id}' declare {twice} twice; "
                    "an id is the key of its values in a result."
                )
        return self

    @model_validator(mode="after")
    def _check_targets(self) -> Self:
        """Check that no derived target is the target of a knob.

        A derived target (`PanelSimulation.derived`) is set by its formula
        alone, so a knob which set it as well would be overwritten by the
        formula, or the formula by the knob, which is no declaration of the
        value the target takes.

        Raises:
            ValueError: naming the panel, the target and the knob, for a
                derived target which a knob of the panel sets.
        """
        for knob in self.knobs:
            if knob.target is not None and knob.target in self.simulation.derived:
                raise ValueError(
                    f"The derived target '{knob.target}' of the panel '{self.id}' is "
                    f"the target of its knob '{knob.id}' as well; a derived target "
                    f"is set by its formula alone."
                )
        return self

    @model_validator(mode="after")
    def _check_curves(self) -> Self:
        """Check that every series which names a curve names a declared one.

        The `curve` of a series is a reference to a curve of the panel, whose
        unit the series is delivered in (`series_unit`): a value which is no
        curve of the panel, e.g. a typo or a curve which was renamed, would be
        delivered in the unit of its own dataset under a curve of another unit,
        which is why it is refused here and not read as the default.

        A series which declares no `curve` belongs under the curve of its own
        `id` (`PanelData.curve_id`) and may have none, which is a series of a
        panel which draws no curve of it.

        Raises:
            ValueError: naming the series, the curve it names and the curves
                the panel declares, for a `curve` which is none of them.
        """
        declared = sorted(curve.id for curve in self.curves)
        for data in self.data:
            if data.curve is not None and data.curve not in declared:
                raise ValueError(
                    f"The series '{data.id}' of the panel '{self.id}' belongs "
                    f"under the curve '{data.curve}', which the panel does not "
                    f"declare; its curves are {declared}."
                )
        return self

    @model_validator(mode="after")
    def _check_scales(self) -> Self:
        """Check that the curves of the panel declare one scale.

        The curves of a panel share one value axis, which a renderer opens on
        the scale they declare (`PanelCurve.scale`): curves which declare
        different scales declare no scale of that axis at all, which is why
        they are refused here and not settled by the order of the curves.

        Raises:
            ValueError: naming the panel and the curves of every scale, for
                curves which declare more than one.
        """
        scales: dict[str, list[str]] = {}
        for curve in self.curves:
            scales.setdefault(curve.scale, []).append(curve.id)
        if len(scales) > 1:
            named = "; ".join(
                f"{scale}: {ids}" for scale, ids in sorted(scales.items())
            )
            raise ValueError(
                f"The curves of the panel '{self.id}' declare different scales "
                f"({named}); they share one value axis, so they declare one."
            )
        return self

    @model_validator(mode="after")
    def _check_limits(self) -> Self:
        """Check the limits of the axes of the panel.

        The units the limits are in are resolved where the model is, see
        `sbmlsim.serve.runner.PanelRunner.units`, which refuses y limits of
        curves of different units and an `x_max` past the end of the
        simulation.

        Raises:
            ValueError: naming the panel and the limits, for a `y_min` which
                is not below `y_max`, for a y limit which is not positive on a
                value axis of the `log` scale, for a y limit of a panel without
                curves, whose value axis has no unit a limit could be in, and
                for an `x_max` which is not positive.
        """
        if not self.curves and (self.y_min is not None or self.y_max is not None):
            raise ValueError(
                f"The panel '{self.id}' limits its value axis but draws no curve; "
                f"a limit of the value axis is in the unit its curves are "
                f"delivered in."
            )
        if (
            self.y_min is not None
            and self.y_max is not None
            and self.y_min >= self.y_max
        ):
            raise ValueError(
                f"The panel '{self.id}' needs y_min < y_max, not "
                f"y_min={self.y_min} and y_max={self.y_max}."
            )
        if any(curve.scale == "log" for curve in self.curves):
            limits = {"y_min": self.y_min, "y_max": self.y_max}
            for name, limit in limits.items():
                if limit is not None and limit <= 0.0:
                    raise ValueError(
                        f"The {name}={limit} of the panel '{self.id}' is not "
                        f"positive, which a value axis of the 'log' scale needs."
                    )
        if self.x_max is not None and self.x_max <= 0.0:
            raise ValueError(
                f"The x_max={self.x_max} of the panel '{self.id}' is not "
                f"positive; the time axis starts at 0."
            )
        return self

    @model_validator(mode="after")
    def _check_data_at(self) -> Self:
        """Check the knob state every series of the study data was measured at.

        The `at` of a series (`PanelData.at`) is the state of the knobs it was
        measured under, and a renderer draws the series only while the knobs
        are at it: a key which is no knob of the panel and a value which is no
        choice of its knob are therefore a series which is drawn nowhere,
        which is why they are refused here.

        A knob of the kind `range` without stops is no condition either: a
        condition is one of a set of conditions and not a point of a
        continuum, and a reader who slides a range knob past the one point of
        its grid which matches would see the series come and go. A panel which
        conditions on a continuous input declares that knob as a `choice` knob
        or gives the range knob named stops (`Knob.choices`), and the value of
        the condition is then one of its stops, which a reader selects.

        Raises:
            ValueError: naming the series, the knob and what the panel
                declares, for a key which is no knob of the panel, for a knob
                of the kind `range` without stops and for a value which is
                none of the choices or the stops of its knob.
        """
        knobs = {knob.id: knob for knob in self.knobs}
        for data in self.data:
            for knob_id, value in (data.at or {}).items():
                knob = knobs.get(knob_id)
                if knob is None:
                    raise ValueError(
                        f"The series '{data.id}' of the panel '{self.id}' was "
                        f"measured at the knob '{knob_id}', which the panel does "
                        f"not declare; its knobs are {sorted(knobs)}."
                    )
                if not knob.choices:
                    raise ValueError(
                        f"The series '{data.id}' of the panel '{self.id}' was "
                        f"measured at the knob '{knob_id}', which is a knob of the "
                        f"kind '{knob.kind}' without stops; a condition is one of "
                        f"a set of conditions, so a knob a series is measured at "
                        f"is a knob of the kind 'choice' or a range knob with "
                        f"named stops."
                    )
                choices = [choice.value for choice in knob.choices]
                if value not in choices:
                    named = "choices" if knob.kind == "choice" else "stops"
                    raise ValueError(
                        f"The series '{data.id}' of the panel '{self.id}' was "
                        f"measured at {value} of the knob '{knob_id}', which is "
                        f"none of its {named} {choices}."
                    )
        return self

    def curve_of(self, data: PanelData) -> PanelCurve | None:
        """Get the curve a series of the study data belongs under.

        Args:
            data: the declaration of the series, one of `data`.

        Returns:
            The curve of `PanelData.curve_id`, which is a curve of the panel
            whenever `PanelData.curve` is set (`_check_curves`), `None` for a
            series whose own `id` is no curve of the panel, which the panel
            gives no unit at all.
        """
        wanted = data.curve_id
        return next((curve for curve in self.curves if curve.id == wanted), None)

    def series_unit(self, data: PanelData) -> str | None:
        """Get the unit a series of the study data declares.

        It is the one place the declared unit of a series lives, so the study
        data of a panel and the curve it is drawn under are delivered in one
        unit, see `sbmlsim.serve.data`.

        Args:
            data: the declaration of the series, one of `data`.

        Returns:
            The unit the series declares, else the unit of the curve it belongs
            under (`curve_of`), `None` for a series whose curve declares no
            unit, which is the unit the model gives that curve, and for one
            whose own `id` is no curve of the panel, which is delivered in the
            unit of its dataset.
        """
        if data.unit is not None:
            return data.unit
        curve = self.curve_of(data)
        return curve.unit if curve is not None else None


def panel_payload(panel: Panel) -> dict[str, Any]:
    """Get the `panel/1` payload of a panel, the shape a renderer sees.

    Args:
        panel: the panel.

    Returns:
        The payload: its `contract`, the texts and the limits of the axes of
        the `panel`, its knobs as the `axes` of the grid, its `curves`, its
        `data`, its `simulation` and the `limits` of a request. Plain JSON
        values only, no object of this package, see `panel_from_payload`.
    """
    return {
        "contract": CONTRACT,
        "panel": {
            "id": panel.id,
            "title": panel.title,
            "subtitle": panel.subtitle,
            "x_label": panel.x_label,
            "y_label": panel.y_label,
            "y_min": panel.y_min,
            "y_max": panel.y_max,
            "x_max": panel.x_max,
        },
        "axes": [knob.model_dump(mode="json") for knob in panel.knobs],
        "curves": [curve.model_dump(mode="json") for curve in panel.curves],
        "data": [data.model_dump(mode="json") for data in panel.data],
        "simulation": panel.simulation.model_dump(mode="json"),
        "limits": {
            "end": MAX_END,
            "points": MAX_POINTS,
            "knobs": MAX_KNOBS,
            "grid": MAX_GRID,
            "curves": MAX_CURVES,
        },
    }


def panel_from_payload(payload: Mapping[str, Any]) -> Panel:
    """Read the panel of a `panel/1` payload, the inverse of `panel_payload`.

    The `limits` of a payload are the ones of the package which wrote it and
    are not read back.

    Args:
        payload: the payload.

    Returns:
        The panel.

    Raises:
        ContractError: for a payload of another contract version.
        pydantic.ValidationError: for a payload which declares no panel,
            naming the field which is wrong or unknown.
    """
    contract = payload.get("contract")
    if contract != CONTRACT:
        raise ContractError(
            f"The payload is of the contract {contract!r} and not of {CONTRACT!r}."
        )
    return Panel.model_validate(
        {
            **dict(payload.get("panel") or {}),
            "knobs": list(payload.get("axes") or ()),
            "curves": list(payload.get("curves") or ()),
            "data": list(payload.get("data") or ()),
            "simulation": dict(payload.get("simulation") or {}),
        }
    )


class Scenario(BaseModel):
    """A named state of the knobs of a page of an app, e.g. an example patient.

    A reader who picks a scenario sets every knob it names to its value at
    once, e.g. the creatinine clearance and the dose of a patient of a
    severe renal impairment; a knob it does not name keeps the value it has.
    The knobs a scenario names are knobs some panel of its page declares, and
    each value is a value of that knob in every panel which declares it,
    which is checked where the models of a service are registered
    (`sbmlsim.serve.runner.ModelRegistry.of`) and not here, since the panels
    are read from the SBML of the model.

    Attributes:
        id: the id of the scenario, an SBML `SId` (`_SID`), distinct within
            its page.
        label: the scenario as it is shown, e.g. `"severe impairment, 12.5
            mg"`; it is not empty.
        values: the value of every knob the scenario sets, by the id of the
            knob, in the unit of the knob; at least one.
    """

    model_config = _CONFIG

    id: str
    label: str
    values: Mapping[str, Annotated[float, Field(allow_inf_nan=False)]]

    @model_validator(mode="after")
    def _check_scenario(self) -> Self:
        """Check the id, the label and the values of the scenario.

        Raises:
            ValueError: naming the id, for an id which is no `SId`, for an
                empty label and for a scenario which sets no knob.
        """
        if _SID.fullmatch(self.id) is None:
            raise ValueError(
                f"The id '{self.id}' of a scenario of an app is no SId: the id of a "
                f"scenario is a letter or '_' and then letters, digits and '_', "
                f"e.g. 'healthy'."
            )
        if not self.label.strip():
            raise ValueError(f"The scenario '{self.id}' of an app needs its label.")
        if not self.values:
            raise ValueError(
                f"The scenario '{self.id}' of an app sets no knob; a scenario is "
                f"the values of the knobs it sets."
            )
        return self


class AppPage(BaseModel):
    """One page of the app of a model: a title and the panels it shows.

    Attributes:
        id: the id of the page, an SBML `SId` (`_SID`): the id a link of the
            app names, so it is no path and no text which has to be escaped.
        title: the page as it is shown, its entry in the navigation; it is not
            empty.
        panels: the ids of the panels of the page, in the order they are
            shown; at least one and every one once. A panel id is a panel the
            model declares, which is checked where the models of a service are
            registered (`sbmlsim.serve.runner.ModelRegistry.of`) and not here,
            see the module.
        scenarios: the named states of the knobs of the page a reader picks
            from, in the order they are shown, e.g. example patients; none
            without, and no two of one id. A knob a scenario names is one
            control of the page, see `Scenario`.
    """

    model_config = _CONFIG

    id: str
    title: str
    panels: tuple[str, ...]
    scenarios: tuple[Scenario, ...] = ()

    @model_validator(mode="after")
    def _check_page(self) -> Self:
        """Check the id, the title and the panels of the page.

        Raises:
            ValueError: naming the id, for an id which is no `SId`; naming the
                page, for an empty title and for a page without a panel;
                naming the ids, for a panel the page lists twice and for two
                scenarios of one id.
        """
        if _SID.fullmatch(self.id) is None:
            raise ValueError(
                f"The id '{self.id}' of a page of an app is no SId: the id of a "
                f"page is a letter or '_' and then letters, digits and '_', e.g. "
                f"'renal', and never a path."
            )
        if not self.title.strip():
            raise ValueError(f"The page '{self.id}' of an app needs its title.")
        if not self.panels:
            raise ValueError(f"The page '{self.id}' of an app needs a panel.")
        panels = list(self.panels)
        twice = sorted({panel for panel in panels if panels.count(panel) > 1})
        if twice:
            raise ValueError(
                f"The page '{self.id}' of an app shows the panels {twice} twice; "
                "a page shows a panel once."
            )
        ids = [scenario.id for scenario in self.scenarios]
        twice = sorted({id_ for id_ in ids if ids.count(id_) > 1})
        if twice:
            raise ValueError(
                f"The page '{self.id}' of an app declares the scenarios {twice} "
                "twice; a reader picks a scenario of a page by its id."
            )
        return self


class AppVariant(BaseModel):
    """One variant of the model a reader of the app switches to.

    Attributes:
        id: the id of the variant, a variant of the project
            (`sbmlsim.project.Project.variants`), which is checked where the
            models of a service are registered
            (`sbmlsim.serve.runner.ModelRegistry.of`) and not here, see the
            module; an SBML `SId` (`_SID`).
        label: the variant as it is shown, e.g. `"renal impairment"`; it is
            not empty.
    """

    model_config = _CONFIG

    id: str
    label: str

    @model_validator(mode="after")
    def _check_variant(self) -> Self:
        """Check the id and the label of the variant.

        Raises:
            ValueError: naming the id, for an id which is no `SId`, and for an
                empty label.
        """
        if _SID.fullmatch(self.id) is None:
            raise ValueError(
                f"The id '{self.id}' of a variant of an app is no SId: the id of a "
                f"variant is a letter or '_' and then letters, digits and '_', "
                f"e.g. 'renal', and never a path."
            )
        if not self.label.strip():
            raise ValueError(f"The variant '{self.id}' of an app needs its label.")
        return self


class App(BaseModel):
    """The app of a model: the pages of its panels, see the module.

    Attributes:
        title: the title of the app, the model as it is shown; it is not empty.
        subtitle: what the model is, shown below the title.
        pages: the pages of the app, in the order they are shown; at least
            one, and no two of them share an id.
        variants: the variants of the model a reader switches between, in the
            order they are shown; none means the default variant of the
            project alone.
        citation: the citation of the model, shown in the provenance footer of
            a page; free text, e.g. `"König et al. (2026)"`.
        doi: the DOI of the model, shown in that footer; free text.
        licenses: the licenses of the model and its data, shown in that
            footer; free text, e.g. `("CC BY-SA 4.0",)`.
    """

    model_config = _CONFIG

    title: str
    subtitle: str = ""
    pages: tuple[AppPage, ...]
    variants: tuple[AppVariant, ...] = ()
    citation: str = ""
    doi: str = ""
    licenses: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_app(self) -> Self:
        """Check the title, the pages and the variants of the app.

        Raises:
            ValueError: for an empty title and for an app without a page;
                naming the ids, for two pages or two variants of one id.
        """
        if not self.title.strip():
            raise ValueError("An app needs its title.")
        if not self.pages:
            raise ValueError("An app needs a page, which is a list of its panels.")
        parts: tuple[tuple[str, list[str]], ...] = (
            ("pages", [page.id for page in self.pages]),
            ("variants", [variant.id for variant in self.variants]),
        )
        for kind, ids in parts:
            twice = sorted({id_ for id_ in ids if ids.count(id_) > 1})
            if twice:
                raise ValueError(
                    f"The {kind} of the app '{self.title}' declare {twice} twice; "
                    "an id is the one a reader of the app names."
                )
        return self


def app_payload(app: App) -> dict[str, Any]:
    """Get the `app/1` payload of the app of a model, the shape a shell sees.

    Args:
        app: the app.

    Returns:
        The payload: its `contract`, the texts of the app, its `pages` with
        their scenarios, its `variants` and the `citation`, the `doi` and the
        `licenses` of the provenance footer. Plain JSON values only, no object
        of this package, see `app_from_payload`.
    """
    return {
        "contract": APP_CONTRACT,
        "title": app.title,
        "subtitle": app.subtitle,
        "pages": [page.model_dump(mode="json") for page in app.pages],
        "variants": [variant.model_dump(mode="json") for variant in app.variants],
        "citation": app.citation,
        "doi": app.doi,
        "licenses": list(app.licenses),
    }


def app_from_payload(payload: Mapping[str, Any]) -> App:
    """Read the app of an `app/1` payload, the inverse of `app_payload`.

    Args:
        payload: the payload.

    Returns:
        The app.

    Raises:
        ContractError: for a payload of another contract version.
        pydantic.ValidationError: for a payload which declares no app, naming
            the field which is wrong or unknown.
    """
    contract = payload.get("contract")
    if contract != APP_CONTRACT:
        raise ContractError(
            f"The payload is of the contract {contract!r} and not of {APP_CONTRACT!r}."
        )
    return App.model_validate(
        {key: value for key, value in payload.items() if key != "contract"}
    )
