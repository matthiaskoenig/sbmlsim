"""The battery hook of the toy credibility package: a dose of A.

The battery simulates the toy model from no A, at its steady state, with a dose
of A at 0, which the second engines repeat, and a late dose, which they do
not; the check `decay` of the hook checks that every dose decays, and its
section `doses` records the amount at the end of every case.
"""

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, override

import pandas as pd

from sbmlsim.credibility import (
    BatteryCase,
    BatteryCollector,
    BatteryContribution,
    BatteryHook,
    BatteryRun,
    BatterySettings,
    Check,
    register_check,
)

#: the check of the hook: every dose decays
DECAY = register_check(Check("decay", "decay of a dose", "numerical_code_verification"))


class DecayBattery(BatteryHook):
    """The doses of A, see the module."""

    name: ClassVar[str] = "decay"
    checks: ClassVar[tuple[Check, ...]] = (DECAY,)
    order: ClassVar[tuple[str, ...]] = (
        "decay",
        "non_negativity",
        "steady_state",
        "cross_simulator",
        "plausibility",
        "rebuild",
    )

    @override
    def settings(self) -> BatterySettings:
        return BatterySettings(horizon=60.0, steps=60)

    @override
    def read(self, variant: str, path: Path) -> object:
        return path.read_text(encoding="utf-8")

    @override
    def sbml(self, data: object) -> str | None:
        import libsbml

        doc = libsbml.readSBMLFromString(str(data))
        species = doc.getModel().getSpecies("A")
        if species.isSetInitialAmount():
            species.setInitialAmount(0.0)
        else:
            species.setInitialConcentration(0.0)
        return libsbml.writeSBMLToString(doc)

    @override
    def cases(self, data: object) -> Sequence[BatteryCase]:
        return (
            BatteryCase(name="dose", changes={0.0: {"A": 1.0}}, added=frozenset({"A"})),
            BatteryCase(
                name="late dose",
                changes={30.0: {"A": 0.5}},
                added=frozenset({"A"}),
                cross_simulator=False,
            ),
        )

    @override
    def contribute(self, runs: Sequence[BatteryRun]) -> BatteryContribution:
        collector = BatteryCollector()
        doses: list[dict[str, Any]] = []
        for run in runs:
            for name, result in run.cases.items():
                values = run.column(result, "A")
                collector.cases += 1
                collector.deviation(float(values[-1] / values.max()), name)
                if not values[-1] < values.max():
                    collector.problems.append(f"{name}: A does not decay")
                doses.append(
                    {"variant": run.build.variant, "case": name, "end": values[-1]}
                )
        return BatteryContribution(
            results=(collector.result(DECAY, "Every dose of A decays."),),
            sections={"doses": doses},
        )

    @override
    def describe(self) -> str:
        return "A dose of 1 of A at 0 and of 0.5 at 30."

    @override
    def report_html(self, sections: Mapping[str, Any]) -> str:
        rows = "".join(
            f"<tr><td>{html.escape(d['case'])}</td><td>{d['end']:.3g}</td></tr>"
            for d in sections.get("doses", ())
        )
        return f'<h3>Doses</h3><table class="decay-doses">{rows}</table>'

    @override
    def report_tables(self, sections: Mapping[str, Any]) -> Mapping[str, pd.DataFrame]:
        return {
            "doses": pd.DataFrame(
                list(sections.get("doses", ())), columns=["variant", "case", "end"]
            )
        }


BATTERY = DecayBattery()
