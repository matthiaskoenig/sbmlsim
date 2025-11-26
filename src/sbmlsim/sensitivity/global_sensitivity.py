"""Global sensitivity analysis.

TODO:
- [ ] multi-output model
- [ ] calculation of pharmacokinetic parameters (improved PK calculation)
- [ ] get all parameter ids from model
- [ ] get defined parameter bounds from model (annotate information);
- [ ] storage of results
- [ ] visualization of results
- [ ] alternative methods:
    - [ ] local sensitivity analysis
    - [ ] SOBOL indices Sobol Sensitivity Analysis (Sobol 2001, Saltelli 2002, Saltelli et al. 2010)
          http://www.sciencedirect.com/science/article/pii/S0378475400002706
          https://www.sciencedirect.com/science/article/pii/S0010465502002801
          https://www.sciencedirect.com/science/article/pii/S0010465509003087
    - [ ] FAST
    - [ ] Morris
    - [ ] Sampling based methods (distribution)
- [ ] report as PDF with references and description (Typst)
- [ ] parallelization ? (benchmark)

"""
import SALib
from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

import numpy as np
import roadrunner
from pathlib import Path
from dataclasses import dataclass

from pkdb_analysis.pk.pharmacokinetics import TimecoursePK
from sbmlutils.console import console
from roadrunner._roadrunner import NamedArray
from pint import UnitRegistry

@dataclass
class SBMLSensitivityAnalysis:
    """Parent class for sensitivity analysis."""
    model_path: Path
    selections: list[str]
    rr: roadrunner.RoadRunner = None

    def __init__(self, model_path: Path, selections: list[str]):
        self.model_path = model_path
        self.selections = selections
        self.rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        self.rr.selections = self.selections
        integrator: roadrunner.Integrator = self.rr.integrator
        integrator.setSetting("variable_step_size", True)
        # state = rr.saveStateS()


    def losartan_simulation(self, changes: dict[str, float]) -> dict[str, float]:
        """Run a given model simulation and create scalar readouts.

        Applies the changes and gets the readouts. This is the general structure of the abstract
        run simulation. This can afterwards be parallelized.

        This must be a small subset of data
        """

        # rr = roadrunner.RoadRunner()
        # rr.loadStateS(state)
        self.rr.resetAll()
        all_changes = {
            "PODOSE_los": 10.0,  # [mg]
            **changes
        }
        for key, value in all_changes.items():
            # print(f"{key=} {value=}")
            self.rr.setValue(key, value)

        s: NamedArray = self.rr.simulate(start=0, end=5 * 24 * 60)
        console.print(type(s))
        # self.plot_simulation(s)

        y: dict[str, float] = self.losartan_pkpd_parameters(s)
        return y



    def losartan_pkpd_parameters(self, s: NamedArray) -> dict[str, float]:
        """This calculation is highly model dependent.

        This requires units.
        """

        y: dict[str, float] = {}

        # pharmacokinetics
        ureg = UnitRegistry()
        Q_ = ureg.Quantity

        # losartan
        time = Q_(s["time"], "min")
        tcpk = TimecoursePK(
            time=time,
            concentration=Q_(s["[Cve_los]"], "mM"),
            substance="losartan",
            ureg=ureg,
            dose=Q_(10, "mg"),
        )
        pk_dict = tcpk.pk.to_dict()
        # console.print(pk_dict)
        for pk_key in [
            "aucinf",
            "cmax",
            "thalf",
            "kel",
        ]:
            y[f"[Cve_los]_{pk_key}"] = pk_dict[pk_key]

        # E3174, L158
        for sid in ["[Cve_e3174]", "[Cve_l158]"]:
            tcpk = TimecoursePK(
                time=time,
                concentration=Q_(s[sid], "mM"),
                substance="losartan",
                ureg=ureg,
                dose=None,
            )
            pk_dict = tcpk.pk.to_dict()
            # console.print(pk_dict)
            for pk_key in [
                "aucinf",
                "cmax",
                "thalf",
                "kel",
            ]:
                y[f"{sid}_{pk_key}"] = pk_dict[pk_key]

        # pharmacodynamics
        for sid, f in [
            ("[ang1]", "max"),
            ("[ang2]", "max"),
            ("[ren]", "max"),
            ("[ald]", "min"),
            ("SBP", "min"),
            ("DBP", "min"),
            ("MAP", "min"),
        ]:
            # minimal and maximal value of readout
            if f == "max":
                y[f"{sid}_max"] = np.min(s[sid])
            elif f == "min":
                y[f"{sid}_min"] = np.max(s[sid])

        return y

    def _plot_simulation(self, s: NamedArray) -> None:

        # plotting
        from matplotlib import pyplot as plt
        plt.plot(
            s["time"], s["[Cve_los]"],
            # df["time"], df["[Cve_los]"],
            marker="o",
            markeredgecolor="black",
            color="tab:blue",
        )
        plt.show()


    def wrapped_run_simulation(self, X, func=losartan_simulation):
        # We transpose to obtain each column (the model factors) as separate variables
        changes: dict[str, float] = {}
        for k, key in enumerate(self.names):
            changes[key] = X[k]

        # Then call the original model
        return list(func(self, changes).values())


    def calculate_sensitivity(self):

        y = self.losartan_simulation(changes={})
        self.outputs = list(y.keys())
        self.names = ['BW']

        # Defining the model inputs
        sp = ProblemSpec({
            'num_vars': len(self.names),
            'names': self.names,
            'bounds': [
                [50, 150],
                # [0.003, 0.005]
            ],
            "outputs": self.outputs,
        })

        # Generate samples
        samples = saltelli.sample(sp, 1024)
        sp.set_samples(samples)


        # Evaluate model
        # sp.evaluate(wrapped_run_simulation)

        Y = np.zeros((samples.shape[0], len(self.outputs)))
        for k, X in enumerate(samples):
             print(k)
             Y[k, :] = self.wrapped_run_simulation(X)
        sp.set_results(Y)


        # Perform Analysis
        Si = sp.analyze(SALib.analyze.sobol)
        print(Si['S1'])
        print(Si['ST'])
        total_Si, first_Si, second_Si = Si.to_df()
        Si.plot()
        from matplotlib import pyplot as plt
        plt.show()




if __name__ == '__main__':
    model_path = Path(__file__).parent / "models" / "losartan" / "losartan_body_flat.xml"

    sa = SBMLSensitivityAnalysis(
        model_path=model_path,
        selections=[
            "time",
            "[Cve_los]",
            "[Cve_e3174]",
            "[Cve_l158]",
            "[ang1]",
            "[ang2]",
            "[ren]",
            "[ald]",
            "SBP",
            "DBP",
            "MAP",
        ]
    )
    y = sa.losartan_simulation(changes={})
    console.print(y)

    # y = run_simulation()
    # print(y)
    sa.calculate_sensitivity()
