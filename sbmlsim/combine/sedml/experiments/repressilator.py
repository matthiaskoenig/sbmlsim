from typing import Dict, List

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.data import Data, DataSet
from sbmlsim.timecourse import Timecourse, TimecourseSim
from sbmlsim.plotting import Figure, Axis


class RepressilatorExperiment(SimulationExperiment):
    @property
    def datasets(self) -> Dict[str, DataSet]:
        d1 = self.load_data(f"{self.sid}_Fig1")
        d1_udict = {key: d1[f"{key}_unit"].unique()[0] for key in
                    ["time", "cpep"]}
        return {
            "fig1": DataSet.from_df(d1, udict=d1_udict, ureg=self.ureg),
        }

    @property
    def simulations(self) -> Dict[str, TimecourseSim]:
        return {
            **self.simulation_cpep(),
        }

    def simulation_cpep(self) -> Dict[str, TimecourseSim]:
        """ Faber1978_Fig1

        - C-peptide, iv, constant infusion, 0.64 [nmol/min], start - end: 60 - 120 min, solution
        - C-peptide, iv, constant infusion, 1.19 [nmol/min], start - end: 120 - 180 min, solution
        - C-peptide, iv, constant infusion, 2.39 [nmol/min], start - end: 180 - 240 min, solution
        """
        Q_ = self.ureg.Quantity

        cpep_inf1 = Q_(0.64, 'nmol/min') * Q_(Mr_cpep, 'g/mol')
        cpep_inf2 = Q_(1.19, 'nmol/min') * Q_(Mr_cpep, 'g/mol')
        cpep_inf3 = Q_(2.39, 'nmol/min') * Q_(Mr_cpep, 'g/mol')

        # FIXME: add glucose clamp ?
        tcsim = TimecourseSim([
            Timecourse(start=0, end=60, steps=120, changes=default_init_conditions(self)),
            Timecourse(start=0, end=60, steps=120,
                       changes={'Ri_cpep': cpep_inf1}
                       ),
            Timecourse(start=0, end=60, steps=120,
                       changes={'Ri_cpep': cpep_inf2}
                       ),
            Timecourse(start=0, end=60, steps=120,
                       changes={'Ri_cpep': cpep_inf3}
                       ),
            Timecourse(start=0, end=120, steps=240,
                       changes={'Ri_cpep': Q_(0, "ng/min")}
                       ),
        ], reset=True, time_offset=0)
        return {'cpep': tcsim}

    @property
    def figures(self) -> Dict[str, Figure]:
        unit_time = "min"
        unit_cpep = "pmol/ml"

        fig = Figure(experiment=self,
                     sid="Fig1", num_cols=1, num_rows=1)
        plots = fig.create_plots(
            xaxis=Axis("time", unit=unit_time),
            yaxis=Axis("c-peptide", unit=unit_cpep),
            legend=True)
        plots[0].set_title(f"{self.sid}_{fig.sid}")
        plots[0].curve(
            x=Data(self, "time", dataset="fig1", unit=unit_time),
            y=Data(self, "cpep", dataset="fig1", unit=unit_cpep),
            label="cpeptide (n=1)", color="black"
        )
        plots[0].curve(
            x=Data(self, "time", simulation="cpep", unit=unit_time),
            y=Data(self, "Cve_cpep", simulation="cpep", unit=unit_cpep),
            label="sim cpeptide", color="black"
        )
        return {fig.sid: fig}
