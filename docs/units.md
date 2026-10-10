# Units

Every SBML model declares units for its parameters, species and compartments. `sbmlsim` reads them and uses [pint](https://pint.readthedocs.io) quantities for changes and results, so a dose given in milligram is converted to the substance unit of the model instead of being assumed to match.

## The unit registry and `Q`

`sbmlsim` has one unit registry, `sbmlsim.units.ureg`, which every model, simulation, dataset, experiment and fit uses, so quantities from different sources can be combined. `Q` is its quantity, which takes the units of pint and not the unit ids of a model (see [Unit ids of a model](#unit-ids-of-a-model)):

```python
from sbmlsim import Q

dose = Q(10, "mmole")
print(dose, dose.to("mole"))
```

Besides the units of pint the registry defines `none` and `item` as counts and `percent`. It defines no international unit (`IU`): an international unit is specific to a substance, an `IU` of insulin is no `IU` of the anti-Xa activity of heparin, so one definition for every substance would convert data of one substance with the factor of another. A model which needs an international unit defines it as a unit definition of its own, which the quantities of the model and of an experiment resolve (see [Unit ids of a model](#unit-ids-of-a-model)), and a user defines it in the registry under a name of the substance or as a count:

```python
from sbmlsim.units import ureg

ureg.define("IU_insulin = 0.0347 * mg")
print(Q(10, "IU_insulin/ml").to("mg/ml"))
```

A unit string with an `IU` which nothing defines raises, as every unit which cannot be read: `Q` raises the `UndefinedUnitError` of pint, and `DataSet.from_df` raises the `ValueError` which names the column when the data is imported; for an undefined `IU`, `µIU`, `mIU` or `IU_per_ml` both say how to define an international unit. Every unit of a `DataSet` column is read on import, also the one unit of a column whose rows all have it, so data in a unit the registry does not define, e.g. `IU/l` or `µIU/ml`, fails on import and not when it is used; define the unit in the registry before the data is read.

## Units of a model

The units are read into a `UnitsInformation`, a mapping from identifier to unit:

```python
from sbmlsim.resources import DEMO_SBML
from sbmlsim.units import UnitsInformation

uinfo = UnitsInformation.from_sbml(DEMO_SBML)
print(uinfo["Vmax_bA"])
print(uinfo["e__A"])  # amount of the species
print(uinfo["[e__A]"])  # concentration of the species
```

A unit of a model is the expression of its unit definition in the units of pint, with the prefixes and names pint knows, e.g. `mmol/min` for a millimole per minute; the ids of the unit definitions of a model are not defined in the registry, so two models which define one id differently keep their own units. A unit whose factor is no prefix, e.g. the `133.322 N/m^2` of a millimeter of mercury, is defined in the registry under its id, or under `<id>_<n>` if another model defines the id differently. Every `RoadrunnerSBMLModel` carries the units of its model as `uinfo`, and every `ScanResult` the units of its variables and coordinates as `units`.

## Unit ids of a model

`Q` does not know the ids of the unit definitions of a model: `Q(0, "mg_per_min")` raises the `UndefinedUnitError` of pint, with a note which names the quantities which know them. The units of a model keep its unit ids with their expressions, `uinfo.unit_ids`, and `uinfo.quantity(value, unit)` replaces every identifier of a unit which is a unit id of the model by its expression in brackets before pint reads it, so a unit id combines with the units of pint, e.g. `mg_per_min/kg`. Only whole identifiers are replaced, `mg` is no unit id of `mg_per_min`, and a unit id wins over a unit of pint of the same name, since the model defines what its id means. `RoadrunnerSBMLModel.Q(value, unit)` is the quantity of a model:

```python
from sbmlsim.model import RoadrunnerSBMLModel

model = RoadrunnerSBMLModel(source=DEMO_SBML)
print(model.uinfo.unit_ids["mole_per_s"])
print(model.Q(0.5, "mole_per_s").to("mmole/min"))
print(model.Q(0.5, "mole_per_s/m3").to("mM/s"))
```

In an experiment `self.Q(value, unit, model=None)` resolves the unit ids of the models of the experiment. The runner loads the models before `initialize`, so it works in `simulations()` and later:

```python
from pathlib import Path
from typing import override

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation


class Dosing(SimulationExperiment):
    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"demo": DEMO_SBML}

    @override
    def simulations(self) -> dict[str, Simulation]:
        # `mole_per_s` is a unit id of the model
        vmax = self.Q(300, "mole_per_s")
        return {"vmax": Simulation(end=10, preinit_changes={"Vmax_bA": vmax})}
```

With `model=` a unit id is the one of that model; without, a unit id must have the same expression in every model of the experiment which has it, else a `ValueError` names the models, their expressions and `model=`. Before the models are loaded `self.Q` takes the units of pint, and a unit id raises an `UndefinedUnitError` whose note says that the models are not loaded.

The unit of a `FitParameter` may have the unit ids of the model of its target as well, e.g. `unit="per_min"`: the fit converts the value with the units of that model, and the versions of a target agree on their unit when their units are one unit, e.g. `mg_per_min` of the model and `mg/min`. The parameter keeps the unit it was defined with, which the tables of a fit show and a PEtab export writes, while the values of a fit, i.e. its parameter sets, its result, the Fisher information and the profiles, carry the unit of pint (`OptimizationProblem.parameter_units`, `1/min` for `per_min`), so the designs of a fit (`fit_parameters`, `profile_parameters`, `fit_repeats`, see [Sampling](sampling.md)) draw quantities without the model.

## Changes with units

The values of a `Simulation`, its `Change`s and the `Dimension`s of a scan are quantities or numbers; a quantity is converted into the unit of its target in the model when the simulation is compiled, and a number is taken in the unit of the model:

```python
import numpy as np

from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
scan = Scan(
    simulation=Simulation(
        end=10,
        steps=100,
        preinit_changes={
            "[e__A]": Q(10, "mM"),
            "[e__B]": Q(1, "mmole/litre"),
            "[e__C]": Q(1, "mole/m**3"),
            "c__A": Q(1e-5, "mole"),
            "c__B": Q(10, "µmole"),
            "Vmax_bA": Q(300.0, "mole/min"),
        },
    ),
    dimensions=[
        Dimension("dim1", values={"[e__A]": Q(np.linspace(5, 15, num=5), "mM")}),
    ],
)
res = simulator.run(DEMO_SBML, scan)
print(res["[e__A]"].values[:, 0])
```

A change with a unit which cannot be converted into the unit of the model raises a `ValueError` which names the target and both units, which is the point: a dose in `mg` for a parameter in `mmole` is a mistake the units catch. The times of a simulation are converted the same way, from its `time_unit` into the time unit of the model, see [Simulations](simulation.md#units-and-times).

## Results with units

The result of a simulation knows the units of its variables, so its values and their statistics are quantities:

```python
print(res.units["[e__A]"])
mean = res.summary("dim1", statistics=["mean"]).quantity("[e__A]")
print(mean.units)
print(mean.to("mole/litre").magnitude[0, :3])
```

Data in a `Data` object or a `DataSet` are converted into requested units the same way, see [Data](data.md); the axes of a plot declare their unit and the curves are converted to it, see [Plots and reports](plotting.md).
