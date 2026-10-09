# Units

Every SBML model declares units for its parameters, species and compartments. `sbmlsim` reads them and uses [pint](https://pint.readthedocs.io) quantities for changes and results, so a dose given in milligram is converted to the substance unit of the model instead of being assumed to match.

## The unit registry and `Q`

`sbmlsim` has one unit registry, `sbmlsim.units.ureg`, which every model, simulation, dataset, experiment and fit uses, so quantities from different sources can be combined. `Q` is its quantity:

```python
from sbmlsim import Q

dose = Q(10, "mmole")
print(dose, dose.to("mole"))
```

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
