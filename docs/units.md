# Units

Every SBML model declares units for its parameters, species and compartments. `sbmlsim` reads them and uses [pint](https://pint.readthedocs.io) quantities for changes and results, so a dose given in milligram is converted to the substance unit of the model instead of being assumed to match.

## Units of a model

The units are read into a `UnitsInformation`, a mapping from identifier to unit string with the unit registry of the model:

```python
from sbmlsim.resources import DEMO_SBML
from sbmlsim.units import UnitsInformation

uinfo = UnitsInformation.from_sbml(DEMO_SBML)
print(uinfo["Vmax_bA"])
print(uinfo["e__A"])  # amount of the species
print(uinfo["[e__A]"])  # concentration of the species
```

Every `RoadrunnerSBMLModel` and every `SimulatorSerial` carry the units of their model as `uinfo`, and `Q_` is the quantity constructor of the registry:

```python
from sbmlsim.simulator import SimulatorSerial

simulator = SimulatorSerial(model=DEMO_SBML)
Q_ = simulator.Q_
dose = Q_(10, "mmole")
print(dose, dose.to("mole"))
```

## Changes with units

Changes of a `Timecourse` or a `Dimension` are quantities; they are converted to the units of the model before the simulation. A float without a unit is taken in the units of the model:

```python
import numpy as np

from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim

scan = ScanSim(
    simulation=TimecourseSim(
        Timecourse(
            start=0,
            end=10,
            steps=100,
            changes={
                "[e__A]": Q_(10, "mM"),
                "[e__B]": Q_(1, "mmole/litre"),
                "[e__C]": Q_(1, "mole/m**3"),
                "c__A": Q_(1e-5, "mole"),
                "c__B": Q_(10, "µmole"),
                "Vmax_bA": Q_(300.0, "mole/min"),
            },
        )
    ),
    dimensions=[
        Dimension("dim1", changes={"[e__A]": Q_(np.linspace(5, 15, num=5), "mM")}),
    ],
)
xres = simulator.run_scan(scan)
print(xres["[e__A]"].values[0])
```

A change with a unit which cannot be converted to the model unit raises a `DimensionalityError`, which is the point: a dose in `mg` for a parameter in `mmole` is a mistake the units catch.

`UnitsInformation.normalize_changes` performs the conversion; the `normalize` methods of `Timecourse`, `TimecourseSim` and `ScanSim` call it before a simulation.

## Results with units

The result of a simulation knows the units of its variables, so reductions return quantities:

```python
print(xres.uinfo["[e__A]"])
mean = xres.dim_mean("[e__A]")
print(mean.units)
print(mean.to("mole/litre").magnitude[:3])
```

Data in a `Data` object or a `DataSet` are converted into requested units the same way, see [Data](data.md); the axes of a plot declare their unit and the curves are converted to it, see [Plots and reports](plotting.md).

## The unit registry

All quantities of a model share one `UnitRegistry`. A `SimulationExperiment` creates a single registry which its models, datasets and results share, so quantities from different sources can be combined. The registry is extended with the units SBML uses, e.g., `mmole` for millimole, in `UnitsInformation._default_ureg`.
