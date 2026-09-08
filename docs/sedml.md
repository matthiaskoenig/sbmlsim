# SED-ML and COMBINE archives

The [Simulation Experiment Description Markup Language](https://sed-ml.org) (SED-ML) describes simulation experiments in a tool independent way: the models with their changes, the simulations and tasks, the data generators computed from the results and the plots and reports. A [COMBINE archive](https://co.mbine.org/standards/omex) (OMEX) packages a SED-ML document with its models and data in one file. `sbmlsim.combine.sedml` reads SED-ML Level 1 Version 4 into a `SimulationExperiment` and executes it, and serializes an experiment to SED-ML, see [References](references.md#standards).

## Executing SED-ML

`execute_sedml` reads a SED-ML file or a COMBINE archive, builds the simulation experiment and runs it, writing the results and figures into the `output_path`. The `working_dir` is the directory the models are resolved against and, for an archive, the directory it is extracted to:

```python
from pathlib import Path

from sbmlsim.combine.sedml.runner import execute_sedml

# from the root of the repository, the paths have to be absolute
sedml_path = Path("examples/sedml/l1v4/algorithm_parameters.sedml").resolve()
output_path = Path("results").resolve() / "algorithm_parameters"
execute_sedml(path=sedml_path, working_dir=sedml_path.parent, output_path=output_path)
print(sorted(p.name for p in output_path.rglob("*") if p.is_file())[:5])
```

The example SED-ML files of Level 1 Version 4 with the corresponding reference figures are in `examples/sedml/l1v4/`, the SED-ML test cases used by the tests in `tests/data/sedml/` and `tests/data/combine/`.

## The SED-ML parser

`SEDMLReader` reads the document from a file, a string or an archive, `SEDMLParser` translates it into an experiment class:

```python
from sbmlsim.combine.sedml.io import SEDMLReader
from sbmlsim.combine.sedml.parser import SEDMLParser

reader = SEDMLReader(source=sedml_path, working_dir=sedml_path.parent)
parser = SEDMLParser(
    sed_doc=reader.sed_doc,
    exec_dir=reader.exec_dir,
    working_dir=sedml_path.parent,
    name="algorithm_parameters",
)
print(parser.models.keys())
print(parser.simulations.keys())
print(parser.tasks.keys())
print(parser.figures.keys())
```

The parser maps the SED-ML objects onto the objects of `sbmlsim`:

| SED-ML | sbmlsim |
| --- | --- |
| `model` with `changeAttribute` changes | `AbstractModel` with changes; the XPath targets are resolved to the ids of the model |
| `uniformTimeCourse`, `oneStep` | `TimecourseSim` |
| `repeatedTask` with `ranges` | `ScanSim` with `Dimension` objects, see `sbmlsim.simulation.range` |
| `algorithm` and `algorithmParameter` | `Algorithm` and `AlgorithmParameter` with their KISAO terms, see `sbmlsim.simulation.kisaos` |
| `dataGenerator` with `variable` and `math` | `Data`, functions of data are evaluated with `sbmlsim.combine.mathml` |
| `dataDescription` with NuML, CSV or TSV data | `DataSet`, see `sbmlsim.combine.sedml.data` |
| `plot2D`, `curve`, `shadedArea`, `style`, `axis` | `Figure`, `Plot`, `Curve`, `ShadedArea`, `Style`, `Axis`, see [Plots and reports](plotting.md) |
| `report` | the report of the experiment, i.e., a table of data generators |

Steady state simulations, `plot3D` and the parameter estimation tasks are not supported.

## Serializing an experiment to SED-ML

`SEDMLSerializer` writes a `SimulationExperiment` as a SED-ML document with the models and, when an `omex_path` is given, as a COMBINE archive. The experiment is run first to resolve the models and their selections:

```py
from sbmlsim.combine.sedml.parser import SEDMLSerializer

SEDMLSerializer(
    exp_class=RepressilatorExperiment,
    working_dir=Path("results") / "omex",
    sedml_filename="repressilator.sedml",
    omex_path=Path("results") / "repressilator.omex",
)
```

`examples/covid/simulate.py` runs the COVID-19 experiments, serializes them to archives and executes the archives again.

## COMBINE archives

Archives are read and written with [pymetadata](https://github.com/matthiaskoenig/pymetadata), which resolves the manifest and the formats of the entries. The master SED-ML file of an archive is executed by `execute_sedml`, or the first SED-ML file if none is flagged as master. The models of the [BioModels](https://www.ebi.ac.uk/biomodels/) database are downloaded as archives with `sbmlutils.biomodels.download_biomodel_omex`, see `examples/covid/omex/download_covid_models.py`.
