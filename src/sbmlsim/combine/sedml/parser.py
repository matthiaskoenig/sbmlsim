"""SED-ML support for sbmlsim.

This modules parses SED-ML based simulation experiments in the sbmlsim
SimulationExperiment format and executes them.

Overview SED-ML
----------------
SED-ML is build of the main classes
- DataDescription
- Model
- Simulation
- Task
- DataGenerator
- Output

DataDescription
---------------
The DataDescription allows to reference external data, and contains a
description on how to access the data, in what format it is, and what subset
of data to extract.

Model
-----
The Model class is used to reference the models used in the simulation
experiment. SED-ML itself is independent of the model encoding underlying the
models. The only requirement is that the model needs to be referenced by
using an unambiguous identifier which allows for finding it, for example
using a MIRIAM URI. To specify the language in which the model is encoded,
a set of predefined language URNs is provided. The SED-ML Change class allows
the application of changes to the referenced models, including changes on the
XML attributes, e.g. changing the value of an observable, computing the change
of a value using mathematics, or general changes on any XML element
of the model representation that is addressable by XPath expressions,
e.g. substituting a piece of XML by an updated one.

Simulation
----------
The Simulation class defines the simulation settings and the steps taken
during simulation. These include the particular type of simulation and the
algorithm used for the execution of the simulation; preferably an unambiguous
reference to such an algorithm should be given, using a controlled vocabulary,
or ontologies. One example for an ontology of simulation algorithms is the
Kinetic Simulation Algorithm Ontology KiSAO. Further information encodable
in the Simulation class includes the step size, simulation duration, and other
simulation-type dependent information.

Task
----
SED-ML makes use of the notion of a Task class to combine a defined model
(from the Model class) and a defined simulation setting
(from the Simulation class). A task always holds one reference each.
To refer to a specific model and to a specific simulation, the corresponding
IDs are used.

DataGenerator
-------------
The raw simulation result sometimes does not correspond to the desired output
of the simulation, e.g. one might want to normalise a plot before output,
or apply post-processing like mean-value calculation.
The DataGenerator class allows for the encoding of such post-processings
which need to be applied to the simulation result before output.
To define data generators, any addressable variable or parameter of any
defined model (from instances of the Model class) may be referenced,
and new entities might be specified using MathML definitions.

Output
-------
The Output class defines the output of the simulation, in the sense that it
specifies what shall be plotted in the output. To do so, an output type is
defined, e.g. 2D-plot, 3D-plot or data table, and the according axes or
columns are all assigned to one of the formerly specified instances of the
DataGenerator class.

For information about SED-ML please refer to http://www.sed-ml.org/
and the SED-ML specification.
"""

import re
import shutil
import warnings
import roadrunner
from collections import defaultdict
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Optional, Type, Union

import libsedml
import pandas as pd
from pint import Quantity
from pymetadata import omex as pyomex
from pymetadata import log

from sbmlsim.combine.mathml import formula_to_astnode
from sbmlsim.combine.sedml.data import DataDescriptionParser
from sbmlsim.combine.sedml.task import TaskNode, TaskTree
from sbmlsim.data import Data, DataSet
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.fit import FitData, FitExperiment, FitMapping, FitParameter
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.model import AbstractModel
from sbmlsim.plot import Axis, Curve, Figure, Plot
from sbmlsim.plot.plotting import (
    AbstractCurve,
    AxisScale,
    ColorType,
    CurveType,
    Fill,
    Line,
    LineType,
    Marker,
    MarkerType,
    ShadedArea,
    Style,
    SubPlot,
    YAxisPosition,
)
from sbmlsim.simulation import AbstractSim, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulation.algorithm import AlgorithmParameter
from sbmlsim.simulation.kisaos import is_supported_algorithm_for_simulation_type
from sbmlsim.task import Task
from sbmlsim.units import UnitRegistry, UnitsInformation


logger = log.get_logger(__file__)


class SBMLModelTargetType(Enum):
    """Supported target types in SBML models."""

    PARAMETER = 0
    COMPARTMENT = 1
    SPECIES = 2
    SPECIES_AMOUNT = 3
    SPECIES_CONCENTRATION = 4
    TIME = 5


class SBMLModelTarget:
    """Target in an SBML model."""

    def __init__(self, selection: str, target_type: SBMLModelTargetType):
        """Initialize SBMLModelTarget."""
        if target_type in {
            SBMLModelTargetType.PARAMETER,
            SBMLModelTargetType.COMPARTMENT,
            SBMLModelTargetType.SPECIES,
            SBMLModelTargetType.SPECIES_AMOUNT,
            SBMLModelTargetType.TIME,
        }:
            sid = selection
        elif target_type == SBMLModelTargetType.SPECIES_CONCENTRATION:
            sid = selection[1:-1]

        self.sid: str = sid
        self.selection: str = selection
        self.target_type: SBMLModelTargetType = target_type

    @property
    def sedml_symbol(self) -> Optional[str]:
        """Get symbol for model target."""
        if self.target_type in {
            SBMLModelTargetType.PARAMETER,
            SBMLModelTargetType.COMPARTMENT,
            SBMLModelTargetType.SPECIES,
        }:
            return None
        elif self.target_type == SBMLModelTargetType.SPECIES_AMOUNT:
            return "urn:sedml:symbol:amount"
        elif self.target_type == SBMLModelTargetType.SPECIES_CONCENTRATION:
            return "urn:sedml:symbol:concentration"
        elif self.target_type == SBMLModelTargetType.TIME:
            return "urn:sedml:symbol:time"

    @property
    def sedml_target(self) -> Optional[str]:
        """Get xpath target."""
        if self.target_type == SBMLModelTargetType.PARAMETER:
            return f"/sbml:sbml/sbml:model/sbml:listOfParameters/sbml:parameter[@id='{self.sid}']"
        elif self.target_type == SBMLModelTargetType.COMPARTMENT:
            return f"/sbml:sbml/sbml:model/sbml:listOfCompartments/sbml:compartment[@id='{self.sid}']"
        elif self.target_type in {
            SBMLModelTargetType.SPECIES,
            SBMLModelTargetType.SPECIES_AMOUNT,
            SBMLModelTargetType.SPECIES_CONCENTRATION,
        }:
            return f"/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='{self.sid}']"
        elif self.target_type == SBMLModelTargetType.TIME:
            return None

    @staticmethod
    def sbmlsim_model_targets(
        r: roadrunner.ExecutableModel,
    ) -> dict[str, "SBMLModelTarget"]:
        """Model targets which are supported by sbmlsim."""
        d: dict[str, "SBMLModelTarget"] = {}

        # time
        d["time"] = SBMLModelTarget(
            selection="time",
            target_type=SBMLModelTargetType.TIME,
        )

        # parameters
        parameter_ids = set(r.getGlobalParameterIds())
        for pid in parameter_ids:
            d[pid] = SBMLModelTarget(
                selection=pid,
                target_type=SBMLModelTargetType.PARAMETER,
            )

        # species
        species_ids = set(r.getBoundarySpeciesIds() + r.getFloatingSpeciesIds())
        for sid in species_ids:
            d[sid] = SBMLModelTarget(
                selection=sid,
                target_type=SBMLModelTargetType.SPECIES_AMOUNT,
            )
            d[f"[{sid}]"] = SBMLModelTarget(
                selection=f"[{sid}]",
                target_type=SBMLModelTargetType.SPECIES_CONCENTRATION,
            )

        # compartments
        compartment_ids = set(r.getCompartmentIds())
        for sid in compartment_ids:
            d[sid] = SBMLModelTarget(
                selection=sid,
                target_type=SBMLModelTargetType.COMPARTMENT,
            )

        return d


class SEDMLSerializer:
    """Serialize SimulationExperiment to SED-ML.

    Creates the SED-ML and the COMBINE archive containing
    all models and data for the simulation experiment.
    """

    def __init__(
        self,
        exp_class: Type[SimulationExperiment],
        working_dir: Path,
        sedml_filename: str,
        omex_path: Path = None,
        data_path: Path = None,
    ):
        """Initialize SED-ML serializer."""
        self.experiment: Type[SimulationExperiment] = exp_class
        self.working_dir: Path = working_dir
        self.sedml_filename: str = sedml_filename
        self.omex_path: Optional[Path] = omex_path

        # initialize experiment
        runner = ExperimentRunner(
            [exp_class],
            simulator=None,
            data_path=data_path,
            base_path=None,
        )
        self.exp: SimulationExperiment = list(runner.experiments.values())[0]
        # lookup of sbmlsim selections
        self.selection_lookup = self._selection_lookup_table()

        # SED-ML document
        self.sed_doc: libsedml.SedDocument = libsedml.SedDocument(1, 4)

        # --- datasets ---
        self.serialize_datasets()

        # --- models ---
        self.serialize_models()

        # --- simulations ---
        self.serialize_simulations()

        # --- tasks ---
        self.serialize_tasks()

        # --- data generators ---
        self.serialize_data()

        # --- figures ---
        self.serialize_figures()

        # --- reports ---

        sedml_path = working_dir / sedml_filename
        libsedml.writeSedML(self.sed_doc, str(sedml_path))

        # package in omex archive
        if omex_path:
            omex = pyomex.Omex.from_directory(working_dir)
            omex.to_omex(omex_path=omex_path)

    def _selection_lookup_table(self) -> dict[str, dict[str, SBMLModelTarget]]:
        """Lookup table for sbmlsim model selections."""
        d: dict[str, dict[str, SBMLModelTarget]] = {}
        for model_id in self.exp.models():
            rrsbml_model: RoadrunnerSBMLModel = self.exp._models[model_id]
            rr_model: roadrunner.ExecutableModel = rrsbml_model.r.model
            d[model_id] = SBMLModelTarget.sbmlsim_model_targets(r=rr_model)
        return d

    def datagenerator_id_from_data(self, data: Data) -> str:
        """Get the data generator id from data."""
        if data.is_task():
            return f"{data.task_id}__{data.index}"
        elif data.is_function():
            return f"{data.index}"
        elif data.is_dataset():
            return f"{data.dset_id}__{data.index}"

    def serialize_datasets(self):
        """Serialize sbmlsim.DataSets to libsedml.DataDescription.

        Write experiment datasets in SedDocument.
        """
        dset_id: str
        dataset: DataSet

        if self.exp._datasets:
            # models are stored in separate directory
            data_dir = self.working_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

        # FIXME: necessary to figure out the minimal set of columns from the data!
        # Only store this subset of data for the experiment.
        # FIXME: Data must be unit converted to the actual plot/report;
        # FIXME: same for the model

        dset_indices: dict[str, set[str]] = defaultdict(set)
        # THIS CREATES PROBLEMS
        # for did, data in self.exp._data.items():
        #     sed_dg: libsedml.SedDataGenerator = self.sed_doc.createDataGenerator()
        #     sed_dg.setId(did)
        #     if data.is_dataset():
        #         dset_indices[data.dset_id].add(data.index)

        from pprint import pprint

        pprint(dset_indices)

        reference = "column_ids"

        for dset_id, dataset in self.exp._datasets.items():
            sed_data_description: libsedml.SedDataDescription = (
                self.sed_doc.createDataDescription()
            )
            sed_data_description.setId(dset_id)

            # resolve source (relative to data dir
            dataset.to_csv(data_dir / f"{dset_id}.tsv", index=False, sep=",")
            sed_data_description.setSource(str(Path(".") / "data" / f"{dset_id}.tsv"))
            sed_data_description.setFormat("urn:sedml:format:csv")

            # Necessary to encode the columns
            """
            <dimensionDescription>
                <compositeDescription indexType="integer" name="index">
                    <compositeDescription indexType="string" name="columns">
                        <atomicDescription valueType="double" name="values"/>
                    </compositeDescription>
                </compositeDescription>
            </dimensionDescription>
            """
            dim_description: libsedml.DimensionDescription = (
                sed_data_description.createDimensionDescription()
            )
            composite_description_index: libsedml.CompositeDescription = (
                dim_description.createCompositeDescription()
            )
            composite_description_index.setIndexType("integer")
            composite_description_index.setName("index")
            composite_description_columns: libsedml.CompositeDescription = (
                composite_description_index.createCompositeDescription()
            )
            composite_description_columns.setIndexType("string")
            composite_description_columns.setName("columns")
            atomic_description: libsedml.AtomicDescription = (
                composite_description_columns.createAtomicDescription()
            )
            atomic_description.setValueType("double")
            atomic_description.setName("values")

            # listOfDataSources
            """
            <listOfDataSources>
                <dataSource id="dataTime">
                    <listOfSlices>
                        <slice reference="ColumnIds" value="time"/>
                    </listOfSlices>
                </dataSource>
                <dataSource id="dataS1">
                    <listOfSlices>
                    <slice reference="ColumnIds" value="S1"/>
                </listOfSlices>
                </dataSource>
            </listOfDataSources>
            """
            for index in dset_indices[dset_id]:
                sed_data_source: libsedml.SedDataSource = (
                    sed_data_description.createDataSource()
                )
                sed_data_source.setId(f"{dset_id}__{index}")
                sed_slice: libsedml.SedSlice = sed_data_source.createSlice()
                sed_slice.setReference(reference)
                sed_slice.setValue(index)

    def serialize_models(self):
        """Serialize models.

        Write experiment models in SedDocument.
        """
        # Get the unresolved model files or URNs
        model: AbstractModel

        if self.exp.models():
            # models are stored in separate directory
            models_dir = self.working_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

        for model_id, model in self.exp.models().items():
            print(model_id, model)
            rrsbml_model: RoadrunnerSBMLModel = self.exp._models[model_id]
            selection_map: dict[str, SBMLModelTarget] = self.selection_lookup[model_id]

            sed_model: libsedml.SedModel = self.sed_doc.createModel()
            sed_model.setId(model_id)
            if model.name:
                sed_model.setName(model.name)

            abstract_model: AbstractModel
            if isinstance(model, Path):
                abstract_model = AbstractModel(source=model)
            elif isinstance(model, AbstractModel):
                abstract_model = model
            else:
                raise ValueError(f"Model type not supported: {type(model)}")

            model_path_src: Path = abstract_model.source.path
            filename = model_path_src.name
            model_path_target = models_dir / filename

            # copy path in models directory
            shutil.copyfile(model_path_src, model_path_target)

            # resolve path relative to working dir
            model_path_rel = Path(".") / "models" / filename
            sed_model.setSource(str(model_path_rel))

            # get normalized changes (to model units)
            changes: dict[str, Quantity] = UnitsInformation.normalize_changes(
                changes=abstract_model.changes, uinfo=rrsbml_model.uinfo
            )

            for selection, value in changes.items():
                sbml_target = selection_map[selection]
                sed_change_attr: libsedml.SedChangeAttribute = (
                    sed_model.createChangeAttribute()
                )
                sed_change_attr.setTarget(sbml_target.sedml_target)

                # FIXME: support amount and concentration
                # https://github.com/SED-ML/sed-ml/issues/141
                # https://github.com/SED-ML/sed-ml/issues/124
                # sed_change_attr.setSymbol(sbml_target.sedml_symbol)

                sed_change_attr.setNewValue(str(value.magnitude))

                # FIXME: Not supported: AddXML, ChangeXML, RemoveXML
                # FIXME: ComputeChange: Not supported

    def serialize_simulations(self):
        """Serialize simulations.

        Write experiment simulations in SedDocument.
        """
        sim_id: str
        simulation: dict[str, AbstractSim]
        for sim_id, simulation in self.exp._simulations.items():
            if isinstance(simulation, (TimecourseSim, ScanSim)):
                if isinstance(simulation, TimecourseSim):
                    tcsim: TimecourseSim = simulation
                elif isinstance(simulation, ScanSim):
                    tcsim = simulation.simulation

                tc: Timecourse
                for k, tc in enumerate(tcsim.timecourses):
                    sed_uniform_tc: libsedml.SedUniformTimeCourse = (
                        self.sed_doc.createUniformTimeCourse()
                    )

                    tc_id: str
                    if k == 0:
                        tc_id = sim_id
                    else:
                        tc_id = f"{sim_id}_{k}"
                        logger.error(
                            f"Concatenated timecourses not supported: '{tc_id}'"
                        )
                    sed_uniform_tc.setId(tc_id)
                    sed_uniform_tc.setInitialTime(tc.start)
                    if tcsim.time_offset is not None:
                        # FIXME: how to handle the time offsets in later simulations
                        output_start_time = tc.start + tcsim.time_offset
                        output_end_time = tc.end + tcsim.time_offset
                    else:
                        output_start_time = tc.start
                        output_end_time = tc.end

                    sed_uniform_tc.setOutputStartTime(output_start_time)
                    sed_uniform_tc.setOutputEndTime(output_end_time)
                    sed_uniform_tc.setNumberOfSteps(tc.steps)

    def serialize_tasks(self):
        """Serialize tasks.

        Write experiment tasks in SedDocument.
        """
        task_id: str
        task: Task
        for task_id, task in self.exp._tasks.items():
            # FIXME: necessary to extract the repeated tasks from the scans
            sed_task: libsedml.SedTask = self.sed_doc.createTask()
            sed_task.setId(task_id)
            sed_task.setModelReference(task.model_id)
            sed_task.setSimulationReference(task.simulation_id)

    def serialize_data(self):
        """Serialize data generators.

        Write experiment data in SedDocument.
        """

        def sed_variable_from_data(
            sed_dg: libsedml.SedDataGenerator, data: Data, var_id: str
        ) -> libsedml.SedVariable:
            """Create sed_variable from given Data."""
            task_id: str = data.task_id
            task: Task = self.exp._tasks[task_id]
            model_id: str = task.model_id

            model_target: SBMLModelTarget = self.selection_lookup[model_id][data.index]
            sed_variable: libsedml.SedVariable = sed_dg.createVariable()
            sed_variable.setId(f"{did}__{var_id}")
            sed_variable.setTaskReference(task_id)

            sed_variable.setModelReference(task.model_id)
            if model_target.sedml_target:
                sed_variable.setTarget(model_target.sedml_target)
            if model_target.sedml_symbol:
                sed_variable.setSymbol(model_target.sedml_symbol)
            return sed_variable

        did: str
        data: Data
        print(self.exp._data.items())
        for did, data in self.exp._data.items():
            print(did, data)
            sed_dg: libsedml.SedDataGenerator = self.sed_doc.createDataGenerator()
            sed_dg.setId(did)

            if data.is_task():
                print("Create Task DataGenerator:", data)
                sed_variable = sed_variable_from_data(
                    sed_dg, data=data, var_id=data.index
                )
                # FIXME: add conversion math for plots
                formula = f"{sed_variable.getId()}"
                math: libsedml.ASTNode = formula_to_astnode(formula)
                sed_dg.setMath(math)

            elif data.is_function():
                formula: str = data.function
                math: libsedml.ASTNode = formula_to_astnode(formula)

                # FIXME: add conversion math for plots
                for var_key, var_data in data.variables.items():
                    sed_variable: libsedml.SedVariable = sed_variable_from_data(
                        sed_dg, data=var_data, var_id=var_key
                    )
                    math.renameSIdRefs(var_key, sed_variable.getId())
                for par_key, par_value in data.parameters:
                    sed_parameter: libsedml.SedParameter = sed_dg.createParameter()
                    sed_parameter_id = f"{did}__{par_key}"
                    sed_parameter.setId(sed_parameter_id)
                    sed_parameter.setValue(float(par_value))
                    math.renameSIdRefs(par_key, sed_parameter_id)

                sed_dg.setMath(math)

            elif data.is_dataset():
                sed_variable: libsedml.SedVariable = sed_dg.createVariable()
                sed_variable.setId(f"{did}__{data.index}")

                # reference DataSource
                sed_variable.setTarget(f"#{data.dset_id}__{data.index}")
                # FIXME: add conversion math for plots
                formula = f"{sed_variable.getId()}"
                math: libsedml.ASTNode = formula_to_astnode(formula)
                sed_dg.setMath(math)

    def serialize_figures(self):
        """Serialize sbmlsim.Figures to libsedml.SedFigures.

        Write experiment figures in SedDocument.
        """

        def set_abstract_curve_attributes(
            acurve: AbstractCurve, sed_acurve: libsedml.SedAbstractCurve
        ) -> None:
            """Set abstract curve attributes."""
            if acurve.sid is not None:
                sed_acurve.setId(acurve.sid)
            if curve.name is not None:
                sed_acurve.setName(acurve.name)
            if curve.x is not None:
                sed_acurve.setXDataReference(self.datagenerator_id_from_data(acurve.x))
            if acurve.order is not None:
                sed_acurve.setOrder(acurve.order)
            if acurve.yaxis_position is not None:
                if acurve.yaxis_position == YAxisPosition.LEFT:
                    sed_acurve.setYAxis("left")
                elif acurve.yaxis_position == YAxisPosition.Right:
                    sed_acurve.setYAxis("right")
            if acurve.style is not None:
                if acurve.style.sid is None:
                    acurve.style.sid = f"style_{sed_acurve.getId()}"

                style_id: str = acurve.style.sid
                sed_style: libsedml.SedStyle = self.sed_doc.getStyle(style_id)
                if sed_style is None:
                    sed_style = self.sed_doc.createStyle()
                    self.serialize_style(acurve.style, sed_style)
                sed_acurve.setStyle(sed_style.getId())

        figure: Figure
        for _, figure in self.exp._figures.items():
            sed_figure: libsedml.SedFigure = self.sed_doc.createFigure()
            sed_figure.setId(figure.sid)
            if figure.name:
                sed_figure.setName(figure.name)
            sed_figure.setNumCols(figure.num_cols)
            sed_figure.setNumRows(figure.num_rows)

            for subplot in figure.subplots:
                sed_subplot: libsedml.SedSubPlot = sed_figure.createSubPlot()
                if subplot.sid:
                    sed_subplot.setId(subplot.sid)
                if subplot.name:
                    sed_subplot.setName(subplot.name)
                sed_subplot.setRow(subplot.row)
                sed_subplot.setCol(subplot.col)
                sed_subplot.setRowSpan(subplot.row_span)
                sed_subplot.setColSpan(subplot.col_span)

                # FIXME: support plot3d

                plot = subplot.plot
                sed_plot2d: libsedml.SedPlot2D = self.sed_doc.createPlot2D()
                if plot.sid:
                    sed_plot2d.setId(plot.sid)
                if plot.name:
                    sed_plot2d.setName(plot.name)

                # legend
                sed_plot2d.setLegend(plot.legend)

                # handle height and width
                plot_height: float = plot.height
                if not plot_height:
                    plot_height = figure.height / figure.num_rows * subplot.row_span
                sed_plot2d.setHeight(plot_height)

                plot_width: float = plot.width
                if not plot_width:
                    plot_width = figure.width / figure.num_cols * subplot.col_span
                sed_plot2d.setWidth(plot_width)

                # axis
                if plot.xaxis:
                    sed_xaxis: libsedml.SedAxis = sed_plot2d.createXAxis()
                    self.serialize_axis(plot.xaxis, sed_xaxis)
                if plot.yaxis:
                    sed_yaxis: libsedml.SedAxis = sed_plot2d.createYAxis()
                    self.serialize_axis(plot.yaxis, sed_yaxis)
                if plot.yaxis_right:
                    sed_yaxis_right: libsedml.SedAxis = sed_plot2d.createRightYAxis()
                    self.serialize_axis(plot.yaxis_right, sed_yaxis_right)

                # curves
                for curve in plot.curves:
                    sed_curve: libsedml.SedCurve = sed_plot2d.createCurve()
                    set_abstract_curve_attributes(acurve=curve, sed_acurve=sed_curve)
                    if curve.y is not None:
                        sed_curve.setYDataReference(
                            self.datagenerator_id_from_data(curve.y)
                        )
                    # FIXME: assymetrical errors
                    if curve.xerr is not None:
                        dg_id_xerr = self.datagenerator_id_from_data(curve.xerr)
                        sed_curve.setXErrorUpper(dg_id_xerr)
                        sed_curve.setXErrorLower(dg_id_xerr)
                    if curve.yerr is not None:
                        dg_id_yerr = self.datagenerator_id_from_data(curve.yerr)
                        sed_curve.setYErrorUpper(dg_id_yerr)
                        sed_curve.setYErrorLower(dg_id_yerr)

                # shaded areas
                for area in plot.areas:
                    sed_area: libsedml.SedShadedArea = sed_plot2d.createShadedArea()
                    set_abstract_curve_attributes(acurve=area, sed_acurve=sed_area)
                    if area.yfrom:
                        sed_area.setYDataReferenceFrom(
                            self.datagenerator_id_from_data(area.yfrom)
                        )
                    if area.yto:
                        sed_area.setYDataReferenceTo(
                            self.datagenerator_id_from_data(area.yto)
                        )

                sed_subplot.setPlot(sed_plot2d.getId())

    def serialize_axis(self, axis: Axis, sed_axis: libsedml.SedAxis) -> None:
        """Serialize sbmlsim.Axis to libsedml.SEDAxis."""
        if axis.sid:
            sed_axis.setId(axis.sid)
        if axis.name:
            sed_axis.setName(axis.name)
        if axis.scale == AxisScale.LINEAR:
            sed_axis.setType(libsedml.SEDML_AXISTYPE_LINEAR)
        elif axis.scale == AxisScale.LOG10:
            sed_axis.setType(libsedml.SEDML_AXISTYPE_LOG10)
        if axis.min:
            sed_axis.setMin(axis.min)
        if axis.max:
            sed_axis.setMax(axis.max)
        if axis.grid:
            sed_axis.setGrid(axis.grid)
        if axis.style is not None:
            if not axis.style.sid:
                axis.style.sid = f"style_{axis.sid}"
            sed_style: libsedml.SedStyle = self.sed_doc.getStyle(axis.style.id)
            if sed_style is None:
                # style does not yet exist, parse style
                sed_style = self.sed_doc.createStyle()
                self.serialize_style(axis.style, sed_style)
            sed_axis.setStyle(sed_style.getId())

    def serialize_style(self, style: Style, sed_style: libsedml.SedStyle) -> None:
        """Serialize sbmlsim.Style to libsedml.Style."""
        sed_style.setId(style.sid)
        if style.name is not None:
            sed_style.setName(style.name)
        if style.base_style is not None:
            sed_basestyle: libsedml.SedStyle = self.sed_doc.getStyle(
                style.base_style.sid
            )
            if sed_basestyle is None:
                sed_basestyle = self.sed_doc.createStyle()
                self.serialize_style(style.base_style, sed_basestyle)
            sed_style.setBaseStyle(sed_basestyle.getId())
        if style.line is not None:
            line = style.line
            sed_line: libsedml.SedLine = sed_style.createLineStyle()
            if style.line.color is not None:
                sed_line.setColor(line.color.color)
            if line.type:
                line_type = line.type
                if line_type == LineType.NONE:
                    sed_line_type = libsedml.SEDML_LINETYPE_NONE
                elif line_type == LineType.SOLID:
                    sed_line_type = libsedml.SEDML_LINETYPE_SOLID
                elif line_type == LineType.DASH:
                    sed_line_type = libsedml.SEDML_LINETYPE_DASH
                elif line_type == LineType.DOT:
                    sed_line_type = libsedml.SEDML_LINETYPE_DOT
                elif line_type == LineType.DASHDOT:
                    sed_line_type = libsedml.SEDML_LINETYPE_DASHDOT
                elif line_type == LineType.DASHDOTDOT:
                    sed_line_type = libsedml.SEDML_LINETYPE_DASHDOTDOT
                sed_line.setType(sed_line_type)
            if line.thickness is not None:
                sed_line.setThickness(line.thickness)

        if style.marker is not None:
            marker = style.marker
            sed_marker: libsedml.SedMarker = sed_style.createMarkerStyle()
            if marker.type:
                marker_type = marker.type
                if marker_type == MarkerType.NONE:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_NONE
                elif marker_type == MarkerType.SQUARE:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_SQUARE
                elif marker_type == MarkerType.CIRCLE:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_CIRCLE
                elif marker_type == MarkerType.DIAMOND:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_DIAMOND
                elif marker_type == MarkerType.XCROSS:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_XCROSS
                elif marker_type == MarkerType.PLUS:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_PLUS
                elif marker_type == MarkerType.PLUS:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_PLUS
                elif marker_type == MarkerType.STAR:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_STAR
                elif marker_type == MarkerType.TRIANGLEUP:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_TRIANGLEUP
                elif marker_type == MarkerType.TRIANGLEDOWN:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_TRIANGLEDOWN
                elif marker_type == MarkerType.TRIANGLELEFT:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_TRIANGLELEFT
                elif marker_type == MarkerType.TRIANGLERIGHT:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_TRIANGLERIGHT
                elif marker_type == MarkerType.HDASH:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_HDASH
                elif marker_type == MarkerType.VDASH:
                    sed_marker_type = libsedml.SEDML_MARKERTYPE_VDASH
                sed_marker.setType(sed_marker_type)
            if marker.size is not None:
                sed_marker.setSize(marker.size)
            if marker.fill:
                sed_marker.setFill(marker.fill.color)
            if marker.line_color:
                sed_marker.setLineColor(marker.line_color)
            if marker.line_thickness is not None:
                sed_marker.setLineThickness(marker.line_thickness)

        if style.fill is not None:
            fill = style.fill
            sed_fill: libsedml.SedFill = sed_style.createFillStyle()
            if fill.color:
                sed_fill.setColor(fill.color.color)
            if fill.second_color:
                sed_fill.setSecondColor(fill.second_color.color)


class SEDMLParser:
    """Parse SED-ML to sbmlsim.SimulationExperiment."""

    def __init__(
        self,
        sed_doc: libsedml.SedDocument,
        exec_dir: Path,
        working_dir: Path,
        name: Optional[str] = None,
    ):
        """Initialize SED-ML parser from SedDocument.

        :param sed_doc: SedDocument
        :param working_dir: working dir to execute the SED-ML
        :param name: class name used for the simulation experiment. Must be valid
                     python class name.
        """
        self.sed_doc: libsedml.SedDocument = sed_doc
        self.exec_dir = exec_dir
        self.working_dir: Path = working_dir
        self.name: str = name

        # unit registry to handle units throughout the simulation
        self.ureg: UnitRegistry = UnitRegistry(on_redefinition="ignore")

        # Reference to the experiment class
        self.exp_class: Type[SimulationExperiment]

        # --- Models ---
        self.models: dict[str, AbstractModel] = {}

        # resolve original model source and changes
        model_sources, model_changes = self.resolve_model_changes()
        sed_model: libsedml.SedModel
        for sed_model in self.sed_doc.getListOfModels():
            mid = sed_model.getId()
            source = model_sources[mid]
            sed_changes = model_changes[mid]
            self.models[mid] = self.parse_model(
                sed_model, source=source, sed_changes=sed_changes
            )
        logger.debug(f"models: {self.models}")

        # --- DataDescriptions ---
        self.data_descriptions: dict[str, dict[str, pd.Series]] = {}
        self.datasets: dict[str, DataSet] = {}
        sed_dd: libsedml.SedDataDescription
        for sed_dd in sed_doc.getListOfDataDescriptions():
            did = sed_dd.getId()
            data_description: dict[str, pd.Series] = DataDescriptionParser.parse(
                sed_dd, self.working_dir
            )
            self.data_descriptions[did] = data_description

            # TODO: fix the dataframe generation
            # pprint(data_description)
            # df = pd.DataFrame(data_description)
            # dset = DataSet.from_df(df=df, ureg=None, udict=None)
            # self.datasets[did] = dset

        logger.debug(f"data_descriptions: {self.data_descriptions}")

        # --- AlgorithmParameters ---
        self.algorithm_parameters: list[AlgorithmParameter] = []
        sed_alg_par: libsedml.SedAlgorithmParameter
        for sed_alg_par in sed_doc.getListOfAlgorithmParameters():
            self.algorithm_parameters.append(
                self.parse_algorithm_parameter(sed_alg_par)
            )
        logger.debug(f"algorithm_parameters: {self.algorithm_parameters}")

        # --- Simulations ---
        self.simulations: dict[str, AbstractSim] = {}
        sed_sim: libsedml.SedSimulation
        for sed_sim in sed_doc.getListOfSimulations():
            self.simulations[sed_sim.getId()] = self.parse_simulation(sed_sim)
        logger.debug(f"simulations: {self.simulations}")

        # --- Tasks ---
        self.tasks: dict[str, Task] = {}
        sed_task: libsedml.SedTask
        for sed_task in sed_doc.getListOfTasks():
            task = self.parse_task(sed_task)
            if isinstance(task, Task):
                self.tasks[sed_task.getId()] = task
            elif isinstance(task, libsedml.SedParameterEstimationTask):
                # --------------------------------------------------------------------
                # Parameter Estimation Task
                # --------------------------------------------------------------------
                print("-" * 80)
                print("Parameter estimation")
                print("-" * 80)
                sed_petask: libsedml.SedParameterEstimationTask = task
                sed_objective: libsedml.SedObjective = sed_petask.getObjective()

                print("*** Objective ***")
                if sed_objective.getTypeCode() == libsedml.SEDML_LEAST_SQUARE_OBJECTIVE:
                    print("LeastSquareOptimization")

                # Fit Experiments
                print("*** FitExperiments & FitMappings ***")
                fit_experiments: list[FitExperiment] = []
                sed_fit_experiment: libsedml.SedFitExperiment
                for sed_fit_experiment in sed_petask.getListOfFitExperiments():
                    pprint(sed_fit_experiment)
                    fit_type = sed_fit_experiment.getType()
                    if fit_type == libsedml.SEDML_EXPERIMENTTYPE_TIMECOURSE:
                        pass
                    elif fit_type == libsedml.SEDML_EXPERIMENTTYPE_STEADYSTATE:
                        # TODO: support steady state fitting
                        raise NotImplementedError(
                            "Steady state parameter fitting is not supported"
                        )
                    else:
                        raise ValueError(f"ExperimentType not supported: {fit_type}")

                    # algorithm
                    # TODO: support algorithms
                    sed_algorithm: libsedml.SedAlgorithm = (  # noqa: F841
                        sed_fit_experiment.getAlgorithm()
                    )

                    # fit_mappings
                    mappings: list[FitMapping] = []
                    sed_fit_mapping: libsedml.SedFitMapping
                    for sed_fit_mapping in sed_fit_experiment.getListOfFitMappings():
                        weight: float = sed_fit_mapping.getWeight()
                        # TODO: support for point weights
                        point_weight: str = (  # noqa: F841
                            sed_fit_mapping.getPointWeight()
                        )

                        # TODO: resolve data from data generator
                        reference: FitData = None
                        observable: FitData = None
                        experiment = None

                        # necessary to map these
                        mapping = FitMapping(
                            experiment=experiment,
                            reference=reference,
                            observable=observable,
                            weight=weight,
                        )
                        mappings.append(mapping)

                    pprint(mappings)

                    # TODO: necessary to create a SimulationExperiment for the fit experiment
                    fit_experiment = FitExperiment(
                        experiment=None, mappings=mappings, fit_parameters=None
                    )
                    fit_experiments.append(fit_experiment)

                # print(fit_experiments)

                # Fit Parameters
                print("*** FitParameters ***")
                parameters: list[FitParameter] = []
                sed_adjustable_parameter: libsedml.SedAdjustableParameter
                for (
                    sed_adjustable_parameter
                ) in sed_petask.getListOfAdjustableParameters():
                    sid = sed_adjustable_parameter.getId()  # noqa: F841
                    # FIXME: this must be the parameter name in the model -> resolve target
                    # The target of an AdjustableParameter must point to an adjustable
                    # element of the Model referenced bythe parent
                    # ParameterEstimationTask.
                    target = sed_adjustable_parameter.getTarget()
                    print(target)
                    pid = "?"

                    initial_value: float = sed_adjustable_parameter.getInitialValue()
                    sed_bounds: libsedml.SedBounds = (
                        sed_adjustable_parameter.getBounds()
                    )
                    lower_bound: float = sed_bounds.getLowerBound()
                    upper_bound: float = sed_bounds.getUpperBound()
                    # FIXME: support scale (only log)
                    scale: str = sed_bounds.getScale()  # noqa: F841

                    parameters.append(
                        FitParameter(
                            pid=pid,
                            start_value=initial_value,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            unit=None,
                        )
                    )

                    # resolve links to experiments!
                    experiment_refs: list[str] = []

                    for (
                        sed_experiment_ref
                    ) in sed_adjustable_parameter.getListOfExperimentRefs():
                        experiment_refs.append(sed_experiment_ref)

                print("*** Objective ***")
                print(sed_objective)

        print("-" * 80)
        logger.debug(f"tasks: {self.tasks}")

        # --- Data ---
        # data is generated in the figures and reports
        self.data: dict[str, Data] = {}

        # --- Styles ---
        self.styles: dict[str, Style] = {}
        sed_style: libsedml.SedStyle
        for sed_style in sed_doc.getListOfStyles():
            self.styles[sed_style.getId()] = self.parse_style(sed_style)

        logger.debug(f"styles: {self.styles}")

        # --- Outputs: Figures/Plots ---
        self.figures: dict[str, Figure] = {}
        sed_output: libsedml.SedOutput

        # which plots are not in figures
        single_plots = set()
        for sed_output in sed_doc.getListOfOutputs():
            if sed_output.getTypeCode() in [
                libsedml.SEDML_OUTPUT_PLOT2D,
                libsedml.SEDML_OUTPUT_PLOT3D,
            ]:
                single_plots.add(sed_output.getId())

        print(single_plots)
        for sed_output in sed_doc.getListOfOutputs():
            type_code = sed_output.getTypeCode()
            if type_code == libsedml.SEDML_FIGURE:
                self.figures[sed_output.getId()] = self.parse_figure(sed_output)
                sed_figure: libsedml.SedFigure = sed_output
                sed_subplot: libsedml.SedSubPlot
                for sed_subplot in sed_figure.getListOfSubPlots():
                    sed_plot_id = sed_subplot.getPlot()
                    print(sed_plot_id, single_plots)
                    single_plots.remove(sed_plot_id)

        # render remaining plots (without figure)
        for sed_output in sed_doc.getListOfOutputs():
            sed_output_id = sed_output.getId()
            if sed_output_id in single_plots:
                self.figures[sed_output_id] = self._wrap_plot_in_figure(sed_output)

        logger.debug(f"figures: {self.figures}")

        # --- Outputs: Reports---
        self.reports: dict[str, dict[str, Data]] = {}

        for sed_output in sed_doc.getListOfOutputs():
            type_code = sed_output.getTypeCode()
            if type_code == libsedml.SEDML_OUTPUT_REPORT:
                sed_report: libsedml.SedReport = sed_output
                report: dict[str, str] = self.parse_report(sed_report=sed_report)
                self.reports[sed_output.getId()] = report

        logger.debug(f"reports: {self.reports}")

        self.exp_class = self._create_experiment_class()
        self.experiment: SimulationExperiment = self.exp_class()
        self.experiment.initialize()

        for figure in self.figures.values():
            figure.experiment = self.experiment

    def _wrap_plot_in_figure(
        self, sed_plot: Union[libsedml.SedPlot2D, libsedml.SedPlot3D]
    ) -> Figure:
        """Create sbmlsim.Plot from libsedml.Plot and wraps in sbmlsim.Figure."""
        typecode = sed_plot.getTypeCode()
        sed_plot_id: str = sed_plot.getId()
        f = Figure(
            experiment=None,
            sid=sed_plot_id,
            num_rows=1,
            num_cols=1,
        )
        if typecode == libsedml.SEDML_OUTPUT_PLOT2D:
            plot = self.parse_plot2d(sed_plot2d=sed_plot)
        elif typecode == libsedml.SEDML_OUTPUT_PLOT3D:
            plot = self.parse_plot3d(sed_plot3d=sed_plot)

        f.add_plots([plot])
        return f

    def _create_experiment_class(self) -> Type[SimulationExperiment]:
        """Create SimulationExperiment class from information.

        See sbmlsim.experiment.Experiment for the expected functions.
        """

        # Create the experiment object
        def f_algorithm_parameters(obj) -> list[AlgorithmParameter]:
            return self.algorithm_parameters

        def f_models(obj) -> dict[str, AbstractModel]:
            return self.models

        def f_datasets(obj) -> dict[str, DataSet]:
            """Dataset definition (experimental data)."""
            return self.datasets

        def f_simulations(obj) -> dict[str, AbstractSim]:
            return self.simulations

        def f_tasks(obj) -> dict[str, Task]:
            return self.tasks

        def f_data(obj) -> dict[str, Data]:
            return self.data

        def f_figures(obj) -> dict[str, Figure]:
            return self.figures

        def f_reports(obj) -> dict[str, dict[str, str]]:
            return self.reports

        class_name = self.name
        if not class_name:
            class_name = "SedmlSimulationExperiment"

        exp_class = type(
            class_name,
            (SimulationExperiment,),
            {
                "algorithm_parameters": f_algorithm_parameters,
                "models": f_models,
                "datasets": f_datasets,
                "simulations": f_simulations,
                "tasks": f_tasks,
                "data": f_data,
                "figures": f_figures,
                "reports": f_reports,
            },
        )
        return exp_class

    def print_info(self) -> None:
        """Print information."""
        info = {
            "algorithm_parameters:": self.algorithm_parameters,
            "models": self.models,
            "simulations": self.simulations,
            "tasks": self.tasks,
            "data": self.data,
            "figures": self.figures,
            "reports": self.reports,
            "styles": self.styles,
        }
        pprint(info)

    # --- MODELS ---
    @staticmethod
    def parse_xpath_target(xpath: str) -> str:
        """Resolve targets in xpath expression.

        Uses a heuristics to figure out the targets.
        """
        # resolve target
        xpath = xpath.replace('"', "'")
        match = re.findall(r"id='(.*?)'", xpath)
        if (match is None) or (len(match) == 0):
            warnings.warn(f"xpath could not be resolved: {xpath}")
        target = match[0]

        return target

    def parse_model(
        self,
        sed_model: libsedml.SedModel,
        source: str,
        sed_changes: list[libsedml.SedChange],
    ) -> AbstractModel:
        """Convert SedModel to AbstractModel.

        :param sed_changes:
        :param source:s
        :param sed_model:
        :return:
        """
        changes = dict()
        for sed_change in sed_changes:
            d = self.parse_change(sed_change)
            for xpath, value in d.items():
                target = self.parse_xpath_target(xpath)
                changes[target] = value

        mid = sed_model.getId()
        language: str
        if sed_model.isSetLanguage():
            language = sed_model.getLanguage()
        else:
            logger.warning("No language attribute set on model, using SBML.")
            language = "urn:sedml:language:sbml"

        model = AbstractModel(
            source=source,
            sid=mid,
            name=sed_model.getName(),
            language=language,
            language_type=None,
            base_path=self.exec_dir,
            changes=changes,
            selections=None,
        )

        return model

    def resolve_model_changes(self):
        """Resolve the original model sources and full change lists.

        Going through the tree of model upwards until root is reached and
        collecting changes on the way (example models m* and changes c*)
        m1 (source) -> m2 (c1, c2) -> m3 (c3, c4)
        resolves to
        m1 (source) []
        m2 (source) [c1,c2]
        m3 (source) [c1,c2,c3,c4]
        The order of changes is important (at least between nodes on different
        levels of hierarchies), because later changes of derived models could
        reverse earlier changes.

        Uses recursive search strategy, which should be okay as long as the
        model tree hierarchy is not getting to deep.
        """

        def find_source(mid: str, changes):
            """Find source.

            Recursive search for original model and store the
            changes which have to be applied in the list of changes.
            """
            # mid is node above
            if mid in model_sources and not model_sources[mid] == mid:
                # add changes for node
                for c in model_changes[mid]:
                    changes.append(c)
                # keep looking deeper
                return find_source(model_sources[mid], changes)
            # the source is no longer a key in the sources, it is the source
            return mid, changes

        # store original source and changes for model
        model_sources = {}
        model_changes = {}

        # collect direct source and changes
        for m in self.sed_doc.getListOfModels():  # type: libsedml.SedModel
            mid = m.getId()
            source = m.getSource()
            model_sources[mid] = source
            changes = []
            # store the changes unique for this model
            for c in m.getListOfChanges():
                changes.append(c)
            model_changes[mid] = changes

        # resolve source and changes recursively
        all_changes = {}
        mids = [m.getId() for m in self.sed_doc.getListOfModels()]
        for mid in mids:
            source, changes = find_source(mid, changes=list())
            model_sources[mid] = source
            all_changes[mid] = changes[::-1]

        return model_sources, all_changes

    def parse_change(self, sed_change: libsedml.SedChange) -> dict:
        """Parse the libsedml.Change.

        Currently only a limited subset of model changes is supported.
        Namely changes of parameters and concentrations within a
        SedChangeAttribute.
        """
        xpath = sed_change.getTarget()

        if sed_change.getTypeCode() == libsedml.SEDML_CHANGE_ATTRIBUTE:
            # simple change which can be directly set in model
            value = float(sed_change.getNewValue())
            return {xpath: value}

        elif sed_change.getTypeCode() == libsedml.SEDML_CHANGE_COMPUTECHANGE:
            # change based on a model calculation (with optional parameters)

            logger.error("ComputeChange not implemented correctly")
            # TODO: implement compute change with model
            """
            # TODO: general calculation on model with amounts and concentrations
            variables = {}
            for par in sed_change.getListOfParameters():  # type: libsedml.SedParameter
                variables[par.getId()] = par.getValue()

            for var in sed_change.getListOfVariables():  # type: libsedml.SedVariable
                vid = var.getId()
                selection = SEDMLParser.selectionFromVariable(var, model)
                expr = selection.id
                if selection.type == "concentration":
                    expr = f"init([{selection.id}])"
                elif selection.type == "amount":
                    expr = f"init({selection.id})"
                variables[vid] = model[expr]

            # value is calculated with the current state of model
            value = evaluableMathML(sed_change.getMath(), variables=variables)
            """
            value = -1.0
            return {xpath: value}

        else:
            logger.error(f"Unsupported change: {sed_change.getElementName()}")
            # TODO: libsedml.SEDML_CHANGE_REMOVEXML
            # TODO: libsedml.SEDML_CHANGE_ADDXML
            # TODO: libsedml.SEDML_CHANGE_CHANGEXML
            return {}

    def parse_algorithm_parameter(
        self, sed_alg_par: libsedml.SedAlgorithmParameter
    ) -> AlgorithmParameter:
        """Parse algorithm parameter information."""
        sid = sed_alg_par.getId() if sed_alg_par.isSetId() else None
        name = sed_alg_par.getName() if sed_alg_par.isSetName() else None
        kisao = sed_alg_par.getKisaoID() if sed_alg_par.isSetKisaoID() else None
        value = sed_alg_par.getValue() if sed_alg_par.isSetValue() else None
        return AlgorithmParameter(sid=sid, name=name, kisao=kisao, value=value)

    def parse_simulation(self, sed_sim: libsedml.SedSimulation) -> TimecourseSim:
        """Parse simulation information."""
        sim_type = sed_sim.getTypeCode()
        algorithm = sed_sim.getAlgorithm()
        if algorithm is None:
            logger.warning(
                "Algorithm missing on simulation, defaulting to "
                "'cvode: KISAO:0000019'"
            )
            algorithm = sed_sim.createAlgorithm()
            algorithm.setKisaoID("KISAO:0000019")

        kisao = algorithm.getKisaoID()

        # is supported algorithm
        if not is_supported_algorithm_for_simulation_type(
            kisao=kisao, sim_type=sim_type
        ):
            logger.error(
                f"Algorithm '{kisao}' unsupported for simulation "
                f"'{sed_sim.getId()}' of  type '{sim_type}'"
            )

        if sim_type == libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE:
            initial_time: float = sed_sim.getInitialTime()
            output_start_time: float = sed_sim.getOutputStartTime()
            output_end_time: float = sed_sim.getOutputEndTime()
            number_of_points: int = sed_sim.getNumberOfPoints()

            # FIXME: handle time offset correctly (either separate presimulation)
            # FIXME: important to have the correct numbers of points
            tcsim = TimecourseSim(
                timecourses=[
                    Timecourse(
                        start=initial_time,
                        end=output_end_time,
                        steps=number_of_points - 1,
                    ),
                ],
                time_offset=output_start_time,
            )
            return tcsim

        elif sim_type == libsedml.SEDML_SIMULATION_ONESTEP:
            step: float = sed_sim.getStep()
            tcsim = TimecourseSim(
                timecourses=[
                    Timecourse(
                        start=0,
                        end=step,
                        steps=2,
                    ),
                ]
            )
            return tcsim

        elif sim_type == libsedml.SEDML_SIMULATION_STEADYSTATE:
            raise NotImplementedError("steady state simulation not yet supported")

        # TODO/FIXME: handle all the algorithm parameters as integrator parameters

    def parse_task(self, sed_task: libsedml.SedAbstractTask) -> Task:
        """Parse arbitrary task (repeated or simple, or simple repeated)."""
        # If no DataGenerator references the task, no execution is necessary
        dgs: list[libsedml.SedDataGenerator] = self.data_generators_for_task(sed_task)
        if len(dgs) == 0:
            logger.warning(
                f"Task '{sed_task.getId()}' is not used in any DataGenerator."
            )

        # tasks contain other subtasks, which can contain subtasks. This
        # results in a tree of task dependencies where the
        # simple tasks are the node leaves. These tree has to be resolved to
        # generate code for more complex task dependencies.

        # resolve task tree (order & dependency of tasks)
        task_tree_root = TaskTree.from_sedml_task(self.sed_doc, root_task=sed_task)

        # go forward through task tree
        tree_nodes = [n for n in task_tree_root]

        for node in tree_nodes:
            task_type = node.task.getTypeCode()

            print(node.task)
            # Create simulation for task
            if task_type == libsedml.SEDML_TASK:
                task = self._parse_simple_task(task_node=node)
                return task

            # Repeated tasks are multi-dimensional scans
            elif task_type == libsedml.SEDML_TASK_REPEATEDTASK:
                self._parse_repeated_task(node=node)

            elif task_type == libsedml.SEDML_TASK_PARAMETER_ESTIMATION:
                return sed_task

            else:
                raise ValueError(f"Unsupported task: {task_type}")

    def _parse_simple_task(self, task_node: TaskNode) -> Task:
        """Parse simple task."""
        sed_task: libsedml.SedTask = task_node.task
        model_id: str = sed_task.getModelReference()
        simulation_id: str = sed_task.getSimulationReference()
        return Task(model=model_id, simulation=simulation_id)

    def _parse_repeated_task(self, node: TaskNode):
        """Parse repeated task.

        Will be translated into a multi-dimensional scan.
        """
        # repeated tasks will be translated into multidimensional scans
        raise NotImplementedError
        # TODO: implement

    def parse_figure(self, sed_figure: libsedml.SedFigure) -> Figure:
        """Parse figure information."""
        figure = Figure(
            experiment=None,
            sid=sed_figure.getId() if sed_figure.isSetId() else None,
            name=sed_figure.getName() if sed_figure.isSetName() else None,
            num_rows=sed_figure.getNumRows() if sed_figure.isSetNumRows() else 1,
            num_cols=sed_figure.getNumCols() if sed_figure.isSetNumCols() else 1,
        )

        panel_height = 0.0
        panel_width = 0.0
        sed_subplot: libsedml.SedSubPlot
        for sed_subplot in sed_figure.getListOfSubPlots():
            sed_output = self.sed_doc.getOutput(sed_subplot.getPlot())
            if sed_output is None:
                raise ValueError(
                    f"Plot could not be resolved. No output exists in "
                    f"listOfOutputs for id='{sed_subplot.getPlot()}'"
                )

            typecode = sed_output.getTypeCode()

            plot: Plot
            if typecode == libsedml.SEDML_OUTPUT_PLOT2D:
                plot = self.parse_plot2d(sed_plot2d=sed_output)
            elif typecode == libsedml.SEDML_OUTPUT_PLOT3D:
                plot = self.parse_plot3d(sed_plot3d=sed_output)
            elif typecode == libsedml.SEDML_OUTPUT_REPORT:
                plot = None
                raise ValueError("Report not supported as subplot.")

            # handle layout
            row = sed_subplot.getRow()
            col = sed_subplot.getCol()
            row_span = sed_subplot.getRowSpan() if sed_subplot.isSetRowSpan() else 1
            col_span = sed_subplot.getColSpan() if sed_subplot.isSetColSpan() else 1

            if not panel_height and plot.height:
                panel_height = plot.height / sed_subplot.getRowSpan()
            if not panel_width and plot.width:
                panel_width = plot.width / sed_subplot.getColSpan()

            # add subplot
            figure.subplots.append(
                SubPlot(
                    plot=plot, row=row, col=col, row_span=row_span, col_span=col_span
                )
            )

        # figure height and width from panels
        if not panel_height:
            panel_height = Figure.panel_height
        figure.height = figure.num_rows * panel_height

        if not panel_width:
            panel_width = Figure.panel_width
        figure.width = figure.num_cols * panel_width

        return figure

    def parse_plot2d(self, sed_plot2d: libsedml.SedPlot2D) -> Plot:
        """Parse the libsedml.Plot2D into a sbmlsim.Plot."""
        plot = Plot(
            sid=sed_plot2d.getId(),
            name=sed_plot2d.getName() if sed_plot2d.isSetName() else None,
            legend=sed_plot2d.getLegend() if sed_plot2d.isSetLegend() else True,
            height=sed_plot2d.getHeight() if sed_plot2d.isSetHeight() else None,
            width=sed_plot2d.getHeight() if sed_plot2d.isSetWidth() else None,
        )

        # axis
        plot.xaxis = self.parse_axis(sed_plot2d.getXAxis())
        plot.yaxis = self.parse_axis(sed_plot2d.getYAxis())
        plot.yaxis_right = self.parse_axis(sed_plot2d.getRightYAxis())

        # curves
        curves: list[Curve] = []
        areas: list[ShadedArea] = []
        for sed_abstract_curve in sed_plot2d.getListOfCurves():
            abstract_curve = self.parse_abstract_curve(sed_abstract_curve)
            if isinstance(abstract_curve, Curve):
                curves.append(abstract_curve)
            elif isinstance(abstract_curve, ShadedArea):
                areas.append(abstract_curve)
        plot.curves = curves
        plot.areas = areas
        return plot

    def parse_plot3d(self, sed_plot3d: libsedml.SedPlot3D) -> Plot:
        """Parse Plot3D."""
        # FIXME: implement
        raise NotImplementedError

    def parse_report(self, sed_report: libsedml.SedReport) -> dict[str, str]:
        """Parse Report.

        :return dictionary of label: dataGenerator.id mapping.
        """
        sed_dataset: libsedml.SedDataSet
        report: dict[str, str] = {}
        for sed_dataset in sed_report.getListOfDataSets():
            sed_dg_id: str = sed_dataset.getDataReference()
            if self.sed_doc.getDataGenerator(sed_dg_id) is None:
                raise ValueError(
                    f"Report '{sed_report.getId()}': Id of DataGenerator "
                    f"does not exist '{sed_dg_id}'"
                )
            if not sed_dataset.isSetLabel():
                logger.error("Required attribute label missing on DataSet in Report.")
                continue
            label = sed_dataset.getLabel()
            if label in report:
                logger.error(f"Duplicate label in report '{report.getId()}': '{label}'")

            report[label] = sed_dg_id
        return report

    def parse_axis(self, sed_axis: libsedml.SedAxis) -> Optional[Axis]:
        """Parse axes information."""
        if sed_axis is None:
            return None

        axis = Axis(
            label=sed_axis.getName() if sed_axis.isSetName else None,
            min=sed_axis.getMin() if sed_axis.isSetMin() else None,
            max=sed_axis.getMax() if sed_axis.isSetMax() else None,
            grid=sed_axis.getGrid() if sed_axis.isSetGrid() else False,
            reverse=sed_axis.getReverse() if sed_axis.isSetReverse() else False,
        )
        axis.sid = sed_axis.getId()
        axis.name = sed_axis.getName()

        scale: AxisScale
        if sed_axis.isSetType():
            sed_type = sed_axis.getType()
            if sed_type == libsedml.SEDML_AXISTYPE_LINEAR:
                scale = AxisScale.LINEAR
            elif sed_type == libsedml.SEDML_AXISTYPE_LOG10:
                scale = AxisScale.LOG10
            elif sed_type == libsedml.SEDML_AXISTYPE_INVALID:
                logger.error("Invalid axis scale encountered, fallback to 'linear'")
                scale = AxisScale.LINEAR
        else:
            scale = AxisScale.LINEAR
        axis.scale = scale

        if sed_axis.isSetStyle():
            style = self.parse_style(sed_axis.getStyle())
            axis.style = style

        return axis

    def parse_abstract_curve(
        self, sed_acurve: libsedml.SedAbstractCurve
    ) -> Union[ShadedArea, Curve]:
        """Parse abstract curve."""
        sid: str = sed_acurve.getId()
        name: Optional[str] = sed_acurve.getName() if sed_acurve.isSetName() else None
        x: Data = self.data_from_datagenerator(sed_acurve.getXDataReference())
        order: int = sed_acurve.getOrder() if sed_acurve.isSetOrder() else None

        # parse yaxis
        yaxis_position = None
        if sed_acurve.isSetYAxis():
            sed_yaxis: str = sed_acurve.getYAxis()
            if sed_yaxis == "left":
                yaxis_position = YAxisPosition.LEFT
            elif sed_yaxis == "right":
                yaxis_position = YAxisPosition.RIGHT
            else:
                raise ValueError(f"Unsupported yAxis on curve: {sed_yaxis}")

        # parse style
        if sed_acurve.isSetStyle():
            # styles are already parsed, used the style
            style = self.styles[sed_acurve.getStyle()]
        else:
            # default style
            style = Style(
                line=Line(),
                marker=Marker(),
                fill=Fill(),
            )

        sed_acurve_type = sed_acurve.getTypeCode()
        if sed_acurve_type == libsedml.SEDML_OUTPUT_CURVE:
            sed_curve: libsedml.SedCurve = sed_acurve
            curve_type: CurveType
            if not sed_curve.isSetType():
                logger.warning(
                    f"No curve.type set on {sed_curve}, "
                    f"defaulting to POINTS. It is highly "
                    f"recommended to set curve.type."
                )
                curve_type = CurveType.POINTS
            else:
                sed_curve_type = sed_curve.getType()
                if sed_curve_type == libsedml.SEDML_CURVETYPE_POINTS:
                    curve_type = CurveType.POINTS
                elif sed_curve_type == libsedml.SEDML_CURVETYPE_BAR:
                    curve_type = CurveType.BAR
                elif sed_curve_type == libsedml.SEDML_CURVETYPE_BARSTACKED:
                    curve_type = CurveType.BARSTACKED
                elif sed_curve_type == libsedml.SEDML_CURVETYPE_HORIZONTALBAR:
                    curve_type = CurveType.HORIZONTALBAR
                elif sed_curve_type == libsedml.SEDML_CURVETYPE_HORIZONTALBARSTACKED:
                    curve_type = CurveType.HORIZONTALBARSTACKED
                elif sed_curve_type == libsedml.SEDML_CURVETYPE_INVALID:
                    raise ValueError(
                        f"Invalid CurveType: {sed_curve.getType()} on " f"{sed_curve}"
                    )
            curve = Curve(
                sid=sid,
                name=name,
                x=x,
                y=self.data_from_datagenerator(sed_curve.getYDataReference()),
                xerr=self.data_from_datagenerator(sed_curve.getXErrorUpper()),
                yerr=self.data_from_datagenerator(sed_curve.getYErrorUpper()),
                type=curve_type,
                order=order,
                yaxis_position=yaxis_position,
                style=style,
            )

            if not curve.name:
                curve.name = f"{curve.y.name}({curve.x.name})"

            return curve
        elif sed_acurve_type == libsedml.SEDML_SHADEDAREA:
            sed_shaded_area: libsedml.SedShadedArea = sed_acurve
            area = ShadedArea(
                sid=sid,
                name=name,
                x=x,
                yfrom=self.data_from_datagenerator(
                    sed_shaded_area.getYDataReferenceFrom()
                ),
                yto=self.data_from_datagenerator(sed_shaded_area.getYDataReferenceTo()),
                order=order,
                yaxis_position=yaxis_position,
                style=style,
            )

            if not area.name:
                area.name = f"{area.yfrom.name}|{area.yto.name}({area.x.name})"

            return area
        else:
            raise ValueError(
                f"Type of AbstractCurve '{sed_acurve}' is not supported: "
                f"'{sed_acurve_type}'"
            )

    def parse_style(self, sed_style: Union[str, libsedml.SedStyle]) -> Optional[Style]:
        """Parse SED-ML style."""
        if not sed_style:
            return None

        # resolve style by style id
        if isinstance(sed_style, str):
            sed_style: libsedml.SedStyle = self.sed_doc.getStyle(sed_style)

        style = Style(
            sid=sed_style.getId(),
            name=sed_style.getName() if sed_style.isSetName() else None,
            base_style=self.parse_style(sed_style.getBaseStyle())
            if sed_style.isSetBaseStyle()
            else None,
        )
        sed_line: libsedml.SedLine = sed_style.getLineStyle()
        style.line = self.parse_line(sed_line)

        sed_marker: libsedml.SedMarker = sed_style.getMarkerStyle()
        style.marker = self.parse_marker(sed_marker)

        sed_fill = libsedml.SedFill = sed_style.getFillStyle()
        style.fill = self.parse_fill(sed_fill)
        return style

    def parse_line(self, sed_line: libsedml.SedLine) -> Optional[Line]:
        """Parse line information."""
        if sed_line is None:
            return None

        line_type: Optional[LineType]
        if not sed_line.isSetType():
            line_type = None
        else:
            sed_line_type = sed_line.getType()
            if sed_line_type == libsedml.SEDML_LINETYPE_NONE:
                line_type = LineType.NONE
            elif sed_line_type == libsedml.SEDML_LINETYPE_SOLID:
                line_type = LineType.SOLID
            elif sed_line_type == libsedml.SEDML_LINETYPE_DASH:
                line_type = LineType.DASH
            elif sed_line_type == libsedml.SEDML_LINETYPE_DOT:
                line_type = LineType.DOT
            elif sed_line_type == libsedml.SEDML_LINETYPE_DASHDOT:
                line_type = LineType.DASHDOT
            elif sed_line_type == libsedml.SEDML_LINETYPE_DASHDOTDOT:
                line_type = LineType.DASHDOTDOT

        return Line(
            type=line_type,
            color=ColorType.parse_color(sed_line.getColor())
            if sed_line.isSetColor()
            else None,
            thickness=sed_line.getThickness() if sed_line.isSetThickness() else None,
        )

    def parse_marker(self, sed_marker: libsedml.SedMarker) -> Optional[Marker]:
        """Parse the line information."""
        if sed_marker is None:
            return None

        marker_type: Optional[MarkerType]
        if not sed_marker.isSetType():
            marker_type = None
        else:
            sed_marker_type = sed_marker.getType()
            if sed_marker_type == libsedml.SEDML_MARKERTYPE_NONE:
                marker_type = MarkerType.NONE
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_SQUARE:
                marker_type = MarkerType.SQUARE
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_CIRCLE:
                marker_type = MarkerType.CIRCLE
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_DIAMOND:
                marker_type = MarkerType.DIAMOND
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_XCROSS:
                marker_type = MarkerType.XCROSS
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_PLUS:
                marker_type = MarkerType.PLUS
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_STAR:
                marker_type = MarkerType.STAR
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_TRIANGLEUP:
                marker_type = MarkerType.TRIANGLEUP
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_TRIANGLEDOWN:
                marker_type = MarkerType.TRIANGLEDOWN
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_TRIANGLELEFT:
                marker_type = MarkerType.TRIANGLELEFT
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_TRIANGLERIGHT:
                marker_type = MarkerType.TRIANGLERIGHT
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_HDASH:
                marker_type = MarkerType.HDASH
            elif sed_marker_type == libsedml.SEDML_MARKERTYPE_VDASH:
                marker_type = MarkerType.VDASH

        marker = Marker(
            size=sed_marker.getSize() if sed_marker.isSetSize() else None,
            type=marker_type,
            fill=ColorType(sed_marker.getFill()) if sed_marker.isSetFill() else None,
            line_thickness=sed_marker.getLineThickness()
            if sed_marker.isSetLineThickness()
            else None,
            line_color=ColorType(sed_marker.getLineColor())
            if sed_marker.isSetLineColor()
            else None,
        )

        return marker

    def parse_fill(self, sed_fill: libsedml.SedFill) -> Optional[Fill]:
        """Parse fill information."""
        if sed_fill is None:
            return None

        return Fill(
            color=ColorType.parse_color(sed_fill.getColor())
            if sed_fill.isSetColor()
            else None,
            second_color=ColorType.parse_color(sed_fill.getSecondColor())
            if sed_fill.isSetSecondColor()
            else None,
        )

    def data_from_datagenerator(self, sed_dg_ref: Optional[str]) -> Optional[Data]:
        """Evaluate DataGenerator with actual data.

        Uses results of SimulationExperiment for evaluation.
        """
        if not sed_dg_ref:
            return None

        sed_dg: libsedml.SedDataGenerator = self.sed_doc.getDataGenerator(sed_dg_ref)
        if sed_dg is None:
            raise ValueError(
                f"DataGenerator with id '{sed_dg_ref}' does not exist "
                f"in listOfDataGenerators:\n"
                f"{[dg.getId() for dg in self.sed_doc.getListOfDataGenerators()]}"
            )

        astnode: libsedml.ASTNode = sed_dg.getMath()
        function: str = libsedml.formulaToL3String(astnode)

        parameters: dict[str, float] = {}
        sed_par: libsedml.SedParameter
        for sed_par in sed_dg.getListOfParameters():
            parameters[sed_par.getId()] = sed_par.getValue()

        variables: dict[str, Data] = {}
        sed_var: libsedml.SedVariable
        for sed_var in sed_dg.getListOfVariables():
            task_id = sed_var.getTaskReference()
            symbol = None
            if sed_var.isSetSymbol():
                if sed_var.getSymbol() == "urn:sedml:symbol:time":
                    index = "time"
                    symbol = Data.Symbols.TIME
            if sed_var.isSetTarget():
                index = self.parse_xpath_target(sed_var.getTarget())
                sed_symbol = sed_var.getSymbol() if sed_var.isSetSymbol() else None
                if sed_symbol:
                    if sed_symbol == "urn:sedml:symbol:amount":
                        symbol = Data.Symbols.AMOUNT
                    elif symbol == "urn:sedml:symbol:concentration":
                        index = Data.Symbols.CONCENTRATION
                    else:
                        logger.error(f"symbol not supported: '{symbol}'")

            d_var = Data(index=index, symbol=symbol, task=task_id)
            # register data
            self.data[d_var.sid] = d_var
            variables[sed_var.getId()] = d_var

        # The simple case of a single variable without math data generator
        if len(variables) == 1 and function == sed_var.getId():
            d = d_var
        else:
            d = Data(
                index=sed_dg.getId(),
                function=function,
                variables=variables,
                parameters=parameters,
            )

        self.data[d.sid] = d
        return d

    def data_generators_for_task(
        self,
        sed_task: libsedml.SedTask,
    ) -> list[libsedml.SedDataGenerator]:
        """Get DataGenerators which reference the given task."""
        sed_dgs = []
        sed_dg: libsedml.SedDataGenerator
        var: libsedml.SedVariable
        for sed_dg in self.sed_doc.getListOfDataGenerators():
            for var in sed_dg.getListOfVariables():
                if var.getTaskReference() == sed_task.getId():
                    sed_dgs.append(sed_dg)
                    # DataGenerator is added, no need to look at rest of variables
                    break
        return sed_dgs

    @staticmethod
    def get_ordered_subtasks(sed_task: libsedml.SedTask) -> list[libsedml.SedTask]:
        """Ordered list of subtasks for task."""
        subtasks = sed_task.getListOfSubTasks()
        subtask_order = [st.getOrder() for st in subtasks]

        # sort by order, if all subtasks have order (not required)
        if all(subtask_order) is not None:
            subtasks = [st for (stOrder, st) in sorted(zip(subtask_order, subtasks))]
        return subtasks
