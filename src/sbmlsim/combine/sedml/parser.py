"""
SED-ML support for sbmlsim
==========================

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

SED-ML in sbmlsim: Supported Features
=====================================
sbmlsim supports SED-ML L1V4 with SBML as model format.
SBML models are fully supported

Supported input for SED-ML are either SED-ML files ('.sedml' extension),
SED-ML XML strings or combine archives ('.sedx'|'.omex' extension).

In the current implementation all SED-ML constructs with exception of
XML transformation changes of the model, i.e.,
- Change.RemoveXML
- Change.AddXML
- Change.ChangeXML
are supported.
"""
import importlib
import logging
import re
import warnings
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Dict, List, Union, Optional, Type, Any

import libsedml
import numpy as np
import pandas as pd
from pprint import pprint

from sbmlsim.combine.sedml.data import DataDescriptionParser
from sbmlsim.combine.sedml.io import read_sedml
from sbmlsim.combine.sedml.kisao import is_supported_algorithm_for_simulation_type
from sbmlsim.combine.sedml.task import Stack, TaskNode, TaskTree
from sbmlsim.data import DataSet, Data
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitParameter, FitExperiment, FitMapping, FitData
from sbmlsim.model import model_resources
from sbmlsim.model.model import AbstractModel
from sbmlsim.plot import Figure, Plot, Axis, Curve
from sbmlsim.plot.plotting import Style, Line, LineStyle, ColorType, Marker, \
    MarkerStyle, Fill, SubPlot, AxisScale, CurveType, YAxisPosition
from sbmlsim.simulation import ScanSim, TimecourseSim, Timecourse, AbstractSim
from sbmlsim.task import Task
from sbmlsim.units import UnitRegistry


logger = logging.getLogger(__file__)

# FIXME: support omex files
'''
def experiment_from_omex(omex_path: Path):
    """Create SimulationExperiments from all SED-ML files."""
    tmp_dir = tempfile.mkdtemp()
    try:
        omex.extractCombineArchive(omex_path, directory=tmp_dir, method="zip")
        locations = omex.getLocationsByFormat(omex_path, "sed-ml")
        sedml_files = [os.path.join(tmp_dir, loc) for loc in locations]

        for k, sedml_file in enumerate(sedml_files):
            pystr = sedmlToPython(sedml_file)
            factory = SEDMLCodeFactory(inputStr, workingDir=workingDir)
            factory.to
            pycode[locations[k]] = pystr

    finally:
        shutil.rmtree(tmp_dir)
    return pycode
'''


class SEDMLParser(object):
    """ Parsing SED-ML in internal format."""

    def __init__(
        self,
        sed_doc: libsedml.SedDocument,
        working_dir: Path,
        name: Optional[str] = None,
    ):
        """Parses information from SedDocument.

        :param sed_doc: SedDocument
        :param working_dir: working dir to execute the SED-ML
        :param name: class name used for the simulation experiment. Must be valid
                     python class name.
        """

        self.sed_doc: libsedml.SedDocument = sed_doc
        self.working_dir: Path = working_dir
        self.name: str = name

        # unit registry to handle units throughout the simulation
        self.ureg: UnitRegistry = UnitRegistry()

        # Reference to the experiment class
        self.exp_class: Type[SimulationExperiment]

        # --- Models ---
        self.models: Dict[str, AbstractModel] = {}

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
        self.data_descriptions: Dict[str, Dict[str, pd.Series]] = {}
        self.datasets: Dict[str, DataSet] = {}
        sed_dd: libsedml.SedDataDescription
        for sed_dd in sed_doc.getListOfDataDescriptions():
            did = sed_dd.getId()
            data_description: Dict[str, pd.Series] = DataDescriptionParser.parse(
                sed_dd, self.working_dir
            )
            self.data_descriptions[did] = data_description

            # TODO: fix the dataframe generation
            # pprint(data_description)
            # df = pd.DataFrame(data_description)
            # dset = DataSet.from_df(df=df, ureg=None, udict=None)
            # self.datasets[did] = dset

        logger.debug(f"data_descriptions: {self.data_descriptions}")

        # --- Simulations ---
        self.simulations: Dict[str, AbstractSim] = {}
        sed_sim: libsedml.SedSimulation
        for sed_sim in sed_doc.getListOfSimulations():
            self.simulations[sed_sim.getId()] = self.parse_simulation(sed_sim)
        logger.debug(f"simulations: {self.simulations}")

        # --- Tasks ---
        self.tasks: Dict[str, Task] = {}
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
                fit_experiments: List[FitExperiment] = []
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
                        raise ValueError(
                            f"ExperimentType not supported: {fit_type}"
                        )

                    # algorithm
                    # TODO: support algorithms
                    sed_algorithm: libsedml.SedAlgorithm = sed_fit_experiment.getAlgorithm()

                    # fit_mappings
                    mappings: List[FitMapping] = []
                    sed_fit_mapping: libsedml.SedFitMapping
                    for sed_fit_mapping in sed_fit_experiment.getListOfFitMappings():
                        weight: float = sed_fit_mapping.getWeight()
                        # TODO: support for point weights
                        point_weight: str = sed_fit_mapping.getPointWeight()

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
                        experiment=None,
                        mappings=mappings,
                        fit_parameters=None
                    )
                    fit_experiments.append(fit_experiment)

                # print(fit_experiments)

                # Fit Parameters
                print("*** FitParameters ***")
                parameters: List[FitParameter] = []
                sed_adjustable_parameter: libsedml.SedAdjustableParameter
                for sed_adjustable_parameter in sed_petask.getListOfAdjustableParameters():

                    sid = sed_adjustable_parameter.getId()
                    # FIXME: this must be the parameter name in the model -> resolve target
                    # The target of an AdjustableParameter must point to an adjustable
                    # element of the Model referenced bythe parent
                    # ParameterEstimationTask.
                    target = sed_adjustable_parameter.getTarget()
                    print(target)
                    pid = "?"

                    initial_value: float = sed_adjustable_parameter.getInitialValue()
                    sed_bounds: libsedml.SedBounds = sed_adjustable_parameter.getBounds()
                    lower_bound: float = sed_bounds.getLowerBound()
                    upper_bound: float = sed_bounds.getUpperBound()
                    scale: str = sed_bounds.getScale()  # FIXME: support scale (only log)

                    parameters.append(
                        FitParameter(
                            pid=pid,
                            start_value=initial_value,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            unit=None
                        )
                    )

                    # resolve links to experiments!
                    experiment_refs: List[str] = []

                    for sed_experiment_ref in sed_adjustable_parameter.getListOfExperimentRefs():
                        experiment_refs.append(sed_experiment_ref)

                print("*** Objective ***")
                print(sed_objective)

        print("-" * 80)
        logger.debug(f"tasks: {self.tasks}")

        # --- Data ---
        # data is generated in the figures
        self.data: Dict[str, Data] = {}

        # --- Styles ---
        self.styles: Dict[str, Style] = {}
        sed_style: libsedml.SedStyle
        for sed_style in sed_doc.getListOfStyles():
            self.styles[sed_style.getId()] = self.parse_style(sed_style)

        logger.debug(f"styles: {self.styles}")

        # --- Outputs: Figures/Plots ---
        fig: Figure
        self.figures: Dict[str, Figure] = {}
        sed_output: libsedml.SedOutput

        # which plots are not in figures
        single_plots = set()
        for sed_output in sed_doc.getListOfOutputs():
            if sed_output.getTypeCode() in [libsedml.SEDML_OUTPUT_PLOT2D, libsedml.SEDML_OUTPUT_PLOT3D]:
                single_plots.add(sed_output.getId())

        for sed_output in sed_doc.getListOfOutputs():
            type_code = sed_output.getTypeCode()
            if type_code == libsedml.SEDML_FIGURE:
                self.figures[sed_output.getId()] = self.parse_figure(sed_output)
                sed_figure: libsedml.SedFigure = sed_output
                sed_subplot: libsedml.SedSubPlot
                for sed_subplot in sed_figure.getListOfSubPlots():
                    sed_plot_id = sed_subplot.getPlot()
                    single_plots.remove(sed_plot_id)


        # render remaining plots (without figure)
        for sed_output in sed_doc.getListOfOutputs():
            sed_output_id = sed_output.getId()
            if sed_output_id in single_plots:
                self.figures[sed_output_id] = self._wrap_plot_in_figure(sed_output)

        logger.debug(f"figures: {self.figures}")

        # --- Outputs: Reports---
        self.reports: Dict[str, Dict[str, Data]] = {}

        for sed_output in sed_doc.getListOfOutputs():
            type_code = sed_output.getTypeCode()
            if type_code == libsedml.SEDML_OUTPUT_REPORT:
                sed_report: libsedml.SedReport = sed_output
                report: Dict[str, str] = self.parse_report(sed_report=sed_report)
                self.reports[sed_output.getId()] = report

        logger.debug(f"reports: {self.reports}")

        self.exp_class = self._create_experiment_class()
        self.experiment: SimulationExperiment = self.exp_class()
        self.experiment.initialize()

        for figure in self.figures.values():
            figure.experiment = self.experiment

    def _wrap_plot_in_figure(self, sed_plot: Union[libsedml.SedPlot2D, libsedml.SedPlot3D]) -> Figure:
        """Creates plot from SED-ML plot and wraps in figure."""
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

        f.add_plots([
            plot
        ])
        return f


    def _create_experiment_class(self) -> Type[SimulationExperiment]:
        """Create SimulationExperiment class from information.

        See sbmlsim.experiment.Experiment for the expected functions.
        """

        # Create the experiment object
        def f_models(obj) -> Dict[str, AbstractModel]:
            return self.models

        def f_datasets(obj) -> Dict[str, DataSet]:
            """Dataset definition (experimental data)."""
            return self.datasets

        def f_simulations(obj) -> Dict[str, AbstractSim]:
            return self.simulations

        def f_tasks(obj) -> Dict[str, Task]:
            return self.tasks

        def f_data(obj) -> Dict[str, Data]:
            return self.data

        def f_figures(obj) -> Dict[str, Figure]:
            return self.figures

        def f_reports(obj) -> Dict[str, Dict[str, str]]:
            return self.reports

        class_name = self.name
        if not class_name:
            class_name = "SedmlSimulationExperiment"

        exp_class = type(
            class_name,
            (SimulationExperiment,),
            {
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
        # FIXME: SED-ML amount/species targets
        """

        # resolve target
        xpath = xpath.replace('"', "'")
        match = re.findall(r"id='(.*?)'", xpath)
        if (match is None) or (len(match) == 0):
            warnings.warn(f"xpath could not be resolved: {xpath}")
        target = match[0]

        if ("model" in xpath) and ("parameter" in xpath):
            # parameter value change
            pass
        elif ("model" in xpath) and ("species" in xpath):
            # species concentration change
            pass
        elif ("model" in xpath) and ("id" in xpath):
            # other
            pass
        else:
            raise ValueError(f"Unsupported target in xpath: {xpath}")

        return target

    def parse_model(
        self,
        sed_model: libsedml.SedModel,
        source: str,
        sed_changes: List[libsedml.SedChange],
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
        model = AbstractModel(
            source=source,
            sid=mid,
            name=sed_model.getName(),
            language=sed_model.getLanguage(),
            language_type=None,
            base_path=self.working_dir,
            changes=changes,
            selections=None,
        )

        return model

    def resolve_model_changes(self):
        """Resolves the original model sources and full change lists.

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

        def findSource(mid, changes):
            """
            Recursive search for original model and store the
            changes which have to be applied in the list of changes

            :param mid:
            :param changes:
            :return:
            """
            # mid is node above
            if mid in model_sources and not model_sources[mid] == mid:
                # add changes for node
                for c in model_changes[mid]:
                    changes.append(c)
                # keep looking deeper
                return findSource(model_sources[mid], changes)
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
            source, changes = findSource(mid, changes=list())
            model_sources[mid] = source
            all_changes[mid] = changes[::-1]

        return model_sources, all_changes

    def parse_change(self, sed_change: libsedml.SedChange) -> Dict:
        """Parses the change.

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

    def parse_simulation(self, sed_sim: libsedml.SedSimulation) -> Union[TimecourseSim]:
        """ Parse simulation information."""
        sim_type = sed_sim.getTypeCode()
        algorithm = sed_sim.getAlgorithm()
        if algorithm is None:
            logger.warning(
                f"Algorithm missing on simulation, defaulting to "
                f"'cvode: KISAO:0000019'"
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
            # FIXME: impoartant to have the correct numbers of points
            tcsim = TimecourseSim(
                timecourses=[
                    Timecourse(
                        start=initial_time,
                        end=output_end_time,
                        steps=number_of_points-1,
                    ),
                ],
                time_offset=output_start_time
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
        """ Parse arbitrary task (repeated or simple, or simple repeated)."""
        # If no DataGenerator references the task, no execution is necessary
        dgs: List[libsedml.SedDataGenerator] = self.data_generators_for_task(sed_task)
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
        lines = []
        node_stack = Stack()
        tree_nodes = [n for n in task_tree_root]

        for kn, node in enumerate(tree_nodes):
            task_type = node.task.getTypeCode()

            # Create simulation for task
            if task_type == libsedml.SEDML_TASK:
                task = self._parse_simple_task(task_node=node)
                return task

            # Repeated tasks are multi-dimensional scans
            elif task_type == libsedml.SEDML_TASK_REPEATEDTASK:
                self._parse_repeated_task(node=node)

            elif task_type == libsedml.SEDML_TASK_SIMPLEREPEATEDTASK:
                self._parse_simple_repeated_task(node=node)

            elif task_type == libsedml.SEDML_TASK_PARAMETER_ESTIMATION:
                return sed_task

            else:
                raise ValueError(f"Unsupported task: {task_type}")

    def _parse_simple_task(self, task_node: TaskNode) -> Task:
        """Parse simple task"""
        sed_task: libsedml.SedTask = task_node.task
        model_id: str = sed_task.getModelReference()
        simulation_id: str = sed_task.getSimulationReference()
        return Task(model=model_id, simulation=simulation_id)

    def _parse_simple_repeated_task(self, node: TaskNode):

        raise NotImplementedError(
            f"Task type is not supported: {node.task.getTypeCode()}"
        )

    def parse_figure(self, sed_figure: libsedml.SedFigure) -> Figure:
        """Parse figure information."""
        figure = Figure(
            experiment=None,
            sid=sed_figure.getId(),
            num_rows=sed_figure.getNumRows() if sed_figure.isSetNumRows() else 1,
            num_cols=sed_figure.getNumCols() if sed_figure.isSetNumCols() else 1,
        )

        sed_subplot: libsedml.SedSubPlot
        for sed_subplot in sed_figure.getListOfSubPlots():
            sed_output = self.sed_doc.getOutput(sed_subplot.getPlot())
            if sed_output is None:
                raise ValueError(f"Plot could not be resolved. No output exists in "
                                 f"listOfOutputs for id='{sed_subplot.getPlot()}'")

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

            # add subplot
            figure.subplots.append(
                SubPlot(plot=plot, row=row, col=col, row_span=row_span, col_span=col_span)
            )

        return figure

    def parse_plot2d(self, sed_plot2d: libsedml.SedPlot2D) -> Plot:
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
        curves: List[Curve] = []
        sed_curve: libsedml.Curve
        for sed_curve in sed_plot2d.getListOfCurves():
            curves.append(self.parse_curve(sed_curve))
        plot.curves = curves
        return plot

    def parse_plot3d(self, sed_plot3d: libsedml.SedPlot3D) -> Plot:
        """ Parse Plot3D."""
        # FIXME: implement
        raise NotImplementedError

    def parse_report(self, sed_report: libsedml.SedReport) -> Dict[str, str]:
        """ Parse Report.

        Returns dictionary of label: dataGenerator.id mapping.
        """
        sed_dataset: libsedml.SedDataSet
        report: Dict[str, str] = {}
        for sed_dataset in sed_report.getListOfDataSets():
            sed_dg_id: str = sed_dataset.getDataReference()
            if self.sed_doc.getDataGenerator(sed_dg_id) is None:
                raise ValueError(f"Report '{sed_report.getId()}': Id of DataGenerator "
                                 f"does not exist '{sed_dg_id}'")
            if not sed_dataset.isSetLabel():
                logger.error(f"Required attribute label missing on DataSet in Report.")
                continue
            label = sed_dataset.getLabel()
            if label in report:
                logger.error(f"Duplicate label in report '{report.getId()}': '{label}'")

            report[label] = sed_dg_id
        return report

    def parse_axis(self, sed_axis: libsedml.SedAxis) -> Axis:
        """Parse axes information."""
        if sed_axis is None:
            axis = Axis()
        else:
            axis = Axis(
                label=sed_axis.getName() if sed_axis.isSetName else None,
                min=sed_axis.getMin() if sed_axis.isSetMin() else None,
                max=sed_axis.getMax() if sed_axis.isSetMax() else None,
                grid=sed_axis.getGrid() if sed_axis.isSetGrid() else False,
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

    def parse_curve(self, sed_curve: libsedml.SedCurve) -> Curve:
        """Parse curve."""
        x: Data
        y: Data
        xerr: Data
        yerr: Data

        sed_curve_type = sed_curve.getTypeCode()
        if sed_curve_type == libsedml.SEDML_OUTPUT_CURVE:

            curve_type: CurveType
            if sed_curve.getType() == libsedml.SEDML_CURVETYPE_POINTS:
                curve_type = CurveType.POINTS
            elif sed_curve.getType() == libsedml.SEDML_CURVETYPE_BAR:
                curve_type = CurveType.BAR
            elif sed_curve.getType() == libsedml.SEDML_CURVETYPE_BARSTACKED:
                curve_type = CurveType.BARSTACKED
            elif sed_curve.getType() == libsedml.SEDML_CURVETYPE_HORIZONTALBAR:
                curve_type = CurveType.HORIZONTALBAR
            elif sed_curve.getType() == libsedml.SEDML_CURVETYPE_HORIZONTALBARSTACKED:
                curve_type = CurveType.HORIZONTALBARSTACKED
            curve = Curve(
                sid=sed_curve.getId(),
                name=sed_curve.getName() if sed_curve.isSetName() else None,
                x=self.data_from_datagenerator(sed_curve.getXDataReference()),
                y=self.data_from_datagenerator(sed_curve.getYDataReference()),
                xerr=self.data_from_datagenerator(sed_curve.getXErrorUpper()),
                yerr=self.data_from_datagenerator(sed_curve.getYErrorUpper()),
                type=curve_type,
                order=sed_curve.getOrder() if sed_curve.isSetOrder() else None,
            )
            # parse yaxis
            yaxis = None
            if sed_curve.isSetYAxis():
                sed_yaxis: str = sed_curve.getYAxis()
                if sed_yaxis == "left":
                    yaxis = YAxisPosition.LEFT
                elif sed_yaxis == "right":
                    yaxis = YAxisPosition.RIGHT
                else:
                    raise ValueError(f"Unsupported yAxis on curve: {sed_yaxis}")
            curve.yaxis = yaxis

            # parse style
            if sed_curve.isSetStyle():
                # styles are already parsed, used the style
                style = self.styles[sed_curve.getStyle()]
            else:
                # default style
                style = Style(
                    line=Line(),
                    marker=Marker(),
                    fill=Fill(),
                )
            curve.style = style

            if not curve.name:
                curve.name = f"{curve.y.index} ~ {curve.x.index}"
        elif sed_curve_type == libsedml.SedShadedArea:
            # FIXME: support shaded area
            logger.error("ShadedArea is not supported.")

        return curve

    def parse_style(self, sed_style: Union[str, libsedml.SedStyle]) -> Optional[Style]:
        """Parse SED-ML style."""
        if not sed_style:
            return None

        # resolve style if string
        if isinstance(sed_style, str):
            sed_style: libsedml.SedStyle = self.sed_doc.getStyle(sed_style)

        # FIXME: get the complete style resolved from basestyle

        style = Style(
            sid=sed_style.getId(),
            name=sed_style.getName() if sed_style.isSetName() else None,
            base_style=self.parse_style(sed_style.getBaseStyle()) if sed_style.isSetBaseStyle() else None,
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

        line_style: Optional[LineStyle]
        if not sed_line.isSetStyle():
            line_style = None
        else:
            sed_line_style = sed_line.getStyle()
            if sed_line_style == libsedml.SEDML_LINETYPE_NONE:
                line_style = LineStyle.NONE
            elif sed_line_style == libsedml.SEDML_LINETYPE_SOLID:
                line_style = LineStyle.SOLID
            elif sed_line_style == libsedml.SEDML_LINETYPE_DASH:
                line_style = LineStyle.DASH
            elif sed_line_style == libsedml.SEDML_LINETYPE_DOT:
                line_style = LineStyle.DOT
            elif sed_line_style == libsedml.SEDML_LINETYPE_DASHDOT:
                line_style = LineStyle.DASHDOT
            elif sed_line_style == libsedml.SEDML_LINETYPE_DASHDOTDOT:
                line_style = LineStyle.DASHDOTDOT

        return Line(
            style=line_style,
            color=ColorType.parse_color(sed_line.getColor()) if sed_line.isSetColor() else None,
            thickness=sed_line.getThickness() if sed_line.isSetThickness() else None
        )

    def parse_marker(self, sed_marker: libsedml.SedMarker) -> Optional[Marker]:
        """Parse the line information."""
        if sed_marker is None:
            return None

        marker_style: Optional[MarkerStyle]
        if not sed_marker.isSetStyle():
            marker_style = None
        else:
            sed_marker_style = sed_marker.getStyle()
            if sed_marker_style == libsedml.SEDML_MARKERTYPE_NONE:
                marker_style = MarkerStyle.NONE
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_SQUARE:
                marker_style = MarkerStyle.SQUARE
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_CIRCLE:
                marker_style = MarkerStyle.CIRCLE
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_DIAMOND:
                marker_style = MarkerStyle.DIAMOND
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_XCROSS:
                marker_style = MarkerStyle.XCROSS
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_PLUS:
                marker_style = MarkerStyle.PLUS
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_STAR:
                marker_style = MarkerStyle.STAR
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_TRIANGLEUP:
                marker_style = MarkerStyle.TRIANGLEUP
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_TRIANGLEDOWN:
                marker_style = MarkerStyle.TRIANGLEDOWN
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_TRIANGLELEFT:
                marker_style = MarkerStyle.TRIANGLELEFT
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_TRIANGLERIGHT:
                marker_style = MarkerStyle.TRIANGLERIGHT
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_HDASH:
                marker_style = MarkerStyle.HDASH
            elif sed_marker_style == libsedml.SEDML_MARKERTYPE_VDASH:
                marker_style = MarkerStyle.VDASH

        marker = Marker(
            size=sed_marker.getSize() if sed_marker.isSetSize() else None,
            style=marker_style if sed_marker.isSetStyle() else None,
            fill=ColorType(sed_marker.getFill()) if sed_marker.isSetFill() else None,
            line_thickness=sed_marker.getLineThickness() if sed_marker.isSetLineThickness() else None,
            line_color=ColorType(sed_marker.getLineColor()) if sed_marker.isSetLineColor() else None,
        )

        return marker

    def parse_fill(self, sed_fill: libsedml.SedFill) -> Optional[Fill]:
        """Parse fill information."""
        if sed_fill is None:
            return None

        return Fill(
            color=ColorType.parse_color(sed_fill.getColor()) if sed_fill.isSetColor() else None,
            second_color=ColorType.parse_color(sed_fill.getSecondColor()) if sed_fill.isSetSecondColor() else None,
        )

    def data_from_datagenerator(self, sed_dg_ref: Optional[str]) -> Optional[Data]:
        """This must all be evaluated with actual data"""
        # FIXME: do all the math on the data-generator
        if not sed_dg_ref:
            return None

        sed_dg: libsedml.SedDataGenerator = self.sed_doc.getDataGenerator(sed_dg_ref)
        if sed_dg is None:
            raise ValueError(f"DataGenerator does not exist in listOfDataGenerators "
                             f"with id: '{sed_dg_ref}'")

        # TODO: Necessary to evaluate the math
        astnode: libsedml.ASTNode = sed_dg.getMath()
        function: str = libsedml.formulaToL3String(astnode)

        parameters: Dict[str, float] = {}
        sed_par: libsedml.SedParameter
        for sed_par in sed_dg.getListOfParameters():
            parameters[sed_par.getId()] = sed_par.getValue()

        variables: Dict[str, Data] = {}
        sed_var: libsedml.SedVariable
        for sed_var in sed_dg.getListOfVariables():
            task_id = sed_var.getTaskReference()
            if sed_var.isSetSymbol():
                index = "time"
            elif sed_var.isSetTarget():
                index = self.parse_xpath_target(sed_var.getTarget())
            d_var = Data(
                index=index,
                task=task_id
            )
            # register data
            self.data[d_var.sid] = d_var
            variables[sed_var.getId()] = d_var

        # FIXME: the simple math should not be evaluated via functions
        d = Data(
            index=sed_dg.getId(),  # FIXME: not sure about this
            function=function,
            variables=variables,
            parameters=parameters,
        )

        self.data[d.sid] = d
        return d

    def _parse_repeated_task(self, node: TaskNode):
        # repeated tasks will be translated into multidimensional scans
        raise NotImplementedError
        # TODO: implement

    def data_generators_for_task(
        self,
        sed_task: libsedml.SedTask,
    ) -> List[libsedml.SedDataGenerator]:
        """ Get the DataGenerators which reference the given task."""
        sed_dgs = []
        for (
            sed_dg
        ) in self.sed_doc.getListOfDataGenerators():  # type: libsedml.SedDataGenerator
            for var in sed_dg.getListOfVariables():  # type: libsedml.SedVariable
                if var.getTaskReference() == sed_task.getId():
                    sed_dgs.append(sed_dg)
                    # DataGenerator is added, no need to look at rest of variables
                    break
        return sed_dgs

    def selections_for_task(self, sed_task: libsedml.SedTask):
        """Populate variable lists from the data generators for the given task.

        These are the timeCourseSelections and steadyStateSelections
        in RoadRunner.

        Search all data generators for variables which have to be part of the simulation.
        """
        model_id = sed_task.getModelReference()
        selections = set()
        for (
            sed_dg
        ) in self.doc.getListOfDataGenerators():  # type: libsedml.SedDataGenerator
            for var in sed_dg.getListOfVariables():
                if var.getTaskReference() == sed_task.getId():
                    # FIXME: resolve with model
                    selection = SEDMLCodeFactory.selectionFromVariable(var, model_id)
                    expr = selection.id
                    if selection.style == "concentration":
                        expr = "[{}]".format(selection.id)
                    selections.add(expr)

        return selections

    @staticmethod
    def get_ordered_subtasks(sed_task: libsedml.SedTask) -> List[libsedml.SedTask]:
        """ Ordered list of subtasks for task."""
        subtasks = sed_task.getListOfSubTasks()
        subtask_order = [st.getOrder() for st in subtasks]

        # sort by order, if all subtasks have order (not required)
        if all(subtask_order) is not None:
            subtasks = [st for (stOrder, st) in sorted(zip(subtask_order, subtasks))]
        return subtasks
