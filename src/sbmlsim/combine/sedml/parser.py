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
from typing import Dict, List, Optional, Union

import libsedml
import numpy as np

from sbmlsim.combine.sedml.data import DataDescriptionParser
from sbmlsim.combine.sedml.io import read_sedml
from sbmlsim.combine.sedml.kisao import is_supported_algorithm_for_simulation_type
from sbmlsim.combine.sedml.task import Stack, TaskNode, TaskTree
from sbmlsim.data import Data, DataSet
from sbmlsim.experiment import ExperimentDict, SimulationExperiment
from sbmlsim.model import model_resources
from sbmlsim.model.model import AbstractModel
from sbmlsim.plot import Axis, Curve, Figure, Plot
from sbmlsim.simulation import AbstractSim, ScanSim, Timecourse, TimecourseSim
from sbmlsim.task import Task
from sbmlsim.units import UnitRegistry


logger = logging.getLogger(__file__)
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
    """Parsing SED-ML in internal format."""

    def __init__(self, sed_doc: libsedml.SedDocument, working_dir: Path):
        """Parses information from SedDocument."""
        self.sed_doc: libsedml.SedDocument = sed_doc
        self.working_dir = working_dir

        # unit registry to handle units throughout the simulation
        self.ureg = UnitRegistry()

        # Reference to the experiment class
        self.exp_class = None

        # --- Models ---
        self.models = {}

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
        print(f"models: {self.models}")

        # --- DataDescriptions ---
        self.data_descriptions = {}
        sed_dd: libsedml.SedDataDescription
        for sed_dd in sed_doc.getListOfDataDescriptions():
            did = sed_dd.getId()
            self.data_descriptions[did] = DataDescriptionParser.parse(
                sed_dd, self.working_dir
            )
        print(f"data_descriptions: {self.data_descriptions}")

        # --- Simulations ---
        self.simulations: Dict[str, AbstractSim] = {}
        sed_sim: libsedml.SedSimulation
        for sed_sim in sed_doc.getListOfSimulations():
            self.simulations[sed_sim.getId()] = self.parse_simulation(sed_sim)
        print(f"simulations: {self.simulations}")

        # --- Tasks ---
        self.tasks: Dict[str, Task] = {}
        sed_task: libsedml.SedTask
        for sed_task in sed_doc.getListOfTasks():
            self.tasks[sed_task.getId()] = self.parse_task(sed_task)
        print(f"tasks: {self.tasks}")

        # --- Figures/Reports ---
        self.figures: Dict[str, Figure] = {}
        sed_output: libsedml.SedOutput
        for sed_output in sed_doc.getListOfOutputs():

            type_code = sed_output.getTypeCode()
            if type_code in [libsedml.SEDML_FIGURE, libsedml.SEDML_OUTPUT_PLOT2D]:
                self.figures[sed_output.getId()] = self.parse_figure(sed_output)
            elif type_code == libsedml.SEDML_OUTPUT_REPORT:
                # FIXME: implement
                logger.error("Output report not implemented.")
        print(f"figures: {self.figures}")

        # Create the experiment object
        def f_models(obj) -> Dict[str, AbstractModel]:
            return ExperimentDict(self.models)

        def f_datasets(obj) -> Dict[str, DataSet]:
            """Dataset definition (experimental data)."""

            # FIXME: convert to DataSets & add units
            return ExperimentDict(self.data_descriptions)

        def f_simulations(obj) -> Dict[str, AbstractSim]:
            return ExperimentDict(self.simulations)

        def f_tasks(obj) -> Dict[str, Task]:
            return ExperimentDict(self.tasks)

        def f_figures(obj) -> Dict[str, Figure]:
            return ExperimentDict(self.figures)

        self.exp_class = type(
            "SedmlExperiment",
            (SimulationExperiment,),
            {
                "models": f_models,
                "datasets": f_datasets,
                "simulations": f_simulations,
                "tasks": f_tasks,
                "figures": f_figures,
            },
        )
        fig: Figure

        self.experiment: SimulationExperiment = self.exp_class()
        self.experiment.initialize()

        # self.exp_class._data = None  # FIXME hack
        for fig in self.figures.values():
            print(fig)
            fig.experiment = self.experiment

            # this must happen automatically
            for plot in fig.plots:
                for curve in plot.curves:
                    if curve.x:
                        curve.x.experiment = self.experiment
                        curve.x._register_data()
                    if curve.y:
                        curve.y.experiment = self.experiment
                        curve.y._register_data()
                    # FIXME: also for xerr and yerr

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
        :param source:
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
        """Parse simulation information."""
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
        dgs = self.data_generators_for_task(sed_task)
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

        print("tree_nodes", tree_nodes)
        print(task_tree_root.info())

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

            else:
                logger.error("Unsupported task: {}".format(task_type))

    def _parse_simple_task(self, task_node: TaskNode) -> Task:
        """Parse simple task"""
        sed_task: libsedml.SedTask = task_node.task
        model_id: str = sed_task.getModelReference()
        simulation_id: str = sed_task.getSimulationReference()
        return Task(model=model_id, simulation=simulation_id)

    def _parse_simple_repeated_task(self, node: TaskNode):
        print("parse simple repeated task")
        raise NotImplementedError(
            f"Task type is not supported: {node.task.getTypeCode()}"
        )
        # TODO: implement

        # for ksub, subtask in enumerate(subtasks):
        #     t = doc.getTask(subtask.getTask())
        #
        #     resultVariable = "__subtask__".format(t.getId())
        #     selections = SEDMLCodeFactory.selectionsForTask(doc=doc, task=task)
        #     if t.getTypeCode() == libsedml.SEDML_TASK:
        #         forLines.extend(
        #             SEDMLCodeFactory.subtaskToPython(
        #                 doc,
        #                 task=t,
        #                 selections=selections,
        #                 resultVariable=resultVariable,
        #             )
        #         )
        #         forLines.append("{}.extend([__subtask__])".format(task.getId()))
        #
        #     elif t.getTypeCode() == libsedml.SEDML_TASK_REPEATEDTASK:
        #         forLines.extend(SEDMLCodeFactory.repeatedTaskToPython(doc, task=t))
        #         forLines.append("{}.extend({})".format(task.getId(), t.getId()))

    def parse_figure(self, sed_output: libsedml.SedOutput) -> Figure:
        """Parse simulation information."""
        type_code = sed_output.getTypeCode()
        if type_code == libsedml.SEDML_OUTPUT_PLOT2D:
            sed_plot2d: libsedml.SedPlot = sed_output
            plots = [self.parse_plot2d(sed_plot2d)]
            f = Figure(
                experiment=None,
                sid=sed_plot2d.getId(),
                num_rows=1,
                num_cols=1,
            )
            f.add_plots(plots)
            return f
        elif type_code == libsedml.SEDML_FIGURE:
            raise NotImplementedError

    def parse_plot2d(self, sed_plot2d: libsedml.SedPlot2D) -> Plot:
        plot = Plot(
            sid=sed_plot2d.getId(),
            name=sed_plot2d.getName(),
            legend=sed_plot2d.getLegend(),
        )
        # axis
        plot.xaxis = self.parse_axis(sed_plot2d.getXAxis())
        plot.yaxis = self.parse_axis(sed_plot2d.getYAxis())

        # curves
        curves: List[Curve] = []
        sed_curve: libsedml.Curve
        for sed_curve in sed_plot2d.getListOfCurves():
            curves.append(self.parse_curve(sed_curve))
        plot.curves = curves
        return plot

    def parse_axis(self, sed_axis: libsedml.SedAxis) -> Axis:
        # FIXME: support style on axis
        if sed_axis is None:
            axis = Axis()
        else:
            axis = Axis(
                label=sed_axis.getName(),
                scale=sed_axis.getType(),
                min=sed_axis.getMin(),
                max=sed_axis.getMax(),
                grid=sed_axis.getGrid(),
            )
        return axis

    def parse_curve(self, sed_curve: libsedml.SedCurve) -> Curve:
        x: Data
        y: Data
        xerr: Data
        yerr: Data

        curve = Curve(
            x=self.data_from_datagenerator(sed_curve.getXDataReference()),
            y=self.data_from_datagenerator(sed_curve.getYDataReference()),
            # FIXME: handle errorbars via lower and upper
            xerr=self.data_from_datagenerator(sed_curve.getXErrorLower()),
            yerr=self.data_from_datagenerator(sed_curve.getYErrorLower()),
        )
        # FIXME: support yaxis
        # FIXME: support type
        # FIXME: parse style
        return curve

    def data_from_datagenerator(self, sed_dg_ref: Optional[str]) -> Optional[Data]:
        """This must all be evaluated with actual data"""
        # FIXME: do all the math on the data-generator
        if not sed_dg_ref:
            return None
        sed_dg: libsedml.SedDataGenerator = self.sed_doc.getDataGenerator(sed_dg_ref)
        print("DataGenerator", sed_dg)

        # sed_dg.getListOfVariables()
        # sed_dg.getMath()
        sed_var: libsedml.SedVariable = sed_dg.getVariable(0)
        task_id = sed_var.getTaskReference()

        sed_symbol = sed_var.isSetSymbol()
        if sed_var.isSetSymbol():
            # FIXME: check symbol
            index = "time"
        elif sed_var.isSetTarget():
            index = self.parse_xpath_target(sed_var.getTarget())

        return Data(experiment=self.exp_class, index=index, task=task_id)

    def _parse_repeated_task(self, node: TaskNode):
        print("repeated task")
        # repeated tasks will be translated into multidimensional scans
        raise NotImplementedError
        # TODO: implement

    def data_generators_for_task(
        self,
        sed_task: libsedml.SedTask,
    ) -> List[libsedml.SedDataGenerator]:
        """Get the DataGenerators which reference the given task."""
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
                    if selection.type == "concentration":
                        expr = "[{}]".format(selection.id)
                    selections.add(expr)

        return selections

    @staticmethod
    def get_ordered_subtasks(sed_task: libsedml.SedTask) -> List[libsedml.SedTask]:
        """Ordered list of subtasks for task."""
        subtasks = sed_task.getListOfSubTasks()
        subtask_order = [st.getOrder() for st in subtasks]

        # sort by order, if all subtasks have order (not required)
        if all(subtask_order) is not None:
            subtasks = [st for (stOrder, st) in sorted(zip(subtask_order, subtasks))]
        return subtasks
