"""Task trees of SED-ML and their translation into python code.

The code generation is the legacy approach of tellurium, it is kept for reference.
"""

import logging
import warnings

import libsedml
import numpy as np
from sbmlutils.converters.mathml import evaluableMathML

logger = logging.getLogger(__name__)


class TaskNode:
    """Tree implementation of task tree."""

    def __init__(self, task: libsedml.SedAbstractTask, depth: int):
        """Initialize the node for the task at the given depth."""
        self.task = task
        self.depth = depth
        self.children = []
        self.parent = None

    def add_child(self, obj):
        """Add a child node."""
        obj.parent = self
        self.children.append(obj)

    def is_leaf(self):
        """Check if the node has no children."""
        return len(self.children) == 0

    def __str__(self) -> str:
        """Render the subtree, one line per node."""
        lines = [f"<[{self.depth}] {self.task.getId()} ({self.task.getElementName()})>"]
        for child in self.children:
            child_str = child.__str__()
            lines.extend([f"\t{line}" for line in child_str.split("\n")])
        return "\n".join(lines)

    def info(self) -> str:
        """Render the node."""
        return f"<[{self.depth}] {self.task.getId()} ({self.task.getElementName()})>"

    def __iter__(self):
        """Depth-first iterator which yields TaskNodes."""
        yield self
        for child in self.children:
            yield from child

    def __repr__(self) -> str:
        """Render the node."""
        return self.info()


class Stack:
    """Stack implementation for nodes."""

    def __init__(self):
        """Initialize the empty stack."""
        self.items = []

    def isEmpty(self):
        """Check if the stack is empty."""
        return self.items == []

    def push(self, item):
        """Push an item on the stack."""
        self.items.append(item)

    def pop(self):
        """Pop the top item."""
        return self.items.pop()

    def peek(self):
        """Return the top item without removing it."""
        return self.items[len(self.items) - 1]

    def size(self):
        """Number of items on the stack."""
        return len(self.items)

    def __str__(self):
        """Render the stack."""
        return "stack: " + str([item.info() for item in self.items])


class TaskTree:
    """Tree of the tasks of a SED-ML document."""

    @staticmethod
    def from_sedml_task(
        sed_task: libsedml.SedDocument, root_task: libsedml.SedAbstractTask
    ) -> TaskNode:
        """Creates task tree for given SedTask.

        The task tree is used to resolve the order of all simulations.
        """

        def add_children(node):
            """Adds task children to given node."""
            typeCode = node.task.getTypeCode()
            if typeCode == libsedml.SEDML_TASK:
                return  # no children
            if typeCode == libsedml.SEDML_TASK_REPEATEDTASK:
                # add the ordered list of subtasks as children
                subtasks = TaskTree.get_ordered_subtasks(node.task)
                for st in subtasks:
                    # get real task for subtask
                    t = sed_task.getTask(st.getTask())
                    child = TaskNode(t, depth=node.depth + 1)
                    node.add_child(child)
                    # recursive adding of children
                    add_children(child)
            elif typeCode == libsedml.SEDML_TASK_PARAMETER_ESTIMATION:
                logger.warning("Skipping parameter estimation task.")
            else:
                raise OSError("Unsupported task type: {node.task_id.getElementName()}")

        # create root
        root = TaskNode(root_task, depth=0)
        # recursive adding of children
        add_children(root)
        return root

    @staticmethod
    def get_ordered_subtasks(
        repeated_task: libsedml.SedRepeatedTask,
    ) -> list[libsedml.SedSubTask]:
        """Ordered list of subtasks for repeated task."""
        subtasks: libsedml.SedListOfSubTasks = repeated_task.getListOfSubTasks()
        subtaskOrder: list[int] = [st.getOrder() for st in subtasks]
        # sort by order, if all subtasks have order (not required)
        if all(subtaskOrder) is not None:
            subtasks = [
                st
                for (stOrder, st) in sorted(zip(subtaskOrder, subtasks, strict=False))
            ]
        return subtasks


# -------------------------------------------------------------------------------------


class SEDMLCodeFactory:
    """Placeholder of the code factory, not implemented."""


class Test:
    """Translation of tasks into python code, kept for reference."""

    @staticmethod
    def simpleTaskToPython(doc, node: TaskNode):
        """Creates the simulation python code for a given taskNode.

        The taskNodes are required to handle the relationships between
        RepeatedTasks, SubTasks and SimpleTasks (Task).

        :param doc: sedml document
        :type doc: SEDDocument
        :param node: taskNode of the current task
        :type node: TaskNode
        :return:
        :rtype:
        """
        lines = []
        task = node.task
        lines.append(f"# Task: <{task.getId()}>")
        lines.append(f"{task.getId()} = [None]")

        mid = task.getModelReference()
        sid = task.getSimulationReference()
        simulation = doc.getSimulation(sid)

        simType = simulation.getTypeCode()
        algorithm = simulation.getAlgorithm()
        if algorithm is None:
            warnings.warn(
                "Algorithm missing on simulation, defaulting to 'cvode: KISAO:0000019'",
                stacklevel=2,
            )
            algorithm = simulation.createAlgorithm()
            algorithm.setKisaoID("KISAO:0000019")
        kisao = algorithm.getKisaoID()

        # is supported algorithm
        if not SEDMLCodeFactory.is_supported_algorithm_for_simulation_type(
            kisao=kisao, sim_type=simType
        ):
            warnings.warn(
                f"Algorithm {kisao} unsupported for simulation {simulation.getId()} type {simType} in task {task.getId()}",
                stacklevel=2,
            )
            lines.append(
                f"# Unsupported Algorithm {kisao} for SimulationType {simulation.getElementName()}"
            )
            return lines

        # set integrator/solver
        integratorName = SEDMLCodeFactory.integrator_from_kisao(kisao)
        if not integratorName:
            warnings.warn(
                f"No integrator exists for {kisao} in roadrunner", stacklevel=2
            )
            return lines

        if simType is libsedml.SEDML_SIMULATION_STEADYSTATE:
            lines.append(f"{mid}.setSteadyStateSolver('{integratorName}')")
        else:
            lines.append(f"{mid}.setIntegrator('{integratorName}')")

        # use fixed step by default for stochastic sims
        if integratorName == "gillespie":
            lines.append(
                "{}.integrator.setValue('{}', {})".format(
                    mid, "variable_step_size", False
                )
            )

        if kisao == "KISAO:0000288":  # BDF
            lines.append("{}.integrator.setValue('{}', {})".format(mid, "stiff", True))
        elif kisao == "KISAO:0000280":  # Adams-Moulton
            lines.append("{}.integrator.setValue('{}', {})".format(mid, "stiff", False))

        # integrator/solver settings (AlgorithmParameters)
        for par in algorithm.getListOfAlgorithmParameters():
            pkey = SEDMLCodeFactory.algorithm_parameter_to_parameter_key(par)
            # only set supported algorithm paramters
            if pkey:
                value = f"'{pkey.value}'" if pkey.dtype is str else pkey.value

                if value == "inf" or pkey.value == float("inf"):
                    value = "float('inf')"
                else:
                    pass

                if simType is libsedml.SEDML_SIMULATION_STEADYSTATE:
                    lines.append(
                        f"{mid}.steadyStateSolver.setValue('{pkey.key}', {value})"
                    )
                else:
                    lines.append(f"{mid}.integrator.setValue('{pkey.key}', {value})")

        if simType is libsedml.SEDML_SIMULATION_STEADYSTATE:
            lines.append(
                f"if {mid}.conservedMoietyAnalysis == False: {mid}.conservedMoietyAnalysis = True"
            )
        else:
            lines.append(
                f"if {mid}.conservedMoietyAnalysis == True: {mid}.conservedMoietyAnalysis = False"
            )

        # get parents
        parents = []
        parent = node.parent
        while parent is not None:
            parents.append(parent)
            parent = parent.parent

        # <selections> of all parents
        # ---------------------------
        selections = SEDMLCodeFactory.selections_for_task(doc=doc, sed_task=node.task)
        for p in parents:
            selections.update(
                SEDMLCodeFactory.selections_for_task(doc=doc, sed_task=p.task_id)
            )

        # <setValues> of all parents
        # ---------------------------
        # apply changes based on current variables, parameters and range variables
        for parent in reversed(parents):
            rangeId = parent.task_id.getRangeId()
            helperRanges = {}
            for r in parent.task_id.getListOfRanges():
                if r.getId() != rangeId:
                    helperRanges[r.getId()] = r

            for setValue in parent.task_id.getListOfTaskChanges():
                variables = {}
                # range variables
                variables[rangeId] = f"__value__{rangeId}"
                for key in helperRanges:
                    variables[key] = f"__value__{key}"
                # parameters
                for par in setValue.getListOfParameters():
                    variables[par.getId()] = par.getValue()
                for var in setValue.getListOfVariables():
                    vid = var.getId()
                    mid = var.getModelReference()
                    selection = SEDMLCodeFactory.selectionFromVariable(var, mid)
                    expr = selection.id
                    if selection.type == "concentration":
                        expr = f"init([{selection.id}])"
                    elif selection.type == "amount":
                        expr = f"init({selection.id})"

                    # create variable
                    lines.append(f"__value__{vid} = {mid}['{expr}']")
                    # variable for replacement
                    variables[vid] = f"__value__{vid}"

                # value is calculated with the current state of model
                lines.append(
                    SEDMLCodeFactory.targetToPython(
                        xpath=setValue.getTarget(),
                        value=evaluableMathML(setValue.getMath(), variables=variables),
                        modelId=setValue.getModelReference(),
                    )
                )

        # handle result variable
        resultVariable = f"{task.getId()}[0]"

        # -------------------------------------------------------------------------
        # <UNIFORM TIMECOURSE>
        # -------------------------------------------------------------------------
        if simType == libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE:
            lines.append(f"{mid}.timeCourseSelections = {list(selections)}")

            initialTime = simulation.getInitialTime()
            outputStartTime = simulation.getOutputStartTime()
            outputEndTime = simulation.getOutputEndTime()
            numberOfPoints = simulation.getNumberOfPoints()

            # reset before simulation (see https://github.com/sys-bio/tellurium/issues/193)
            lines.append(f"{mid}.reset()")

            # throw some points away
            if abs(outputStartTime - initialTime) > 1e-6:
                lines.append(
                    f"{mid}.simulate(start={initialTime}, end={outputStartTime}, points=2)"
                )
            # real simulation
            lines.append(
                f"{resultVariable} = {mid}.simulate(start={outputStartTime}, end={outputEndTime}, steps={numberOfPoints})"
            )
        # -------------------------------------------------------------------------
        # <ONESTEP>
        # -------------------------------------------------------------------------
        elif simType == libsedml.SEDML_SIMULATION_ONESTEP:
            lines.append(f"{mid}.timeCourseSelections = {list(selections)}")
            step = simulation.getStep()
            lines.append(
                f"{resultVariable} = {mid}.simulate(start={0.0}, end={step}, points=2)"
            )

        # -------------------------------------------------------------------------
        # <STEADY STATE>
        # -------------------------------------------------------------------------
        elif simType == libsedml.SEDML_SIMULATION_STEADYSTATE:
            lines.append(
                "{}.steadyStateSolver.setValue('{}', {})".format(
                    mid, "allow_presimulation", False
                )
            )
            lines.append(f"{mid}.steadyStateSelections = {list(selections)}")
            lines.append(
                f"{mid}.simulate()"
            )  # for stability of the steady state solver
            lines.append(f"{resultVariable} = {mid}.steadyStateNamedArray()")
            # no need to turn this off because it will be checked before the next simulation
            # lines.append("{}.conservedMoietyAnalysis = False".format(mid))

        # -------------------------------------------------------------------------
        # <OTHER>
        # -------------------------------------------------------------------------
        else:
            lines.append(f"# Unsupported simulation: {simType}")

        return lines

    @staticmethod
    def repeatedTaskToPython(doc, node):
        """Create python for RepeatedTask.

        Must create
        - the ranges (Ranges)
        - apply all changes (SetValues)
        """
        # storage of results
        task = node.task_id
        lines = ["", f"{task.getId()} = []"]

        # <Range Definition>
        # master range
        rangeId = task.getRangeId()
        masterRange = task.getRange(rangeId)
        if masterRange.getTypeCode() == libsedml.SEDML_RANGE_UNIFORMRANGE:
            lines.extend(SEDMLCodeFactory.uniformRangeToPython(masterRange))
        elif masterRange.getTypeCode() == libsedml.SEDML_RANGE_VECTORRANGE:
            lines.extend(SEDMLCodeFactory.vectorRangeToPython(masterRange))
        elif masterRange.getTypeCode() == libsedml.SEDML_RANGE_FUNCTIONALRANGE:
            warnings.warn(
                "FunctionalRange for master range not supported in task.", stacklevel=2
            )
        # lock-in ranges
        for r in task.getListOfRanges():
            if r.getId() != rangeId:
                if r.getTypeCode() == libsedml.SEDML_RANGE_UNIFORMRANGE:
                    lines.extend(SEDMLCodeFactory.uniformRangeToPython(r))
                elif r.getTypeCode() == libsedml.SEDML_RANGE_VECTORRANGE:
                    lines.extend(SEDMLCodeFactory.vectorRangeToPython(r))

        # <Range Iteration>
        # iterate master range
        lines.append(
            f"for __k__{rangeId}, __value__{rangeId} in enumerate(__range__{rangeId}):"
        )

        # Everything from now on is done in every iteration of the range
        # We have to collect & intent all lines in the loop)
        forLines = []

        # definition of lock-in ranges
        helperRanges = {}
        for r in task.getListOfRanges():
            if r.getId() != rangeId:
                helperRanges[r.getId()] = r
                if r.getTypeCode() in [
                    libsedml.SEDML_RANGE_UNIFORMRANGE,
                    libsedml.SEDML_RANGE_VECTORRANGE,
                ]:
                    forLines.append(
                        f"__value__{r.getId()} = __range__{r.getId()}[__k__{rangeId}]"
                    )

                # <functional range>
                if r.getTypeCode() == libsedml.SEDML_RANGE_FUNCTIONALRANGE:
                    variables = {}
                    # range variables
                    variables[rangeId] = f"__value__{rangeId}"
                    for key in helperRanges:
                        variables[key] = f"__value__{key}"
                    # parameters
                    for par in r.getListOfParameters():
                        variables[par.getId()] = par.getValue()
                    for var in r.getListOfVariables():
                        vid = var.getId()
                        mid = var.getModelReference()
                        selection = SEDMLCodeFactory.selectionFromVariable(var, mid)
                        expr = selection.id
                        if selection.type == "concentration":
                            expr = f"[{selection.id}]"
                        lines.append(f"__value__{vid} = {mid}['{expr}']")
                        variables[vid] = f"__value__{vid}"

                    # value is calculated with the current state of model
                    value = evaluableMathML(r.getMath(), variables=variables)
                    forLines.append(f"__value__{r.getId()} = {value}")

        # <resetModels>
        # models to reset via task tree below node
        mids = set()
        for child in node:
            t = child.task_id
            if t.getTypeCode() == libsedml.SEDML_TASK:
                mids.add(t.getModelReference())
        # reset models referenced in tree below task
        for mid in mids:
            if task.getResetModel():
                # reset before every iteration
                forLines.append(f"{mid}.reset()")
            else:
                # reset before first iteration
                forLines.append(f"if __k__{rangeId} == 0:")
                forLines.append(f"    {mid}.reset()")

        # add lines
        lines.extend("    " + line for line in forLines)

        return lines

    @staticmethod
    def uniformRangeToPython(r):
        """Create python lines for a uniform range."""
        lines = []
        rId = r.getId()
        rStart = r.getStart()
        rEnd = r.getEnd()
        rPoints = r.getNumberOfPoints() + 1  # One point more than number of points
        rType = r.getType()
        if rType in ["Linear", "linear"]:
            lines.append(
                f"__range__{rId} = np.linspace(start={rStart}, stop={rEnd}, num={rPoints})"
            )
        elif rType in ["Log", "log"]:
            lines.append(
                f"__range__{rId} = np.logspace(start={rStart}, stop={rEnd}, num={rPoints})"
            )
        else:
            warnings.warn(
                f"Unsupported range type in UniformRange: {rType}", stacklevel=2
            )
        return lines

    @staticmethod
    def vectorRangeToPython(r):
        """Create python lines for a vector range."""
        lines = []
        __range = np.zeros(shape=[r.getNumValues()])
        for k, v in enumerate(r.getValues()):
            __range[k] = v
        lines.append(f"__range__{r.getId()} = {list(__range)}")
        return lines
