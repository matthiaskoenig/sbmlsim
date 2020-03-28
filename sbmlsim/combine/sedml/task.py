class Task(object):



    @staticmethod
    def selectionFromVariable(var, model):
        """ Resolves the selection for the given variable.

        First checks if the variable is a symbol and returns the symbol.
        If no symbol is set the xpath of the target is resolved
        and used in the selection

        :param var: variable to resolve
        :type var: SedVariable
        :return: a single selection
        :rtype: Selection (namedtuple: id type)
        """
        Selection = namedtuple('Selection', 'id type')

        # parse symbol expression
        if var.isSetSymbol():
            cvs = var.getSymbol()
            astr = cvs.rsplit("symbol:")
            sid = astr[1]
            return Selection(sid, 'symbol')
        # use xpath
        elif var.isSetTarget():
            xpath = var.getTarget()
            target = SEDMLParser._resolveXPath(xpath, model)
            return Selection(target.id, target.type)

        else:
            warnings.warn(f"Unrecognized Selection in variable: {var}")
            return None



    @staticmethod
    def parse_task(doc: libsedml.SedDocument, sed_task: libsedml.SedAbstractTask):
        """ Create python for arbitrary task (repeated or simple)."""
        # If no DataGenerator references the task, no execution is necessary
        dgs = SEDMLParser.get_data_generators_for_tasks(doc, sed_task)
        if len(dgs) == 0:
            logger.warning(f"Task '{sed_task.getId()}' is not part of any DataGenerator. "
                           f"Task will not be executed.")

        # tasks contain other subtasks, which can contain subtasks. This
        # results in a tree of task dependencies where the
        # simple tasks are the node leaves. These tree has to be resolved to
        # generate code for more complex task dependencies.

        # resolve task tree (order & dependency of tasks) & generate code
        taskTree = SEDMLParser.createTaskTree(doc, rootTask=sed_task)
        return SEDMLParser.parse_task_tree(doc, tree=taskTree)

    @staticmethod
    def get_data_generators_for_tasks(doc: libsedml.SedDocument, sed_task: libsedml.SedTask) -> List[libsedml.SedDataGenerator]:
        """ Get the DataGenerators which reference the given task."""
        dgs = []
        for dg in doc.getListOfDataGenerators():  # type: libsedml.SedDataGenerator
            for var in dg.getListOfVariables():  # type: libsedml.SedVariable
                if var.getTaskReference() == sed_task.getId():
                    dgs.append(dg)
                    break  # the DataGenerator is added, no need to look at rest of variables
        return dgs

    class TaskNode(object):
        """ Tree implementation of task tree. """
        def __init__(self, task, depth):
            self.task = task
            self.depth = depth
            self.children = []
            self.parent = None

        def add_child(self, obj):
            obj.parent = self
            self.children.append(obj)

        def is_leaf(self):
            return len(self.children) == 0

        def __str__(self):
            lines = ["<[{}] {} ({})>".format(self.depth, self.task.getId(), self.task.getElementName())]
            for child in self.children:
                child_str = child.__str__()
                lines.extend(["\t{}".format(line) for line in child_str.split('\n')])
            return "\n".join(lines)

        def info(self):
            return "<[{}] {} ({})>".format(self.depth, self.task.getId(), self.task.getElementName())

        def __iter__(self):
            """ Depth-first iterator which yields TaskNodes."""
            yield self
            for child in self.children:
                for node in child:
                    yield node

    class Stack(object):
        """ Stack implementation for nodes."""
        def __init__(self):
            self.items = []

        def isEmpty(self):
            return self.items == []

        def push(self, item):
            self.items.append(item)

        def pop(self):
            return self.items.pop()

        def peek(self):
            return self.items[len(self.items)-1]

        def size(self):
            return len(self.items)

        def __str__(self):
            return "stack: " + str([item.info() for item in self.items])

    @staticmethod
    def createTaskTree(doc: libsedml.SedDocument, rootTask: libsedml.SedAbstractTask) -> TaskNode:
        """ Creates the task tree.
        The task tree is used to resolve the order of all simulations.
        """
        def add_children(node):
            typeCode = node.task.getTypeCode()
            if typeCode == libsedml.SEDML_TASK:
                return  # no children
            elif typeCode == libsedml.SEDML_TASK_REPEATEDTASK:
                # add the ordered list of subtasks as children
                subtasks = SEDMLParser.getOrderedSubtasks(node.task)
                for st in subtasks:
                    # get real task for subtask
                    t = doc.getTask(st.getTask())
                    child = SEDMLParser.TaskNode(t, depth=node.depth+1)
                    node.add_child(child)
                    # recursive adding of children
                    add_children(child)
            else:
                raise IOError('Unsupported task type: {}'.format(node.task.getElementName()))

        # create root
        root = SEDMLParser.TaskNode(rootTask, depth=0)
        # recursive adding of children
        add_children(root)
        return root

    @staticmethod
    def getOrderedSubtasks(task):
        """ Ordered list of subtasks for task."""
        subtasks = task.getListOfSubTasks()
        subtaskOrder = [st.getOrder() for st in subtasks]
        # sort by order, if all subtasks have order (not required)
        if all(subtaskOrder) != None:
            subtasks = [st for (stOrder, st) in sorted(zip(subtaskOrder, subtasks))]
        return subtasks

    @staticmethod
    def parse_task_tree(doc: libsedml.SedDocument, tree: TaskNode):
        """ Python code generation from task tree. """

        # go forward through task tree
        lines = []
        nodeStack = SEDMLParser.Stack()
        treeNodes = [n for n in tree]

        # iterate over the tree
        for kn, node in enumerate(treeNodes):
            task_type = node.task.getTypeCode()

            # Create information for task
            # We are going down in the tree
            if task_type == libsedml.SEDML_TASK_REPEATEDTASK:
                taskLines = SEDMLCodeFactory.repeatedTaskToPython(doc, node=node)

            elif task_type == libsedml.SEDML_TASK:
                tid = node.task.getId()
                taskLines = SEDMLCodeFactory.simpleTaskToPython(doc=doc, node=node)
            else:
                lines.append("# Unsupported task: {}".format(task_type))
                warnings.warn("Unsupported task: {}".format(task_type))

            lines.extend(["    "*node.depth + line for line in taskLines])

            # Collect information
            # We have to go back up
            # Look at next node in the treeNodes (this is the next one to write)
            if kn == (len(treeNodes)-1):
                nextNode = None
            else:
                nextNode = treeNodes[kn+1]

            # The next node is further up in the tree, or there is no next node
            # and still nodes on the stack
            if (nextNode is None) or (nextNode.depth < node.depth):

                # necessary to pop nodes from the stack and close the code
                test = True
                while test is True:
                    # stack is empty
                    if nodeStack.size() == 0:
                        test = False
                        continue
                    # try to pop next one
                    peek = nodeStack.peek()
                    if (nextNode is None) or (peek.depth > nextNode.depth):
                        # TODO: reset evaluation has to be defined here
                        # determine if it's steady state
                        # if taskType == libsedml.SEDML_TASK_REPEATEDTASK:
                        # print('task {}'.format(node.task.getId()))
                        # print('  peek {}'.format(peek.task.getId()))
                        if node.task.getTypeCode() == libsedml.SEDML_TASK_REPEATEDTASK:
                        # if peek.task.getTypeCode() == libsedml.SEDML_TASK_REPEATEDTASK:
                            # sid = task.getSimulationReference()
                            # simulation = doc.getSimulation(sid)
                            # simType = simulation.getTypeCode()
                            # if simType is libsedml.SEDML_SIMULATION_STEADYSTATE:
                            terminator = 'terminate_trace({})'.format(node.task.getId())
                        else:
                            terminator = '{}'.format(node.task.getId())

                        lines.extend([
                            "",
                            # "    "*node.depth + "{}.extend({})".format(peek.task.getId(), terminator),
                            "    " * node.depth + "{}.extend({})".format(peek.task.getId(), node.task.getId()),
                        ])
                        node = nodeStack.pop()

                    else:
                        test = False
            else:
                # we are going done or next subtask -> put node on stack
                nodeStack.push(node)

        return "\n".join(lines)


    @staticmethod
    def simpleTaskToPython(doc, node: TaskNode):
        """ Creates the simulation python code for a given taskNode.

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
        lines.append("# Task: <{}>".format(task.getId()))
        lines.append("{} = [None]".format(task.getId()))

        mid = task.getModelReference()
        sid = task.getSimulationReference()
        simulation = doc.getSimulation(sid)

        simType = simulation.getTypeCode()
        algorithm = simulation.getAlgorithm()
        if algorithm is None:
            warnings.warn("Algorithm missing on simulation, defaulting to 'cvode: KISAO:0000019'")
            algorithm = simulation.createAlgorithm()
            algorithm.setKisaoID("KISAO:0000019")
        kisao = algorithm.getKisaoID()

        # is supported algorithm
        if not SEDMLCodeFactory.isSupportedAlgorithmForSimulationType(kisao=kisao, simType=simType):
            warnings.warn("Algorithm {} unsupported for simulation {} type {} in task {}".format(kisao, simulation.getId(), simType, task.getId()))
            lines.append("# Unsupported Algorithm {} for SimulationType {}".format(kisao, simulation.getElementName()))
            return lines

        # set integrator/solver
        integratorName = SEDMLCodeFactory.getIntegratorNameForKisaoID(kisao)
        if not integratorName:
            warnings.warn("No integrator exists for {} in roadrunner".format(kisao))
            return lines

        if simType is libsedml.SEDML_SIMULATION_STEADYSTATE:
            lines.append("{}.setSteadyStateSolver('{}')".format(mid, integratorName))
        else:
            lines.append("{}.setIntegrator('{}')".format(mid, integratorName))

        # use fixed step by default for stochastic sims
        if integratorName == 'gillespie':
            lines.append("{}.integrator.setValue('{}', {})".format(mid, 'variable_step_size', False))

        if kisao == "KISAO:0000288":  # BDF
            lines.append("{}.integrator.setValue('{}', {})".format(mid, 'stiff', True))
        elif kisao == "KISAO:0000280":  # Adams-Moulton
            lines.append("{}.integrator.setValue('{}', {})".format(mid, 'stiff', False))

        # integrator/solver settings (AlgorithmParameters)
        for par in algorithm.getListOfAlgorithmParameters():
            pkey = SEDMLCodeFactory.algorithmParameterToParameterKey(par)
            # only set supported algorithm paramters
            if pkey:
                if pkey.dtype is str:
                    value = "'{}'".format(pkey.value)
                else:
                    value = pkey.value

                if value == str('inf') or pkey.value == float('inf'):
                    value = "float('inf')"
                else:
                    pass

                if simType is libsedml.SEDML_SIMULATION_STEADYSTATE:
                    lines.append("{}.steadyStateSolver.setValue('{}', {})".format(mid, pkey.key, value))
                else:
                    lines.append("{}.integrator.setValue('{}', {})".format(mid, pkey.key, value))

        if simType is libsedml.SEDML_SIMULATION_STEADYSTATE:
            lines.append("if {model}.conservedMoietyAnalysis == False: {model}.conservedMoietyAnalysis = True".format(model=mid))
        else:
            lines.append("if {model}.conservedMoietyAnalysis == True: {model}.conservedMoietyAnalysis = False".format(model=mid))

        # get parents
        parents = []
        parent = node.parent
        while parent is not None:
            parents.append(parent)
            parent = parent.parent

        # <selections> of all parents
        # ---------------------------
        selections = SEDMLCodeFactory.selectionsForTask(doc=doc, task=node.task)
        for p in parents:
            selections.update(SEDMLCodeFactory.selectionsForTask(doc=doc, task=p.task))

        # <setValues> of all parents
        # ---------------------------
        # apply changes based on current variables, parameters and range variables
        for parent in reversed(parents):
            rangeId = parent.task.getRangeId()
            helperRanges = {}
            for r in parent.task.getListOfRanges():
                if r.getId() != rangeId:
                    helperRanges[r.getId()] = r

            for setValue in parent.task.getListOfTaskChanges():
                variables = {}
                # range variables
                variables[rangeId] = "__value__{}".format(rangeId)
                for key in helperRanges.keys():
                    variables[key] = "__value__{}".format(key)
                # parameters
                for par in setValue.getListOfParameters():
                    variables[par.getId()] = par.getValue()
                for var in setValue.getListOfVariables():
                    vid = var.getId()
                    mid = var.getModelReference()
                    selection = SEDMLCodeFactory.selectionFromVariable(var, mid)
                    expr = selection.id
                    if selection.type == 'concentration':
                        expr = "init([{}])".format(selection.id)
                    elif selection.type == 'amount':
                        expr = "init({})".format(selection.id)

                    # create variable
                    lines.append("__value__{} = {}['{}']".format(vid, mid, expr))
                    # variable for replacement
                    variables[vid] = "__value__{}".format(vid)

                # value is calculated with the current state of model
                lines.append(SEDMLCodeFactory.targetToPython(xpath=setValue.getTarget(),
                                                             value=evaluableMathML(setValue.getMath(), variables=variables),
                                                             modelId=setValue.getModelReference())
                             )

        # handle result variable
        resultVariable = "{}[0]".format(task.getId())

        # -------------------------------------------------------------------------
        # <UNIFORM TIMECOURSE>
        # -------------------------------------------------------------------------
        if simType == libsedml.SEDML_SIMULATION_UNIFORMTIMECOURSE:
            lines.append("{}.timeCourseSelections = {}".format(mid, list(selections)))

            initialTime = simulation.getInitialTime()
            outputStartTime = simulation.getOutputStartTime()
            outputEndTime = simulation.getOutputEndTime()
            numberOfPoints = simulation.getNumberOfPoints()

            # reset before simulation (see https://github.com/sys-bio/tellurium/issues/193)
            lines.append("{}.reset()".format(mid))

            # throw some points away
            if abs(outputStartTime - initialTime) > 1E-6:
                lines.append("{}.simulate(start={}, end={}, points=2)".format(
                                    mid, initialTime, outputStartTime))
            # real simulation
            lines.append("{} = {}.simulate(start={}, end={}, steps={})".format(
                                    resultVariable, mid, outputStartTime, outputEndTime, numberOfPoints))
        # -------------------------------------------------------------------------
        # <ONESTEP>
        # -------------------------------------------------------------------------
        elif simType == libsedml.SEDML_SIMULATION_ONESTEP:
            lines.append("{}.timeCourseSelections = {}".format(mid, list(selections)))
            step = simulation.getStep()
            lines.append("{} = {}.simulate(start={}, end={}, points=2)".format(resultVariable, mid, 0.0, step))

        # -------------------------------------------------------------------------
        # <STEADY STATE>
        # -------------------------------------------------------------------------
        elif simType == libsedml.SEDML_SIMULATION_STEADYSTATE:
            lines.append("{}.steadyStateSolver.setValue('{}', {})".format(mid, 'allow_presimulation', False))
            lines.append("{}.steadyStateSelections = {}".format(mid, list(selections)))
            lines.append("{}.simulate()".format(mid))  # for stability of the steady state solver
            lines.append("{} = {}.steadyStateNamedArray()".format(resultVariable, mid))
            # no need to turn this off because it will be checked before the next simulation
            # lines.append("{}.conservedMoietyAnalysis = False".format(mid))

        # -------------------------------------------------------------------------
        # <OTHER>
        # -------------------------------------------------------------------------
        else:
            lines.append("# Unsupported simulation: {}".format(simType))

        return lines

    @staticmethod
    def repeatedTaskToPython(doc, node):
        """ Create python for RepeatedTask.

        Must create
        - the ranges (Ranges)
        - apply all changes (SetValues)
        """
        # storage of results
        task = node.task
        lines = ["", "{} = []".format(task.getId())]

        # <Range Definition>
        # master range
        rangeId = task.getRangeId()
        masterRange = task.getRange(rangeId)
        if masterRange.getTypeCode() == libsedml.SEDML_RANGE_UNIFORMRANGE:
            lines.extend(SEDMLCodeFactory.uniformRangeToPython(masterRange))
        elif masterRange.getTypeCode() == libsedml.SEDML_RANGE_VECTORRANGE:
            lines.extend(SEDMLCodeFactory.vectorRangeToPython(masterRange))
        elif masterRange.getTypeCode() == libsedml.SEDML_RANGE_FUNCTIONALRANGE:
            warnings.warn("FunctionalRange for master range not supported in task.")
        # lock-in ranges
        for r in task.getListOfRanges():
            if r.getId() != rangeId:
                if r.getTypeCode() == libsedml.SEDML_RANGE_UNIFORMRANGE:
                    lines.extend(SEDMLCodeFactory.uniformRangeToPython(r))
                elif r.getTypeCode() == libsedml.SEDML_RANGE_VECTORRANGE:
                    lines.extend(SEDMLCodeFactory.vectorRangeToPython(r))

        # <Range Iteration>
        # iterate master range
        lines.append("for __k__{}, __value__{} in enumerate(__range__{}):".format(rangeId, rangeId, rangeId))

        # Everything from now on is done in every iteration of the range
        # We have to collect & intent all lines in the loop)
        forLines = []

        # definition of lock-in ranges
        helperRanges = {}
        for r in task.getListOfRanges():
            if r.getId() != rangeId:
                helperRanges[r.getId()] = r
                if r.getTypeCode() in [libsedml.SEDML_RANGE_UNIFORMRANGE,
                                       libsedml.SEDML_RANGE_VECTORRANGE]:
                    forLines.append("__value__{} = __range__{}[__k__{}]".format(r.getId(), r.getId(), rangeId))

                # <functional range>
                if r.getTypeCode() == libsedml.SEDML_RANGE_FUNCTIONALRANGE:
                    variables = {}
                    # range variables
                    variables[rangeId] = "__value__{}".format(rangeId)
                    for key in helperRanges.keys():
                        variables[key] = "__value__{}".format(key)
                    # parameters
                    for par in r.getListOfParameters():
                        variables[par.getId()] = par.getValue()
                    for var in r.getListOfVariables():
                        vid = var.getId()
                        mid = var.getModelReference()
                        selection = SEDMLCodeFactory.selectionFromVariable(var, mid)
                        expr = selection.id
                        if selection.type == 'concentration':
                            expr = "[{}]".format(selection.id)
                        lines.append("__value__{} = {}['{}']".format(vid, mid, expr))
                        variables[vid] = "__value__{}".format(vid)

                    # value is calculated with the current state of model
                    value = evaluableMathML(r.getMath(), variables=variables)
                    forLines.append("__value__{} = {}".format(r.getId(), value))

        # <resetModels>
        # models to reset via task tree below node
        mids = set([])
        for child in node:
            t = child.task
            if t.getTypeCode() == libsedml.SEDML_TASK:
                mids.add(t.getModelReference())
        # reset models referenced in tree below task
        for mid in mids:
            if task.getResetModel():
                # reset before every iteration
                forLines.append("{}.reset()".format(mid))
            else:
                # reset before first iteration
                forLines.append("if __k__{} == 0:".format(rangeId))
                forLines.append("    {}.reset()".format(mid))

        # add lines
        lines.extend('    ' + line for line in forLines)

        return lines



    @staticmethod
    def selectionsForTask(doc, task):
        """ Populate variable lists from the data generators for the given task.

        These are the timeCourseSelections and steadyStateSelections
        in RoadRunner.

        Search all data generators for variables which have to be part of the simulation.
        """
        modelId = task.getModelReference()
        selections = set()
        for dg in doc.getListOfDataGenerators():
            for var in dg.getListOfVariables():
                if var.getTaskReference() == task.getId():
                    selection = SEDMLCodeFactory.selectionFromVariable(var, modelId)
                    expr = selection.id
                    if selection.type == "concentration":
                        expr = "[{}]".format(selection.id)
                    selections.add(expr)

        return selections

    @staticmethod
    def uniformRangeToPython(r):
        """ Create python lines for uniform range.
        :param r:
        :type r:
        :return:
        :rtype:
        """
        lines = []
        rId = r.getId()
        rStart = r.getStart()
        rEnd = r.getEnd()
        rPoints = r.getNumberOfPoints()+1  # One point more than number of points
        rType = r.getType()
        if rType in ['Linear', 'linear']:
            lines.append("__range__{} = np.linspace(start={}, stop={}, num={})".format(rId, rStart, rEnd, rPoints))
        elif rType in ['Log', 'log']:
            lines.append("__range__{} = np.logspace(start={}, stop={}, num={})".format(rId, rStart, rEnd, rPoints))
        else:
            warnings.warn("Unsupported range type in UniformRange: {}".format(rType))
        return lines

    @staticmethod
    def vectorRangeToPython(r):
        lines = []
        __range = np.zeros(shape=[r.getNumValues()])
        for k, v in enumerate(r.getValues()):
            __range[k] = v
        lines.append("__range__{} = {}".format(r.getId(), list(__range)))
        return lines