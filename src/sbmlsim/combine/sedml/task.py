"""Task trees of SED-ML documents."""

import logging

import libsedml

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
        subtasks: list[libsedml.SedSubTask] = list(repeated_task.getListOfSubTasks())
        subtask_order: list[int] = [st.getOrder() for st in subtasks]
        # sort by order, if all subtasks have order (not required)
        if all(subtask_order) is not None:
            subtasks = [
                st
                for (_order, st) in sorted(zip(subtask_order, subtasks, strict=False))
            ]
        return subtasks


# -------------------------------------------------------------------------------------
