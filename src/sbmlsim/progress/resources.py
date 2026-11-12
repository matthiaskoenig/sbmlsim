"""The cores and the memory a run uses, for its progress.

The progress of a long run says what of the machine it uses: the cores it may
use (the CPUs of the process bounded by the quota of its cgroup, see
`sbmlsim.cgroup`), the processes it runs on, the CPU it uses, the memory of its
processes and the memory of the machine. A run which is held up by the cores
and a run which is held up by the memory then look different on the console, in
the log and in the events.

The sample is taken from `/proc` on linux: the stat of the process and of every
process below it gives the seconds of the CPU and the resident memory of the
tree, `meminfo` the memory of the machine. The pool of a run is a tree of
children, so the sample covers the workers as well, which
`resource.getrusage` does not: it counts a child when it ends. Without `/proc`
(a platform which has none) the CPU is the one of `getrusage` and the memory is
left out, which the reporters leave out of their text.

The sample costs a scan of `/proc`, so it is taken at most every second and
every reporter of a process takes the same one, see `SAMPLER`.
"""

from __future__ import annotations

import logging
import os
import resource
from dataclasses import dataclass, replace
from pathlib import Path

from sbmlsim.cgroup import available_cpus
from sbmlsim.progress.core import _now

logger = logging.getLogger(__name__)

#: where the processes and the memory of the machine are read, `/proc` on linux
PROC = Path("/proc")

#: the seconds between two samples, a scan of `/proc` each
INTERVAL = 1.0

#: the bytes of a page of memory, for the resident memory of `/proc/<pid>/stat`
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

#: the ticks of a second of the CPU, for the times of `/proc/<pid>/stat`
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")

#: the bytes of a gigabyte, for the text of a sample
GIGABYTE = 1024**3


@dataclass(frozen=True, slots=True)
class Resources:
    """What of the machine a run uses.

    Attributes:
        cores: the cores the run may use.
        workers: the processes it runs on, `None` if it does not say.
        cpu_percent: the CPU it used since the sample before, in percent of a
            core, `None` for the first sample; `1180` are twelve busy cores.
        rss: the resident bytes of its processes, `None` without `/proc`.
        memory_used: the bytes of the memory of the machine which are used,
            `None` without `/proc`.
        memory_total: the bytes of the memory of the machine, `None` without
            `/proc`.
    """

    cores: int
    workers: int | None = None
    cpu_percent: float | None = None
    rss: int | None = None
    memory_used: int | None = None
    memory_total: int | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        """Get the sample as JSON, the field `resources` of a progress event."""
        return {
            "cores": self.cores,
            "workers": self.workers,
            "cpu_percent": (
                None if self.cpu_percent is None else round(self.cpu_percent, 1)
            ),
            "rss": self.rss,
            "memory_used": self.memory_used,
            "memory_total": self.memory_total,
        }


def _tree(root: Path, pid: int) -> tuple[float, int] | None:
    """Get the CPU seconds and the resident bytes of a process and its children.

    Args:
        root: where the processes are read, `/proc` on linux.
        pid: the process at the top of the tree.

    Returns:
        The seconds of the CPU and the resident bytes, `None` without `/proc`.
    """
    if not root.is_dir():
        return None
    #: the children of every process, and what every process used
    children: dict[int, list[int]] = {}
    used: dict[int, tuple[float, int]] = {}
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            # the process ended while its tree was read
            continue
        # the name of a process may hold spaces and brackets, the fields
        # behind it are counted from the last bracket, see `proc_pid_stat(5)`
        fields = stat[stat.rfind(")") + 1 :].split()
        if len(fields) < 22:
            continue
        try:
            ppid = int(fields[1])
            times = (int(fields[11]) + int(fields[12])) / CLOCK_TICKS
            memory = int(fields[21]) * PAGE_SIZE
        except ValueError:
            continue
        current = int(entry.name)
        used[current] = (times, memory)
        children.setdefault(ppid, []).append(current)
    if pid not in used:
        return None
    seconds = 0.0
    memory = 0
    stack = [pid]
    while stack:
        current = stack.pop()
        times, resident = used.get(current, (0.0, 0))
        seconds += times
        memory += resident
        stack.extend(children.get(current, ()))
    return seconds, memory


def _machine(root: Path) -> tuple[int, int] | None:
    """Get the used and the total bytes of the memory of the machine.

    Args:
        root: where the memory is read, `/proc` on linux.

    Returns:
        The bytes which are used and the bytes in total, `None` without
        `/proc`.
    """
    try:
        text = (root / "meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        fields = rest.split()
        if name in {"MemTotal", "MemAvailable"} and fields:
            try:
                values[name] = int(fields[0]) * 1024
            except ValueError:
                continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None
    return max(total - available, 0), total


def _own_cpu() -> float:
    """Get the CPU seconds of the process and of the children which ended."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


class Sampler:
    """The samples of the resources of a process, one per interval.

    Attributes:
        sample_: the sample of the interval which runs, `None` before the
            first one.
    """

    def __init__(self, now: object = _now) -> None:
        """Create the sampler without a sample.

        Args:
            now: the monotonic clock, for the tests.
        """
        self._now = now
        self._taken: float | None = None
        self._cpu: float | None = None
        self.sample_: Resources | None = None

    def sample(
        self, workers: int | None = None, interval: float = INTERVAL
    ) -> Resources:
        """Get what of the machine the run uses, at most one sample per interval.

        Args:
            workers: the processes the run runs on, for the text of the
                sample.
            interval: the seconds a sample is kept before the next one is
                taken.

        Returns:
            The sample, the one of the interval which runs if it is young
            enough.
        """
        now = float(self._now())  # ty: ignore[call-non-callable]
        current = self.sample_
        if (
            current is not None
            and self._taken is not None
            and now - self._taken < interval
        ):
            return current if workers is None else replace(current, workers=workers)
        tree = _tree(PROC, os.getpid())
        machine = _machine(PROC)
        cpu = tree[0] if tree is not None else _own_cpu()
        percent: float | None = None
        if self._cpu is not None and self._taken is not None and now > self._taken:
            percent = 100.0 * (cpu - self._cpu) / (now - self._taken)
        self._taken = now
        self._cpu = cpu
        self.sample_ = Resources(
            cores=available_cpus(),
            workers=workers,
            cpu_percent=percent,
            rss=None if tree is None else tree[1],
            memory_used=None if machine is None else machine[0],
            memory_total=None if machine is None else machine[1],
        )
        return self.sample_


#: the sampler of the process, which every reporter of it takes its sample from
SAMPLER = Sampler()


def format_resources(sample: Resources) -> str:
    """Write what of the machine a run uses, what a platform gives of it.

    Args:
        sample: the sample.

    Returns:
        The text, e.g. `cores 4/12 cpu 380% mem 7.4 GB, machine 18.3/31.0 GB`.
    """
    parts = [
        f"cores {sample.cores}"
        if sample.workers is None
        else f"cores {sample.workers}/{sample.cores}"
    ]
    if sample.cpu_percent is not None:
        parts.append(f"cpu {sample.cpu_percent:.0f}%")
    if sample.rss is not None:
        parts.append(f"mem {sample.rss / GIGABYTE:.1f} GB,")
    if sample.memory_used is not None and sample.memory_total is not None:
        parts.append(
            f"machine {sample.memory_used / GIGABYTE:.1f}"
            f"/{sample.memory_total / GIGABYTE:.1f} GB"
        )
    return " ".join(parts)
