"""CPU quota of the cgroup of the process.

A container limits its CPUs by a quota of its cgroup, which `os.process_cpu_count()` does not see: it counts the CPUs of the host. `available_cpus` is the number of processes a run may use. This module imports only the standard library.
"""

import math
import os
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_CGROUP = Path("/proc/self/cgroup")


def _read(path: Path) -> str | None:
    """Read a text file, `None` when it is missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None


def _cpus(quota: str, period: str) -> float | None:
    """Get a quota in CPUs, `None` for a value which is no finite positive number."""
    try:
        quota_us, period_us = float(quota), float(period)
    except ValueError:
        return None
    if not (math.isfinite(quota_us) and math.isfinite(period_us)):
        return None
    if quota_us <= 0 or period_us <= 0:
        return None
    return quota_us / period_us


def _v2_quota(root: Path, proc_cgroup: Path) -> float | None:
    """Get the smallest quota in CPUs of the cgroup v2 path of the process and its parents."""
    text = _read(proc_cgroup)
    if text is None:
        return None
    path = None
    for line in text.splitlines():
        if line.startswith("0::"):
            path = line[3:]
    if path is None:
        return None
    directory = root / path.lstrip("/")
    quotas: list[float] = []
    while True:
        content = _read(directory / "cpu.max")
        if content is not None:
            parts = content.split()
            cpus = _cpus(parts[0], parts[1]) if len(parts) == 2 else None
            if cpus is not None:
                quotas.append(cpus)
        if directory == root or root not in directory.parents:
            break
        directory = directory.parent
    return min(quotas) if quotas else None


def _v1_quota(root: Path) -> float | None:
    """Get the quota in CPUs of cgroup v1, as the container sees its own cgroup."""
    for name in ("cpu", "cpu,cpuacct"):
        quota = _read(root / name / "cpu.cfs_quota_us")
        period = _read(root / name / "cpu.cfs_period_us")
        if quota is None or period is None:
            continue
        cpus = _cpus(quota, period)
        if cpus is not None:
            return cpus
    return None


def cpu_quota(root: Path = CGROUP_ROOT, proc_cgroup: Path = PROC_CGROUP) -> int | None:
    """Get the CPU quota of the cgroup of the process.

    Args:
        root: the mount of the cgroup file system.
        proc_cgroup: the file with the cgroup of the process.

    Returns:
        The floor of the smallest quota of the cgroup and its parents (cgroup v2) or of its quota (v1), at least 1, `None` without a quota. A quota is `max` (v2) or `-1` (v1) without a limit; a file which is no quota, e.g. garbage, a value which is no finite positive number or a directory, sets no quota.
    """
    quota = _v2_quota(root, proc_cgroup)
    if quota is None:
        quota = _v1_quota(root)
    if quota is None:
        return None
    return max(1, math.floor(quota))


def available_cpus() -> int:
    """Get the number of CPUs a run may use: the CPUs of the process bounded by the quota of its cgroup, at least 1."""
    cpus = os.process_cpu_count() or 1
    quota = cpu_quota()
    return max(1, cpus if quota is None else min(cpus, quota))
