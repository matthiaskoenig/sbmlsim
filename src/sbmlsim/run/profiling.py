"""The profile of a run, with the sampling profiler of python 3.15.

`--profile` runs a command of `python -m sbmlsim.run` which writes into an
output directory (`run`, `unit`, `merge`) under the statistical sampling
profiler of python 3.15 (`profiling.sampling`, PEP 799) and writes the profile
into the folder `profiling/` of the output:

```bash
python -m sbmlsim.run simulate spec.yaml --out results --profile
```

The profiler samples the stack of the command at a fixed rate from a process
of its own, so the command runs at nearly its normal speed. The command runs as
a child of the profiler (`python -m profiling.sampling run -m sbmlsim.run
...`), whose binary profile is converted afterwards (`replay`) into the files
of the folder `profiling/<command>_<kind>_<time>/` of the output:

- `flamegraph.html`: the interactive flame graph of the sampled stacks;
- `heatmap/`: the source code of the sampled lines with their sample counts;
- `profile.pstats`: the statistics of the functions, for `pstats` and its
  viewers;
- `summary.txt`: the functions with the most samples;
- `profile.bin`: the samples, from which other formats are made and against
  which a later profile is compared (`replay --diff-flamegraph profile.bin`).

The profiler writes its samples outside of the output while the command runs,
since a run refuses an output which is not empty, and the folder of the profile
is written after the command. The child gets the thread variables of
`sbmlsim.run.__main__` set, so it runs without starting itself again, which
the profiler would not follow.

`run_profiled` profiles any entry point which runs as `python -m <module>`,
`sbmlsim.run` by default, e.g. the command line of a framework of its own. The
profiler exits with 0 whatever the command did, so the entry point writes its
exit code into the file of the environment variable `EXIT_CODE_FILE` with
`record_exit_code` before it exits, which `sbmlsim.run.__main__` does; this is
the hook of every other entry point. The exit code of a profiled run is the
code the command recorded, 1 if it recorded none (it did not end by itself or
does not record its code). A failure of the profiler or of the conversion of
the profile is logged and does not change the exit code.

The profiler runs as `python -m sbmlsim.run.profiling`, the command line of
`profiling.sampling` with one change (`sample`): python 3.15.2 reads now and
then a name of a frame of the command while it changes, a string with a lone
surrogate, and the writer of the binary profile broke with a
`UnicodeEncodeError` on such a sample, which ended the profiler and the
command (one run in three of a short command). Such a sample is not written;
the binary profile does not count it (its `collect_failed_sample` does
nothing), so the profiler reports the number of these samples on stderr. Which
versions of python fix it is not known, so the change applies whenever the
profiler has the collector it changes (`BinaryCollector.collect` of
`profiling.sampling.binary_collector`) and the profiler runs unchanged with a
warning otherwise.

The profiler is part of python 3.15; with an older python `--profile` is an
error and the command does not run.
"""

from __future__ import annotations

import datetime
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "EXIT_CODE_FILE",
    "FORMATS",
    "PROFILER_PYTHON",
    "PROFILE_DIR",
    "PROFILE_OPTION",
    "ProfilerUnavailable",
    "profile_dir",
    "profiler_available",
    "record_exit_code",
    "run_profiled",
    "sample",
]

logger = logging.getLogger(__name__)

#: the option of the command line which runs a command under the profiler
PROFILE_OPTION = "--profile"

#: the python which has the sampling profiler
PROFILER_PYTHON = (3, 15)

#: the folder of the profiles in the output of a run
PROFILE_DIR = "profiling"

#: the environment variable with the file into which the profiled command
#: writes its exit code: the profiler exits with 0 whatever the command did
EXIT_CODE_FILE = "SBMLSIM_PROFILE_EXIT_CODE"

#: the formats which the binary profile is converted into, as the option of
#: `replay` and the file in the folder of the profile
FORMATS: tuple[tuple[str, str], ...] = (
    ("--flamegraph", "flamegraph.html"),
    ("--heatmap", "heatmap"),
    ("--pstats", "profile.pstats"),
)


class ProfilerUnavailable(RuntimeError):
    """The python has no sampling profiler, `profiling.sampling` of python 3.15."""


def profiler_available() -> bool:
    """Check whether the python has the sampling profiler `profiling.sampling`."""
    if sys.version_info < PROFILER_PYTHON:
        return False
    try:
        return importlib.util.find_spec("profiling.sampling") is not None
    except ModuleNotFoundError:
        return False


def record_exit_code(code: int | str | None) -> None:
    """Write the exit code of a profiled command, see `EXIT_CODE_FILE`.

    Nothing is written outside of a profiled command.

    Args:
        code: the code of the command or of its `SystemExit`: `None` is 0, a
            message is 1.
    """
    path = os.environ.get(EXIT_CODE_FILE)
    if not path:
        return
    value = 0 if code is None else code if isinstance(code, int) else 1
    Path(path).write_text(str(value), encoding="utf-8")


def profile_dir(out: Path, command: str, kind: str, unit: int | None = None) -> Path:
    """Get a new folder of the profile of a command in the output of a run.

    Args:
        out: the output of the run.
        command: the command, e.g. `run`.
        kind: the kind, e.g. `simulate` or `module:attribute` of a kind.
        unit: the index of the unit of the command `unit`, so the units of a
            run which compute into one output at once get folders of their own.

    Returns:
        `profiling/<command>_<kind>_<time>` in the output, of a unit
        `profiling/unit_<kind>_<index>_<time>`, not created; the characters of
        the kind which are no letters, digits, `.`, `-` or `_` become `_`.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = re.sub(r"[^\w.-]", "_", kind)
    index = "" if unit is None else f"_{unit}"
    return Path(out) / PROFILE_DIR / f"{command}_{name}{index}_{stamp}"


def _child_environment(exit_code: Path) -> dict[str, str]:
    """Get the environment of the profiled command.

    The thread variables are set, so `sbmlsim.run.__main__` does not start
    the command again (`os.execv`), which the profiler would not follow.
    """
    from sbmlsim.run.__main__ import THREAD_VARIABLES

    environment = {**os.environ, EXIT_CODE_FILE: str(exit_code)}
    for name in THREAD_VARIABLES:
        environment.setdefault(name, "1")
    environment.setdefault("MPLBACKEND", "Agg")
    return environment


def _convert(binary: Path, folder: Path) -> None:
    """Convert a binary profile into the files of its folder, see the module.

    Raises:
        RuntimeError: if a conversion fails.
    """
    profiler = [sys.executable, "-m", "profiling.sampling"]
    for option, name in FORMATS:
        replay = subprocess.run(
            [*profiler, "replay", option, "-o", str(folder / name), str(binary)],
            check=False,
            capture_output=True,
            text=True,
        )
        if replay.returncode:
            raise RuntimeError(
                f"replay {option} failed:\n{replay.stderr}{replay.stdout}"
            )
    summary = subprocess.run(
        [*profiler, "replay", "--pstats", "--sort", "cumtime", "-l", "40", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    if summary.returncode:
        raise RuntimeError(f"replay --pstats failed:\n{summary.stderr}{summary.stdout}")
    # the statistics, without the progress of the replay before them
    text = summary.stdout
    start = text.find("Profile Stats:")
    (folder / "summary.txt").write_text(
        text[start:] if start >= 0 else text, encoding="utf-8"
    )


def run_profiled(
    arguments: Sequence[str], folder: Path, *, module: str = "sbmlsim.run"
) -> int:
    """Run an entry point under the sampling profiler, see the module.

    Args:
        arguments: the arguments of `python -m <module>`, for `sbmlsim.run`
            with the command and without `--profile`.
        folder: the folder of the profile, created after the command ran, see
            `profile_dir`.
        module: the module of the entry point, which records its exit code
            with `record_exit_code`.

    Returns:
        The exit code the command recorded, 1 if it recorded none; a failure
        of the profiler or of the conversion of the profile is logged and does
        not change it.

    Raises:
        ProfilerUnavailable: if the python has no sampling profiler.
    """
    if not profiler_available():
        version = ".".join(map(str, PROFILER_PYTHON))
        raise ProfilerUnavailable(
            f"{PROFILE_OPTION} needs the sampling profiler of python {version} "
            f"(profiling.sampling), this is python {sys.version.split()[0]}."
        )
    with tempfile.TemporaryDirectory(prefix="sbmlsim-profile-") as tmp:
        binary = Path(tmp) / "profile.bin"
        exit_code = Path(tmp) / "exit_code"
        returned = subprocess.run(
            [
                *(sys.executable, "-m", __name__),
                *("run", "--binary", "-o", str(binary)),
                *("-m", module, *arguments),
            ],
            check=False,
            env=_child_environment(exit_code),
        ).returncode
        # the code of the command, which the profiler does not pass on
        if exit_code.is_file():
            code = int(exit_code.read_text(encoding="utf-8"))
        else:
            logger.error(
                "The command '%s' recorded no exit code (record_exit_code): it "
                "did not end by itself or does not record it, its code is 1.",
                module,
            )
            code = 1
        if returned or not binary.is_file():
            logger.error(
                "The profiler failed with the exit code %d and wrote no profile "
                "of the command.",
                returned,
            )
            return code
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / binary.name
        # the temporary directory may be on another file system than the out
        shutil.move(binary, saved)
    try:
        _convert(saved, folder)
    except (RuntimeError, OSError) as err:
        logger.error("The profile '%s' was not converted: %s", saved, err)
        return code
    logger.info("The profile: file://%s", (folder / "flamegraph.html").resolve())
    return code


def _encodable(value: object) -> bool:
    """Check that every string of a sample of the profiler is valid UTF-8.

    A sample is a list of tuples (structure sequences) of strings and numbers.
    """
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True
    if isinstance(value, (tuple, list)):
        return all(_encodable(item) for item in value)
    return True


def _patched_collector() -> Callable[[], int] | None:
    """Skip the samples the binary profile cannot encode, see the module.

    Returns:
        The number of the samples skipped so far, `None` if the profiler has
        no collector to change.
    """
    try:
        binary_collector = importlib.import_module(
            "profiling.sampling.binary_collector"
        )
    except ImportError:
        binary_collector = None
    collector = getattr(binary_collector, "BinaryCollector", None)
    collect = getattr(collector, "collect", None)
    if collector is None or not callable(collect):
        sys.stderr.write(
            "Warning: profiling.sampling has no BinaryCollector.collect, the "
            "profiler runs unchanged.\n"
        )
        return None
    skipped = 0

    def checked(self: Any, stack_frames: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal skipped
        if _encodable(stack_frames):
            return collect(self, stack_frames, *args, **kwargs)
        skipped += 1
        return None

    collector.collect = checked
    return lambda: skipped


def sample() -> None:
    """Run the command line of the sampling profiler, see the module.

    A sample with a name the binary profile cannot encode, a name the profiler
    read while the command changed it, is not written instead of breaking the
    writer of the profile, which ended the profiler and the command; their
    number is reported on stderr.
    """
    cli = importlib.import_module("profiling.sampling.cli")
    skipped = _patched_collector()
    try:
        cli.main()
    finally:
        if skipped is not None and skipped():
            sys.stderr.write(
                f"Warning: {skipped()} samples with a name which is no valid "
                "UTF-8 were not written into the profile.\n"
            )


if __name__ == "__main__":
    sample()
