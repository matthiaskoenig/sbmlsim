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
the profiler would not follow. The profiler exits with 0 whatever the command
did, so the command writes its exit code into the file of the environment
variable `EXIT_CODE_FILE` (`record_exit_code`).

The profiler runs as `python -m sbmlsim.run.profiling`, the command line of
`profiling.sampling` with one change (`sample`): python 3.15.2 reads now and
then a name of a frame of the command while it changes, a string with a lone
surrogate, and the writer of the binary profile broke with a
`UnicodeEncodeError` on such a sample, which ended the profiler and the
command (one run in three of a short command); such a sample is not written
and counts as a failed one, as the profiler counts a sample it cannot read.

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
from collections.abc import Sequence
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


def profile_dir(out: Path, command: str, kind: str) -> Path:
    """Get a new folder of the profile of a command in the output of a run.

    Args:
        out: the output of the run.
        command: the command, e.g. `run`.
        kind: the kind, e.g. `simulate` or `module:attribute` of a kind.

    Returns:
        `profiling/<command>_<kind>_<time>` in the output, not created; the
        characters of the kind which are no letters, digits, `.`, `-` or `_`
        become `_`.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = re.sub(r"[^\w.-]", "_", kind)
    return Path(out) / PROFILE_DIR / f"{command}_{name}_{stamp}"


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


def run_profiled(arguments: Sequence[str], folder: Path) -> int:
    """Run a command of the run layer under the sampling profiler, see the module.

    Args:
        arguments: the arguments of `python -m sbmlsim.run`, with the command
            and without `--profile`.
        folder: the folder of the profile, created after the command ran, see
            `profile_dir`.

    Returns:
        The exit code of the command.

    Raises:
        ProfilerUnavailable: if the python has no sampling profiler.
        RuntimeError: if a profile is not converted.
    """
    if not profiler_available():
        version = ".".join(map(str, PROFILER_PYTHON))
        raise ProfilerUnavailable(
            f"{PROFILE_OPTION} needs the sampling profiler of python {version} "
            f"(profiling.sampling), this is python {sys.version.split()[0]}."
        )
    profiler = [sys.executable, "-m", "profiling.sampling"]
    sampler = [sys.executable, "-m", __name__]
    with tempfile.TemporaryDirectory(prefix="sbmlsim-profile-") as tmp:
        binary = Path(tmp) / "profile.bin"
        exit_code = Path(tmp) / "exit_code"
        returned = subprocess.run(
            [
                *sampler,
                *("run", "--binary", "-o", str(binary)),
                *("-m", "sbmlsim.run", *arguments),
            ],
            check=False,
            env=_child_environment(exit_code),
        ).returncode
        # the code of the command, which the profiler does not pass on
        code = (
            int(exit_code.read_text(encoding="utf-8"))
            if exit_code.is_file()
            else returned or 1
        )
        if returned or not binary.is_file():
            logger.error(
                "The profiler failed with the exit code %d and wrote no profile "
                "of the command.",
                returned,
            )
            return code or 1
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / binary.name
        # the temporary directory may be on another file system than the out
        shutil.move(binary, saved)
    for option, name in FORMATS:
        replay = subprocess.run(
            [*profiler, "replay", option, "-o", str(folder / name), str(saved)],
            check=False,
            capture_output=True,
            text=True,
        )
        if replay.returncode:
            raise RuntimeError(
                f"replay {option} failed:\n{replay.stderr}{replay.stdout}"
            )
    summary = subprocess.run(
        [*profiler, "replay", "--pstats", "--sort", "cumtime", "-l", "40", str(saved)],
        check=True,
        capture_output=True,
        text=True,
    )
    # the statistics, without the progress of the replay before them
    text = summary.stdout
    start = text.find("Profile Stats:")
    (folder / "summary.txt").write_text(
        text[start:] if start >= 0 else text, encoding="utf-8"
    )
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


def sample() -> None:
    """Run the command line of the sampling profiler, see the module.

    A sample with a name the binary profile cannot encode, a name the profiler
    read while the command changed it, counts as a failed sample instead of
    breaking the writer of the profile, which ended the profiler and the
    command.
    """
    binary_collector = importlib.import_module("profiling.sampling.binary_collector")
    cli = importlib.import_module("profiling.sampling.cli")
    collector = binary_collector.BinaryCollector
    collect = collector.collect

    def checked(self: Any, stack_frames: Any, timestamp_us: int | None = None) -> None:
        if _encodable(stack_frames):
            collect(self, stack_frames, timestamp_us)
        else:
            self.collect_failed_sample()

    collector.collect = checked
    cli.main()


if __name__ == "__main__":
    sample()
