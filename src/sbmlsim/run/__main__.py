"""The command line of the run layer: `python -m sbmlsim.run`, see `sbmlsim.run.cli`.

The process runs the numerical libraries with one thread each: a run uses its
CPUs through the processes of its pools, and threads of BLAS in every one of
them would oversubscribe the CPUs. The thread variables `THREAD_VARIABLES` are
read when BLAS is loaded, which `import sbmlsim` does already, so the process
sets every one which is unset to `1` and runs itself again, once, with the
flags of its interpreter (`sys.orig_argv`): the new process sees them set. On
POSIX it replaces itself (`os.execv`); on windows, where `os.execv` starts a
new process and ends the old one with 0, it runs the new process and exits
with its code. `MPLBACKEND` is set to `Agg` unless set, so the workers of the
pools draw with `Agg` as the process does (`matplotlib.use("Agg")`, #317). The
messages are logged on stderr, so stdout carries only the output of `plan` and
`list`.
"""

import os
import subprocess
import sys

#: the variables of the threads of the numerical libraries, `1` unless set
THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _single_threaded() -> None:
    """Set the unset thread variables to `1` and run the process again.

    Returns only when every thread variable was set; otherwise the process is
    replaced, or on windows exits with the code of the new process.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    unset = [name for name in THREAD_VARIABLES if name not in os.environ]
    if not unset:
        return
    for name in unset:
        os.environ[name] = "1"
    argv = [sys.executable, *sys.orig_argv[1:]]
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        sys.exit(subprocess.run(argv, check=False).returncode)
    os.execv(sys.executable, argv)


def main() -> None:
    """Run the command line and exit with its code."""
    _single_threaded()

    import matplotlib

    matplotlib.use("Agg")

    from rich.console import Console

    from sbmlsim.log import enable_rich_logging
    from sbmlsim.run.cli import main as cli_main

    enable_rich_logging(console=Console(stderr=True))
    sys.exit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
