"""The command line of the run layer: `python -m sbmlsim.run`, see `sbmlsim.run.cli`.

The process runs the numerical libraries with one thread each: a run uses its
CPUs through the processes of its pools, and threads of BLAS in every one of
them would oversubscribe the CPUs. The thread variables `THREAD_VARIABLES` are
read when BLAS is loaded, which `import sbmlsim` does already, so the process
sets every one which is unset to `1` and executes itself again, once: the new
process sees them set. matplotlib draws with `Agg`, and the messages are
logged on stderr, so stdout carries only the output of `plan` and `list`.
"""

import os
import sys

#: the variables of the threads of the numerical libraries, `1` unless set
THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _single_threaded() -> None:
    """Set the unset thread variables to `1` and execute the process again."""
    unset = [name for name in THREAD_VARIABLES if name not in os.environ]
    if not unset:
        return
    for name in unset:
        os.environ[name] = "1"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-m", "sbmlsim.run", *sys.argv[1:]])


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
