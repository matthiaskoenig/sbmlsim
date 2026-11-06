"""An entry point of another framework for the profiled runs of the tests.

`python -m tests.run.profiled_entry <code> [record]` exits with the code and
records it for the profiler with `record` (`record_exit_code`).
"""

import sys

from sbmlsim.run.profiling import record_exit_code

if __name__ == "__main__":
    code = int(sys.argv[1])
    if sys.argv[2:] == ["record"]:
        record_exit_code(code)
    sys.exit(code)
