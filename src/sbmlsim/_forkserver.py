"""The last module the forkserver of a pool imports, see `sbmlsim.parallel`.

Its import preloads the experiment classes the forkserver imported
(`sbmlsim.parallel.preload_experiments`), so the workers it forks inherit what
they load. No other process imports it.
"""

from sbmlsim.parallel import preload_experiments

preload_experiments()
