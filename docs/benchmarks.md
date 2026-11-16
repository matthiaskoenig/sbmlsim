# Benchmarks

`sbmlsim` runs a simulation with one of four engines, see [Engines](simulation.md#engines): [roadrunner](simulation.md#roadrunner), [COPASI](simulation.md#copasi), [AMICI](simulation.md#amici) and the Rust engine [core](simulation.md#core) ([sbmlsim-core](https://github.com/matthiaskoenig/sbmlsim-core), compiled to WebAssembly, see [Optional packages](installation.md#optional-packages)). This page is what each of them costs. `scripts/engine_benchmark.py` loads, simulates and scans every packaged model in every engine which is installed, compares the results with roadrunner and prints the table below.

Read a cell together with its configuration. Every number belongs to one machine, one model, one number of output times, one number of points and one number of workers, and the engines change places when any of those change: the two sections after the table measure the same scan again, cold against warm and over the width of the output grid, and the engines do not come out in the same order in all of them. The table answers "what does this cost here", it does not rank the engines.

## The run

```bash
# every model in every engine which is installed, pinned to eight cores
taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py --workers 8
```

That command printed the table below: the four packaged models, every engine which is installed, and the scans over the default of 1000 points on 8 workers. `--out FILE` writes the table into a file and `--json FILE` writes the measurement with every number as it was measured; `--models`, `--engines`, `--points`, `--repetitions` and `--rounds` narrow or widen the run, and `--cold` and `--steps` are the measurements of the two sections after it.

`--no-sync` keeps the environment as it is. AMICI comes from the extra `amici` and the bindings of the engine core from a checkout of sbmlsim-core next to this one, see [Optional packages](installation.md#optional-packages), and neither of them is in `uv.lock`; a plain `uv run`, which syncs the environment to the lock, uninstalls both and measures whichever engines are left. An engine which is not installed is skipped and named below the table, so a shorter table is never a slow engine.

The machine: a 12th Gen Intel Core i7-12700 (12 cores, 20 hardware threads) with 62 GB of memory, Ubuntu 24.04.5 with the kernel 7.0.0-43, python 3.14.10. Every measurement on this page was pinned to one hardware thread of each of the eight performance cores (`taskset -c 0,2,4,6,8,10,12,14`), which is where the eight workers of a parallel scan run. It is an interactive desktop: the one minute load average was 3.4 of the 20 hardware threads before the run, mostly a videoconference which took about one core throughout, and the CPU governor is `powersave`, so the clocks are the kernel's choice and not a fixed frequency.

What that leaves of a cell: the engines take turns within a round and every cell is the median of the round medians, so a change of the load reaches every engine alike, which makes a comparison within one run much steadier than its absolute seconds. It does not make it exact. The warm parallel scan of HCTZ is measured three times on this page, in the table, in the cold section and at 101 output times in the grid section, and gives 0.497, 0.522 and 0.494 seconds for roadrunner and 0.409, 0.399 and 0.417 for the engine core, a spread of about 5 %: an ordering holds where the difference is larger than that, and two cells within a few percent of each other are not ordered by this method. Without the pinning it is worse, and not only by a constant factor: an earlier unpinned measurement of this page had roadrunner ahead of the engine core on the parallel scan of HCTZ, 0.506 seconds against 0.622, the opposite of all three pinned runs.

## The table

| model | engine | load s | compile s | instantiate s | first compile s | simulate ms | scan serial s | scan parallel s | parallel on | max rel. diff. | max abs. diff. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| repressilator | roadrunner | 0.0203 |  |  | 0.041 | 1.12 | 0.926 | 0.196 | 8 processes | 0 |  |
| repressilator | copasi | 0.0541 |  |  | 0.056 | 1.77 | 2.06 | 0.468 | 8 processes | 4.31e-09 |  |
| repressilator | amici | 0.0298 |  |  | 0.0303 | 3.17 | 3.44 | 0.76 | 8 processes | 3.87e-09 |  |
| repressilator | core | 0.0219 | 0.000549 | 0.000947 | 0.143 | 1.2 | 0.868 | 0.236 | 8 threads | 1.78e-10 |  |
| demo | roadrunner | 0.0245 |  |  | 0.0431 | 0.5 | 0.3 | 0.0888 | 8 processes | 0 | 0 |
| demo | copasi | 0.0546 |  |  | 0.0503 | 1.15 | 1.54 | 0.373 | 8 processes | 0 | 0 |
| demo | amici | 0.0327 |  |  | 0.032 | 2.45 | 2.71 | 0.664 | 8 processes | 0 | 0 |
| demo | core | 0.026 | 0.000723 | 0.00103 | 0.0278 | 0.469 | 0.0927 | 0.0578 | 8 threads | 0 | 0 |
| midazolam | roadrunner | 0.0553 |  |  | 0.2 | 1.6 | 1.92 | 0.426 | 8 processes | 0 | 0 |
| midazolam | copasi | 0.156 |  |  | 0.148 | 13.3 | 15.2 | 2.97 | 8 processes | 2.68e-07 | 1.64e-23 |
| midazolam | amici | 0.0787 |  |  | 0.0783 | 11.6 | 14.9 | 3.64 | 8 processes | 2.72e-07 | 0 |
| midazolam | core | 0.0572 | 0.00305 | 0.005 | 0.0645 | 1.74 | 1.23 | 0.524 | 8 threads | 1.70e-08 | 0 |
| hctz | roadrunner | 0.109 |  |  | 0.2 | 2.43 | 2.45 | 0.497 | 8 processes | 0 | 0 |
| hctz | copasi | 0.241 |  |  | 0.268 | 12.5 | 13.6 | 2.74 | 8 processes | 1.60e-08 | 1.47e-22 |
| hctz | amici | 0.137 |  |  | 0.136 | 11.2 | 13.9 | 3.29 | 8 processes | 1.63e-08 | 0 |
| hctz | core | 0.102 | 0.00324 | 0.00402 | 0.141 | 2 | 1.31 | 0.409 | 8 threads | 2.42e-11 | 0 |

Measured with python 3.14.10, sbmlsim 0.25.0, libroadrunner 2.11.1.dev655, python-copasi 4.48.311, amici 1.2.0, sbmlsim-core 0.2.0: the median of 20 simulations after 3 warm ups in 3 rounds (the median of the round medians, the engines taking turns), the scans over 1000 points.

- `load s` is `load_model` with the model already compiled, `first compile s` the first load of that model in this process, the one which compiles it (the LLJIT of roadrunner, the module of the engine core); the runtime of the engine core is created once per process, so only the first model of a run pays it in this column. AMICI is loaded from a cache which was warmed before the measurement, so its compilation from cold, minutes per model, is not measured, and COPASI, which compiles the model on every load, has the same measurement twice.
- `compile s` and `instantiate s` are the compilation of the SBML into the module of the engine core and its instantiation in the runtime, which `load s` of the other engines does not break out. They are no part of `load s` of the core either, which finds the module compiled (the engine keeps it per SBML and process), and they are part of its `first compile s`.
- `simulate ms` is one `Simulator.simulate` at the defaults of `Simulator`, i.e. the tolerances a run has and not the tighter ones of the agreement below, which is measured in simulations of its own: every speed of this table is the speed of a run as a user gets it. `scan serial s` and `scan parallel s` are that simulation over the points of the scan on one worker and on the workers of `parallel on`.
- The scan times are warm: every scan runs once before it is timed, so they exclude the pool of worker processes, which is kept per process and number of workers, the model every worker loads and compiles and the first batch of the threads of the engine core. A measurement which includes that first run is not the same measurement: it is several times larger for an engine whose parallel scan runs in worker processes, which carries the pool and one compilation of the model per worker, while the engine core, which starts no pool, pays only its own compilation there.
- The simulation of a model is its horizon with an equidistant output, which every engine computes alike (the steps of the integrator are each engine's own), in the time unit of the model: `repressilator` to 100 in 101 output times, its scan over `n` from 1.5 to 4.0; `demo` to 100 in 101 output times, its scan over `Km_A` from 0.5 to 2.0; `midazolam` to 1440 in 101 output times, its scan over `PODOSE_mid` from 5.0 to 100.0; `hctz` to 1440 in 101 output times, its scan over `PODOSE_hctz` from 5.0 to 100.0. How much output a point has matters for the parallel scan of the engine core, see `parallel on`.
- `parallel on` says what the workers are: python processes for roadrunner, COPASI and AMICI, which load the model in every worker, and threads of the engine inside one python process for the engine core, which starts no pool. The two do not scale alike: a process runs the whole point, a thread of the core integrates while the result of the run is assembled in its one python process, so the more output a point has, the less the threads gain.
- `max rel. diff.` and `max abs. diff.` are the agreement with roadrunner at atol 1e-12 and rtol 1e-10: the largest difference of a column divided by the largest magnitude of that column in roadrunner, and, for a column which is identically zero in roadrunner, where a relative difference is undefined, the largest absolute difference. roadrunner is the reference, COPASI is not bitwise reproducible across loads of a model.
- An empty cell is a measurement which was not made.

## What the table says

- **Loading a model.** roadrunner and the engine core load a compiled model in the same time within a few percent (102 ms against 109 ms for HCTZ), COPASI takes two to three times as long and compiles the model on every load, and AMICI imports its compiled extension in about 1.3 to 1.5 times roadrunner's load. Compiling is where they differ: roadrunner's LLJIT adds about 90 ms to the load of HCTZ (200 ms against 109 ms), while the engine core writes the WebAssembly module of that model in 3 ms and instantiates it in 4 ms, a few milliseconds on a load of about 100 ms. Read `first compile s` with one more thing in mind: the first load of a model in a process can cost more than a later one for reasons which are no compilation. COPASI, whose every load compiles the model, shows that as 268 ms against 241 ms on HCTZ, and the 141 ms of the engine core on that model are about 30 ms above its load plus its compilation, while on `demo` and midazolam its first load is its load plus its compilation to within a millisecond. The 143 ms of the first model of the engine core, the repressilator, also carry the runtime of the engine, which is created once per process and takes about 50 ms here. AMICI is the exception the table cannot show: its first compilation of a model is minutes, once per machine, and the measurement warms that cache before it times anything.
- **One simulation.** roadrunner and the engine core are within about 10 % of each other on the repressilator, `demo` and midazolam, and there the ordering does not reproduce: the engine core is 9 % slower than roadrunner on midazolam in this table and 1 % faster in a second pinned run of that model. On HCTZ it is 18 % faster (2.0 ms against 2.43 ms, and 1.83 ms against 2.27 ms at the same output times in the grid section), which does reproduce. COPASI and AMICI take between four and a half and eight and a half times as long on the two physiologically based models and one and a half to five times as long on the two small ones.
- **A serial scan.** The engine core runs the 1000 points of HCTZ in 1.31 s against roadrunner's 2.45 s, and the 1000 points of `demo` in 0.09 s against 0.30 s, far more than the 18 % and 6 % which separate one simulation of those two models. The reason is where the python is: a serial scan of roadrunner runs point by point through python, while a scan of the engine core hands a whole chunk, here up to 250 points, to the engine as one batch, so the per-point python is paid once per chunk. The smaller the model, the more that matters, which is why `demo` gains most and the repressilator, whose oscillating integration is the work, gains almost nothing (0.87 s against 0.93 s).
- **A parallel scan.** On eight workers the engine core is ahead on `demo` (0.058 s against 0.089 s) and HCTZ (0.409 s against 0.497 s) and roadrunner is ahead on the repressilator (0.196 s against 0.236 s) and midazolam (0.426 s against 0.524 s). A second pinned run of the repressilator and midazolam gives the same ordering with about the same margin (0.165 against 0.189 and 0.394 against 0.477), and the three pinned runs of HCTZ above give the engine core the lead in all three, so these four orderings are the method's and not one run's. The two parallelize different things: roadrunner's eight worker processes each run whole points, their integration and the python around them, while the engine core runs eight threads of the engine inside one python process, which assembles every result alone. The table's own serial column is what decides the outcome: the models the engine core wins in parallel are the two it wins the serial scan by the widest margin (`demo` at 0.31 of roadrunner's time, HCTZ at 0.53), while on midazolam (0.64) and the repressilator (0.94) that lead is too small to pay for the one python process. The two sections below measure that process directly, cold against warm and over the width of the output grid.
- **The agreement.** At atol 1e-12 and rtol 1e-10 every engine agrees with roadrunner to 2.8e-07 relative or better, and on `demo` every one of them reproduces roadrunner exactly. The engine core is the closest, by a factor of 16 to 660 on the three models which differ at all (2.42e-11 on HCTZ against 1.60e-08 for COPASI and 1.63e-08 for AMICI), because it integrates with the vector of absolute tolerances which roadrunner computes per state, while COPASI and AMICI have a single absolute tolerance and are given the smallest one with a safety factor, see [Absolute tolerances](simulation.md#absolute-tolerances). Agreeing with roadrunner more closely is not the same as being more accurate: the roadrunner rows are 0 because roadrunner is the reference of the column, not because it is the truth, and the error of the engine core is not bounded by roadrunner's, see [core](simulation.md#core).

## A cold scan and a warm scan are different measurements

`scan serial s` and `scan parallel s` are warm: every scan of the table runs once before it is timed. That first run is where a parallel scan of roadrunner, COPASI or AMICI starts its pool of worker processes and every worker loads and compiles the model, and where the engine core compiles its module and starts its first batch of threads. `sbmlsim.parallel` keeps a pool per process and number of workers, so the second scan of a session finds all of it ready.

`--cold` times that first run as well and reports it next to the warm median. It is a cold measurement of the first model and engine of a process only, since the pool and the runtime of the engine core are kept per process, so each engine was measured in a run of its own, the HCTZ model over 1000 points on 8 workers:

```bash
# the cold scans of one engine next to its warm ones, in a process of its own
for engine in roadrunner copasi amici core; do
  taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py \
      --models hctz --engines "$engine" --workers 8 --cold
done
```

| engine | scan parallel cold s | scan parallel s | cold / warm |
| --- | ---: | ---: | ---: |
| roadrunner | 1.97 | 0.522 | 3.8 |
| copasi | 4.11 | 2.87 | 1.4 |
| amici | 6.49 | 3.52 | 1.8 |
| core | 0.503 | 0.399 | 1.3 |

The first parallel scan of a process costs roadrunner 3.8 times its warm one, because it starts eight worker processes and compiles the model in each of them, while the engine core, which starts no pool, pays about 100 ms for its first batch of threads. So the engines change places with the question being asked: the first scan of 1000 points in a process costs 0.50 s in the engine core and 1.97 s in roadrunner, while in a session or a fit which scans again and again every later scan costs 0.40 s in the engine core and 0.52 s in roadrunner. A single number for "how much faster" says nothing unless it says which of the two it is, and at which output grid, see the next section.

A serial scan has no cold run worth the name: the same four runs give 2.53 s cold against 2.64 s warm for roadrunner, 2.08 against 1.97 for the engine core, 14.1 against 13.6 for COPASI and 15.6 against 17.0 for AMICI, two of them below their warm median, which is the spread of the method and not a cold cost. The pool of worker processes is what a cold scan pays for.

## The width of the output grid

A thread of the engine core integrates while the one python process of the run assembles every result, so the wider the output grid of a point, the less the threads gain. `--steps` scans every model on another equidistant grid, here the same scan of HCTZ, 1000 points on 8 workers, warm, with the two engines taking turns in one process:

```bash
# the same scan on four output grids, `steps + 1` output times per point
for steps in 10 25 100 400; do
  taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py \
      --models hctz --engines roadrunner core --workers 8 --steps "$steps"
done
```

| output times per point | roadrunner s | core s | core / roadrunner |
| ---: | ---: | ---: | ---: |
| 11 | 0.353 | 0.301 | 0.85 |
| 26 | 0.413 | 0.308 | 0.75 |
| 101 | 0.494 | 0.417 | 0.84 |
| 401 | 0.795 | 0.905 | 1.14 |

Both engines get slower with a wider grid and the engine core gets slower faster: it is ahead by 15 to 25 % up to about a hundred output times per point and behind by about 15 % at 401. How large that lead is moves with the run, 0.85, 0.75 and 0.84 of roadrunner's time on the three narrower grids; what reproduces is the lead itself and the crossing between 101 and 401 output times. That is the mechanism behind the column `parallel on`: threads of the engine waiting on the one python process of the run. The serial scans of the same four runs show the other side of it, 1.10, 1.14, 1.31 and 2.03 seconds for the engine core against 1.86, 1.88, 2.30 and 4.26 for roadrunner, so the engine keeps its serial advantage at every grid and it is the threads which pay for the width. The table at the top of the page uses 101 output times for every model. A single simulation has no such threads either, so none of this reaches it.

## Measuring your own model

The numbers which matter are the ones of your model on your machine. Everything of the benchmark is in `scripts/engine_benchmark.py`:

```bash
# one model, a smaller scan, the table and the measurement written out
mkdir -p results
uv run --no-sync python scripts/engine_benchmark.py --models hctz --points 200 --out results/table.md --json results/benchmark.json
```

`results/` is ignored by the repository, so the two files do not turn up as changes of a checkout. The method is the one of a microbenchmark: the simulations of a cell are timed `--repetitions` times after three untimed ones (`WARMUPS` of the script), that is one round, the median of a round is taken, and the median of `--rounds` round medians is the cell; the configurations take turns within a round, so the machine treats them alike. A scan is seconds rather than milliseconds, so it runs once before the timed runs and then once per round. The agreement is measured in simulations of its own at atol 1e-12 and rtol 1e-10, where the engines agree, see [Comparing the engines](simulation.md#comparing-the-engines), while every speed of the table is measured at the defaults of `Simulator`, i.e. at the tolerances of a run. Pin the process to a fixed set of cores and leave the machine alone, as the runs of this page did, if you want numbers which compare across runs.
