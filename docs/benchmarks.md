# Benchmarks

`sbmlsim` runs a simulation with one of four engines, see [Engines](simulation.md#engines): the Rust engine [core](simulation.md#core) ([sbmlsim-core](https://github.com/matthiaskoenig/sbmlsim-core), compiled to WebAssembly, see [Optional packages](installation.md#optional-packages)), [roadrunner](simulation.md#roadrunner), [COPASI](simulation.md#copasi) and [AMICI](simulation.md#amici). This page is what each of them costs. `scripts/engine_benchmark.py` loads, simulates and scans four packaged models and one large model in every engine which is installed, compares the results with roadrunner and prints the tables below.

Read a cell together with its configuration. Every number belongs to one machine, one model, one number of output times, one number of points and one number of workers, and the engines change places when any of those change: the two sections after the tables measure the same scan again, cold against warm and over the width of the output grid, and the engines do not come out in the same order in all of them. The tables answer "what does this cost here", they do not rank the engines.

## The run

```bash
# every model in every engine which is installed, pinned to eight cores
taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py --workers 8 \
    --out tables.md --figure docs/images/benchmarks/engines.png
```

That command printed the tables below and drew the figure: the four packaged models and the large model, every engine which is installed, and the scans over the default of 1000 points on 8 workers. `--out FILE` writes the tables into a file, `--figure FILE` draws the comparison and `--json FILE` writes the measurement with every number as it was measured; `--models`, `--engines`, `--points`, `--repetitions` and `--rounds` narrow or widen the run, and `--cold` and `--steps` are the measurements of the two sections after it. Nothing regenerates this page, so a new measurement is a run of the script and a commit of its tables and its figure.

The large model is `benchmarks/models/machine_perfusion_liver_flat.xml`, the liver of a normothermic machine perfusion: 839 species, 1007 reactions, 570 parameters, 387 rules and 43 compartments in 2.6 MB, which is twelve times the largest packaged model, scanned over the flow of its pump. It is in the repository and no part of the package, so no install carries it, see `benchmarks/models/README.md`. One of its simulations is tens of seconds in COPASI, so it is scanned over as many points as the parallel scan has workers and measured in fewer and shorter rounds than the packaged models, which the first table of the run says of every model.

`--no-sync` keeps the environment as it is. COPASI comes with the extra `dev` and the engine core with every install, since its bindings `sbmlsim-core` are a dependency of sbmlsim in `uv.lock`. AMICI comes from the extra `amici` (`uv sync --extra dev --extra amici`), which is not in `uv.lock`, see [Optional packages](installation.md#optional-packages); a plain `uv run`, which syncs the environment to the lock, uninstalls it and measures the three other engines. An engine which is not installed is skipped and named below the table, so a shorter table is never a slow engine.

The machine: a 12th Gen Intel Core i7-12700 (12 cores, 20 hardware threads) with 62 GB of memory, Ubuntu 24.04.5 with the kernel 7.0.0-43, python 3.14.10. Every measurement on this page was pinned to one hardware thread of each of the eight performance cores (`taskset -c 0,2,4,6,8,10,12,14`), which is where the eight workers of a parallel scan run. It is an interactive desktop and the CPU governor is `powersave`, so the clocks are the kernel's choice and not a fixed frequency.

What that leaves of a cell: a cell is the mean of its rounds and their sample standard deviation, the engines of a model take turns within a round, and the `±` is what the method can say about itself. A deviation which is a few percent of its mean is a cell to read; one which is a quarter of it is a cell which was measured while the machine was doing something else, and the figure shows both at a glance. Pinning is what makes the small deviations possible at all: an earlier unpinned measurement of this page had roadrunner ahead of the engine core on the parallel scan of HCTZ, 0.506 seconds against 0.622, the opposite of every pinned run since. Two cells whose intervals overlap are not ordered by this method, however much their means differ.

The tables come from two runs of that command. An unrelated job on the machine disturbed a different part of each of them, and every measure of every model is taken from the run whose four engines deviate least there, so the engines of a comparison always took their turns within one run, which is the only comparison this page makes. The parallel scans are the noisiest measure in both runs, at a tenth of their mean, because eight workers and the process which collects them share eight hardware threads.

## The tables

### What every model was measured with

| model | output times | scan points | simulations | warm ups | rounds |
|---|---:|---:|---:|---:|---:|
| repressilator | 101 | 1000 | 20 | 3 | 5 |
| demo | 101 | 1000 | 20 | 3 | 5 |
| midazolam | 101 | 1000 | 20 | 3 | 5 |
| hctz | 101 | 1000 | 20 | 3 | 5 |
| perfusion | 101 | 8 | 3 | 1 | 3 |

### Loading a model

| model | engine | load s | first compile s | compile s | instantiate s |
|---|---|---:|---:|---:|---:|
| repressilator | core | 0.0207 ± 0.00015 | 0.591 | 0.000545 ± 2.3e-06 | 0.00348 ± 1.3e-05 |
| repressilator | roadrunner | 0.0189 ± 0.00017 | 0.0383 |  |  |
| repressilator | copasi | 0.0504 ± 0.00026 | 0.0509 |  |  |
| repressilator | amici | 0.0281 ± 0.00017 | 0.0286 |  |  |
| demo | core | 0.0244 ± 0.00011 | 0.0297 | 0.000718 ± 9.6e-06 | 0.00337 ± 4.8e-05 |
| demo | roadrunner | 0.0227 ± 9.3e-05 | 0.0407 |  |  |
| demo | copasi | 0.0502 ± 0.00041 | 0.049 |  |  |
| demo | amici | 0.0306 ± 0.00035 | 0.0307 |  |  |
| midazolam | core | 0.0533 ± 0.00042 | 0.0693 | 0.00284 ± 2.6e-05 | 0.0115 ± 6.8e-05 |
| midazolam | roadrunner | 0.051 ± 0.00045 | 0.178 |  |  |
| midazolam | copasi | 0.144 ± 0.00069 | 0.139 |  |  |
| midazolam | amici | 0.0732 ± 0.00061 | 0.0703 |  |  |
| hctz | core | 0.0943 ± 0.00082 | 0.107 | 0.00297 ± 3.5e-05 | 0.00955 ± 0.00015 |
| hctz | roadrunner | 0.1 ± 0.0011 | 0.191 |  |  |
| hctz | copasi | 0.225 ± 0.0022 | 0.22 |  |  |
| hctz | amici | 0.127 ± 0.0013 | 0.123 |  |  |
| perfusion | core | 3.56 ± 0.067 | 4.14 | 0.155 ± 0.0021 | 0.389 ± 0.0041 |
| perfusion | roadrunner | 4.15 ± 0.022 | 6.01 |  |  |
| perfusion | copasi | 8.05 ± 0.034 | 6.99 |  |  |
| perfusion | amici | 4.36 ± 0.13 | 4.49 |  |  |

### One simulation and a scan

| model | engine | simulate ms | scan serial s | scan parallel s |
|---|---|---:|---:|---:|
| repressilator | core | 1.26 ± 0.014 | 1.11 ± 0.009 | 0.201 ± 0.015 |
| repressilator | roadrunner | 0.913 ± 0.0096 | 0.862 ± 0.0064 | 0.149 ± 0.015 |
| repressilator | copasi | 1.43 ± 0.019 | 1.84 ± 0.0067 | 0.3 ± 0.016 |
| repressilator | amici | 2.71 ± 0.027 | 2.79 ± 0.045 | 0.438 ± 0.044 |
| demo | core | 0.371 ± 0.0022 | 0.0751 ± 0.003 | 0.0414 ± 0.0023 |
| demo | roadrunner | 0.395 ± 0.0021 | 0.269 ± 0.0026 | 0.064 ± 0.0091 |
| demo | copasi | 0.987 ± 0.0092 | 1.36 ± 0.032 | 0.222 ± 0.018 |
| demo | amici | 2.31 ± 0.017 | 2.27 ± 0.049 | 0.338 ± 0.021 |
| midazolam | core | 1.92 ± 0.0079 | 1.41 ± 0.0095 | 0.356 ± 0.011 |
| midazolam | roadrunner | 1.56 ± 0.0072 | 1.68 ± 0.013 | 0.311 ± 0.041 |
| midazolam | copasi | 12 ± 0.085 | 12.4 ± 0.059 | 1.85 ± 0.058 |
| midazolam | amici | 10.5 ± 0.092 | 10.6 ± 0.062 | 2.12 ± 0.042 |
| hctz | core | 2.32 ± 0.029 | 1.63 ± 0.0087 | 0.505 ± 0.044 |
| hctz | roadrunner | 2.3 ± 0.028 | 2.15 ± 0.0048 | 0.56 ± 0.04 |
| hctz | copasi | 12 ± 0.18 | 11.5 ± 0.018 | 2.82 ± 0.086 |
| hctz | amici | 11 ± 0.26 | 10.6 ± 0.082 | 2.63 ± 0.042 |
| perfusion | core | 123 ± 4.1 | 0.936 ± 0.0059 | 0.536 ± 0.022 |
| perfusion | roadrunner | 2.72e+03 ± 75 | 21.3 ± 0.58 | 5.58 ± 0.18 |
| perfusion | copasi | 5.56e+04 ± 1.4e+03 | 353 ± 12 | 83.2 ± 3.4 |
| perfusion | amici | 958 ± 1.4 | 7.74 ± 0.74 | 2.95 ± 0.086 |

![The simulation engines of sbmlsim: one panel per model and measure, the dot of an engine its mean and the whisker its standard deviation, on a logarithmic axis](images/benchmarks/engines.png)

The figure is the two columns of seconds above, a panel per model and measure. Each panel carries the axis of its own model, since the models are five orders of magnitude apart and none of them is comparable with another: each is its own simulation over its own number of points. A dot and not a bar, because a bar on a logarithmic axis encodes its length as the logarithm of its value.

### The agreement with roadrunner

| model | engine | max rel. diff. | max abs. diff. |
|---|---|---:|---:|
| repressilator | core | 1.78e-10 |  |
| repressilator | roadrunner | 0 |  |
| repressilator | copasi | 4.31e-09 |  |
| repressilator | amici | 3.87e-09 |  |
| demo | core | 0 | 0 |
| demo | roadrunner | 0 | 0 |
| demo | copasi | 0 | 0 |
| demo | amici | 0 | 0 |
| midazolam | core | 1.70e-08 | 0 |
| midazolam | roadrunner | 0 | 0 |
| midazolam | copasi | 2.68e-07 | 1.64e-23 |
| midazolam | amici | 2.72e-07 | 0 |
| hctz | core | 2.42e-11 | 0 |
| hctz | roadrunner | 0 | 0 |
| hctz | copasi | 1.62e-08 | 1.17e-21 |
| hctz | amici | 1.63e-08 | 0 |
| perfusion | core | 5.38e-06 | 0 |
| perfusion | roadrunner | 0 | 0 |
| perfusion | copasi | 7.81e-03 | 0 |
| perfusion | amici | 5.46e-06 | 0 |

Measured with python 3.14.10, sbmlsim 0.29.0, libroadrunner 2.11.1.dev655, sbmlsim-core 0.7.0, python-copasi 4.48.311, amici 1.2.0: every cell is the mean and the sample standard deviation of its rounds, a round the median of its timed calls after its warm ups, and the engines take turns within a model, so a change of the load of the machine reaches every engine of a comparison alike. The run measured 20 simulations after 3 warm ups in 5 rounds and scanned 1000 points unless the model above says otherwise, and the parallel scans ran on 8 workers.

- A cell is `mean ± sd` over the rounds of its model and the mean alone below 3 rounds, where a deviation of so small a sample says nothing. A deviation which is a large part of its mean is a cell to measure again on an idle machine rather than one to read.
- `load s` is `load_model` with the model already compiled, `first compile s` the first load of that model in this process, the one which compiles it (the LLJIT of roadrunner, the module of the engine core); the runtime of the engine core is created once per process, so only the first model of a run pays it in this column. AMICI is loaded from a cache which was warmed before the measurement, so its compilation from cold, minutes per model, is not measured, and COPASI, which compiles the model on every load, has the same measurement twice.
- `compile s` and `instantiate s` are the compilation of the SBML into the module of the engine core and its instantiation in the runtime, which `load s` of the other engines does not break out. They are no part of `load s` of the core either, which finds the module compiled (the engine keeps it per SBML and process), and they are part of its `first compile s`.
- `simulate ms` is one `Simulator.simulate` at the defaults of `Simulator`, i.e. the tolerances a run has and not the tighter ones of the agreement below, which is measured in simulations of its own: every speed of this table is the speed of a run as a user gets it. `scan serial s` and `scan parallel s` are that simulation over the points of the scan on one worker and on the workers of the parallel scan below.
- The scan times are warm: every scan runs once before it is timed, so they exclude the pool of worker processes, which is kept per process and number of workers, the model every worker loads and compiles and the first batch of the threads of the engine core. A measurement which includes that first run is not the same measurement: it is several times larger for an engine whose parallel scan runs in worker processes, which carries the pool and one compilation of the model per worker, while the engine core, which starts no pool, pays only its own compilation there.
- The simulation of a model is its horizon with an equidistant output, which every engine computes alike (the steps of the integrator are each engine's own), in the time unit of the model: `repressilator` to 100 in 101 output times, its scan over `n` from 1.5 to 4.0; `demo` to 100 in 101 output times, its scan over `Km_A` from 0.5 to 2.0; `midazolam` to 1440 in 101 output times, its scan over `PODOSE_mid` from 5.0 to 100.0; `hctz` to 1440 in 101 output times, its scan over `PODOSE_hctz` from 5.0 to 100.0; `perfusion` to 60 in 101 output times, its scan over `Q_pump` from 0.0015 to 0.006. How much output a point has matters for the parallel scan of the engine core, see the workers of a parallel scan below.
- The workers of a parallel scan are 8 python processes for roadrunner, COPASI and AMICI, which load the model in every worker, and for the engine core the threads of the engine inside one python process: `Simulator.run` hands the points of a scan to the engine in batches which it integrates on 8 threads of its own (`worker._run_chunk_core`, at most `CORE_MAX_CHUNK` points per batch), so a run of the engine core starts no pool of processes at all and loads the model once. The two do not scale alike: a process runs the whole point, a thread of the engine integrates while the result of the run is assembled in its one python process, so the more output a point has, the less the threads gain.
- `max rel. diff.` and `max abs. diff.` are the agreement with roadrunner at atol 1e-12 and rtol 1e-10: the largest difference of a column divided by the largest magnitude of that column in roadrunner, and, for a column which is identically zero in roadrunner, where a relative difference is undefined, the largest absolute difference. roadrunner is the reference, COPASI is not bitwise reproducible across loads of a model.
- An empty cell is a measurement which was not made.

## What the tables say

- **Loading a model.** On the four packaged models roadrunner and the engine core load an already compiled model within a few percent of each other (94.3 ms against 100 ms for HCTZ), COPASI takes two to three times as long and compiles the model on every load, and AMICI imports its compiled extension in about 1.3 times roadrunner's load. The large model reorders them: the engine core loads it in 3.56 s, roadrunner in 4.15 s, AMICI in 4.36 s and COPASI in 8.05 s. Compiling is where they differ most. roadrunner's LLJIT adds about 91 ms to the load of HCTZ (191 ms against 100 ms) and 1.9 s to the load of the perfusion model (6.01 s against 4.15 s), while the engine core writes the WebAssembly module of HCTZ in 3 ms and instantiates it in 10 ms, and of the perfusion model in 0.16 s and 0.39 s. The 0.59 s of the first model of a run, the repressilator, is almost all the runtime of the engine, which compiles the engine itself and is created once per process. AMICI is the exception the table cannot show: its first compilation of a model is minutes, once per machine, and the measurement warms that cache before it times anything.
- **One simulation.** On the small models the two fast engines trade places and the margins are small: roadrunner leads on the repressilator (0.913 ms against 1.26 ms) and midazolam (1.56 ms against 1.92 ms), the engine core leads on `demo` (0.371 ms against 0.395 ms), and HCTZ is a tie within the deviation of the cells (2.32 ms against 2.30 ms). COPASI and AMICI take five to eight times as long on the two physiologically based models. The large model is a different regime altogether: the engine core simulates it in 123 ms, AMICI in 0.96 s, roadrunner in 2.7 s and COPASI in 56 s - the engine core is 22 times faster than roadrunner and 450 times faster than COPASI on the model where a simulation actually costs something.
- **A serial scan.** The engine core runs the 1000 points of `demo` in 0.075 s against roadrunner's 0.269 s and the 1000 points of HCTZ in 1.63 s against 2.15 s, more than the margins of one simulation of those models. The reason is where the python is: a serial scan of roadrunner runs point by point through python, while a scan of the engine core hands a whole chunk, here up to 250 points, to the engine as one batch, so the per-point python is paid once per chunk. The smaller the model, the more that matters, which is why `demo` gains most and the repressilator, whose oscillating integration is the work, gains nothing at all (1.11 s against 0.862 s). On the eight points of the perfusion model, where the integration is everything, the engine core needs 0.94 s against AMICI's 7.7 s, roadrunner's 21.3 s and COPASI's 353 s.
- **A parallel scan.** On eight workers the engine core is ahead on `demo` (0.041 s against 0.064 s) and HCTZ (0.505 s against 0.56 s) and roadrunner is ahead on the repressilator (0.149 s against 0.201 s) and midazolam (0.311 s against 0.356 s), the same four orderings as the serial scans and for the same reason. The two parallelize different things: roadrunner's eight worker processes each run whole points, their integration and the python around them, while the engine core runs eight threads of the engine inside one python process, which assembles every result alone. That shows on the large model, where the engine core goes from 0.94 s to 0.54 s, a factor of 1.7 on eight threads, while roadrunner goes from 21.3 s to 5.6 s, a factor of 3.8 on eight processes - and the engine core still finishes the scan ten times sooner. The section below measures that one python process directly, cold against warm.
- **The agreement.** At atol 1e-12 and rtol 1e-10 every engine agrees with roadrunner to 2.8e-07 relative or better on the packaged models, and on `demo` every one of them reproduces roadrunner exactly. The engine core is the closest there, by a factor of 16 to 660 on the models which differ at all (2.42e-11 on HCTZ against 1.68e-08 for COPASI and 1.63e-08 for AMICI), because it integrates with the vector of absolute tolerances which roadrunner computes per state, while COPASI and AMICI have a single absolute tolerance and are given the smallest one with a safety factor, see [Absolute tolerances](simulation.md#absolute-tolerances). The large model separates them: the engine core and AMICI stay at 5.4e-06 of roadrunner, while COPASI is at 7.8e-03, five orders of magnitude further away, which is a model to compare engine by engine before trusting a number, see [Comparing the engines](simulation.md#comparing-the-engines). Agreeing with roadrunner more closely is not the same as being more accurate: the roadrunner rows are 0 because roadrunner is the reference of the column, not because it is the truth, and the error of the engine core is not bounded by roadrunner's, see [core](simulation.md#core).

## A cold scan and a warm scan are different measurements

`scan serial s` and `scan parallel s` are warm: every scan of the table runs once before it is timed. That first run is where a parallel scan of roadrunner, COPASI or AMICI starts its pool of worker processes and every worker loads and compiles the model, and where the engine core compiles its module and starts its first batch of threads. `sbmlsim.parallel` keeps a pool per process and number of workers, so the second scan of a session finds all of it ready.

`--cold` times that first run as well and reports it next to the warm measurement. It is a cold measurement of the first model and engine of a process only, since the pool and the runtime of the engine core are kept per process, so each engine was measured in a run of its own, the HCTZ model over 1000 points on 8 workers:

```bash
# the cold scans of one engine next to its warm ones, in a process of its own
for engine in core roadrunner copasi amici; do
  taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py \
      --models hctz --engines "$engine" --workers 8 --cold
done
```

| engine | scan parallel cold s | scan parallel s | cold / warm |
| --- | ---: | ---: | ---: |
| core | 0.508 | 0.486 ± 0.044 | 1.0 |
| roadrunner | 2.48 | 0.643 ± 0.023 | 3.9 |
| copasi | 4.28 | 2.99 ± 0.2 | 1.4 |
| amici | 9.85 | 4.02 ± 0.76 | 2.4 |

The first parallel scan of a process costs roadrunner 3.9 times its warm one and AMICI 2.4 times, because they start eight worker processes and load the model in each of them, while the engine core, which starts no pool, pays nothing measurable for its first batch of threads. So the engines change places with the question being asked: the first scan of 1000 points in a process costs 0.51 s in the engine core and 2.5 s in roadrunner, a factor of five, while in a session or a fit which scans again and again every later scan costs 0.49 s in the engine core and 0.64 s in roadrunner, a factor of 1.3. A single number for "how much faster" says nothing unless it says which of the two it is, and at which output grid, see the next section.

A serial scan has no cold run worth the name: the same four runs give 2.15 s cold against 2.31 s warm for the engine core, 2.9 against 2.87 for roadrunner, 17.5 against 15.9 for COPASI and 11.8 against 12.0 for AMICI, two of them below their warm mean, which is the spread of the method and not a cold cost. The pool of worker processes is what a cold scan pays for.

## The width of the output grid

A thread of the engine core integrates while the one python process of the run assembles every result, so the wider the output grid of a point, the less the threads gain. `--steps` scans every model on another equidistant grid, here the same scan of HCTZ, 1000 points on 8 workers, warm, with the two engines taking turns in one process:

```bash
# the same scan on four output grids, `steps + 1` output times per point
for steps in 10 25 100 400; do
  taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py \
      --models hctz --engines core roadrunner --workers 8 --steps "$steps"
done
```

| output times per point | core s | roadrunner s | core / roadrunner |
| ---: | ---: | ---: | ---: |
| 11 | 0.301 | 0.353 | 0.85 |
| 26 | 0.308 | 0.413 | 0.75 |
| 101 | 0.417 | 0.494 | 0.84 |
| 401 | 0.905 | 0.795 | 1.14 |

This is the one table on the page which was measured with sbmlsim-core 0.2.0 and sbmlsim 0.25.0 rather than the versions below the tables above; the measurement which would replace it came out while the machine was busy, with deviations of up to a whole mean, and a noisy table says less than a dated one. Both engines get slower with a wider grid and the engine core gets slower faster: it is ahead by 15 to 25 % up to about a hundred output times per point and behind by about 15 % at 401. How large that lead is moves with the run, 0.85, 0.75 and 0.84 of roadrunner's time on the three narrower grids; what reproduces is the lead itself and the crossing between 101 and 401 output times. That is the mechanism behind the workers of a parallel scan: threads of the engine waiting on the one python process of the run. The serial scans of the same four runs show the other side of it, 1.10, 1.14, 1.31 and 2.03 seconds for the engine core against 1.86, 1.88, 2.30 and 4.26 for roadrunner, so the engine keeps its serial advantage at every grid and it is the threads which pay for the width. The tables at the top of the page use 101 output times for every model. A single simulation has no such threads either, so none of this reaches it.

## Measuring your own model

The numbers which matter are the ones of your model on your machine. Everything of the benchmark is in `scripts/engine_benchmark.py`:

```bash
# one model, a smaller scan, the tables, the figure and the measurement written out
mkdir -p results
uv run --no-sync python scripts/engine_benchmark.py --models hctz --points 200 \
    --out results/tables.md --figure results/engines.png --json results/benchmark.json
```

`results/` is ignored by the repository, so the files do not turn up as changes of a checkout. The method is the one of a microbenchmark: the simulations of a cell are timed `--repetitions` times after three untimed ones (`WARMUPS` of the script), that is one round whose median is its time, and the cell is the mean of `--rounds` such rounds and their sample standard deviation, which is the `±` of every timed cell (the mean alone below three rounds, where a deviation of so small a sample says nothing); the engines of a model take turns within a round, so the machine treats them alike, and a model is measured on its own with the points and rounds it carries. A scan is seconds rather than milliseconds, so it runs once before the timed runs and then once per round. The agreement is measured in simulations of its own at atol 1e-12 and rtol 1e-10, where the engines agree, see [Comparing the engines](simulation.md#comparing-the-engines), while every speed of the table is measured at the defaults of `Simulator`, i.e. at the tolerances of a run. Pin the process to a fixed set of cores and leave the machine alone, as the runs of this page did, if you want numbers which compare across runs: every disturbed cell on this page announced itself as a deviation of a quarter of its mean or more, which is what the `±` is there for.
