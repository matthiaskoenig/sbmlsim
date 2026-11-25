# Benchmarks

`sbmlsim` runs a simulation with one of four engines, see [Engines](simulation.md#engines): the Rust engine [sbmlsim-core](simulation.md#sbmlsim-core) ([the package](https://github.com/matthiaskoenig/sbmlsim-core), compiled to WebAssembly, see [Optional packages](installation.md#optional-packages)), [roadrunner](simulation.md#roadrunner), [COPASI](simulation.md#copasi) and [AMICI](simulation.md#amici). This page is what each of them costs. `scripts/engine_benchmark.py` loads, simulates and scans four packaged models and one large model in every engine which is installed, compares the results with roadrunner and prints the tables below.

Read a cell together with its configuration. Every number belongs to one machine, one model, one number of output times, one number of points and one number of workers, and the engines change places when any of those change: the two sections after the tables measure the same scan again, cold against warm and over the width of the output grid, and the same two engines run between 6.8 times apart and 1.4 times apart across them. The tables answer "what does this cost here", they do not rank the engines.

## The run

```bash
# every model in every engine which is installed, pinned to eight cores
taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py --workers 8 \
    --out tables.md --figure docs/images/benchmarks/engines.svg
```

That command printed the tables below and drew the figures: the four packaged models and the large model, every engine which is installed, and the scans over the default of 1000 points on 8 workers. `--out FILE` writes the tables into a file, `--figure FILE` draws the comparison and `--json FILE` writes the measurement with every number as it was measured; `--figure` names a stem and not a file, since a model gets a figure of its own, drawn by the renderer of sbmlsim-core and written as SVG, so the command above writes `engines-hctz.svg` and the four others beside it, and `--from-json FILE` draws them again from a stored measurement without measuring anything; the images below are the `engines-<model>-light.png` and `-dark.png` files of the previous measurement, and the next run writes `engines-<model>.svg` instead, which the page then embeds; `--models`, `--engines`, `--points`, `--repetitions` and `--rounds` narrow or widen the run, and `--cold` and `--steps` are the measurements of the two sections after it. Nothing regenerates this page, so a new measurement is a run of the script and a commit of its tables and its figure.

The rows of the large model are a second run of that command with `--models perfusion`, merged into the first with `--from-json`. In the first run another process on the machine took the eight cores while the simulations of the large model were timed, which put its single simulation at twice what the serial scan of the same run reports for a point of the same model, so those four rows were measured again an hour later on an idle machine and the two measurements were merged into the tables and the figures below. Every other cell is of the first run.

The large model is `benchmarks/models/machine_perfusion_liver_flat.xml`, the liver of a normothermic machine perfusion: 839 species, 1007 reactions, 570 parameters, 387 rules and 43 compartments in 2.6 MB, which is twelve times the largest packaged model, scanned over the flow of its pump. It is in the repository and no part of the package, so no install carries it, see `benchmarks/models/README.md`. One of its simulations is tens of seconds in COPASI, so it is scanned over as many points as the parallel scan has workers and measured in fewer and shorter rounds than the packaged models, which the first table of the run says of every model.

`--no-sync` keeps the environment as it is. COPASI comes with the extra `dev` and sbmlsim-core with every install, since its bindings are a dependency of sbmlsim in `uv.lock`. AMICI comes from the extra `amici` (`uv sync --extra dev --extra amici`), which is not in `uv.lock`, see [Optional packages](installation.md#optional-packages); a plain `uv run`, which syncs the environment to the lock, uninstalls it and measures the three other engines. An engine which is not installed is skipped and named below the table, so a shorter table is never a slow engine.

The machine: a 12th Gen Intel Core i7-12700 (12 cores, 20 hardware threads) with 62 GB of memory, Ubuntu 24.04.5 with the kernel 7.0.0-43, python 3.14.10. Every measurement on this page was pinned to one hardware thread of each of the eight performance cores (`taskset -c 0,2,4,6,8,10,12,14`), which is where the eight workers of a parallel scan run. It is an interactive desktop and the CPU governor is `powersave`, so the clocks are the kernel's choice and not a fixed frequency.

What that leaves of a cell: a cell is the median of its rounds, each round the median of its timed calls, and the engines of a model take turns within a round. A median of medians is what makes a measurement on a machine which is not perfectly idle worth reading at all: a round which was slow because something else ran moves a mean and does not move a median, and an earlier version of this page reported means with their deviations, where a disturbed cell announced itself with a deviation of a quarter of its mean and had to be measured again. The ratios against sbmlsim-core are the other half of the same idea. A ratio of two medians measured in the same turns of the same run survives a machine which is uniformly busy, while the seconds do not, so the ratios are the part of this page which carries to another machine. Pinning is what makes either possible: an earlier unpinned measurement had roadrunner ahead of sbmlsim-core on the parallel scan of HCTZ, 0.506 seconds against 0.622, the opposite of every pinned run since.

## The figures

A figure per model, a panel per measure and a row per engine, on a logarithmic axis of its own. The dot of an engine is the median of its rounds, the same number the tables below carry, so a figure is the shape of a comparison and the tables are its digits. A dot and not a bar, because a bar on a logarithmic axis encodes its length as the logarithm of its value, and the engine of every row is named beside it, so a mark is never told apart by its colour alone. The colours are the ones of the [benchmark of sbmlsim-core](https://matthiaskoenig.github.io/sbmlsim-core/benchmarks/): sbmlsim-core is its blue and roadrunner is the pink it has there, so one engine is one colour across both projects.

The models have a figure each and not a panel each of one figure, because they are five orders of magnitude apart and are not comparable with each other anyway: each is its own simulation over its own number of points, which the first table of the next section says of every model. The panels of a figure do not share their axis either, for the same reason within a model: a scan of 1000 points and one simulation are three orders apart.

### repressilator

The repressilator, the oscillator of three genes.

![The engines of repressilator: one simulation and the scans](images/benchmarks/engines-repressilator-light.png#only-light)
![The engines of repressilator: one simulation and the scans](images/benchmarks/engines-repressilator-dark.png#only-dark)

### demo

`demo`, the smallest packaged model.

![The engines of demo: one simulation and the scans](images/benchmarks/engines-demo-light.png#only-light)
![The engines of demo: one simulation and the scans](images/benchmarks/engines-demo-dark.png#only-dark)

### midazolam

Midazolam, a physiologically based model of the drug.

![The engines of midazolam: one simulation and the scans](images/benchmarks/engines-midazolam-light.png#only-light)
![The engines of midazolam: one simulation and the scans](images/benchmarks/engines-midazolam-dark.png#only-dark)

### hctz

HCTZ, the model of the tutorial and of the fitting example.

![The engines of hctz: one simulation and the scans](images/benchmarks/engines-hctz-light.png#only-light)
![The engines of hctz: one simulation and the scans](images/benchmarks/engines-hctz-dark.png#only-dark)

### perfusion

The liver of a machine perfusion, the large model.

![The engines of perfusion: one simulation and the scans](images/benchmarks/engines-perfusion-light.png#only-light)
![The engines of perfusion: one simulation and the scans](images/benchmarks/engines-perfusion-dark.png#only-dark)

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
| repressilator | sbmlsim-core | 0.0209 | 0.0296 | 0.000601 | 0.000168 |
| repressilator | roadrunner | 0.0191 | 0.0377 |  |  |
| repressilator | copasi | 0.0512 | 0.0515 |  |  |
| repressilator | amici | 0.0284 | 0.0286 |  |  |
| demo | sbmlsim-core | 0.0248 | 0.0263 | 0.00104 | 0.000148 |
| demo | roadrunner | 0.0232 | 0.0406 |  |  |
| demo | copasi | 0.0512 | 0.0494 |  |  |
| demo | amici | 0.0312 | 0.0304 |  |  |
| midazolam | sbmlsim-core | 0.0571 | 0.0847 | 0.00532 | 0.000709 |
| midazolam | roadrunner | 0.055 | 0.201 |  |  |
| midazolam | copasi | 0.163 | 0.199 |  |  |
| midazolam | amici | 0.0778 | 0.0949 |  |  |
| hctz | sbmlsim-core | 0.0952 | 0.107 | 0.00494 | 0.000481 |
| hctz | roadrunner | 0.1 | 0.183 |  |  |
| hctz | copasi | 0.225 | 0.223 |  |  |
| hctz | amici | 0.128 | 0.125 |  |  |
| perfusion | sbmlsim-core | 3.41 | 3.52 | 0.223 | 0.00741 |
| perfusion | roadrunner | 3.95 | 5.4 |  |  |
| perfusion | copasi | 7.86 | 6.51 |  |  |
| perfusion | amici | 4.13 | 4.06 |  |  |

### One simulation and a scan

| model | engine | simulate ms | scan serial s | scan parallel s |
|---|---|---:|---:|---:|
| repressilator | sbmlsim-core | 0.563 | 0.571 | 0.102 |
| repressilator | roadrunner | 0.858 | 0.821 | 0.128 |
| repressilator | copasi | 1.44 | 1.84 | 0.273 |
| repressilator | amici | 2.77 | 2.84 | 0.397 |
| demo | sbmlsim-core | 0.113 | 0.0428 | 0.0275 |
| demo | roadrunner | 0.38 | 0.275 | 0.0667 |
| demo | copasi | 0.977 | 1.4 | 0.281 |
| demo | amici | 2.32 | 2.3 | 0.439 |
| midazolam | sbmlsim-core | 0.656 | 0.624 | 0.179 |
| midazolam | roadrunner | 1.37 | 1.27 | 0.261 |
| midazolam | copasi | 12.9 | 13.5 | 2.29 |
| midazolam | amici | 11.6 | 11.1 | 2.27 |
| hctz | sbmlsim-core | 0.877 | 0.76 | 0.162 |
| hctz | roadrunner | 1.68 | 1.5 | 0.237 |
| hctz | copasi | 12.1 | 12 | 1.67 |
| hctz | amici | 11 | 10.6 | 1.85 |
| perfusion | sbmlsim-core | 58.8 | 0.465 | 0.0982 |
| perfusion | roadrunner | 84.9 | 0.671 | 0.148 |
| perfusion | copasi | 4.82e+04 | 351 | 80.7 |
| perfusion | amici | 899 | 7.2 | 2.81 |

### Against sbmlsim-core

| model | engine | load | one simulation | a serial scan | a parallel scan |
|---|---|---:|---:|---:|---:|
| repressilator | sbmlsim-core | 1 | 1 | 1 | 1 |
| repressilator | roadrunner | 0.917 | 1.53 | 1.44 | 1.25 |
| repressilator | copasi | 2.46 | 2.56 | 3.22 | 2.66 |
| repressilator | amici | 1.36 | 4.92 | 4.98 | 3.88 |
| demo | sbmlsim-core | 1 | 1 | 1 | 1 |
| demo | roadrunner | 0.937 | 3.36 | 6.43 | 2.42 |
| demo | copasi | 2.07 | 8.64 | 32.8 | 10.2 |
| demo | amici | 1.26 | 20.6 | 53.8 | 15.9 |
| midazolam | sbmlsim-core | 1 | 1 | 1 | 1 |
| midazolam | roadrunner | 0.963 | 2.09 | 2.04 | 1.46 |
| midazolam | copasi | 2.85 | 19.7 | 21.6 | 12.8 |
| midazolam | amici | 1.36 | 17.7 | 17.8 | 12.7 |
| hctz | sbmlsim-core | 1 | 1 | 1 | 1 |
| hctz | roadrunner | 1.06 | 1.92 | 1.97 | 1.47 |
| hctz | copasi | 2.37 | 13.8 | 15.8 | 10.4 |
| hctz | amici | 1.34 | 12.5 | 13.9 | 11.4 |
| perfusion | sbmlsim-core | 1 | 1 | 1 | 1 |
| perfusion | roadrunner | 1.16 | 1.44 | 1.44 | 1.5 |
| perfusion | copasi | 2.3 | 821 | 753 | 822 |
| perfusion | amici | 1.21 | 15.3 | 15.5 | 28.6 |

### The agreement with roadrunner

| model | engine | max rel. diff. | max abs. diff. |
|---|---|---:|---:|
| repressilator | sbmlsim-core | 4.28e-12 |  |
| repressilator | roadrunner | 0 |  |
| repressilator | copasi | 4.33e-09 |  |
| repressilator | amici | 3.69e-09 |  |
| demo | sbmlsim-core | 0 | 0 |
| demo | roadrunner | 0 | 0 |
| demo | copasi | 0 | 0 |
| demo | amici | 0 | 0 |
| midazolam | sbmlsim-core | 1.70e-08 | 0 |
| midazolam | roadrunner | 0 | 0 |
| midazolam | copasi | 2.68e-07 | 1.64e-23 |
| midazolam | amici | 2.72e-07 | 0 |
| hctz | sbmlsim-core | 1.20e-11 | 0 |
| hctz | roadrunner | 0 | 0 |
| hctz | copasi | 1.72e-08 | 5.94e-22 |
| hctz | amici | 1.63e-08 | 0 |
| perfusion | sbmlsim-core | 7.49e-06 | 0 |
| perfusion | roadrunner | 0 | 0 |
| perfusion | copasi | 3.51e-02 | 0 |
| perfusion | amici | 7.45e-06 | 0 |

Measured with python 3.14.10, sbmlsim 0.31.1, libroadrunner 2.12.0.dev700, sbmlsim-core 0.13.0, python-copasi 4.48.311, amici 1.2.0: every cell is the median of its rounds, a round the median of its timed calls after its warm ups, and the engines take turns within a model, so a change of the load of the machine which outlasts their turns reaches every engine of a comparison alike, and one which is shorter than a phase of the run can still move a single cell. The run measured 20 simulations after 3 warm ups in 5 rounds and scanned 1000 points unless the model above says otherwise, and the parallel scans ran on 8 workers.

- A cell is the median of the rounds of its model, each round the median of its timed calls: a round which was slow because the machine was doing something else moves a mean and not a median, which is why the tables report medians and no deviation.
- The table against sbmlsim-core is the median of an engine divided by the median of sbmlsim-core on the same model and measure, so sbmlsim-core is 1, a number above it is that many times its time and a number below it is a cell it loses. The ratios are what carries from one machine to another; the seconds of the tables above belong to this one.
- `load s` is `load_model` with the model already compiled, `first compile s` the first load of that model in this process, the one which compiles it (the LLJIT of roadrunner, the module of sbmlsim-core); the runtime of sbmlsim-core is created once per process, so only the first model of a run pays it in this column. AMICI is loaded from a cache which was warmed before the measurement, so its compilation from cold, minutes per model, is not measured, and COPASI, which compiles the model on every load, has the same measurement twice.
- `compile s` and `instantiate s` are the compilation of the SBML into the module of sbmlsim-core and its instantiation in the runtime, which `load s` of the other engines does not break out. They are no part of `load s` of the core either, which finds the module compiled (the engine keeps it per SBML and process), and they are part of its `first compile s`.
- `simulate ms` is one `Simulator.simulate` at the defaults of `Simulator`, i.e. the tolerances a run has and not the tighter ones of the agreement below, which is measured in simulations of its own: every speed of this table is the speed of a run as a user gets it. `scan serial s` and `scan parallel s` are that simulation over the points of the scan on one worker and on the workers of the parallel scan below.
- The scan times are warm: every scan runs once before it is timed, so they exclude the pool of worker processes, which is kept per process and number of workers, the model every worker loads and compiles and the first batch of the threads of sbmlsim-core. A measurement which includes that first run is not the same measurement: it is several times larger for an engine whose parallel scan runs in worker processes, which carries the pool and one compilation of the model per worker, while sbmlsim-core, which starts no pool, pays only its own compilation there.
- The simulation of a model is its horizon with an equidistant output, which every engine computes alike (the steps of the integrator are each engine's own), in the time unit of the model: `repressilator` to 100 in 101 output times, its scan over `n` from 1.5 to 4.0; `demo` to 100 in 101 output times, its scan over `Km_A` from 0.5 to 2.0; `midazolam` to 1440 in 101 output times, its scan over `PODOSE_mid` from 5.0 to 100.0; `hctz` to 1440 in 101 output times, its scan over `PODOSE_hctz` from 5.0 to 100.0; `perfusion` to 60 in 101 output times, its scan over `Q_pump` from 0.0015 to 0.006. How much output a point has matters for the parallel scan of sbmlsim-core, see the workers of a parallel scan below.
- The workers of a parallel scan are 8 python processes for roadrunner, COPASI and AMICI, which load the model in every worker, and for sbmlsim-core the threads of the engine inside one python process: `Simulator.run` hands the points of a scan to the engine in batches which it integrates on 8 threads of its own (`worker._run_chunk_core`, at most `CORE_MAX_CHUNK` points per batch), so a run of sbmlsim-core starts no pool of processes at all and loads the model once. The two do not scale alike: a process runs the whole point, its integration and the python around it, while a thread of the engine integrates only and the one python process of the run assembles the result of every point itself, so the more output a point has, the less the threads gain.
- `max rel. diff.` and `max abs. diff.` are the agreement with roadrunner at atol 1e-12 and rtol 1e-10: the largest difference of a column divided by the largest magnitude of that column in roadrunner, and, for a column which is identically zero in roadrunner, where a relative difference is undefined, the largest absolute difference. roadrunner is the reference, COPASI is not bitwise reproducible across loads of a model.
- An empty cell is a measurement which was not made.

## What the tables say

**roadrunner 2.12 changed this page.** Up to libroadrunner 2.11 CVODE factorized a dense Jacobian, and the large model, whose Jacobian is 0.45 % dense at 919 states, spent nearly all of its time there: one simulation took 2.7 s and sbmlsim-core was 22 times faster. roadrunner 2.12 ([#1450](https://github.com/sys-bio/roadrunner/pull/1450)) added the sparse direct solver KLU and a colored sparse difference quotient Jacobian, and both are its defaults (`linear_solver="klu"`, `jacobian="sparse_difference_quotient"`). The same simulation now takes 84.9 ms, a factor of 28, which is what brought roadrunner back within reach of sbmlsim-core on that model: it trails its single simulation by 1.4 and its parallel scan by 1.5. The two settings only work together: with either one alone CVODE falls back to the dense factorization and the model takes 2.4 s, since at 919 states the factorization is the whole cost and the Jacobian evaluation is not.

- **Loading a model.** On the four packaged models roadrunner and sbmlsim-core load an already compiled model within a few percent of each other (0.100 s against 0.0952 s for HCTZ), COPASI takes a bit over twice as long and compiles the model on every load, and AMICI imports its compiled extension in about 1.3 times roadrunner's load. The large model keeps that order: sbmlsim-core loads it in 3.41 s, roadrunner in 3.95 s, AMICI in 4.13 s and COPASI in 7.86 s. Compiling is where they differ most. roadrunner's LLJIT adds about 80 ms to the load of HCTZ (0.183 s against 0.100 s) and 1.4 s to the load of the perfusion model (5.40 s against 3.95 s), while sbmlsim-core writes the WebAssembly module of HCTZ in 4.9 ms and instantiates it in 0.5 ms, and of the perfusion model in 223 ms and 7.4 ms. AMICI is the exception the table cannot show: its first compilation of a model is minutes, once per machine, and the measurement warms that cache before it times anything.
- **One simulation.** sbmlsim-core leads every model of this run, by 1.4 to 3.4 times: the repressilator 0.563 ms against roadrunner's 0.858 ms, `demo` 0.113 ms against 0.380 ms, midazolam 0.656 ms against 1.37 ms, HCTZ 0.877 ms against 1.68 ms and the large model 58.8 ms against 84.9 ms. It did not lead before sbmlsim-core 0.9.0 and sbmlsim 0.30.0: the wheel ran `engine.wasm` in wasmtime, the slowest host the engine has, and sbmlsim built an instance, a vector of tolerances and the targets of the changes again for every simulation, which on HCTZ was most of the 2.34 ms this page used to report. COPASI and AMICI take 12 to 20 times sbmlsim-core on the two physiologically based models. The large model is where the numbers separate: sbmlsim-core 58.8 ms, roadrunner 84.9 ms, AMICI 899 ms and COPASI 48.2 s, so COPASI is 820 times sbmlsim-core on the model where a simulation actually costs something.
- **A serial scan.** sbmlsim-core runs the 1000 points of `demo` in 0.0428 s against roadrunner's 0.275 s, six times less, and HCTZ in 0.760 s against 1.50 s. The reason is where the python is: a serial scan of roadrunner runs point by point through python, while a scan of sbmlsim-core hands a whole chunk, here up to 250 points, to the engine as one batch, so the per-point python is paid once per chunk. The smaller the model, the more that matters, which is why `demo` gains most and the repressilator, whose oscillating integration is the work, gains least (0.571 s against 0.821 s). On the eight points of the perfusion model, where the integration is everything, sbmlsim-core needs 0.465 s against roadrunner's 0.671 s, AMICI's 7.20 s and COPASI's 351 s.
- **A parallel scan.** On eight workers sbmlsim-core takes all five models, the large one by 1.5 (0.0982 s against roadrunner's 0.148 s). The two parallelize different things: roadrunner's eight worker processes each run whole points, their integration and the python around them, while sbmlsim-core hands the points of a chunk to the engine as one batch, which spreads them over eight threads of its own, and the one python process of the run assembles every result itself. That shows on the large model, where roadrunner goes from 0.671 s to 0.148 s, a factor of 4.5 on eight processes, and sbmlsim-core from 0.465 s to 0.0982 s, a factor of 4.7 on eight threads. What is left of its run is the slowest point and the assembly: a point of this scan costs 56 to 91 ms when it runs alone, so eight threads cannot finish before the slowest of them, and the 3301 variables of the result are assembled in that one python process. The factor was 1.8 until sbmlsim 0.32.0, when this cell read 0.268 s: a chunk was sized for a pool of processes, four chunks per worker, which handed the engine two of the eight points at a time and ran the four batches one after the other, so six of the eight threads idled through each of them ([#576](https://github.com/matthiaskoenig/sbmlsim/pull/576)). The closest race of the five is the repressilator, 0.102 s against 0.128 s. The section below measures that one python process directly, cold against warm.
- **The agreement.** At atol 1e-12 and rtol 1e-10 every engine agrees with roadrunner to 2.7e-07 relative or better on the packaged models, and on `demo` every one of them reproduces roadrunner exactly. sbmlsim-core is the closest there, by a factor of 16 on midazolam and of about 1400 on HCTZ and 1000 on the repressilator, because it integrates with the vector of absolute tolerances which roadrunner computes per state, while COPASI and AMICI have a single absolute tolerance and are given the smallest one with a safety factor, see [Absolute tolerances](simulation.md#absolute-tolerances). The large model separates them differently: sbmlsim-core and AMICI are level there, both at 7.5e-06 of roadrunner, while COPASI is at 7.8e-03, three orders of magnitude further away, which is a model to compare engine by engine before trusting a number, see [Comparing the engines](simulation.md#comparing-the-engines). Agreeing with roadrunner more closely is not the same as being more accurate: the roadrunner rows are 0 because roadrunner is the reference of the column, not because it is the truth, and the error of sbmlsim-core is not bounded by roadrunner's, see [sbmlsim-core](simulation.md#sbmlsim-core).

## The sparse solver of roadrunner

The factor of 28 above is worth its own measurement, since it is the difference between a page where roadrunner is last on large models and one where it is first. `linear_solver` and `jacobian` are settings of `Simulator`, so the four combinations are measured in one process on the perfusion model, each the median of three rounds:

| `linear_solver` | `jacobian` | one simulation |
| --- | --- | ---: |
| `klu` | `sparse_difference_quotient` | 82.5 ms |
| `dense` | `difference_quotient` | 2410 ms |
| `dense` | `sparse_difference_quotient` | 2380 ms |
| `klu` | `difference_quotient` | 2430 ms |

The first row is what roadrunner 2.12 does by default and the second is what 2.11 did. Neither half helps alone: KLU with a dense difference quotient Jacobian falls back to the dense solver, and the sparse Jacobian with the dense solver saves nothing worth measuring, because the factorization and not the Jacobian evaluation is where a model of 919 states spends its time. A model with no sparsity pattern, for instance one under conserved moiety analysis, takes the dense path whatever these settings say. The packaged models are too small for any of this to matter: HCTZ has 22 states, where the dense factorization is a rounding error. The four rows were measured back to back in one process, each the median of three rounds of three simulations, which is what the ratio rests on.

## A cold scan and a warm scan are different measurements

`scan serial s` and `scan parallel s` are warm: every scan of the table runs once before it is timed. That first run is where a parallel scan of roadrunner, COPASI or AMICI starts its pool of worker processes and every worker loads and compiles the model, and where sbmlsim-core compiles its module and starts its first batch of threads. `sbmlsim.parallel` keeps a pool per process and number of workers, so the second scan of a session finds all of it ready.

`--cold` times that first run as well and reports it next to the warm measurement. It is a cold measurement of the first model and engine of a process only, since the pool and the runtime of sbmlsim-core are kept per process, so each engine was measured in a run of its own, the HCTZ model over 1000 points on 8 workers:

```bash
# the cold scans of one engine next to its warm ones, in a process of its own
for engine in core roadrunner copasi amici; do
  taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py \
      --models hctz --engines "$engine" --workers 8 --cold
done
```

| engine | scan parallel cold s | scan parallel s | cold / warm |
| --- | ---: | ---: | ---: |
| sbmlsim-core | 0.180 | 0.160 | 1.1 |
| roadrunner | 1.22 | 0.246 | 5.0 |
| copasi | 2.74 | 1.63 | 1.7 |
| amici | 3.13 | 1.92 | 1.6 |

The first parallel scan of a process costs roadrunner 5.0 times its warm one, because it starts eight worker processes and loads the model in each of them, while sbmlsim-core, which starts no pool and whose native engine has nothing to compile at a first batch, pays a tenth of its scan for it. So the question being asked changes the margin by a factor of four, even where it does not change the order: the first scan of 1000 points in a process costs 0.180 s in sbmlsim-core against 1.22 s in roadrunner, 6.8 times, while in a session or a fit which scans again and again every later scan costs 0.160 s against 0.246 s, 1.5 times. A single number for "how much faster" says nothing unless it says which of the two it is, and at which output grid, see the next section. Until sbmlsim 0.30.0 the two engines changed places here, sbmlsim-core winning the cold scan and roadrunner the warm one; what removed that crossing was the native engine of the wheel and the instance, tolerances and targets which sbmlsim now keeps per model.

A serial scan has no cold run worth the name: the same four runs give 0.758 s cold against 0.755 s warm for sbmlsim-core, 1.52 against 1.50 for roadrunner, 11.5 against 11.4 for COPASI and 9.81 against 9.83 for AMICI, every one of them within the noise of its own measurement. The pool of worker processes is what a cold scan pays for.

## The width of the output grid

A thread of sbmlsim-core integrates while the one python process of the run assembles every result, so the wider the output grid of a point, the more of the run is that assembly. `--steps` scans every model on another equidistant grid, here the same scan of HCTZ, 1000 points on 8 workers, warm, with the two engines taking turns in one process:

```bash
# the same scan on four output grids, `steps + 1` output times per point
for steps in 10 25 100 400; do
  taskset -c 0,2,4,6,8,10,12,14 uv run --no-sync python scripts/engine_benchmark.py \
      --models hctz --engines core roadrunner --workers 8 --steps "$steps"
done
```

| output times per point | sbmlsim-core s | roadrunner s | roadrunner / sbmlsim-core |
| ---: | ---: | ---: | ---: |
| 11 | 0.120 | 0.167 | 1.39 |
| 26 | 0.124 | 0.186 | 1.50 |
| 101 | 0.169 | 0.263 | 1.56 |
| 401 | 0.255 | 0.539 | 2.11 |

Both engines get slower with a wider grid, sbmlsim-core leads at every width, and the grid no longer eats its lead: 1.39 at eleven output times, 1.56 at 101 and 2.11 at 401, where roadrunner's parallel scan more than doubles between the last two widths and sbmlsim-core's grows by half. Until sbmlsim 0.30.0 this table read 1.68, 1.56, 1.23 and 0.99, a lead the grid ate until the two engines ran level at 401 output times. The serial scans of the same four runs are 0.672, 0.689, 0.745 and 0.959 seconds for sbmlsim-core against 1.12, 1.18, 1.51 and 2.70 for roadrunner, a lead of 1.7 to 2.8 which grows with the width without a dip.

What changed is sbmlsim and not the engine. The values of a point went from the solution into the result through four passes, two of which were waste: a selection of every column in order, which copied a whole solution to keep all of it, and a transpose of every chunk into the cube, which a chunk in the layout of the cube does not need. They are gone, and a run in the calling process now writes the values of a point straight into the result, so they are copied once instead of four times: the parallel scan of 401 output times went from 0.619 s to 0.247 s, and the assembly of the result, which was 112 ms of that scan, is 2 ms. roadrunner's serial scan gained a little from the same change, which was in its path too; its parallel scan did not, because its eight worker processes were already copying in parallel with each other. That is the part sbmlsim-core could not hide: it runs its threads in one process, so it paid the whole of the assembly in the open, and the measurement which looked like a limit of the engine was three unnecessary passes over 613 MB.

The tables at the top use 101 output times for every model. A single simulation assembles one solution, so none of this reaches it: there sbmlsim-core leads by 1.6 to 2.6 across the same four widths.

## Measuring your own model

The numbers which matter are the ones of your model on your machine. Everything of the benchmark is in `scripts/engine_benchmark.py`:

```bash
# one model, a smaller scan, the tables, the figure and the measurement written out
mkdir -p results
uv run --no-sync python scripts/engine_benchmark.py --models hctz --points 200 \
    --out results/tables.md --figure results/engines.svg --json results/benchmark.json
```

`results/` is ignored by the repository, so the files do not turn up as changes of a checkout. The method is the one of a microbenchmark: the simulations of a cell are timed `--repetitions` times after three untimed ones (`WARMUPS` of the script), that is one round whose median is its time, and the cell is the median of `--rounds` such rounds; the engines of a model take turns within a round, so the machine treats them alike, and a model is measured on its own with the points and rounds it carries. A scan is seconds rather than milliseconds, so it runs once before the timed runs and then once per round. The agreement is measured in simulations of its own at atol 1e-12 and rtol 1e-10, where the engines agree, see [Comparing the engines](simulation.md#comparing-the-engines), while every speed of the table is measured at the defaults of `Simulator`, i.e. at the tolerances of a run. Pin the process to a fixed set of cores, as the runs of this page did, if you want numbers which compare across runs, and read the ratios rather than the seconds if you want to compare with the tables above: the medians survive a machine which is busy now and then, and the ratios survive a machine which is slower than this one throughout.
