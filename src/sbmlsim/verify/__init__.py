"""Verify that a change of a model changes only what it is meant to change.

A refactoring of a model, an update of sbmlsim or a refresh of the study data
should leave the simulations of the model as they were, and a fix should change
them only where it is meant to. The subcommands of
`python -m sbmlsim.run verify` make this checkable; they are a family of
commands of the run layer like `plan` and `list`, not a kind: they compare
results of local checkouts and write only where they are told to
(`--out`):

- `dump` runs the simulation experiments of a project (`--project
  module:attribute`, optionally `--root`, the directory put first on the python
  path, e.g. a checkout) and stores the results, the datasets, the fit
  mappings and the figures in a pickle,
- `compare` compares two dumps: the simulations on the matched rows of their
  output times, a row at a time which only one dump has (the steps of the
  integrator of another run) with the other dump interpolated at its time,
  relative to the scale of every array (a row outside of the times of the other
  dump is a difference; an array whose interpolation cannot resolve the
  threshold is unresolved, not equal), everything else exactly; with
  `--renames FILE` the expected renames of a change (metadata fields and
  members, added fields, texts such as legend keys, ids) and the expected
  differences are applied to the old dump first, see `renames.load_renames`,
- `ranges` lists the fit mappings of a dump whose data lie outside of the
  simulated range, e.g. data times beyond the end of the simulation,
- `fits` builds the optimization problem of every fit definition of a project
  and runs a minimal fit,
- `definitions` writes the fit definitions of a project as json: the settings,
  the parameters and the mappings of the collections, so that two checkouts are
  compared with `diff`,
- `settings` writes what a run of the experiments of a project hands to
  sbmlsim, so that the settings of two checkouts are compared with `diff`,
- `png` compares the PNG figures of two directories pixel by pixel,
- `sbml` compares SBML models by their elements, ignoring the order and the
  annotations,
- `scorecard` reads the credibility records of many models (`credibility.json`
  of the kind `credibility`) into one table, pools their ratio pairs by a value
  of the metadata with the evidence extension named by `--pool` and compares the
  table with a baseline, see `sbmlsim.credibility.scorecard`.

A model before and after a change, with tight tolerances so that the solver does
not hide a difference:

```bash
git worktree add --detach ../base develop
# the experiments simulate the models of the project, which are generated:
# create them with the factory of each checkout right before its dump
python -m sbmlsim.run verify dump --root ../base --project pkg.models.hctz:PROJECT \
    --out base.pkl --atol 1e-16 --rtol 1e-12 --grid 1000
python -m sbmlsim.run verify dump --root . --project pkg.models.hctz:PROJECT \
    --out new.pkl --order base.pkl --atol 1e-16 --rtol 1e-12 --grid 1000
python -m sbmlsim.run verify compare base.pkl new.pkl
```

`--grid N` outputs every simulation at `N + 1` equidistant times from its start
to its end and at the times of its changes (`dump.on_grid`) instead of the steps
of the integrator, which differ between two models or tolerances: both dumps
share their times and `compare` matches them exactly. Without it the rows of a
time which only one dump has are compared by interpolation, which on the steps
of a real model resolves about 1e-4 to 1e-3 (unresolved arrays are reported, not
passed); a compare of two models therefore dumps both with the same grid.

The variant of the model a task runs on is stored per task (`builds`): the
variant `SimulationExperiment.variant_of` names, else the variant of the project
whose file is the source of the model of the task, so a study which moves to
the build of another variant is a difference of its own.

The command only needs the python packages of the environment it runs in, so a
checkout which needs another version of sbmlsim is dumped with the python of an
environment with that version, as long as it has the experiments of the project
API of this version. The exit code is 0 when nothing differs, 1 when a
difference is found and 2 for a command which cannot be run as it was asked; the
result of a command goes to stdout, the progress and the problems are logged on
stderr.
"""
