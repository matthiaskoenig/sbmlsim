# The models of the benchmark

The models here are measured by `scripts/engine_benchmark.py` and are no part of the package: hatchling builds `src/sbmlsim`, so nothing of this directory is installed, and `sbmlsim.resources` names the four packaged models the benchmark shares with the tests, the examples and the documentation.

A large model belongs here rather than in the package, since every install would carry it for a measurement alone.

## machine_perfusion_liver_flat.xml

The liver of a normothermic machine perfusion: 839 species, 1007 reactions, 570 parameters, 387 rules and 43 compartments in 2.6 MB, the time in minutes, which is 12 times the largest packaged model (HCTZ, 223 kB).

Its source is `pkdb_models/misc/machine_perfusion/models/output/models/machine_perfusion_liver_flat.xml` of [matthiaskoenig/pkdb_models](https://github.com/matthiaskoenig/pkdb_models) at `a509edcd5`, prepared with `sbmlsim.webapp.compile.prepare_sbml`, which is what the web app of a model runs on: the model is flattened and its `distrib` annotations are removed, which the compiler of sbmlsim-core requires (`required package is not supported (distrib)`) and which no engine's semantics depend on, since `distrib` states the uncertainty of a parameter and no math of this model reads a distribution. Every engine of the benchmark gets this one file, so the comparison is of the engines and not of what each of them reads.

Regenerate it with:

```bash
python -c "
from pathlib import Path
from sbmlsim.webapp.compile import prepare_sbml
src = Path('<pkdb_models>/pkdb_models/misc/machine_perfusion/models/output/models/machine_perfusion_liver_flat.xml')
out = Path('benchmarks/models/machine_perfusion_liver_flat.xml')
out.write_text(prepare_sbml(src.read_text(encoding='utf-8'), base_path=src.parent), encoding='utf-8')
"
```
