# Lotka-Volterra with a network in the right hand side

The case `sciml_problem_import/001` of the [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) (commit `0622bbfc5e12eb9b482659eabd1756ca0e87dfc8`, MIT license), unchanged: the Lotka-Volterra model `lv.xml` whose interaction term of the predator, `gamma`, is the output of the feed forward network `net1.yaml` (three linear layers of five units with `tanh`) with the species as inputs, the arrays of the network in `net1_ps.hdf5`, and the tables of PEtab v2 with the `sciml` extension in `problem.yaml`.

`python -m examples.sciml.lotka_volterra_fit` reads it, improves its values with a short fit which starts from them, reports the fit and writes the fitted problem as PEtab SciML again. It needs the extra `sciml`: `pip install sbmlsim[sciml]`.
