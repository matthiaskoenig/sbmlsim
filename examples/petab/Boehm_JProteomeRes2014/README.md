# Boehm_JProteomeRes2014

The PEtab problem of

> Boehm ME, Adlung L, Schilling M, Roth S, Klingmüller U, Lehmann WD.
> *Identification of isoform-specific dynamics in phosphorylation-dependent STAT5 dimerization by quantitative mass spectrometry and mathematical modeling.*
> J Proteome Res. 2014;13(12):5685-94. https://doi.org/10.1021/pr5006923

as it is published in the [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/master/Benchmark-Models/Boehm_JProteomeRes2014), which is licensed BSD-3-Clause. The files are the **PEtab 1.0** problem of the collection, unchanged.

The model is the dimerization of STAT5A and STAT5B after stimulation with erythropoietin. Its three observables are the relative amounts of the phosphorylated and of the total protein, i.e. **formulas over several species** such as

    (100 * pApB + 200 * pApA * specC17) / (pApB + STAT5A * specC17 + 2 * pApA * specC17)

which is what makes it interesting here: `sbmlsim` observes what roadrunner selects, so `sbmlsim.fit.petab_v2.observables` writes a copy of the model in which every such observable is a parameter with an assignment rule, and the fit selects that parameter.

Simulated at the nominal parameters, the observables agree with the
`simulatedData_Boehm_JProteomeRes2014.tsv` of the collection to `2e-4` at a
relative tolerance of `1e-9` of the integrator.
