# Perelson_Science1996

The PEtab problem of

> Perelson AS, Neumann AU, Markowitz M, Leonard JM, Ho DD.
> *HIV-1 dynamics in vivo: virion clearance rate, infected cell life-span, and viral generation time.*
> Science. 1996;271(5255):1582-6. https://doi.org/10.1126/science.271.5255.1582

as it is published in the [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/c44d5ede9bf979db5aa8ae4e94fc31fba9776675/problems/Perelson_Science1996), which is licensed BSD-3-Clause. The collection has the problem in PEtab v1 only, `problems/Perelson_Science1996/v1` at the commit `c44d5ede`; the files are that problem converted to **PEtab v2** with `petab.v2.petab1to2` (`sbmlsim.fit.petab_v2.conversion.convert_v1`), since `sbmlsim` reads PEtab v2 only.

The model is the viral dynamics of HIV-1 after the start of a protease inhibitor: the viral load `V` is measured over the first week of treatment, and the clearance rate of the virions `c` and the loss rate of the infected cells `delta` are estimated.
