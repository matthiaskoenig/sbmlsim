# Perelson_Science1996

The PEtab problem of

> Perelson AS, Neumann AU, Markowitz M, Leonard JM, Ho DD.
> *HIV-1 dynamics in vivo: virion clearance rate, infected cell life-span, and viral generation time.*
> Science. 1996;271(5255):1582-6. https://doi.org/10.1126/science.271.5255.1582

as it is published in the [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/master/Benchmark-Models/Perelson_Science1996), which is licensed BSD-3-Clause. The files are the **PEtab 1.0** problem of the collection, unchanged.

`sbmlsim` reads PEtab 2.0, so `examples/petab/benchmark_perelson.py` converts the problem with `petab.v2.petab1to2` before it fits it, i.e. the conversion is part of the example and the files here stay the ones of the collection.

The model is the viral dynamics of HIV-1 after the start of a protease inhibitor: the viral load `V` is measured over the first week of treatment, and the clearance rate of the virions `c` and the loss rate of the infected cells `delta` are estimated.
