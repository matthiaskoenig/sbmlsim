# References

`sbmlsim` builds on the standards of the [COMBINE](https://co.mbine.org) community and on the simulation and analysis libraries of the scientific python ecosystem. These are the publications behind them; cite them when you describe a simulation experiment, and cite `sbmlsim` itself as described in [Home](index.md#how-to-cite).

## Simulation

**libRoadRunner.** The SBML simulation engine all simulations run on.

> Welsh C, Xu J, Smith L, König M, Choi K, Sauro HM.
> **libRoadRunner 2.0: a high performance SBML simulation and analysis library.**
> *Bioinformatics.* 2023;39(1):btac770.
> [doi:10.1093/bioinformatics/btac770](https://doi.org/10.1093/bioinformatics/btac770)

> Somogyi ET, Bouteiller JM, Glazier JA, König M, Medley JK, Swat MH, Sauro HM.
> **libRoadRunner: a high performance SBML simulation and analysis library.**
> *Bioinformatics.* 2015;31(20):3315-3321.
> [doi:10.1093/bioinformatics/btv363](https://doi.org/10.1093/bioinformatics/btv363)

## Standards

**SBML Level 3.** The format of the models.

> Keating SM, Waltemath D, König M, Zhang F, Dräger A, Chaouiya C, Bergmann FT, Finney A, Gillespie CS, Helikar T, Hoops S, Malik-Sheriff RS, Moodie SL, Moraru II, Myers CJ, Naldi A, Olivier BG, Sahle S, Schaff JC, Smith LP, Swat MJ, Thieffry D, Watanabe L, Wilkinson DJ, Blinov ML, Begley K, Faeder JR, Gómez HF, Hamm TM, Inagaki Y, Liebermeister W, Lister AL, Lucio D, Mjolsness E, Proctor CJ, Raman K, Rodriguez N, Shaffer CA, Shapiro BE, Stelling J, Swainston N, Tanimura N, Wagner J, Meier-Schellersheim M, Sauro HM, Palsson B, Bolouri H, Kitano H, Funahashi A, Hermjakob H, Doyle JC, Hucka M; SBML Level 3 Community members.
> **SBML Level 3: an extensible format for the exchange and reuse of biological models.**
> *Molecular Systems Biology.* 2020;16(8):e9110.
> [doi:10.15252/msb.20199110](https://doi.org/10.15252/msb.20199110)

**SED-ML.** The description of simulation experiments which `sbmlsim.combine.sedml` reads and executes, see [SED-ML and COMBINE archives](sedml.md).

> Smith LP, Bergmann FT, Garny A, Helikar T, Karr J, Nickerson D, Sauro H, Waltemath D, König M.
> **The simulation experiment description markup language (SED-ML): language specification for level 1 version 4.**
> *Journal of Integrative Bioinformatics.* 2021;18(3):20210021.
> [doi:10.1515/jib-2021-0021](https://doi.org/10.1515/jib-2021-0021)

> Waltemath D, Adams R, Bergmann FT, Hucka M, Kolpakov F, Miller AK, Moraru II, Nickerson D, Sahle S, Snoep JL, Le Novère N.
> **Reproducible computational biology experiments with SED-ML — the Simulation Experiment Description Markup Language.**
> *BMC Systems Biology.* 2011;5:198.
> [doi:10.1186/1752-0509-5-198](https://doi.org/10.1186/1752-0509-5-198)

**COMBINE archive.** The container for models, simulation experiments and data.

> Bergmann FT, Adams R, Moodie S, Cooper J, Glont M, Golebiewski M, Hucka M, Laibe C, Miller AK, Nickerson DP, Olivier BG, Rodriguez N, Sauro HM, Scharm M, Soiland-Reyes S, Waltemath D, Yvon F, Le Novère N.
> **COMBINE archive and OMEX format: one file to share all information to reproduce a modeling project.**
> *BMC Bioinformatics.* 2014;15:369.
> [doi:10.1186/s12859-014-0369-z](https://doi.org/10.1186/s12859-014-0369-z)

**KISAO.** The ontology of simulation algorithms and their parameters, see `sbmlsim.simulation.kisaos`.

> Courtot M, Juty N, Knüpfer C, Waltemath D, Zhukova A, Dräger A, Dumontier M, Finney A, Golebiewski M, Hastings J, Hoops S, Keating S, Kell DB, Kerrien S, Lawson J, Lister A, Lu J, Machne R, Mendes P, Pocock M, Rodriguez N, Villeger A, Wilkinson DJ, Wimalaratne S, Laibe C, Hucka M, Le Novère N.
> **Controlled vocabularies and semantics in systems biology.**
> *Molecular Systems Biology.* 2011;7:543.
> [doi:10.1038/msb.2011.77](https://doi.org/10.1038/msb.2011.77)

**PEtab.** The specification of parameter estimation problems which `sbmlsim.fit.petab_omex` packages, see [Parameter fitting](fitting.md).

> Schmiester L, Schälte Y, Bergmann FT, Camba T, Dudkin E, Egert J, Fröhlich F, Fuhrmann L, Hauber AL, Kemmer S, Lakrisenko P, Loos C, Merkt S, Müller W, Pathirana D, Raimúndez E, Refisch L, Rosenblatt M, Stapor PL, Städter P, Wang D, Wieland FG, Banga JR, Timmer J, Villaverde AF, Sahle S, Kreutz C, Hasenauer J, Weindl D.
> **PEtab — Interoperable specification of parameter estimation problems in systems biology.**
> *PLoS Computational Biology.* 2021;17(1):e1008646.
> [doi:10.1371/journal.pcbi.1008646](https://doi.org/10.1371/journal.pcbi.1008646)

## Sensitivity analysis

The global methods of `sbmlsim.sensitivity` are the implementations of [SALib](https://salib.readthedocs.io), see [Sensitivity analysis](sensitivity.md).

> Herman J, Usher W.
> **SALib: An open-source Python library for Sensitivity Analysis.**
> *Journal of Open Source Software.* 2017;2(9):97.
> [doi:10.21105/joss.00097](https://doi.org/10.21105/joss.00097)

> Iwanaga T, Usher W, Herman J.
> **Toward SALib 2.0: Advancing the accessibility and interpretability of global sensitivity analyses.**
> *Socio-Environmental Systems Modelling.* 2022;4:18155.
> [doi:10.18174/sesmo.18155](https://doi.org/10.18174/sesmo.18155)

**Sobol indices.** Variance based first order and total effect indices.

> Sobol' IM.
> **Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates.**
> *Mathematics and Computers in Simulation.* 2001;55(1-3):271-280.
> [doi:10.1016/S0378-4754(00)00270-6](https://doi.org/10.1016/S0378-4754(00)00270-6)

> Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S.
> **Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index.**
> *Computer Physics Communications.* 2010;181(2):259-270.
> [doi:10.1016/j.cpc.2009.09.018](https://doi.org/10.1016/j.cpc.2009.09.018)

**Morris method.** Elementary effects screening.

> Morris MD.
> **Factorial sampling plans for preliminary computational experiments.**
> *Technometrics.* 1991;33(2):161-174.
> [doi:10.1080/00401706.1991.10484804](https://doi.org/10.1080/00401706.1991.10484804)

> Campolongo F, Cariboni J, Saltelli A.
> **An effective screening design for sensitivity analysis of large models.**
> *Environmental Modelling & Software.* 2007;22(10):1509-1518.
> [doi:10.1016/j.envsoft.2006.10.004](https://doi.org/10.1016/j.envsoft.2006.10.004)

**FAST.** The Fourier amplitude sensitivity test and its extended form.

> Cukier RI, Fortuin CM, Shuler KE, Petschek AG, Schaibly JH.
> **Study of the sensitivity of coupled reaction systems to uncertainties in rate coefficients. I Theory.**
> *The Journal of Chemical Physics.* 1973;59(8):3873-3878.
> [doi:10.1063/1.1680571](https://doi.org/10.1063/1.1680571)

> Saltelli A, Tarantola S, Chan KPS.
> **A quantitative model-independent method for global sensitivity analysis of model output.**
> *Technometrics.* 1999;41(1):39-56.
> [doi:10.1080/00401706.1999.10485594](https://doi.org/10.1080/00401706.1999.10485594)

## Data structures

**xarray.** Simulation results are stored as labeled N-dimensional arrays, see `sbmlsim.result.xresult`.

> Hoyer S, Hamman J.
> **xarray: N-D labeled arrays and datasets in Python.**
> *Journal of Open Research Software.* 2017;5(1):10.
> [doi:10.5334/jors.148](https://doi.org/10.5334/jors.148)
