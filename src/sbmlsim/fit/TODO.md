# TODO
Important features:
- support simple cluster deployment (full serialization of problem & settings)
- (support complex calculated data AUC, ...)

*IO*
- [ ] serialization of problem(s) to PEtab (v2) + additional metadata (i.e. all simulations, data, metadata); i.e. the complete problem is stored in PEtab; store problem and metadata (for filtering & validation) => full problem => create subset problems
- [ ] inject model parameters/changes and serialize SBML
- [ ] generation of subsets of problems easily (filter) => new PEtab problems; all serialized; 

*Optimization*
- simulate only timepoints which are necessary
- clear access to optimization function;
- robust handling of failing optimizations;

* Evaluation & reports*
- simple calculation of metrics based on a set of paramters; i.e PETab problem + parameter set allow to calculate all metrics
  - [ ] Calculate metrics: PRED, IPRED, IRES, AIC, RMSE (based on given set of parameters) and report; Access to cost function;
- store results of parameter optimization (parameter sets with metrics) -> merge results
- confidence intervals using profile Likelihoods of cost function; => subsequent analysis
- improve plots; Bland Altmann; Waterfall
- separate reports from parameter optimization (just needs problem definition & model parameters); Creates report for given parameter set, trainings data & evaluation data
- interactive reports (quarto); => use best of both worlds R, nlmixr2 , Pypesto; petab; much better interactive/static reports as overview

Profile-Likelihood analysis
- Parameter identifiability analysis; 
