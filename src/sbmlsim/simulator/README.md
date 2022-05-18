TODO:
* [ ] implement multiprocessing solution (state)
* improve handling of units for models (i.e. conversions)



* simulator should not use any units
* xresults should not use any units



* run simulation experiments -> XResults
* unify model/RoadrunnerModel

# Refactoring


## Simulations -> Results

The following core simulations should be supported
* `timecourse` task (model & simulation)
* `steadystate taks` (model & simulation)

In addition these simulations can be wrapped in a more complex scan
* `dimensions`
This creates higher dimensional scans

## Observables
- handle addition of observables to the model as functions; have to be added to the
  model or calculated on the results






## Postprocessing (Results -> Results)
This should be functions which allow to take the Results data and 
create new results data.

