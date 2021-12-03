import roadrunner

r: roadrunner.RoadRunner = roadrunner.RoadRunner("nan_species.xml")

# setting a valid volume, now the model is completely specified and can be
# numerically integrated
r.setValue("Vext", 1.0)

# Instead of `initialConcentration="0.0" defined in the model there is an
# incorrect `initialAmount="NaN"` on the `dex_ext` species probably due to some
# pre-processing steps of the model
sbml_str = r.getCurrentSBML()
print("-" * 80)
print(sbml_str)
print("-" * 80)

# As a consequence the model does not integrate
s = r.simulate(start=0, end=10, steps=10)

