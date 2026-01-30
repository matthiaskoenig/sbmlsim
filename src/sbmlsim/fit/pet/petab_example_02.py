"""Example using pypesto, petab, amici."""

# import matplotlib and increase image resolution
import matplotlib as mpl

import pypesto
import pypesto.optimize as optimize
import pypesto.petab

mpl.rcParams["figure.dpi"] = 300


# directory of the PEtab problem
petab_yaml = "./boehm_JProteomeRes2014/Boehm_JProteomeRes2014.yaml"

importer = pypesto.petab.PetabImporter.from_yaml(petab_yaml)
problem = importer.create_problem()

# Set gradient computation method to adjoint
# problem.objective.amici_solver.setSensitivityMethod(
#     amici.SensitivityMethod.adjoint
# )

# choose optimizer
optimizer = optimize.ScipyOptimizer()

# # do the optimization
# result = optimize.minimize(problem=problem,
#                            optimizer=optimizer,
#                            n_starts=10)
#
# # E.g. best model fit was obtained by the following optimization run:
# result.optimize_result.list[0]

# paralleization:
# Parallelize
engine = pypesto.engine.MultiProcessEngine()

# Optimize
result = optimize.minimize(
    problem=problem, optimizer=optimizer, engine=engine, n_starts=100
)
