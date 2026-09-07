"""Example using pypesto, petab, amici."""

# import matplotlib and increase image resolution
import matplotlib as mpl
import numpy as np
import pypesto  # ty: ignore[unresolved-import]
import pypesto.optimize as optimize  # ty: ignore[unresolved-import]
import pypesto.petab  # ty: ignore[unresolved-import]

mpl.rcParams["figure.dpi"] = 300


# define objective function
def f(x: np.ndarray) -> float:
    return x[0] ** 2 + x[1] ** 2


# define gradient
def grad(x: np.ndarray) -> np.ndarray:
    return 2 * x


# define objective
custom_objective = pypesto.Objective(fun=f, grad=grad)

# define optimization bounds
lb = np.array([-10, -10])
ub = np.array([10, 10])

# create problem
custom_problem = pypesto.Problem(objective=custom_objective, lb=lb, ub=ub)

# choose optimizer
optimizer = optimize.ScipyOptimizer()

# do the optimization
result_custom_problem = optimize.minimize(
    problem=custom_problem, optimizer=optimizer, n_starts=10
)

# E.g. The best model fit was obtained by the following optimization run:
print(result_custom_problem.optimize_result.list[0])
