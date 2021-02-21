import pandas as pd
import roadrunner
from roadrunner import SelectionRecord


# Loading model and simulating
model = roadrunner.RoadRunner(
    "initial_assignment.xml"
)  # type: roadrunner.ExecutableModel
model.selections = ["time", "D", "A1", "[A1]"]

s = model.simulate(0, 100, steps=11)
s = pd.DataFrame(s, columns=s.colnames)
print(s.head())

del model
del s

print("-" * 80)

# Changing initial value of D (re-evalutating initial assignments)
model = roadrunner.RoadRunner("initial_assignment.xml")
model.selections = ["time", "D", "A1", "[A1]"]

# Setting initial value of D
# This should allow to update the initial value of A1 which is
#     init(A1) = 2 * D
model["init(D)"] = 2
model.resetAll()

s = model.simulate(0, 100, steps=11)
s = pd.DataFrame(s, columns=s.colnames)
print(s.head())

del model
del s

# Changing initial value of D (re-evalutating initial assignments)
model = roadrunner.RoadRunner(
    "initial_assignment.xml"
)  # type: roadrunner.ExecutableModel
model.selections = ["time", "D", "A1", "[A1]"]

# Setting initial value of D
# This should allow to update the initial value of A1 which is
#     init(A1) = 2 * D
model["init(D)"] = 2
model.resetAll()
model.reset(SelectionRecord.DEPENDENT_FLOATING_AMOUNT)
model.reset(SelectionRecord.DEPENDENT_INITIAL_GLOBAL_PARAMETER)

s = model.simulate(0, 100, steps=11)
s = pd.DataFrame(s, columns=s.colnames)
print(s.head())
