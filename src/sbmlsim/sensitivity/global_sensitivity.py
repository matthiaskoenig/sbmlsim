"""Global sensitivity analysis.

TODO:
- [ ] multi-output model
- [ ] calculation of pharmacokinetic parameters
- [ ] get all parameter ids from model
- [ ] get defined bounds from model;
- [ ] storage of results
- [ ] visualization of results
- [ ] alternative methods: FAST, Morris, local sensitivity analysis
- [ ] report as PDF with references and description (Typst)
- [ ] parallelization ? (benchmark)

"""
import SALib
from SALib import ProblemSpec
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

import numpy as np
import roadrunner
from pathlib import Path

model_path = Path(
    __file__).parent / "models" / "losartan" / "losartan_body_flat.xml"
rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
rr.selections = ["time", "[Cve_los]"]
integrator: roadrunner.Integrator = rr.integrator
integrator.setSetting("variable_step_size", True)
state = rr.saveStateS()



def run_simulation(BW: float, FVKi) -> float:
    # rr = roadrunner.RoadRunner()
    # rr.loadStateS(state)
    rr.resetAll()
    changes = {
        "PODOSE_los": 10.0,  # [mg]
        "BW": BW,  # [kg]
        "FVki": FVKi,  # [kg]
    }
    for key, value in changes.items():
        print(f"{key=} {value=}")
        rr.setValue(key, value)

    s = rr.simulate(start=0, end=24 * 60)
    # import pandas as pd
    # df = pd.DataFrame(s, columns=s.colnames)
    # print(df.to_string())

    # # plotting
    # from matplotlib import pyplot as plt
    # plt.plot(
    #     s["time"], s["[Cve_los]"],
    #     # df["time"], df["[Cve_los]"],
    #     marker="o",
    #     markeredgecolor="black",
    #     color="tab:blue",
    # )
    # plt.show()

    y = np.max(s["[Cve_los]"])
    return y


def wrapped_run_simulation(X, func=run_simulation):
    # We transpose to obtain each column (the model factors) as separate variables
    BW, FVKi = X.T

    # Then call the original model
    return func(BW, FVKi)


def calculate_sensitivity():

    # Defining the model inputs
    sp = ProblemSpec({
        'num_vars': 2,
        'names': ['BW', 'FVki'],
        'bounds': [
            [50, 150],
            [0.003, 0.005]
        ],
        "outputs": ["Y"],
    })

    # Generate samples
    samples = saltelli.sample(sp, 1024)
    sp.set_samples(samples)


    # Evaluate model
    sp.evaluate(wrapped_run_simulation)

    # Y = np.zeros([param_values.shape[0]])
    # for k, X in enumerate(param_values):
    #     print(k)
    #     Y[k] = wrapped_run_simulation(X)


    # Perform Analysis
    Si = sp.analyze(SALib.analyze.sobol)
    print(Si['S1'])
    print(Si['ST'])
    total_Si, first_Si, second_Si = Si.to_df()
    Si.plot()
    from matplotlib import pyplot as plt
    plt.show()




if __name__ == '__main__':
    # y = run_simulation()
    # print(y)
    calculate_sensitivity()
