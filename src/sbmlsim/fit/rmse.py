"""Calculation of statistics."""
import numpy as np
import pandas as pd
from pymetadata.console import console


def rmse(mse: float):
    """Root Mean Square Error."""
    return np.sqrt(mse)

def aic(mse: float, N: int, k: int):
    """Akaike Information Criterion (AIC).

    N: datapoints
    k: parameters
    """
    return N * np.log(mse) + 2*k



if __name__ == "__main__":
    items = [
        {
            "uid": "20250708_183921__4fba0",
            "name": "LOSARTAN_LSQ_PK",
            "N": 225,
            "k": 12,
            "MSE": 70.150056,
        },
        {
            "uid": "20250711_231400__8d0b3",
            "name": "LOSARTAN_LSQ_PD",
            "N": 460,
            "k": 4,
            "MSE": 73.095901,
         }
    ]
    for item in items:
        item["RMSE"] = rmse(item["MSE"])
        item["AIC"] = aic(mse=item["MSE"], N=item["N"], k=item["k"])

    df = pd.DataFrame(items)
    console.print(df)

    console.rule()
    tex = df.to_latex(None, index=False, float_format="{:.2f}".format)
    tex = tex.replace("_", "\_")
    console.print(tex)

    console.rule()
