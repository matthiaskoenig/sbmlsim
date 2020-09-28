import lhsmdu
import pandas as pd
from matplotlib import pyplot as plt


def lhsmdu_runtime():
    runtimes = []
    import time

    for k in range(0, 7):
        ts = time.time()
        samples = 2 ** k
        lhsmdu.sample(
            2, samples
        )  # Latin Hypercube Sampling of two variables, and 10 samples each
        te = time.time()
        res = {"samples": samples, "time": te - ts}
        print(res)
        runtimes.append(res)
    df = pd.DataFrame(runtimes)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
    for ax in axes:
        ax.plot(df.samples, df.time, "-o", markersize=10)
        ax.set_xlabel("sample size")
        ax.set_ylabel("runtime [s]")
        ax.grid(True)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    fig.savefig("lhsmdu_runtime.png")
    plt.show()


if __name__ == "__main__":
    lhsmdu_runtime()
