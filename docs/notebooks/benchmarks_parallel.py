"""
Scripts for benchmarking
"""

from sbmlsim.examples.example_parallel import example_parallel_timecourse

if __name__ == "__main__":
    benchmarks = []
    for nsim in [2**k for k in range(0, 11)]:
        for k in range(3):
            sim_info = example_parallel_timecourse(nsim=nsim, actor_count=15)
            for item in sim_info:
                item["repeat"] = k
            benchmarks.extend(sim_info)

    import pandas as pd
    df = pd.DataFrame(benchmarks)
    print(df)
    df.to_csv("benchmarks_v03.tsv", sep="\t", index=False)
