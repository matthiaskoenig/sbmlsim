"""Multiprocessing simulation example."""
from multiprocessing import Process

import roadrunner


def run_simulations(r, size):
    """Run simulations."""
    for _ in range(size):
        print("simulate")
        res = r.simulate(0, 100, steps=100)
        print(res)


if __name__ == "__main__":
    rr = roadrunner.RoadRunner("icg_body_flat.xml")
    size = 10
    p = Process(target=run_simulations, args=(rr, size))
    p.start()
    p.join()
