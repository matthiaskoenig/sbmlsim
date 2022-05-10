"""Multiprocessing simulation example."""
from multiprocessing import Process

from sbmlsim import RESOURCES_DIR
from sbmlsim.model.model_roadrunner import roadrunner


def run_simulations(r: roadrunner.RoadRunner, size: int) -> None:
    """Run simulations."""
    for _ in range(size):
        print("simulate")
        res = r.simulate(0, 100, steps=5)
        print(res)


def multiprocessing_example() -> None:
    """Run multiprocessing example."""
    model_path = RESOURCES_DIR / "testdata" / "models" / "icg_body_flat.xml"
    rr: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
    p = Process(target=run_simulations, args=(rr, 10))
    p.start()
    p.join()


if __name__ == "__main__":
    multiprocessing_example()
