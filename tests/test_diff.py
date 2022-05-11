"""Test difference."""
from sbmlsim.resources.testdata.diff import simulate_examples


def test_run_simulations() -> None:
    """Run simulations."""
    simulate_examples.run_simulations(create_files=False)


def test_run_comparisons() -> None:
    """Run comparisons."""
    simulate_examples.run_comparisons(create_files=False)
