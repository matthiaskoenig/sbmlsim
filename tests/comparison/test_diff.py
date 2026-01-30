"""Test difference."""

import pytest

from data.diff import simulate_examples


@pytest.mark.skip(reason="no diff support")
def test_run_simulations() -> None:
    """Run simulations."""
    simulate_examples.run_simulations(create_files=False)


@pytest.mark.skip(reason="no diff support")
def test_run_comparisons() -> None:
    """Run comparisons."""
    simulate_examples.run_comparisons(create_files=False)
