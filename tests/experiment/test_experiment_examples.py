"""Testing experiment examples."""

from pathlib import Path

import pytest

from examples.demo.demo import run_demo_experiments
from examples.glucose.glucose import run_glucose_experiments
from examples.repressilator.repressilator import (
    run_repressilator_example,
)


@pytest.mark.skip(reason="no experiment support")
def test_demo_example(tmp_path: Path) -> None:
    """Test demo simulation experiment."""
    run_demo_experiments(tmp_path)


@pytest.mark.skip(reason="no experiment support")
def test_glucose_example(tmp_path: Path) -> None:
    """Test glucose simulation experiment."""
    run_glucose_experiments(tmp_path)


@pytest.mark.skip(reason="no experiment support")
def test_repressilator_example(tmp_path: Path) -> None:
    """Test repressilator simulation experiment."""
    run_repressilator_example(output_path=tmp_path)
