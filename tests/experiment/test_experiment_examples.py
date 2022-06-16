"""Testing experiment examples."""
from pathlib import Path

import pytest

from sbmlsim.examples.experiments.demo.demo import run_demo_experiments
from sbmlsim.examples.experiments.glucose.glucose import run_glucose_experiments
from sbmlsim.examples.experiments.midazolam.simulate import run_midazolam_experiments
from sbmlsim.examples.experiments.repressilator.repressilator import (
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
def test_midazolam_example(tmp_path: Path) -> None:
    """Test midazolam simulation experiment."""
    run_midazolam_experiments(tmp_path)


@pytest.mark.skip(reason="no experiment support")
def test_repressilator_example(tmp_path: Path) -> None:
    """Test repressilator simulation experiment."""
    run_repressilator_example(output_path=tmp_path)
