import pytest
from pathlib import Path

from sbmlsim.examples.experiments.covid.simulate import run_covid_examples


@pytest.mark.skip("SED-ML relative paths")
def test_covid_example(tmp_path: Path) -> None:
    """Test covid simulation experiment."""
    run_covid_examples(tmp_path)
