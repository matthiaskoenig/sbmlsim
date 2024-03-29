"""Testing fitting examples."""
import pytest

from sbmlsim.examples.experiments.midazolam.fitting_example import (
    fitting_example,
    op_mid1oh_iv,
)


@pytest.mark.skip(reason="no fit support")
def test_example() -> None:
    """Test running the example fitting."""
    fitting_example(op_factory=op_mid1oh_iv, n_cores=1, size=1)
