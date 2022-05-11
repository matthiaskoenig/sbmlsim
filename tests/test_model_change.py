"""Test model changes."""
from sbmlsim.examples import example_model_change


def test_example1() -> None:
    """Test example1."""
    example_model_change.run_model_change_example1()


def test_clamp1() -> None:
    """Test clamp1."""
    example_model_change.run_model_clamp1()


def test_clamp2() -> None:
    """Test clamp2."""
    example_model_change.run_model_clamp2()
