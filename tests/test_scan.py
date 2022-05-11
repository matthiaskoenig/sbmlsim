"""Test scans."""
from sbmlsim.examples import example_scan


def test_scan0d() -> None:
    """Run scan0d."""
    example_scan.run_scan0d()


def test_scan1d() -> None:
    """Run scan1d."""
    example_scan.run_scan1d()


def test_scan2d() -> None:
    """Run scan2d."""
    example_scan.run_scan2d()


def test_scan1d_distribution() -> None:
    """Run scan1d distribution."""
    example_scan.run_scan1d_distribution()
