from sbmlsim.examples import example_scan


def test_scan0d():
    example_scan.run_scan0d()


def test_scan1d():
    example_scan.run_scan1d()


def test_scan2d():
    example_scan.run_scan2d()


def test_scan1d_distribution():
    example_scan.run_scan1d_distribution()
