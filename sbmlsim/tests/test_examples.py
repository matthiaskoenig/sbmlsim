"""
Run all the example files in tests.
"""
from sbmlsim.examples import example_clamp, example_scan, example_sensitivity, example_timecourse


def test_clamp():
    example_clamp.run_clamp()


def test_scan():
    example_scan.run_parameter_scan()


def test_sensitivity():
    example_sensitivity.run_sensitivity()


def test_timecourse():
    example_timecourse.run_timecourse_examples()
