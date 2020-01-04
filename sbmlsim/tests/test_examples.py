"""
Run all the example files in tests.
"""
from sbmlsim.examples import example_clamp, example_scan, example_sensitivity, example_timecourse, \
    example_parallel, example_units


def test_clamp():
    example_clamp.run_clamp()


def test_scan():
    example_scan.run_parameter_scan()


def test_sensitivity():
    example_sensitivity.run_sensitivity()


def test_timecourse():
    example_timecourse.run_timecourse_examples()


def test_parallel_1():
    example_parallel.example_single_actor()


def test_parallel_2():
    example_parallel.example_multiple_actors()


def test_parallel_3():
    example_parallel.example_parallel_timecourse(nsim=20, actor_count=5)


def test_units():
    example_units.run_demo_example()
