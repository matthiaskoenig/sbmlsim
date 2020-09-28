import pytest

from sbmlsim.examples import example_parallel


def test_parallel_1():
    example_parallel.example_single_actor()


def test_parallel_2():
    example_parallel.example_multiple_actors()


def test_parallel_3():
    example_parallel.example_parallel_timecourse(nsim=20, actor_count=5)
