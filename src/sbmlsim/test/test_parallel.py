import pytest


try:
    import ray
except ImportError:
    ray = None

if not ray:
    # ray not available on windows
    from sbmlsim.examples import example_parallel


@pytest.mark.skip(msg="skipping parallel tests due to hanging CI")
def test_parallel_1():
    example_parallel.example_single_actor()


@pytest.mark.skip(msg="skipping parallel tests due to hanging CI")
def test_parallel_2():
    example_parallel.example_multiple_actors()


@pytest.mark.skip(msg="skipping parallel tests due to hanging CI")
def test_parallel_3():
    example_parallel.example_parallel_timecourse(nsim=20, actor_count=5)
