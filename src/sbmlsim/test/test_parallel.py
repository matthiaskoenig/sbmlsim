import pytest

from sbmlsim.examples import example_parallel


# @pytest.mark.skip(msg="skipping parallel tests due to hanging CI")
def test_parallel_single_actor() -> None:
    """Test single actor."""
    example_parallel.example_single_actor()


# @pytest.mark.skip(msg="skipping parallel tests due to hanging CI")
def test_parallel_multiple_actors() -> None:
    """Test multiple actors."""
    example_parallel.example_multiple_actors()


# @pytest.mark.skip(msg="skipping parallel tests due to hanging CI")
def test_parallel_timecourse() -> None:
    """Test parallel timecourses."""
    example_parallel.example_parallel_timecourse(nsim=20, actor_count=5)
