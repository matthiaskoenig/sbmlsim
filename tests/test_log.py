"""Tests of the logging helpers."""

from sbmlsim.log import some_ids


def test_some_ids_lists_the_first_ids_and_the_count() -> None:
    assert some_ids(["a", "b"]) == "['a', 'b']"
    assert (
        some_ids([f"x{k}" for k in range(7)], n=5)
        == "['x0', 'x1', 'x2', 'x3', 'x4'] ... (7 in total)"
    )
    assert some_ids([], n=5) == "[]"
    # exactly `n` ids are all listed, one more is counted
    assert some_ids(["a"] * 5, n=5) == str(["a"] * 5)
    assert some_ids(["a"] * 6, n=5) == f"{['a'] * 5} ... (6 in total)"
