import pytest

from sbmlsim.run.units import MissingUnits, unit_blocks


def test_no_items_no_blocks() -> None:
    assert unit_blocks(0) == []


def test_few_items_one_per_block() -> None:
    assert unit_blocks(10) == [range(i, i + 1) for i in range(10)]
    assert unit_blocks(64) == [range(i, i + 1) for i in range(64)]


def test_blocks_of_the_ceiling() -> None:
    blocks = unit_blocks(130)
    assert len(blocks) == 44
    assert blocks[0] == range(3)
    assert blocks[-1] == range(129, 130)
    assert [i for block in blocks for i in block] == list(range(130))


def test_one_past_the_maximum() -> None:
    blocks = unit_blocks(65)
    assert len(blocks) == 33
    assert blocks[0] == range(2)
    assert blocks[-1] == range(64, 65)


def test_max_units() -> None:
    assert unit_blocks(10, max_units=3) == [range(4), range(4, 8), range(8, 10)]


@pytest.mark.parametrize(("n", "max_units"), [(-1, 64), (10, 0)])
def test_invalid_arguments(n: int, max_units: int) -> None:
    with pytest.raises(ValueError):
        unit_blocks(n, max_units)


def test_missing_units_are_named() -> None:
    error = MissingUnits([2, 5], 10)
    assert error.missing == (2, 5)
    assert "2 of 10" in str(error)
    assert "2, 5" in str(error)
