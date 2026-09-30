"""Tests of the ids of the units, the inputs and the outputs of a network."""

import re

import pytest

from sbmlsim.sciml.network import (
    element_id,
    index_id,
    input_id,
    output_id,
    parse_io_id,
    unit_id,
)

#: an SBML `SId`
SID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def test_the_ids_of_the_design() -> None:
    """The ids are the ones of the table of the design."""
    assert element_id("net1", "layer1", "weight", (0, 1)) == "net1__layer1__weight__0_1"
    assert unit_id("net1", "tanh_1", (3,)) == "net1__tanh_1__3"
    assert input_id("net1", 0, (1,)) == "net1__input0__1"
    assert input_id("net1", 0) == "net1__input0"
    assert output_id("net1", 0, (0,)) == "net1__output0__0"


def test_the_index_of_an_id() -> None:
    """The axes are joined by `_`, a value without axes has the index `0`."""
    assert index_id((0, 1, 2)) == "0_1_2"
    assert index_id((7,)) == "7"
    assert index_id(()) == "0"
    assert unit_id("net1", "flatten", ()) == "net1__flatten__0"


@pytest.mark.parametrize(
    "sid",
    [
        unit_id("net1", "block.0", (1, 2)),
        unit_id("net1", "layer-1", (0,)),
        input_id("net.1", 2, (0, 0)),
        output_id("net1", 1, (3,)),
    ],
)
def test_an_id_is_an_sid(sid: str) -> None:
    """A character which is not part of an `SId` is replaced."""
    assert SID.fullmatch(sid)


def test_a_negative_position_or_index() -> None:
    """An id of a negative position or axis does not exist."""
    with pytest.raises(ValueError, match=r"Network 'net1'.*input '-1'"):
        input_id("net1", -1, (0,))
    with pytest.raises(ValueError, match=r"Network 'net1'.*output '-2'"):
        output_id("net1", -2, (0,))
    with pytest.raises(ValueError, match=r"\(0, -1\).*negative"):
        unit_id("net1", "tanh", (0, -1))


@pytest.mark.parametrize(
    ("kind", "k", "index"),
    [
        ("input", 0, (1,)),
        ("input", 12, (0, 3, 10)),
        ("input", 1, None),
        ("output", 0, (0,)),
        ("output", 3, (2, 1)),
    ],
)
def test_an_id_is_read_back(kind: str, k: int, index: tuple[int, ...] | None) -> None:
    """The position and the index of an id are the ones it was built from."""
    sid = (
        input_id("net1", k, index)
        if kind == "input"
        else output_id("net1", k, index or ())
    )
    assert parse_io_id("net1", kind, sid) == (k, index)


@pytest.mark.parametrize(
    "sid",
    [
        "net1__input0__",
        "net1__input__0",
        "net1__inputs0__0",
        "net1__input0__a",
        "net1__input0__0_",
        "net1__input0__0__1",
        "net1_input0__0",
        "net2__input0__0",
        "xnet1__input0__0",
        "net1__output0__0",
        "prey",
        "",
        # an id has one spelling: ascii digits without leading zeros
        "net1__input01__0",
        "net1__input0__01",
        "net1__input0__0_00",
        "net1__input\u0661__0",
        "net1__input0__\u0661",
    ],
)
def test_an_id_which_is_not_an_input(sid: str) -> None:
    """An id which is not the id of an input names the form of one."""
    with pytest.raises(ValueError, match=r"Network 'net1'.*net1__input0__1") as excinfo:
        parse_io_id("net1", "input", sid)
    assert f"'{sid}'" in str(excinfo.value)


def test_the_id_of_a_network_with_a_pattern_character() -> None:
    """The id of the network is text and not a pattern."""
    assert parse_io_id("net.1", "input", "net.1__input0__1") == (0, (1,))
    with pytest.raises(ValueError, match="is not the id of an input"):
        parse_io_id("net.1", "input", "netx1__input0__1")
