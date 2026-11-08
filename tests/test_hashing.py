"""The hashes which are persisted and compared across runs never change.

The values were computed with the serialization of every site before it
moved to `sbmlsim.hashing`: a changed value invalidates the runs, scans and
records written before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sbmlsim.fit.provenance import file_checksums
from sbmlsim.hashing import canonical_json_sha256, file_sha256
from sbmlsim.run.context import spec_hash
from sbmlsim.run.kinds import SplitSpec
from sbmlsim.run.scan_kinds import _digest


class _Spec(SplitSpec):
    name: str
    values: list[float]
    path: Path | None = None


#: a record with a numpy array, numpy scalars, a text which is not ASCII,
#: `None` and keys out of order
RECORD = {
    "b": np.arange(3),
    "a": [np.float64(0.1), "µ", None],
    "c": {"z": 1, "y": np.int64(2)},
}


def test_spec_hash() -> None:
    """The hash of a spec, with and without an excluded field."""
    spec = _Spec(
        name="Dose µg", values=[1.0, 0.5, float("inf")], path=Path("a/b"), units=4
    )
    assert spec_hash(spec) == (
        "ee2229e88100a1eb50bddfedd2ca4abb15d133bc17e98665f9b5aafab2cdaf8b"
    )
    assert spec_hash(spec, exclude={"path"}) == (
        "4953979633bd8feaf08d50557768e1fb309b294514662384b5aa6d593efc3f8f"
    )


def test_digest_of_a_scan_record() -> None:
    """The digest of the record of a scan, whose numpy values are lists."""
    assert _digest(RECORD) == (
        "15fafe92d598eb4491307dbace6b3c7303a01a3055193f4ed2953b52cc59a0de"
    )


def test_canonical_json_sha256() -> None:
    """Sorted keys, no whitespace, ASCII escapes and `NaN` as written by json."""
    assert canonical_json_sha256({"b": [1, float("nan")], "a": "µ"}) == (
        "570ef2ebb49bfceb5cfd077e1dc1836489adcba169658ba7d4e03e76fde37e02"
    )


def test_file_sha256(tmp_path: Path) -> None:
    """The sha256 of the bytes of a file, also as a checksum of a provenance."""
    path = tmp_path / "model.xml"
    path.write_bytes(b"<sbml>\r\n\xc2\xb5</sbml>\n")
    digest = "e656baeefb9b3007691b3f5899d8a09eb41870164fba5d15df503de964db701e"
    assert file_sha256(path) == digest
    assert file_checksums([path], root=tmp_path) == {"model.xml": digest}
