"""The license texts of a model, byte for byte the files of pkdb_models."""

import hashlib
from pathlib import Path

import pytest

from sbmlsim.licenses import License, license_text, write_license

#: sha256 of the license files of the models of pkdb_models, which the texts of
#: the package have to reproduce byte for byte
CC_BY_SA_SHA256 = "28a9529c7d0bb4dc51f4bf5c116a3d16ef247a052f7591466768ddf563fd1cf5"
MIT_SHA256 = {
    # pkdb_models/models/montelukast/models/results/mit.txt
    "2026 Matthias König": (
        "ccf1f21b1b04471c7675e404c5c192ab79a0e4d803682679fb8a5f91b82c117a"
    ),
    # pkdb_models/models/hctz/models/results/mit.txt
    "2024-2026 Matthias König": (
        "47da781a2269219ead5df575b89c6d7f6c5d19c8f7efb105bb4edb930f733e32"
    ),
    # pkdb_models/models/glimepiride/models/results/mit.txt
    "2024-2025 Matthias König": (
        "87cdbfc2a95335ffb545c8af6f38c24dfa024b8ce8ae9f48828717d470ca7f19"
    ),
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_filenames() -> None:
    assert License.CC_BY_SA_4_0.filename == "cc-by-sa-4.0.txt"
    assert License.MIT.filename == "mit.txt"


def test_cc_by_sa_text() -> None:
    text = license_text(License.CC_BY_SA_4_0)
    assert text.startswith("Attribution-ShareAlike 4.0 International\n")
    assert sha256(text.encode("utf-8")) == CC_BY_SA_SHA256


@pytest.mark.parametrize("copyright", MIT_SHA256)
def test_mit_text(copyright: str) -> None:
    text = license_text(License.MIT, copyright)
    assert text.splitlines()[0] == f"Copyright (c) {copyright}"
    assert sha256(text.encode("utf-8")) == MIT_SHA256[copyright]


def test_mit_texts_differ_only_in_the_copyright() -> None:
    texts = [license_text(License.MIT, copyright) for copyright in MIT_SHA256]
    assert len({text.split("\n", 1)[1] for text in texts}) == 1
    assert "Permission is hereby granted" in texts[0]


def test_license_by_its_value(tmp_path: Path) -> None:
    assert license_text("mit", "2026 Matthias König") == license_text(
        License.MIT, "2026 Matthias König"
    )
    assert write_license(tmp_path, "cc-by-sa-4.0") == tmp_path / "cc-by-sa-4.0.txt"
    with pytest.raises(ValueError, match="is not a valid License"):
        license_text("lgpl-3.0")


def test_mit_requires_copyright() -> None:
    with pytest.raises(ValueError, match="copyright"):
        license_text(License.MIT)


def test_cc_by_sa_has_no_copyright() -> None:
    with pytest.raises(ValueError, match="no copyright line"):
        license_text(License.CC_BY_SA_4_0, "2026 Matthias König")


@pytest.mark.parametrize(
    "copyright",
    [
        "",
        "  ",
        "2026 Matthias König\n2025 Somebody",
        "Copyright (c) 2026 Matthias König",
    ],
)
def test_invalid_copyright(copyright: str) -> None:
    with pytest.raises(ValueError, match="copyright"):
        license_text(License.MIT, copyright)


def test_write_cc_by_sa(tmp_path: Path) -> None:
    path = write_license(tmp_path, License.CC_BY_SA_4_0)
    assert path == tmp_path / "cc-by-sa-4.0.txt"
    assert sha256(path.read_bytes()) == CC_BY_SA_SHA256


@pytest.mark.parametrize("copyright", MIT_SHA256)
def test_write_mit(tmp_path: Path, copyright: str) -> None:
    """The file is the one of the model byte for byte, on every platform."""
    path = write_license(tmp_path, License.MIT, copyright)
    assert path == tmp_path / "mit.txt"
    content = path.read_bytes()
    assert b"\r" not in content
    assert sha256(content) == MIT_SHA256[copyright]


def test_write_replaces_existing_file(tmp_path: Path) -> None:
    (tmp_path / "mit.txt").write_text("outdated", encoding="utf-8")
    path = write_license(tmp_path, License.MIT, "2026 Matthias König")
    assert sha256(path.read_bytes()) == MIT_SHA256["2026 Matthias König"]


def test_write_into_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_license(tmp_path / "missing", License.CC_BY_SA_4_0)
