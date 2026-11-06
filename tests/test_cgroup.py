"""Tests of the CPU quota of the cgroup."""

import os
from pathlib import Path

import pytest

from sbmlsim import cgroup, parallel
from sbmlsim.fit.runner import resolve_n_cores


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _v2(tmp_path: Path, cgroup_path: str = "/", **files: str) -> int | None:
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{cgroup_path}\n", encoding="utf-8")
    for name, text in files.items():
        _write(tmp_path / "root" / name.replace("__", "/"), text)
    (tmp_path / "root").mkdir(exist_ok=True)
    return cgroup.cpu_quota(tmp_path / "root", proc)


@pytest.mark.parametrize(
    ("content", "expected"),
    [("150000 100000", 1), ("400000 100000", 4), ("50000 100000", 1)],
)
def test_v2_quota_is_floored(tmp_path: Path, content: str, expected: int) -> None:
    assert _v2(tmp_path, **{"cpu.max": content}) == expected


def test_v2_max_is_no_quota(tmp_path: Path) -> None:
    assert _v2(tmp_path, **{"cpu.max": "max 100000"}) is None


def test_v2_smallest_quota_of_the_path_wins(tmp_path: Path) -> None:
    result = _v2(
        tmp_path,
        "/a/b",
        **{"a__cpu.max": "200000 100000", "a__b__cpu.max": "max 100000"},
    )
    assert result == 2


@pytest.mark.parametrize(("quota", "expected"), [("300000", 3), ("-1", None)])
def test_v1_cfs_quota(tmp_path: Path, quota: str, expected: int | None) -> None:
    root = tmp_path / "root"
    _write(root / "cpu" / "cpu.cfs_quota_us", quota)
    _write(root / "cpu" / "cpu.cfs_period_us", "100000")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("4:cpu,cpuacct:/x\n", encoding="utf-8")
    assert cgroup.cpu_quota(root, proc) == expected


def test_no_files_is_no_quota(tmp_path: Path) -> None:
    assert cgroup.cpu_quota(tmp_path / "root", tmp_path / "proc") is None


def test_available_cpus_takes_the_smaller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 8)
    monkeypatch.setattr(cgroup, "cpu_quota", lambda: 2)
    assert cgroup.available_cpus() == 2
    monkeypatch.setattr(cgroup, "cpu_quota", lambda: None)
    assert cgroup.available_cpus() == 8


def test_resolve_workers_uses_the_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cgroup, "available_cpus", lambda: 3)
    assert parallel.resolve_workers(None, 10_000) == 3
    assert resolve_n_cores(None) == 2


@pytest.mark.parametrize(
    "content",
    [
        "garbage",
        "1 2 3",
        "",
        "inf 100000",
        "nan 100000",
        "-inf 100000",
        "100000 inf",
        "100000 nan",
        "100000 0",
        "0 100000",
        "-100000 100000",
    ],
)
def test_v2_malformed_quota_is_no_quota(tmp_path: Path, content: str) -> None:
    assert _v2(tmp_path, **{"cpu.max": content}) is None


def test_v2_directory_in_place_of_the_file_is_no_quota(tmp_path: Path) -> None:
    (tmp_path / "root" / "cpu.max").mkdir(parents=True)
    assert _v2(tmp_path) is None


def test_v2_malformed_child_keeps_the_quota_of_its_parent(tmp_path: Path) -> None:
    result = _v2(
        tmp_path,
        "/a/b",
        **{"a__cpu.max": "300000 100000", "a__b__cpu.max": "inf 100000"},
    )
    assert result == 3


def test_v2_parent_quota_below_the_one_of_its_child_wins(tmp_path: Path) -> None:
    result = _v2(
        tmp_path,
        "/a/b",
        **{"a__cpu.max": "200000 100000", "a__b__cpu.max": "400000 100000"},
    )
    assert result == 2


@pytest.mark.parametrize("quota", ["inf", "nan", "garbage", ""])
def test_v1_malformed_quota_is_no_quota(tmp_path: Path, quota: str) -> None:
    root = tmp_path / "root"
    _write(root / "cpu" / "cpu.cfs_quota_us", quota)
    _write(root / "cpu" / "cpu.cfs_period_us", "100000")
    assert cgroup.cpu_quota(root, tmp_path / "proc_cgroup") is None
