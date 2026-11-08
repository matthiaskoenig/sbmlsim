"""Test that the memory the C allocator holds free goes back to the system."""

import ast
import ctypes
import sys
from pathlib import Path

import pytest

from sbmlsim import memory


def _rss_mib() -> int:
    """Get the resident memory of the process in MiB (linux)."""
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1]) // 1024
    raise AssertionError("no VmRSS")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc only")
def test_freed_small_allocations_go_back_to_the_system() -> None:
    """The heap of many freed small allocations, as a compilation leaves it, shrinks."""
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    blocks = [libc.malloc(1024) for _ in range(200_000)]
    for block in blocks:
        ctypes.memset(block, 1, 1024)
    # the last block stays, so the freed ones are not at the top of the heap
    for block in blocks[:-1]:
        libc.free(block)
    held = _rss_mib()
    memory.release_freed_memory()
    assert _rss_mib() < held - 100
    libc.free(blocks[-1])


def test_nothing_happens_without_glibc(monkeypatch: pytest.MonkeyPatch) -> None:
    memory._malloc_trim.cache_clear()
    monkeypatch.setattr(sys, "platform", "darwin")
    try:
        assert memory._malloc_trim() is None
        memory.release_freed_memory()
    finally:
        memory._malloc_trim.cache_clear()


def test_the_module_imports_the_standard_library_only() -> None:
    path = Path(memory.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names)
