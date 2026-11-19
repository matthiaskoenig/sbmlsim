"""Vendor the components of sbmlsim-web which the web app of a model embeds.

    uv run python scripts/web_assets.py            # the version of VERSION below
    uv run python scripts/web_assets.py --check    # the vendored files are the ones of VERSION

The web app of a model (`sbmlsim.webapp`) is one HTML file which runs without
the network, so it embeds the components of sbmlsim-web: the IIFE build of
the package of npm (`dist/sbmlsim-web.iife.js`, the global `sbmlsimWeb`) and
its license. They are downloaded from the registry of npm at the pinned
version, the tarball checked against the integrity the registry gives it
(the sha512 which npm and the provenance of its release record), and written
into `src/sbmlsim/resources/web/sbmlsim-web/` with `VERSION.json`, the version,
the integrity of the tarball and the sha256 of every vendored file, which the
tests compare (`tests/webapp/test_page.py`). A new release of sbmlsim-web is
vendored by raising `VERSION` and running the script; the files are never
edited by hand.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

#: the version of sbmlsim-web which is vendored
VERSION = "0.5.0"
#: the files of the package which are vendored, by their path in its tarball
FILES = {
    "package/dist/sbmlsim-web.iife.js": "sbmlsim-web.iife.js",
    "package/LICENSE": "LICENSE",
}
REGISTRY = "https://registry.npmjs.org/sbmlsim-web"
TARGET = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "sbmlsim"
    / "resources"
    / "web"
    / "sbmlsim-web"
)


def _get(url: str) -> bytes:
    """Get the bytes of a URL of the registry."""
    request = urllib.request.Request(url, headers={"User-Agent": "sbmlsim-web-assets"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def vendored(version: str) -> tuple[dict[str, bytes], dict[str, object]]:
    """Get the vendored files of a version and the record of them.

    Raises:
        ValueError: if the tarball is not the one of the integrity of the
            registry or misses a file.
    """
    meta = json.loads(_get(f"{REGISTRY}/{version}"))
    integrity = meta["dist"]["integrity"]
    algorithm, digest = integrity.split("-", 1)
    if algorithm != "sha512":
        raise ValueError(f"the integrity of sbmlsim-web {version} is no sha512")
    tarball = _get(meta["dist"]["tarball"])
    if base64.b64encode(hashlib.sha512(tarball).digest()).decode() != digest:
        raise ValueError(f"the tarball of sbmlsim-web {version} is not the one of npm")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        for member, name in FILES.items():
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"sbmlsim-web {version} has no {member}")
            files[name] = extracted.read()
    record = {
        "package": "sbmlsim-web",
        "version": version,
        "integrity": integrity,
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    return files, record


def main() -> int:
    """Vendor or check the files, see the module."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only compare")
    args = parser.parse_args()
    files, record = vendored(VERSION)
    if args.check:
        stale = [
            name
            for name, data in files.items()
            if not (TARGET / name).exists() or (TARGET / name).read_bytes() != data
        ]
        if stale:
            print(f"stale: {stale}; run scripts/web_assets.py", file=sys.stderr)
            return 1
        return 0
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (TARGET / name).write_bytes(data)
    (TARGET / "VERSION.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"sbmlsim-web {VERSION} vendored into {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
