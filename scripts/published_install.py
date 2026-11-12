"""Check that a released version of sbmlsim installs from PyPI and runs.

The release is installed as `docs/installation.md` describes it, in a new
environment of a python managed by uv, with the index of the builds of the
develop branch of libroadrunner:

- `pip`: the build of libroadrunner from the index of the builds first,
  `pip install --no-deps --index-url <index> libroadrunner`, and then
  `pip install sbmlsim==<version>` from PyPI;
- `uv`: a project with the explicit index and libroadrunner pinned to it in
  its `pyproject.toml`, and `uv add sbmlsim==<version> libroadrunner`.

The installed version of sbmlsim must be the released one, and
`scripts/plain_install.py` runs a scan with it. With `--wait`, the check first
waits until PyPI serves the version. The job `published install` of the
workflow `ci-cd.yml` runs it after a release:

```bash
uv run --no-project scripts/published_install.py --installer uv --python 3.14 --version 0.21.0
```
"""

import argparse
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

#: the package index of the builds of the develop branch of libroadrunner
INDEX = "https://matthiaskoenig.github.io/roadrunner-wheels/simple/"

#: the lines of `docs/installation.md` which a uv project adds
PROJECT_INDEX = """
[tool.uv.sources]
libroadrunner = {{ index = "roadrunner-dev" }}

[[tool.uv.index]]
name = "roadrunner-dev"
url = "{index}"
explicit = true
"""

#: the script which runs a scan in the installed environment
SCAN = Path(__file__).with_name("plain_install.py")


def wait_for(version: str, seconds: float) -> None:
    """Wait until PyPI serves a version of sbmlsim.

    Args:
        version: the version.
        seconds: the time to wait at most.

    Raises:
        SystemExit: if PyPI does not serve the version in time.
    """
    url = "https://pypi.org/simple/sbmlsim/"
    deadline = time.monotonic() + seconds
    while True:
        try:
            request = urllib.request.Request(
                url, headers={"Cache-Control": "no-cache", "Accept": "text/html"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                if f"sbmlsim-{version}-" in response.read().decode():
                    return
        except urllib.error.URLError as err:
            print(f"{url}: {err}")
        if time.monotonic() > deadline:
            raise SystemExit(f"PyPI does not serve sbmlsim {version} after {seconds} s")
        time.sleep(20)


def run(*args: str | Path, cwd: Path | None = None) -> None:
    """Run a command, failing the check when it fails."""
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=cwd)


def install(*args: str | Path, cwd: Path | None = None, attempts: int = 10) -> None:
    """Run an install, again while the mirrors of PyPI do not serve the release yet.

    PyPI lists a new release in its simple index before every mirror of its
    CDN serves it: an install right after the publication, on another
    mirror than the one `wait_for` asked, does not find the version.
    """
    for attempt in range(1, attempts + 1):
        try:
            run(*args, cwd=cwd)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"the install failed, again in 30 s ({attempt}/{attempts})")
            time.sleep(30)


def venv_python(venv: Path) -> Path:
    """Get the python of a virtual environment."""
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def check(installer: str, python: str, version: str, index: str) -> None:
    """Install the release with `installer` and run a scan with it.

    Args:
        installer: `pip` or `uv`.
        python: the python version.
        version: the released version of sbmlsim.
        index: the package index of the builds of libroadrunner.
    """
    os.environ.update({"UV_NO_CACHE": "1", "UV_PYTHON_PREFERENCE": "only-managed"})
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        requirement = f"sbmlsim=={version}"
        if installer == "uv":
            run("uv", "init", "--bare", "--python", python, cwd=work)
            with (work / "pyproject.toml").open("a", encoding="utf-8") as f:
                f.write(PROJECT_INDEX.format(index=index))
            install(
                "uv", "add", "--python", python, requirement, "libroadrunner", cwd=work
            )
            venv = work / ".venv"
        else:
            venv = work / "venv"
            run("uv", "venv", "--seed", "--python", python, venv)
            # the build of libroadrunner from the index of the builds first,
            # as `docs/installation.md` describes it: its numpy comes from
            # PyPI with sbmlsim
            install(
                venv_python(venv),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--index-url",
                index,
                "libroadrunner",
            )
            install(
                venv_python(venv),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                requirement,
            )
        python_path = venv_python(venv)
        run(
            python_path,
            "-I",
            "-c",
            "import importlib.metadata as m, sys; "
            "v = m.version('sbmlsim'); "
            "print('sbmlsim', v, 'libroadrunner', m.version('libroadrunner')); "
            f"sys.exit(v != {version!r})",
            cwd=work,
        )
        os.environ.setdefault("MPLBACKEND", "Agg")
        run(python_path, "-I", SCAN, cwd=work)


def main() -> None:
    """Parse the command line and run the check."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", required=True, help="the released version")
    parser.add_argument("--installer", choices=["uv", "pip"], required=True)
    parser.add_argument("--python", default="3.14")
    parser.add_argument("--index", default=INDEX)
    parser.add_argument(
        "--wait", type=float, default=0, help="seconds to wait for PyPI"
    )
    args = parser.parse_args()
    if args.wait:
        wait_for(args.version, args.wait)
    check(args.installer, args.python, args.version, args.index)


if __name__ == "__main__":
    main()
