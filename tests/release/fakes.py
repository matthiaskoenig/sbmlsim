"""The fakes of the tests of the release: the tools and the publishing.

`FakeRunner` runs git and fakes uv, the analyses of the run layer, docker and
gh by writing what they would write; `FakePublisher` records the calls of the
publishing.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.release.source import ReleaseError, Runner, run
from tests.project_toy import TOY_PATH

VERSION = "0.2.0"
PYPI = '{ registry = "https://pypi.org/simple" }'
#: the sbmlsim of the lock of the source repository (`tests/data/release/uv.lock`)
SBMLSIM = "0.8.0"


def lock_text() -> str:
    """A lock of the release with the core libraries from PyPI."""
    packages = [("toy-model", VERSION, '{ editable = "." }')]
    packages += [("sbmlsim", SBMLSIM, PYPI), ("sbmlutils", "0.9.0", PYPI)]
    blocks = [
        f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {source}\n'
        for name, version, source in packages
    ]
    return 'version = 1\nrequires-python = ">=3.14"\n\n' + "\n".join(blocks)


class FakeRunner:
    """Runs git; fakes uv, the analyses of the run layer, docker and gh."""

    def __init__(
        self,
        fail_on: str | None = None,
        interrupt_on: str | None = None,
        factory_changes: dict[str, str] | None = None,
    ) -> None:
        """Fake the tools.

        Args:
            fail_on: a part of the command which fails.
            interrupt_on: a part of the command which is interrupted.
            factory_changes: files the factory writes otherwise than the
                package has them, by their path relative to the package.
        """
        self.commands: list[list[str]] = []
        self.fail_on = fail_on
        self.interrupt_on = interrupt_on
        self.factory_changes = factory_changes or {}

    def __call__(self, args: Sequence[str], cwd: Path, *, stream: bool = False) -> str:
        command = [str(a) for a in args]
        self.commands.append(command)
        joined = " ".join(command)
        if self.fail_on and self.fail_on in joined:
            raise ReleaseError(f"'{joined}' failed in {cwd}")
        if self.interrupt_on and self.interrupt_on in joined:
            raise KeyboardInterrupt
        if command[0] == "git":
            return run(command, cwd)
        if command[:2] == ["uv", "lock"]:
            (cwd / "uv.lock").write_text(lock_text(), encoding="utf-8")
        elif "sbmlsim.run" in command:
            self._analysis(command[5], command[6], cwd / command[8], cwd)
        return ""

    def _analysis(self, kind: str, spec: str, out: Path, cwd: Path) -> None:
        if kind == "factory":
            models = out / "models"
            models.mkdir(parents=True, exist_ok=True)
            # the factory builds the models of the package
            exported = sorted((cwd / "src").glob("**/models/toy.xml"))
            source = exported[0] if exported else TOY_PATH / "models" / "toy.xml"
            shutil.copy(source, models / "toy.xml")
            for relative, text in self.factory_changes.items():
                self._write(out / relative, text)
            changed = sorted(self.factory_changes)
            (out / "summary.json").write_text(
                json.dumps({"changed": changed}), encoding="utf-8"
            )
            (models / "toy.md").write_text("# toy\n", encoding="utf-8")
            (out / "toy.omex").write_text("omex\n", encoding="utf-8")
            self._write(out / "figures" / "toy_model.png", "png")
        elif kind == "simulate":
            self._write(out / "Decay" / "Decay.html")
            self._write(out / "index.html")
        elif kind == "report":
            # a report per adopted fit, from the records of the exported package
            for record in (cwd / "src").glob("**/parameters.json"):
                report = out / record.parent.name
                self._write(report / "index.html")
                self._write(report / "report.pdf", "%PDF")
        elif kind == "credibility":
            assert json.loads((cwd / spec).read_text(encoding="utf-8"))["source_commit"]
            self._write(out / "credibility" / "index.html")
        (out / "run.json").write_text("{}", encoding="utf-8")

    @staticmethod
    def _write(path: Path, text: str = "<html/>") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def ran(self, word: str) -> bool:
        """Tell whether a command with this word ran."""
        return any(word in command for command in self.commands)


class FakePublisher:
    """Records the calls of the publishing steps."""

    def __init__(self) -> None:
        self.calls: list[tuple[ReleaseSettings, Path, str, Path]] = []

    def __call__(
        self,
        settings: ReleaseSettings,
        release_path: Path,
        version: str,
        archive: Path,
        runner: Runner,
    ) -> list[str]:
        self.calls.append((settings, release_path, version, archive))
        return []
