"""The semantic cases of the SBML Test Suite on disk.

A case is a directory `cases/semantic/NNNNN/` with the model in several
encodings, `NNNNN-settings.txt` with the simulation and the tolerances,
`NNNNN-results.csv` with the expected results and `NNNNN-model.m` with the
description and the tags. `SemanticCase` reads the two metadata files, and
`SemanticSuite` is a directory of cases, downloaded from a release of the
suite and cached.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: version of the SBML Test Suite the tests and the documentation run against.
#: The release job resolves the newest release instead, see
#: `SemanticSuite.latest_version`; a scheduled job opens a pull request when
#: upstream is newer, which is where new failures are seen
SUITE_VERSION = "3.5.0"

#: releases of the suite, `{version}` is a tag such as `3.5.0`
SUITE_URL = (
    "https://github.com/sbmlteam/sbml-test-suite/releases/download/"
    "{version}/semantic_tests_v{version}.zip"
)

#: the newest release of the suite
LATEST_RELEASE_URL = (
    "https://api.github.com/repos/sbmlteam/sbml-test-suite/releases/latest"
)

#: SBML encodings of a case, the newest first: a case is run once, in the
#: newest encoding it provides. Reading the older encodings tests the reader,
#: not the simulation, and is covered by `sbmlutils`
ENCODINGS: tuple[str, ...] = (
    "l3v2",
    "l3v1",
    "l2v5",
    "l2v4",
    "l2v3",
    "l2v2",
    "l2v1",
    "l1v2",
)

#: keys of `NNNNN-model.m` which are read, all of them single line
_TAG_PATTERN = re.compile(
    r"^(componentTags|testTags|testType|generatedBy|packagesPresent):(.*)$",
    re.MULTILINE,
)


def _split(value: str) -> list[str]:
    """Split a comma separated value of a metadata file.

    Args:
        value: the text behind the key, may be empty.

    Returns:
        The entries without surrounding whitespace, empty for an empty value.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class SemanticCase:
    """A semantic case of the SBML Test Suite.

    Attributes:
        cid: identifier of the case, i.e. its directory name such as `00028`.
        path: directory of the case.
        model_path: the model which is simulated, the newest encoding present.
        start: start time of the simulation.
        duration: end time of the simulation.
        steps: number of steps, so the results have `steps + 1` points.
        variables: the columns the results are compared on, without `time`.
        amount: the variables which are compared as an amount.
        concentration: the variables which are compared as a concentration.
        absolute_tolerance: absolute tolerance of the comparison.
        relative_tolerance: relative tolerance of the comparison.
        component_tags: SBML components the case uses, e.g. `EventWithDelay`.
        test_tags: what the case tests, e.g. `NonConstantParameter`.
        test_type: kind of the case, `TimeCourse` for the cases which are run.
        generated_by: `Analytic` or `Numeric`, how the results were produced.
        packages: SBML packages the case needs, e.g. `fbc`.
    """

    cid: str
    path: Path
    model_path: Path
    start: float
    duration: float
    steps: int
    variables: list[str]
    amount: list[str]
    concentration: list[str]
    absolute_tolerance: float
    relative_tolerance: float
    component_tags: frozenset[str] = field(default_factory=frozenset)
    test_tags: frozenset[str] = field(default_factory=frozenset)
    test_type: str = ""
    generated_by: str = ""
    packages: frozenset[str] = field(default_factory=frozenset)

    @staticmethod
    def settings(path: Path) -> dict[str, str]:
        """Read the settings file of a case.

        Args:
            path: `NNNNN-settings.txt` of the case.

        Returns:
            The keys of the file with their raw values, values may be empty.
        """
        settings: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if _:
                settings[key.strip()] = value.strip()
        return settings

    @staticmethod
    def tags(path: Path) -> dict[str, str]:
        """Read the tags of the description of a case.

        The description is prose with a header of `key: value` lines; only the
        single line keys of the header are read, i.e. not the synopsis.

        Args:
            path: `NNNNN-model.m` of the case.

        Returns:
            The tag keys with their raw values.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        return {key: value.strip() for key, value in _TAG_PATTERN.findall(text)}

    @classmethod
    def from_directory(cls, path: Path) -> SemanticCase | None:
        """Read a case from its directory.

        A case which is not a timecourse simulation is not read: the flux
        balance cases have no duration and the stochastic cases are not part
        of the semantic suite.

        Args:
            path: directory of the case, named after its identifier.

        Returns:
            The case, or `None` if it is not a timecourse case or has no
            model in a known encoding.
        """
        cid = path.name
        settings_path = path / f"{cid}-settings.txt"
        if not settings_path.exists():
            logger.debug("'%s': no settings file, the case is skipped", cid)
            return None

        settings = cls.settings(settings_path)
        # a case without a duration is not a timecourse, i.e. it is a flux
        # balance case, which needs a steady state and not a simulation
        if not settings.get("duration"):
            return None

        model_path = cls.model_of(path, cid)
        if model_path is None:
            logger.debug("'%s': no model in a known encoding", cid)
            return None

        model_file = path / f"{cid}-model.m"
        tags = cls.tags(model_file) if model_file.exists() else {}

        return cls(
            cid=cid,
            path=path,
            model_path=model_path,
            start=float(settings.get("start") or 0.0),
            duration=float(settings["duration"]),
            steps=int(settings.get("steps") or 0),
            variables=_split(settings.get("variables", "")),
            amount=_split(settings.get("amount", "")),
            concentration=_split(settings.get("concentration", "")),
            absolute_tolerance=float(settings.get("absolute") or 1e-9),
            relative_tolerance=float(settings.get("relative") or 1e-6),
            component_tags=frozenset(_split(tags.get("componentTags", ""))),
            test_tags=frozenset(_split(tags.get("testTags", ""))),
            test_type=tags.get("testType", ""),
            generated_by=tags.get("generatedBy", ""),
            packages=frozenset(_split(tags.get("packagesPresent", ""))),
        )

    @staticmethod
    def model_of(path: Path, cid: str) -> Path | None:
        """Get the model of a case in the newest encoding it provides.

        Args:
            path: directory of the case.
            cid: identifier of the case.

        Returns:
            The SBML file, or `None` if the case has none.
        """
        for encoding in ENCODINGS:
            model_path = path / f"{cid}-sbml-{encoding}.xml"
            if model_path.exists():
                return model_path
        return None

    @property
    def encoding(self) -> str:
        """Get the SBML encoding which is simulated, e.g. `l3v2`."""
        match = re.search(r"-sbml-(l\d+v\d+)\.xml$", self.model_path.name)
        return match.group(1) if match else ""

    @property
    def selections(self) -> list[str]:
        """Get the roadrunner selections of the compared variables.

        A variable of the `concentration` list is the concentration of a
        species, i.e. `[S1]`, everything else is read under its own
        identifier. `time` is always selected first.
        """
        concentration = set(self.concentration)
        return [
            "time",
            *(f"[{v}]" if v in concentration else v for v in self.variables),
        ]

    def expected(self) -> pd.DataFrame:
        """Read the results a correct simulator produces.

        The time column is `time` in most cases and `Time` in others, so it is
        renamed; the columns of the variables are named as the settings name
        them.

        Returns:
            The expected results with a `time` column and one column per
            variable of the case.
        """
        df = pd.read_csv(self.path / f"{self.cid}-results.csv")
        time_columns = {c: "time" for c in df.columns if c.strip().lower() == "time"}
        return df.rename(columns=time_columns)


@dataclass(frozen=True)
class SemanticSuite:
    """The semantic cases of a release of the SBML Test Suite.

    Attributes:
        path: directory which holds the case directories.
        version: release of the suite, e.g. `3.5.0`.
    """

    path: Path
    version: str

    def __len__(self) -> int:
        """Get the number of timecourse cases of the suite."""
        return len(list(self.cases()))

    def cases(self) -> Iterator[SemanticCase]:
        """Iterate the timecourse cases of the suite, by identifier.

        Yields:
            Every case which is a timecourse simulation, in the order of the
            case identifiers.
        """
        for directory in sorted(self.path.iterdir()):
            if not directory.is_dir():
                continue
            case = SemanticCase.from_directory(directory)
            if case is not None:
                yield case

    def case(self, cid: str) -> SemanticCase:
        """Get a single case by its identifier.

        Args:
            cid: identifier of the case, e.g. `00028`.

        Returns:
            The case.

        Raises:
            ValueError: if the suite has no such timecourse case.
        """
        case = SemanticCase.from_directory(self.path / cid)
        if case is None:
            raise ValueError(f"'{cid}': no timecourse case in the suite '{self.path}'")
        return case

    # ------------------------------------------------------------------
    # the release of the suite
    # ------------------------------------------------------------------
    @staticmethod
    def cache_path(version: str) -> Path:
        """Get the directory a release of the suite is unpacked into.

        `SBMLSIM_TEST_SUITE_PATH` overrides it, e.g. for an offline machine
        which has the cases somewhere else. Otherwise it is the user cache,
        i.e. `XDG_CACHE_HOME` or `~/.cache`.

        Args:
            version: release of the suite.

        Returns:
            The directory the cases of the release live in.
        """
        override = os.environ.get("SBMLSIM_TEST_SUITE_PATH")
        if override:
            return Path(override)
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        return cache / "sbmlsim" / "test-suite" / version / "semantic"

    @classmethod
    def latest_version(cls) -> str:
        """Get the newest release of the suite from GitHub.

        Returns:
            The tag of the newest release, e.g. `3.5.0`.
        """
        import json

        with urllib.request.urlopen(LATEST_RELEASE_URL, timeout=60) as response:
            release = json.load(response)
        version: str = release["tag_name"]
        return version

    @classmethod
    def cached(cls, version: str = SUITE_VERSION) -> SemanticSuite | None:
        """Get a release of the suite if it is already on this machine.

        Args:
            version: release of the suite.

        Returns:
            The suite, or `None` if it was not downloaded yet.
        """
        path = cls.cache_path(version)
        return cls(path=path, version=version) if path.is_dir() else None

    @classmethod
    def load(cls, version: str = SUITE_VERSION) -> SemanticSuite:
        """Get a release of the suite, downloading it if it is not cached.

        Args:
            version: release of the suite.

        Returns:
            The suite with its cases unpacked in the cache.

        Raises:
            OSError: if the release cannot be downloaded.
        """
        suite = cls.cached(version)
        if suite is not None:
            return suite

        path = cls.cache_path(version)
        url = SUITE_URL.format(version=version)
        logger.info("Downloading the SBML Test Suite '%s' from '%s'", version, url)

        # unpack next to the target and move it into place, so an interrupted
        # download does not leave a directory which looks like a cached suite
        staging = path.parent / f".{path.name}.incomplete"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        archive = staging / "semantic.zip"
        try:
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(staging)
            archive.unlink()
            cases = cls._cases_dir(staging)
            path.parent.mkdir(parents=True, exist_ok=True)
            cases.replace(path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return cls(path=path, version=version)

    @staticmethod
    def _cases_dir(staging: Path) -> Path:
        """Get the directory of the case directories of an unpacked archive.

        The archive of a release holds the cases under a `semantic` directory;
        a differently packed archive is searched one level deep.

        Args:
            staging: directory the archive was unpacked into.

        Returns:
            The directory which holds the case directories.

        Raises:
            OSError: if it holds no cases.
        """
        candidates = [staging, *(p for p in staging.iterdir() if p.is_dir())]
        for candidate in candidates:
            if any(p.is_dir() and p.name.isdigit() for p in candidate.iterdir()):
                return candidate
        raise OSError(f"No case directories in the unpacked suite '{staging}'")
